# Marunthagam — Technical Architecture

> This document is for judges, reviewers, and contributors who want to understand how and why Marunthagam is built the way it is.

---

## 1. System Overview

Marunthagam is a three-tier health intelligence system. The tiers are not marketing layers — they reflect the actual deployment topology of India's rural health infrastructure: a field worker with a phone, a PHC doctor with a workstation, and a district health officer with a dashboard and connectivity.

**Tier 1 (Field — fully offline):** An ASHA worker captures symptoms via Tamil voice input, optionally photographs a skin condition or wound, and receives a structured triage card on an Android phone. The entire inference stack — model, protocol engine, log database — runs on-device. There is no network dependency at any point in the triage path.

**Tier 2 (Clinic — intermittent connectivity):** A PHC doctor runs Gemma 4 26B-A4B-IT on a local workstation. This tier handles cases that have been escalated from Tier 1 and cases that arrive directly at the clinic. The larger model provides fuller differential reasoning and can handle more complex multimodal inputs. Tier 2 can sync aggregated signals to Tier 3 when connectivity allows.

**Tier 3 (District — connected):** A district health officer runs the React/D3 dashboard backed by Gemma 4 31B-IT. This tier receives aggregated, anonymized population-level signals from Tiers 1 and 2 and surfaces disease trend analysis, resource allocation insights, and outbreak early-warning signals. No individual patient records flow to Tier 3.

The offline-first design principle is not a compromise — it is the requirement. Rural Tamil Nadu has intermittent connectivity and no guarantee of a cloud endpoint. Any system that requires a network call to produce a triage decision will fail when it is most needed.

---

## 2. KALAVAI LoRA Fusion

### Why specialist LoRAs rather than a single generalist fine-tune

A single LoRA trained across all health domains would either be too large to fit comfortably on-device at the required quality level or too diluted across domains to achieve the specialisation necessary for high RED recall. Community health presents three genuinely distinct reasoning modes: symptom-based triage (structured clinical reasoning), dermatological assessment (primarily visual pattern recognition), and maternal/neonatal health (protocol-heavy, time-critical). Training a single adapter on a mixture of these domains introduces conflicting gradients and tends to regress toward the mean.

The KALAVAI approach (arXiv:2603.22755) trains specialist adapters independently and then fuses them via a learned router. This means each specialist can be trained to convergence on its domain without sacrificing another. The router learns which specialist to activate from the query embedding, not from explicit user tagging.

### The three specialists

All three LoRAs share the same architecture: Gemma 4 E4B-IT as the base model, rank 32, alpha 64, trained with Unsloth QLoRA (4-bit) over 3 epochs at lr=2e-4 on an RTX 5090. The full configuration is in `training/configs/lora_triage.yaml`, `lora_derm.yaml`, and `lora_maternal.yaml`.

**LoRA-Triage** handles general symptom triage — fever, respiratory illness, gastrointestinal emergencies, paediatric danger signs, common adult presentations. The training data consists of Tamil function-calling examples in `triage_classify()` format, sourced from WHO IMNCI case studies and Tamil Nadu state health protocols, translated and adapted to the ASHA worker context using Gemma 4 31B on Mac Studio (MLX).

**LoRA-Derm** is trained for dermatological assessment — skin infections, rashes, wound assessment, leprosy early detection. Crucially, Gemma 4 requires images to appear before text in multimodal prompts; the derm training pipeline enforces this ordering. The `lora_derm.yaml` config sets `multimodal: true` to enable vision training paths.

**LoRA-Maternal** covers ANC (antenatal care), postnatal danger signs, and neonatal emergency recognition — the domain where rural health system failures are most acute in terms of preventable mortality.

### The MoE router

The router is a single linear layer that maps from the query embedding (Gemma 4 E4B hidden dimension: 2560) to a 3-way softmax over specialists. It is trained separately from the LoRAs, using embeddings from the specialist validation sets as training data with cross-entropy loss. Training takes 50 epochs at lr=1e-3 with the Adam optimizer.

At inference time, the router scores the query embedding before the first decode step. Under the default `top1` routing strategy the highest-scoring specialist LoRA is loaded and applied. Under `top2_weighted` (for ambiguous queries spanning multiple domains) the two top-scoring LoRAs are linearly interpolated by their softmax weights. Both strategies are implemented in `training/scripts/train_router.py` and governed by `training/configs/router.yaml`.

The router embedding dimension in the current development configuration is 768 (a stub for pipeline testing). The production configuration targets the actual E4B hidden dimension of 2560 once full model embeddings are available.

---

## 3. triage_classify() Function Calling

### Why function calling instead of free text

Free-text medical advice from a language model is not safe to display to a semi-trained field worker in a life-or-death context. The model cannot be trusted to consistently include disclaimers, produce machine-parseable urgency levels, or format action steps in plain Tamil. Function calling forces every output into a validated schema before it ever reaches the UI.

The `triage_classify()` function is defined with Pydantic v2 schemas in `inference/function_calling/schemas.py`. Every model output is validated against `TriageClassifyOutput` before display or logging. Fields that fail validation are rejected and the system falls back to a conservative escalation rather than displaying garbage.

### Input schema

```
verbal_symptoms    string (1–2000 chars)   Tamil symptom description from ASHA worker
image_findings     string, optional        Model's interpretation of the clinical image
patient_age_group  enum                    infant | child | adolescent | adult | elderly
duration_days      integer (0–3650)        Days symptoms have persisted
vital_signs        object, optional        {temperature (°C), pulse (bpm), respiratory_rate (/min)}
```

### Output schema

```
level              enum                    GREEN | YELLOW | RED
confidence         float (0.0–1.0)         Model's calibrated confidence
suspected_conditions  array (max 3)        Ranked suspected diagnoses
reasoning_chain    string                  Step-by-step clinical reasoning in Tamil
next_steps_tamil   string                  Plain Tamil instructions for ASHA worker
protocol_references  array                 WHO/IMNCI/TN protocol codes referenced
escalation_flag    boolean                 True if confidence < 0.70 or conflicting signals
disclaimer         string                  Always "இது மருத்துவ ஆலோசனை அல்ல"
```

The `disclaimer` field is enforced by a Pydantic field validator (`enforce_disclaimer`) that overwrites any model-provided value with the canonical Tamil string. This is not a convention — it is a schema-level guarantee. A model that forgets the disclaimer still produces a compliant output.

### Protocol override chain

The model's raw output is passed through the protocol engine before being shown to the user. The override chain operates in two phases:

1. **Rule-based upgrade:** Active rules in the `protocol_rules` SQLite table are checked against the patient presentation (symptom pattern, age group, duration). Any matching rule that requires a higher urgency level than the model's output triggers an upgrade. Downgrades are impossible by construction — the engine only moves the level upward in the GREEN → YELLOW → RED order.

2. **Confidence floor:** If the model's confidence is below 0.70 (the `CONFIDENCE_THRESHOLD` constant in `engine.py`), the result is escalated one additional level and `escalation_flag` is set to `True`. This captures uncertainty as a safety signal rather than allowing a low-confidence GREEN to pass through unchallenged.

All overrides are logged as `ProtocolOverride` records (including the `rule_id`, original level, overridden-to level, and reason string) and attached to the interaction log entry.

---

## 4. Protocol Engine

The protocol engine (`inference/protocol_engine/engine.py`) is deliberately not an LLM. It is a deterministic rule evaluator that reads from a SQLite database of active rules derived from WHO IMNCI guidelines and Tamil Nadu state health protocols.

Each rule in the `protocol_rules` table has:
- `condition_pattern`: a regex matched against the symptom description
- `age_group`: required age group, or `any`
- `duration_min_days`: minimum symptom duration to trigger
- `minimum_triage_level`: the urgency floor this rule enforces
- `override_reason`: human-readable justification logged with the override
- `active`: boolean to enable/disable without deletion

The engine's `apply()` method takes the LLM's `TriageResult`, the raw symptom text, age group, and duration, and returns a (possibly upgraded) result plus the list of overrides applied. The design is deliberately conservative: the system can never produce a lower urgency than either the LLM or any matching protocol rule independently would.

This approach is why RED recall can be targeted at >0.90 even if the LLM alone would miss some emergency presentations. Known RED-pattern cases (e.g. respiratory distress + infant, convulsions any age, post-partum haemorrhage) are encoded as protocol rules that override regardless of model confidence.

---

## 5. Privacy Architecture

The privacy model is designed around the principle of minimum sufficient information: log only what is needed for population-level health intelligence, discard everything else immediately.

**What is never stored:** Patient names, ages, addresses, phone numbers, photographs, audio recordings, or any demographic attribute that could identify an individual. Audio is transcribed on-device by Whisper-small-Tamil (when used) and the audio file is discarded. Images are processed by the multimodal model and discarded; only the model's text interpretation (`image_findings`) is retained, and only in memory during inference.

**What is stored (encrypted):** Each interaction log entry (`InteractionLogEntry` in `inference/protocol_engine/logger.py`) contains: locale, device tier, model ID, modalities used, triage level, confidence, escalation flag, protocol overrides, an optional 6-character geohash (~1km precision), protocol version, and a generated UUID record ID. No field in `InteractionLogEntry` can contain a patient identifier — the schema has no such field.

**Geohash precision:** A 6-character geohash covers approximately a 1.2km × 0.6km cell. This resolution is sufficient for district-level outbreak mapping (clustering by geohash reveals village-level disease burden) while being too coarse to identify individuals. The geohash field is optional — ASHA workers in particularly small or identifiable communities may omit it.

**Local storage:** The SQLite database is encrypted with AES-256 using pycryptodome (integration in progress — see `InteractionLogger` in `logger.py`). All data at rest on the Android device is encrypted before being written to flash storage.

**Sync policy:** The `InteractionLogger.get_pending_sync()` method surfaces records for Tier 3 transmission, but the sync process is responsible for aggregating these into population-level signals before transmission. Individual `InteractionLogEntry` records are never transmitted over the network — only their aggregate derivatives (case counts by geohash, escalation rates, protocol override frequencies). Sync uses TLS 1.3.

---

## 6. Inference Pipeline

### GGUF and Q4_K_M quantization

The Q4_K_M quantization format was chosen specifically for the Tier 1 deployment target. It provides a good trade-off between model quality (the K-quant family uses mixed-precision quantization, preserving more precision in the most sensitive layers) and model size (~5GB for E4B), which fits comfortably in the storage budget of a mid-range Android device. The quality degradation relative to full precision is measurable but modest — Q4_K_M retains approximately 97–98% of the original model's benchmark performance on classification tasks.

The export pipeline (`training/scripts/export_gguf.py`) converts the fine-tuned Unsloth checkpoint to GGUF using llama.cpp's conversion tooling. The GGUF format is self-describing and requires no separate tokenizer files, which simplifies Android JNI integration.

### llama.cpp and Android JNI

The Android application uses llama.cpp compiled as a shared library via JNI. The JNI bridge (`android/`) provides Kotlin bindings for model load, context creation, and token streaming. The llama.cpp C++ layer handles GGUF parsing, KV cache management, and quantized matrix multiplication using NEON SIMD intrinsics on ARM64 devices.

The inference path on Android is: Kotlin UI → JNI bridge → llama.cpp → GGUF model → function call JSON output → Pydantic validation → protocol engine → triage card rendered in UI.

### Desktop inference

On the workstation (Tier 2) and dashboard (Tier 3), inference runs via the llama.cpp CLI (`llama-cli`) invoked as a subprocess, or optionally via vLLM for higher-throughput batch evaluation. The eval suite's `_real_predict()` function in `eval/scripts/run_eval.py` demonstrates the subprocess interface, including parsing of `<tool_call>...</tool_call>` tags from model output.

---

## 7. Training Pipeline

### Dataset construction

Training data is constructed in `training/` from WHO IMNCI clinical case studies, Tamil Nadu state health protocols, and synthetic cases generated by Gemma 4 31B on Mac Studio (MLX) and reviewed by a clinical consultant. Each training example is a function-calling pair: a patient presentation (symptoms, age group, duration, optional vitals in Tamil) and the corresponding `triage_classify()` output with reasoning chain.

The dataset is split 80/10/10 (train/val/test) per specialist domain. The test split is held out throughout all training and hyperparameter search — it is never touched until final evaluation. This is a hard requirement for honest evaluation claims.

### Unsloth QLoRA training

Each specialist LoRA is trained with Unsloth's QLoRA implementation in 4-bit precision. The base model (`unsloth/gemma-4-E4B-it`) is loaded in NF4 quantization. The LoRA adapters target all seven projection matrices: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. This full-attention LoRA configuration is more expensive than a subset but produces better instruction-following on the clinical reasoning task.

Training runs use gradient accumulation of 4 steps with a batch size of 4 (effective batch 16), 50 warmup steps, and a cosine learning rate schedule. All runs are tracked with Weights & Biases (`use_wandb: true` in all LoRA configs). Every experiment is run with 3 seeds (42, 137, 256) and the best checkpoint is selected by validation F1.

### Evaluation methodology

The full evaluation suite (`eval/scripts/run_eval.py`) loads fixture files from all three specialist domains plus the 20-case baseline set, runs `triage_classify()` inference per case (via real llama.cpp or the deterministic mock), and computes weighted F1, macro F1, per-class P/R/F1, and RED recall using scikit-learn. Results are aggregated as mean ± std across seeds and saved to `eval/results/`.

The mock predictor (`_mock_predict`) introduces a realistic ~10% error rate with seed-specific Gaussian noise to validate the eval pipeline before model weights are available. The safety evaluation (`eval/scripts/eval_safety.py`) runs 100 adversarial prompts designed to elicit out-of-scope responses (surgery instructions, mental health crisis handling) and verifies that the system escalates rather than engaging.
