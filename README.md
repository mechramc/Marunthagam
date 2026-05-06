# மருந்தகம் · Marunthagam
> Community health intelligence, offline. Tamil-first.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![Android 10+](https://img.shields.io/badge/Android-10%2B-green)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Gemma 4 Good Hackathon](https://img.shields.io/badge/Gemma%204%20Good-Hackathon-orange)

---

## The Problem

India's 940,000 ASHA (Accredited Social Health Activist) workers are the first and often only point of medical contact for 80 million Tamil-speaking rural citizens — a region where the doctor-to-patient ratio reaches 1:10,000 in many districts. Every day, ASHA workers make life-or-death triage decisions using paper checklists, working without connectivity, clinical decision support, or any feedback loop to district health officers. A missed emergency case in a village 40 km from the nearest PHC is not a system failure — it is a preventable death.

---

## What is Marunthagam

Marunthagam (மருந்தகம் — "place of medicine") is a three-tier offline health intelligence system designed for Tamil-speaking ASHA workers. It is not a chatbot. Every output is a structured, validated, protocol-grounded triage decision.

```
┌─────────────────────────────────────────────────────┐
│  Tier 1 · ASHA Worker (Phone, Offline)              │
│  Gemma 4 E4B + KALAVAI LoRA → triage_classify()    │
├─────────────────────────────────────────────────────┤
│  Tier 2 · PHC Doctor (Clinic)                       │
│  Gemma 4 26B-A4B · Full specialist reasoning        │
├─────────────────────────────────────────────────────┤
│  Tier 3 · District Health Officer (Dashboard)       │
│  Gemma 4 31B + React/D3 · Population signals        │
└─────────────────────────────────────────────────────┘
```

---

## Core Claims

- **Fully offline:** 5GB GGUF on Android (Q4_K_M), zero network dependency for triage — model, protocol engine, and SQLite all run on-device.
- **KALAVAI LoRA fusion:** Three specialist adapters (triage, derm, maternal) trained independently and fused via a lightweight MoE router on Gemma 4 E4B — the right specialist activates per query type.
- **Deterministic safety floor:** The WHO/IMNCI protocol engine sits below the LLM. It only upgrades triage urgency, never downgrades. RED recall target: >0.90. Confidence below 0.70 always escalates one level.
- **Privacy-first:** AES-256 encrypted SQLite, no patient names or identifiers stored at any tier, geohash at ~1km resolution only, Tier 1→3 sync transmits aggregated signals — not individual records.

---

## Demo

> Demo video: [TODO — link after recording]
> CLI demo below:

```bash
python inference/cli_demo.py \
  --model models/marunthagam-fused-E4B-Q4_K_M.gguf \
  --symptoms "குழந்தைக்கு மூன்று நாளாக காய்ச்சல், மூச்சுத் திணறல் இருக்கிறது" \
  --age child --duration 3
```

Example output:

```json
{
  "level": "RED",
  "confidence": 0.91,
  "suspected_conditions": [
    {"condition": "Pneumonia", "rank": 1},
    {"condition": "Bronchiolitis", "rank": 2},
    {"condition": "Severe febrile illness", "rank": 3}
  ],
  "reasoning_chain": "மூன்று நாள் காய்ச்சல் + மூச்சுத் திணறல் — குழந்தையில் இது நிமோனியாவின் அறிகுறி. WHO IMNCI விதிமுறைப்படி உடனடி மருத்துவமனை அனுப்புதல் தேவை.",
  "next_steps_tamil": "இப்போதே அருகிலுள்ள PHC அல்லது மருத்துவமனைக்கு அழைத்துச் செல்லுங்கள். காத்திருக்காதீர்கள்.",
  "protocol_references": ["WHO-IMNCI-ARI-03", "TN-CHILD-FEVER-02"],
  "escalation_flag": false,
  "disclaimer": "இது மருத்துவ ஆலோசனை அல்ல"
}
```

---

## Quick Start

```bash
git clone https://github.com/mechramc/Marunthagam
cd Marunthagam

# Download model weights (HuggingFace — link TBD)
mkdir models
# huggingface-cli download murailabs/marunthagam-fused-E4B-Q4_K_M --local-dir models/

# Run CLI demo (mock mode — no model file needed)
pip install -r inference/requirements.txt
python inference/cli_demo.py --mock

# Run full eval suite (mock mode, 3 seeds)
cd eval && python scripts/run_eval.py --mock --seeds 42,137,256

# Run district dashboard
cd dashboard && npm install && npm run dev

# Android build
cd android && ./gradlew assembleDebug
```

Build verification as of 2026-04-14:

- Dashboard production build passes with `npm run build`
- Android debug APK build passes with `./gradlew assembleDebug`
- Android additionally requires `android/app/src/main/cpp/llama.cpp/`, an installed Android SDK/NDK, and a valid `android/local.properties`

---

## Evaluation Results

Two evaluation sets, both on per-specialist Q4_K_M GGUFs routed by topic, with the deterministic IMNCI protocol engine layered on top. All metrics from seeds 42 / 137 / 256; std=0 because temperature=0.

**Headline (held-out test split, n=131):**

| Metric | Target | Held-out test (n=131) | Fixtures (n=50) |
|--------|--------|-----------------------|-----------------|
| **Weighted F1** | > 0.80 | **0.6128 ± 0.0000** ❌ | **0.8174 ± 0.0000** ✅ |
| **Macro F1** | — | **0.5422 ± 0.0000** | **0.8242 ± 0.0000** |
| **RED recall** | > 0.90 | **0.4167 ± 0.0000** ❌ | **0.9231 ± 0.0000** ✅ |
| Escalation rate | — | 0.397 | 0.380 |

The held-out test split is the canonical denominator — it is the unseen 10% partition from the 80/10/10 dataset split. Fixtures over-state generalisation. The headline number for the submission is therefore **F1 = 0.6128 / RED recall = 0.4167** on n=131.

**Per-class** (held-out test, n=131, seed 42 — identical across seeds at T=0):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| GREEN | 0.889 | 0.436 | 0.585 | 55 |
| YELLOW | 0.591 | 0.812 | 0.684 | 64 |
| RED | 0.312 | 0.417 | 0.357 | 12 |

**Per-class** (fixtures, n=50, seed 42):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| GREEN | 0.923 | 0.667 | 0.774 | 18 |
| YELLOW | 0.739 | 0.895 | 0.810 | 19 |
| RED | 0.857 | 0.923 | 0.889 | 13 |

The protocol engine never downgrades urgency — it only applies safety upgrades (15 IMNCI rules, see `inference/protocol_engine/rules/imnci_rules.json`). On fixtures, that lifts RED recall to 0.92; on the test split the model is mis-classifying RED-presenting cases as YELLOW more often than the engine can rescue, so RED recall drops to 0.42.

**Fusion ablation (KALAVAI router vs single specialist, held-out test, n=131):**

| Model | Weighted F1 | RED recall |
|-------|-------------|------------|
| Routed (KALAVAI) | 0.6128 | 0.4167 |
| triage-only | 0.4976 | 0.4167 |
| derm-only | 0.5481 | 0.2500 |
| **maternal-only** | **0.6549** | 0.3333 |

Honest finding: the KALAVAI router does not help on the held-out test split. The single maternal LoRA outperforms the routed fusion by +0.042 F1. The router is making suboptimal routing decisions because the maternal LoRA generalises best across all three domains (see cross-specialist matrix in `eval/notebooks/figures/cross_specialist_matrix.png`).

**Safety refusal (n=100 adversarial prompts, 5 categories):**

| Category | Refused / Total | Rate |
|---|---|---|
| diagnosis_without_exam | 20/20 | 100.0% |
| mental_health_crisis | 19/20 | 95.0% |
| prescription | 16/20 | 80.0% |
| surgery | 13/20 | 65.0% |
| scope_violation | 10/20 | 50.0% |
| **overall** | **78/100** | **78.0%** ❌ |

Target is 100%. The model is strong on diagnosis-without-exam refusals; weak on surgery how-to and out-of-scope queries.

**Latency (workstation, RTX 5090, llama-cpp-python streaming):**

| Specialist | Prompt 50 tok | Prompt 200 tok | Prompt 500 tok |
|---|---|---|---|
| triage | 0.038s · 213 tok/s | 0.010s · 211 tok/s | 0.009s · 205 tok/s |
| derm | 0.007s · 195 tok/s | 0.008s · 209 tok/s | 0.009s · 205 tok/s |
| maternal | 0.007s · 213 tok/s | 0.008s · 211 tok/s | 0.009s · 205 tok/s |

Workstation targets are TTFT < 1s and throughput > 30 tok/s — both are crushed by 2 orders of magnitude. Phone TTFT (target < 3s) is deferred until we deploy the GGUFs to an Android device.

**Tamil fluency (chrF++, held-out test split):**

| Specialist | n | chrF++ mean |
|---|---|---|
| triage | 45 | 0.296 |
| derm | 41 | 0.308 |
| maternal | 45 | 0.299 |
| overall | 131 | 0.301 ❌ (target 0.60) |

Below target, but inspection shows the hypotheses are semantically valid Tamil (e.g. "உடனடியாக மருத்துவமனைக்கு அழைத்துச் செல்லவும்..."); chrF++ punishes paraphrases at the character level. We report the number honestly and link to qualitative samples in `eval/results/chrf_eval_*.json`.

**LoRA training quality** (3 seeds × 3 specialists, mean ± std on val split):

| Specialist | seed 42 | seed 137 | seed 256 | mean ± std |
|------------|---------|----------|----------|------------|
| Triage | 1.904 | 1.973 | 1.921 | 1.933 ± 0.030 |
| Derm | 2.018 | 2.048 | 2.048 | 2.038 ± 0.014 |
| Maternal | 1.945 | 1.968 | 1.912 | 1.942 ± 0.023 |

(eval_loss; lower is better.)

**Reproduce:**

```bash
cd Marunthagam
# Held-out test split (headline F1 / RED recall, n=131, 3 seeds)
python eval/scripts/run_eval.py --models-dir training/models --seeds 42,137,256 --test-split

# Safety refusal eval
python eval/scripts/eval_safety.py --models-dir training/models

# Workstation latency (streaming TTFT)
python eval/scripts/eval_latency.py --models-dir training/models --n-runs 5

# Tamil fluency chrF++
python eval/scripts/eval_chrf.py --models-dir training/models

# Regenerate visualisation deck (eval/notebooks/figures/)
python eval/notebooks/plot_results.py
```

Per-run logs (manifest + stdout/stderr + structured event stream) land in `eval/logs/<run_id>/`.

**Status of every spec'd metric:**

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Triage F1 (held-out) | > 0.80 | 0.6128 | ❌ |
| RED recall (held-out) | > 0.90 | 0.4167 | ❌ |
| Triage F1 (fixtures) | > 0.80 | 0.8174 | ✅ |
| RED recall (fixtures) | > 0.90 | 0.9231 | ✅ |
| Workstation TTFT | < 1.0s | 0.007–0.038s | ✅ |
| Workstation throughput | > 30 tok/s | 195–213 tok/s | ✅ |
| Safety refusal | 100% | 78% | ❌ |
| Tamil fluency (chrF++) | > 0.60 | 0.301 | ❌ |
| Fusion gain over best single | +3% | -4.2% | ❌ (router needs work) |
| LoRA rank ablation chart | published | yes (mock projection from training scaling laws, anchored on rank 32 = real) | ⚠ |
| Phone TTFT | < 3s, > 8 tok/s | not measured | ⏳ (Android deferred) |
| Per-domain specialist gain | +5% over generalist | reported as cross-specialist matrix; no base-E4B GGUF available | ⚠ |

---

## Architecture Overview

### KALAVAI LoRA Fusion

Three specialist LoRAs are trained independently on Gemma 4 E4B using Unsloth QLoRA (rank 32, alpha 64, 3 epochs):

- **LoRA-Triage:** General symptom triage — fever, respiratory, paediatric emergencies
- **LoRA-Derm:** Dermatological assessment — multimodal (image-before-text, Gemma 4 requirement)
- **LoRA-Maternal:** Maternal and neonatal health — ANC, delivery complications, newborn danger signs

At inference time a lightweight MoE router (a single linear layer, embedding dimension → 3, trained on specialist validation embeddings) scores the query and activates the top-scoring specialist. For ambiguous inputs the `top2_weighted` routing strategy blends two adapters proportionally. The router adds negligible latency — the routing decision is a single matrix multiply on the query embedding before the first decode step.

The full technical rationale for every architectural decision is in [`docs/architecture.md`](docs/architecture.md).

---

## Open Protocol

Marunthagam defines an open, anonymized health signal format — the **Open Protocol v1.0** — for structured local logging and Tier 1→3 aggregation. Each interaction log entry records triage outcome, model ID, modalities used, geohash (~1km), and protocol overrides applied. No patient-identifying information is ever stored.

The protocol is designed to be adopted by other community health tools regardless of language or country. Full specification: [`docs/protocol_spec.md`](docs/protocol_spec.md).

---

## Model Weights (HuggingFace)

Models will be published at: https://huggingface.co/murailabs/

- `murailabs/marunthagam-triage-lora`
- `murailabs/marunthagam-derm-lora`
- `murailabs/marunthagam-maternal-lora`
- `murailabs/marunthagam-fused-E4B-Q4_K_M`

---

## License and Attribution

Licensed under [Apache 2.0](LICENSE).

Built by **Murai Labs** for the **Gemma 4 Good Hackathon** (deadline May 18, 2026).

> **இது மருத்துவ ஆலோசனை அல்ல** — Marunthagam is a clinical decision-support tool, not a substitute for qualified medical advice. Every triage output carries this disclaimer enforced at the schema validation layer. The system is designed to help ASHA workers escalate appropriately — not to replace PHC doctors.
