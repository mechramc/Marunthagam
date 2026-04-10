# Marunthagam — Claude Code Configuration

> மருந்தகம் · Community health intelligence, offline.
> Gemma 4 Good Hackathon · Deadline: May 18, 2026

---

## Project Overview

Marunthagam is an **offline community health AI system** for Tamil-speaking ASHA workers in rural India. It is NOT a chatbot — it is a three-tier health intelligence system.

**Three tiers:**
- **Tier 1 (Phone):** Gemma 4 E4B-IT (Q4 GGUF, ~5GB) — fully offline, ASHA worker
- **Tier 2 (Clinic):** Gemma 4 26B-A4B-IT (Q4 GGUF, ~18GB) — PHC doctor
- **Tier 3 (Dashboard):** Gemma 4 31B-IT + React/D3 — district health officer

**Core innovation:** KALAVAI LoRA fusion — three specialist LoRAs (triage, derm, maternal) fused via MoE router on Gemma 4 E4B.

---

## Repo Structure

```
training/       Dataset construction + Unsloth fine-tuning + KALAVAI router
inference/      llama.cpp integration + function calling + protocol engine
android/        Android app (Kotlin + llama.cpp JNI + SQLite)
dashboard/      React + D3 district dashboard
eval/           Evaluation scripts + ablation notebooks
protocol/       Open Protocol JSON schema + documentation
docs/plans/     Implementation plans (YYYY-MM-DD-<name>.md)
```

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Fine-tuning | Unsloth (QLoRA on Gemma 4 E4B) |
| Training hardware | RTX 5090 (32GB VRAM) |
| Inference (phone) | llama.cpp JNI (GGUF Q4_K_M) |
| Inference (desktop) | llama.cpp or vLLM |
| Android app | Kotlin + llama.cpp JNI |
| Dashboard | React + D3.js |
| Local storage | AES-256 encrypted SQLite |
| Eval | Python — numpy, pandas, scikit-learn |
| Protocol engine | SQLite + deterministic rules (Python) |
| Dataset translation | Gemma 4 31B on Mac Studio (MLX) |

---

## Key Commands

### Training
```bash
# Install training dependencies
cd training && pip install -r requirements.txt

# Train a specialist LoRA
python scripts/train_lora.py --specialist triage --seed 42

# Train KALAVAI router
python scripts/train_router.py --config configs/router.yaml

# Export to GGUF
python scripts/export_gguf.py --checkpoint outputs/triage-lora-best
```

### Evaluation
```bash
# Run full eval suite
cd eval && python scripts/run_eval.py --model outputs/fused-E4B-Q4.gguf

# Ablation: LoRA rank comparison
python scripts/ablation_rank.py

# Triage accuracy + RED recall
python scripts/eval_triage.py --split test --model fused
```

### Dashboard
```bash
cd dashboard
npm install
npm run dev         # Local dev
npm run build       # Production build
```

### Android
```bash
cd android
./gradlew assembleDebug
./gradlew installDebug   # Install to connected device
```

### Protocol validation
```bash
cd protocol
python validate_schema.py --record examples/sample_record.json
```

---

## Function Calling Schema

The primary function is `triage_classify()`:

**Input:**
- `verbal_symptoms: string` — Tamil symptom description
- `image_findings: string` — model's image interpretation
- `patient_age_group: enum` — infant | child | adolescent | adult | elderly
- `duration_days: integer`
- `vital_signs: object` (optional) — {temperature, pulse, respiratory_rate}

**Output:**
- `level: enum` — GREEN | YELLOW | RED
- `confidence: float` — 0.0–1.0
- `suspected_conditions: array` — up to 3 ranked
- `reasoning_chain: string` — step-by-step in Tamil
- `next_steps_tamil: string` — plain Tamil instructions
- `protocol_references: array` — WHO/IMNCI/TN protocol codes
- `escalation_flag: boolean` — true if confidence < 0.7

**Protocol override rules:**
- LLM says GREEN but protocol requires YELLOW → upgrade to YELLOW, log override
- confidence < 0.7 → always escalate one level, set escalation_flag = true
- safety refusal on out-of-scope queries (surgery, mental health crisis) → escalate

---

## Data Privacy Rules

- NO patient names or identifiers stored anywhere
- Audio and images: processed ephemerally on-device, discarded after triage
- Local storage: AES-256 encrypted SQLite only
- Sync: aggregated signals only (not individual records), TLS 1.3
- Geohash at ~1km resolution only

**Every triage output must include disclaimer:**
> "இது மருத்துவ ஆலோசனை அல்ல" ("This is not medical advice")

---

## Evaluation Targets

| Metric | Target |
|--------|--------|
| Triage F1 (overall) | > 0.80 |
| RED recall (emergency) | > 0.90 |
| Per-domain specialist gain | +5% over generalist |
| Fusion gain over best specialist | +3% |
| Tamil fluency (chrF++) | > 0.60 |
| Safety refusal rate | 100% (100 adversarial prompts) |
| Phone TTFT | < 3s, > 8 tok/s |
| Workstation TTFT | < 1s, > 30 tok/s |

Always run 3 seeds (42, 137, 256) and report confidence intervals.

---

## Do's and Don'ts

**Do:**
- Always include Tamil disclaimer in every triage output
- Use `gemma-4` chat template (non-thinking for E4B)
- Image-before-text ordering in multimodal prompts (Gemma 4 requirement)
- Run 3 seeds per experiment; report best checkpoint + mean/std
- Use QLoRA (4-bit) for E4B training — fits ~17GB VRAM on RTX 5090
- Export to GGUF Q4_K_M for phone deployment
- Log all training runs with config + metric snapshots

**Don't:**
- Don't store patient-identifiable data anywhere
- Don't use cloud APIs in the inference path — everything must be offline-capable
- Don't generate free text for triage — always use function calling
- Don't claim triage accuracy without held-out test set (80/10/10 split)
- Don't skip ablation studies — they are part of the submission

---

## Minimum Viable Submission (cut scope here first)

1. Gemma 4 E4B + LoRA-Triage (Tamil triage)
2. `triage_classify()` function calling with structured output
3. Protocol grounding (SQLite + WHO/IMNCI rules)
4. Evaluation: P/R/F1 per class, 3 seeds
5. CLI demo on llama.cpp
6. Technical write-up
7. Demo video (CLI acceptable if Android not ready)
8. GitHub + HuggingFace weights

---

## Sprint Timeline

- **Week 1 (Apr 10–16):** Foundation — baseline, dataset construction begins, function calling schema, protocol engine, Android skeleton
- **Week 2 (Apr 17–23):** Fine-tuning — all three LoRAs (3 seeds each), KALAVAI router, GGUF export
- **Week 3 (Apr 24–30):** App build — Android UI, audio input, protocol grounding, local logging
- **Week 4 (May 1–7):** Evaluation + polish — full eval suite, ablations, app polish, Tier 2/3 demo
- **Week 5 (May 8–18):** Submission — GitHub repo, HuggingFace weights, write-up, video, submit

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| E4B Tamil quality insufficient | Demo primarily on 26B-A4B; E4B as secondary |
| Native audio not in GGUF | Whisper-small-Tamil fallback (still offline) |
| KALAVAI fusion shows no gain | Report honestly; experiment = contribution |
| Scope too ambitious | Ruthless prioritization to MVP above |
| Android llama.cpp JNI issues | Fallback to CLI demo on laptop |
