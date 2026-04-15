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

*Target values — actual results pending LoRA training (Week 2 of sprint). All metrics reported as mean ± std across seeds 42, 137, 256 on held-out test split (80/10/10).*

| Metric | Target | Status |
|--------|--------|--------|
| Triage F1 (weighted) | > 0.80 | Pending |
| RED recall | > 0.90 | Pending |
| Tamil fluency (chrF++) | > 0.60 | Pending |
| Safety refusal rate | 100% (100 adversarial prompts) | Pending |
| Per-domain specialist gain | +5% over generalist | Pending |
| Ablation: fusion gain | +3% over best specialist | Pending |
| Phone TTFT (E4B) | < 3s, > 8 tok/s | Pending |
| Workstation TTFT | < 1s, > 30 tok/s | Pending |

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
