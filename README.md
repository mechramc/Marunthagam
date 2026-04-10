# மருந்தகம் — Marunthagam

**Community health intelligence, offline.**

> An open protocol and reference implementation for deploying offline community health AI in zero-connectivity environments.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hackathon: Gemma 4 Good](https://img.shields.io/badge/Gemma_4_Good_Hackathon-2026-green)]()
[![Protocol Version](https://img.shields.io/badge/Protocol-v1.0.0-orange)]()

---

## The Problem

940,000 ASHA workers. 600,000 villages. 80 million Tamil speakers. 1 doctor per 10,000 people. Zero AI tools that work offline in their language.

Marunthagam fixes that.

---

## What It Is

Not a chatbot. A three-tier health intelligence system using the full Gemma 4 model family:

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: FIELD (ASHA Worker — Android Phone)                │
│  Gemma 4 E4B-IT · Q4 GGUF · ~5GB · Fully Offline          │
│  Tamil voice input + clinical photo → Triage card           │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: CLINIC (PHC Doctor — Workstation)                  │
│  Gemma 4 26B-A4B-IT · Q4 GGUF · ~18GB · Intermittent      │
│  Higher-accuracy assessment + differential reasoning        │
├─────────────────────────────────────────────────────────────┤
│  TIER 3: DISTRICT (Health Officer — Dashboard)              │
│  Gemma 4 31B-IT · React + D3 · Connected                   │
│  Population health signals + Tamil intelligence briefs      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Claims

- **First KALAVAI-fused specialist LoRA system on an edge model:** Three domain-specific LoRAs (triage, dermatology, maternal health) fused via learned MoE router on Gemma 4 E4B.
- **Full Gemma 4 multimodal showcase:** Native audio (Tamil speech), native vision (clinical images), native function calling — all on-device.
- **Three-tier deployment across the full Gemma 4 family:** One protocol, three model sizes, one coherent system.
- **Open protocol for global adoption:** JSON schema for community health interactions. Any language, any country.

---

## Quick Start (CLI Demo)

```bash
# 1. Clone the repo
git clone https://github.com/murailabs/marunthagam
cd marunthagam

# 2. Install inference dependencies
pip install -r inference/requirements.txt

# 3. Download the fused model
# huggingface-cli download murailabs/marunthagam-fused-E4B-GGUF \
#   marunthagam-fused-E4B-Q4_K_M.gguf --local-dir models/

# 4. Run triage demo
python inference/demo_cli.py \
  --model models/marunthagam-fused-E4B-Q4_K_M.gguf \
  --symptoms "குழந்தைக்கு 3 நாட்களாக காய்ச்சல் மற்றும் தோலில் சிவந்த புள்ளிகள்" \
  --age child \
  --duration 3
```

---

## Evaluation Results

| Metric | Target | Result |
|--------|--------|--------|
| Triage F1 (weighted) | > 0.80 | TBD |
| RED Recall (emergency detection) | > 0.90 | TBD |
| Tamil fluency (chrF++) | > 0.60 | TBD |
| Safety refusal rate | 100% | TBD |
| Specialist gain over generalist | +5% | TBD |
| Fusion gain over best specialist | +3% | TBD |
| Phone inference (TTFT) | < 3s | TBD |

*All results reported as mean ± std across 3 seeds (42, 137, 256).*

---

## Repository Structure

```
training/     Dataset construction + Unsloth fine-tuning + KALAVAI router
inference/    llama.cpp integration + function calling + protocol engine
android/      Android app (Kotlin + llama.cpp JNI + SQLite)
dashboard/    React + D3 district health dashboard
eval/         Evaluation scripts + ablation notebooks
protocol/     Marunthagam Open Protocol (JSON schema + specification)
docs/         Architecture documentation + implementation plans
models/       Downloaded GGUF models (not tracked in git)
```

---

## Models on HuggingFace

| Model | Description |
|-------|-------------|
| `murailabs/marunthagam-triage-lora` | LoRA-Triage adapter |
| `murailabs/marunthagam-derm-lora` | LoRA-Derm adapter (multimodal) |
| `murailabs/marunthagam-maternal-lora` | LoRA-Maternal adapter |
| `murailabs/marunthagam-fused-E4B-GGUF` | Fused model + router (Q4_K_M, ~5GB) |
| `murailabs/marunthagam-tamil-medical-dataset` | Curated training data (CC-BY-SA) |

---

## Open Protocol

Every triage interaction produces a record conforming to the [Marunthagam Open Protocol v1.0](protocol/schemas/interaction_record_v1.json).

**Privacy guarantees:** No patient identifiers stored. Audio and images processed ephemerally on-device. Sync transmits aggregated signals only. AES-256 local storage. TLS 1.3 sync.

**Disclaimer:** Every triage output includes: *"இது மருத்துவ ஆலோசனை அல்ல"* (This is not medical advice.)

---

## License

Apache 2.0 — See [LICENSE](LICENSE)

---

**Murai Labs · முறை · murailabs.com**

*Gemma 4 Good Hackathon · Kaggle · Google DeepMind · May 2026*
