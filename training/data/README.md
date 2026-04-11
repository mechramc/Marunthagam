# Marunthagam Training Data

This directory contains the curated Tamil medical dataset for training three specialist LoRA adapters on Gemma 4 E4B.

**⚠ PRIVACY: No real patient data. All pairs are synthetic clinical vignettes translated from English medical Q&A datasets.**

---

## 1. Data Sources

| Specialist | Source Datasets | Target Size |
|------------|----------------|-------------|
| LoRA-Triage | MEDIQA, MedQuAD, HealthSearchQA | ~2,000 pairs |
| LoRA-Derm | DermNet, ISIC, Fitzpatrick17k | ~1,200 image-text pairs |
| LoRA-Maternal | WHO IMNCI, Tamil Nadu state health guidelines, maternal health Q&A | ~1,500 pairs |
| Function Calling | Synthetic protocol-grounded triage_classify() examples | ~600 pairs |
| Safety Guardrails | Adversarial + out-of-scope (shared across all specialists) | ~500 pairs |

**Total: ~5,800 training pairs across three specialists + shared safety set.**

---

## 2. Translation Pipeline

1. **Source collection**: Download English medical Q&A datasets (MEDIQA, MedQuAD, HealthSearchQA)
2. **Automated translation**: Run `scripts/translate_dataset.py` using Gemma 4 31B on Mac Studio (MLX/llama.cpp). Outputs JSONL with `review_status: "pending"`
3. **Human review**: 3 Tamil-speaking medical reviewers mark each pair as `"approved"`, `"rejected"`, or `"needs_revision"`
4. **Format conversion**: Run `scripts/format_training_data.py` on approved pairs — converts to Gemma 4 chat template with function calling format
5. **Split**: 80/10/10 train/val/test, stratified by triage level

---

## 3. Review Instructions (For Tamil Reviewers)

Each reviewed JSONL file has entries with these fields. Update `review_status` to one of:
- `"approved"` — medically accurate Tamil translation, correct triage level, natural language
- `"rejected"` — mistranslation, factual error, or inappropriate content
- `"needs_revision"` — minor issues, return with comments in `reviewer_notes` field

**Critical checks:**
- Medical terminology matches Tamil Nadu government health communications
- Triage level is correct per WHO IMNCI guidelines
- Tamil text is natural and accessible (not over-medicalized)
- Function calling format is correct (triage_classify arguments match the patient case)

---

## 4. Train/Val/Test Split

- **80% train**: Used for gradient updates during QLoRA fine-tuning
- **10% val**: Used for early stopping (eval_loss monitored per epoch)
- **10% test**: HELD OUT — only used for final evaluation metrics

The split is stratified by triage level (GREEN/YELLOW/RED) to ensure balanced evaluation.

---

## 5. Directory Structure

```
training/data/
├── README.md               (this file)
├── fixtures/               (10 hand-crafted examples per specialist for testing)
│   ├── triage_reviewed.jsonl
│   ├── derm_reviewed.jsonl
│   └── maternal_reviewed.jsonl
├── raw/                    (GITIGNORED — downloaded English source data)
├── reviewed/               (GITIGNORED — human-reviewed JSONL, pending/approved/rejected)
└── formatted/              (GITIGNORED — Gemma 4 chat template JSONL, ready for training)
```

The `raw/`, `reviewed/`, and `formatted/` directories are gitignored (~5GB total). Weights are tracked on HuggingFace Hub.

---

## 6. Gemma 4 Format Requirements

All training examples must follow these Gemma 4 requirements:
- **Chat template**: Use `gemma-4` non-thinking template (not gemma-4-thinking)
- **Multimodal ordering**: For derm specialist — images MUST appear BEFORE text in the prompt (Gemma 4 requirement)
- **Function calling format**: Use `<tool_call>...</tool_call>` tags for model function calls
- **Max sequence length**: 4096 tokens

---

## 7. Privacy Rules

- **No real patient data**: All pairs are synthetic or translated clinical vignettes
- **No identifiers**: No names, IDs, locations, or demographic data beyond age group
- **Ephemeral images**: Clinical images are referenced but not stored in training data
- **India DPDPA 2023 compliance**: Data handling follows India's Digital Personal Data Protection Act 2023

---

## 8. HuggingFace Dataset

The curated Tamil medical dataset is published as:
`murailabs/marunthagam-tamil-medical-dataset` (CC-BY-SA 4.0)

This includes the approved JSONL pairs (post-human-review) and the formatted training splits.
