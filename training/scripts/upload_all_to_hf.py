"""
One-shot uploader for all Marunthagam artifacts to HuggingFace Hub.

Handles:
  - 3 model repos (one per specialist) with LoRA adapter + Q4_K_M GGUF
    + multimodal mmproj GGUF + dataset card with full eval numbers
  - 1 dataset repo with all training data (3 specialists × train/val/test)
    + adversarial safety prompts + safety classifier validation set
    + the user-completed label-quality spotcheck CSVs

All uploads target the authenticated user's namespace (default: mechramc).
By default repos are public (this is the submission upload).

Usage:
    python training/scripts/upload_all_to_hf.py            # all artifacts, public
    python training/scripts/upload_all_to_hf.py --models   # models only
    python training/scripts/upload_all_to_hf.py --datasets # datasets only
    python training/scripts/upload_all_to_hf.py --private  # private upload (testing)

Requires `hf auth login` (or HF_TOKEN env var) under the target user.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

REPO = Path(__file__).resolve().parents[2]
TRAIN_DATA = REPO / "training" / "data" / "formatted"
EVAL_DATA = REPO / "eval" / "data"
ANALYSIS_DIR = REPO / "eval" / "analysis"
MODELS_DIR = REPO / "training" / "models"
ADAPTERS_DIR = REPO / "training" / "outputs"

SPECIALISTS = ("triage", "derm", "maternal")

MODEL_README_TEMPLATE = """\
---
license: apache-2.0
language:
  - ta
  - en
base_model: unsloth/gemma-4-E4B-it
tags:
  - gemma-4
  - tamil
  - medical
  - triage
  - lora
  - gguf
  - q4_k_m
  - low-resource
  - clinical-decision-support
library_name: gguf
---

# Marunthagam — {specialist_title} Specialist (Gemma 4 E4B Q4_K_M)

Tamil community-health triage specialist fine-tuned on Gemma 4 E4B with
Unsloth QLoRA, exported to Q4_K_M GGUF for offline phone deployment in
rural Tamil Nadu by ASHA workers.

Part of the **Marunthagam** project (Gemma 4 Good Hackathon 2026): a
three-tier offline health intelligence system. See the [project README](https://github.com/mechramc/Marunthagam)
for the KALAVAI fusion architecture, dataset construction, the
diagnostic methodology that surfaces label-quality and morphology
issues, and the production-stack evaluation results.

## Files

- `gemma-4-e4b-it.Q4_K_M.gguf` — quantised model (~5 GB), Q4_K_M
- `gemma-4-e4b-it.BF16-mmproj.gguf` — multimodal projector (~1 GB) — Gemma 4 multimodal requires this when image inputs are used
- `Modelfile` — Ollama Modelfile for `ollama create`

{adapter_section}

## Training

| | |
|---|---|
| Base | `unsloth/gemma-4-E4B-it` (4-bit QLoRA) |
| Rank / alpha / dropout | 32 / 64 / 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | {epochs} |
| Optimizer | AdamW, lr 2e-4, cosine schedule, 50 warmup steps |
| Train rows | {train_rows} |
| Test rows (held-out 80/10/10) | {test_rows} |
| Seed | 42 |
| Best eval_loss | {eval_loss} |

{sprint2_block}

## Held-out evaluation (n=131, single seed 42, T=0)

The number that matters for ASHA-worker safety is the **missed-emergency
rate** — gold-RED cases that ended up GREEN. The protocol engine + confidence
floor + escalation logic catches **0/12** in the production routed config.

| Metric | Result |
|---|---|
| Weighted F1 | 0.6491 |
| RED recall | 0.5833 |
| Missed-emergency rate (RED→GREEN) | **0/12** |
| Adversarial safety refusal | 100/100 |
| Workstation TTFT / throughput | 0.007–0.038s · 195–213 tok/s |
| Tamil semantic similarity (multilingual mpnet cosine) | 0.6687 |

## Production stack (Sprint 2 final)

- **Routed inference** by case specialist
- **Triage**: B-retrained LoRA (6 epochs plain SFT on relabeled data)
- **Derm + Maternal**: Sprint 1 LoRAs (sprint-1 derm beat the contamination-cleaned variant in head-to-head)
- **Protocol engine v2.1**: 21 active rules (15 migrated v1 IMNCI + 6 new adult-emergency rules with Tamil case-inflected forms)
- **Multilingual safety classifier v2**: ~135 indicators across English / Hindi (Devanagari) / Gujarati / Tamil with morphological coverage

## Diagnostic findings worth re-using

This release is a hackathon submission, but three findings generalise beyond it:

1. **Triage GREEN labels need clinical relabeling.** A clinical reviewer
   judged 18% of triage GREEN labels as YELLOW or higher. The minority class
   in this kind of dataset carries the most labeling noise; relabel before
   you retrain.
2. **Tamil regex needs morphology-aware patterns.** The original safety
   classifier had 22/22 false negatives because it only covered locative
   case `மருத்துவரிடம்` and missed accusative `மருத்துவரை அணுக`,
   instrumental `நாயினால்`, and Hindi/Gujarati script when the model
   code-switches.
3. **Schema-consumer audit catches silent data loss.** The eval pipeline
   discarded engine override traces in a throwaway local variable; took
   two sprint-internal patches to surface and fix.

## Disclaimer

> இது மருத்துவ ஆலோசனை அல்ல (this is not medical advice).

Always include the disclaimer in user-facing surfaces. The Pydantic schema
in the project enforces this at the schema-validation layer.

## License

Apache 2.0. Built by mechramc for the Gemma 4 Good Hackathon (deadline
2026-05-18).
"""

ADAPTER_SECTION_TRIAGE = """\

### LoRA adapter (B-retrained, Sprint 2 production)

In addition to the GGUF, this repo includes the LoRA adapter weights at
`adapter/` so researchers can apply it to a different quantization or
inference backend:

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `adapter/tokenizer.json`, `adapter/tokenizer_config.json`, `adapter/chat_template.jinja`

Load with HuggingFace `peft`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-4-E4B-it", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "mechramc/marunthagam-triage-E4B-Q4_K_M", subfolder="adapter")
tok = AutoTokenizer.from_pretrained("mechramc/marunthagam-triage-E4B-Q4_K_M", subfolder="adapter")
```

The adapter is the relabel + 6-epoch retraining product. The GGUF in this
repo is the same model post-merge + Q4_K_M quantization for phone deployment.
"""

ADAPTER_SECTION_DEFAULT = """\

### LoRA adapter

This repo includes the LoRA adapter weights at `adapter/` (sprint-1 training,
3 epochs plain SFT on the original training distribution).
"""

SPRINT2_TRIAGE = """\

## Sprint 2 retraining

The Sprint 2 production stack uses a **B-retrained** triage LoRA. The
training data was clinician-relabeled before this run: 113 triage GREEN
cases were reviewed by a clinical rater, and 20 (18%) were re-labeled
YELLOW for medically under-triaged cases (cardiac patterns, post-fall
syncope, persistent chest discomfort, etc.). The model was retrained
for 6 epochs on the post-relabel data.

Compared to the original Sprint 1 triage LoRA on the same held-out test
split, the Sprint 2 B-retrained version has:

- F1 0.6033 vs 0.4972 on triage rows (+0.106)
- Higher RED-at-RED catch rate
- Same missed-emergency rate (0/12)

### What's in this repo right now

- `gemma-4-e4b-it.Q4_K_M.gguf`: **Sprint 1** triage GGUF (3-epoch plain SFT
  on the original training distribution) — uploaded first because the
  Sprint 2 GGUF export was still in flight at submission time
- `adapter/`: **Sprint 2 B-retrained** LoRA adapter (6-epoch SFT on
  relabeled data) — production-shipped weights

A Sprint 2 B-retrained Q4_K_M GGUF will replace the Sprint 1 GGUF in this
repo as a follow-up commit. Until then, the most production-faithful
inference path is the adapter at `adapter/` applied to the 4-bit Unsloth
base via `peft` (see snippet above).
"""

SPRINT2_DEFAULT = """"""

SPECIALIST_META = {
    "triage": {
        "epochs": 6,
        "eval_loss": 1.893,
        "train_rows": 351,
        "test_rows": 45,
        "adapter_section": ADAPTER_SECTION_TRIAGE,
        "sprint2_block": SPRINT2_TRIAGE,
        "adapter_path": ADAPTERS_DIR / "triage-relabel-seed42-6ep" / "final",
    },
    "derm": {
        "epochs": 3,
        "eval_loss": 2.018,
        "train_rows": 328,
        "test_rows": 41,
        "adapter_section": ADAPTER_SECTION_DEFAULT,
        "sprint2_block": SPRINT2_DEFAULT,
        "adapter_path": ADAPTERS_DIR / "derm-seed42" / "final",
    },
    "maternal": {
        "epochs": 3,
        "eval_loss": 1.912,
        "train_rows": 353,
        "test_rows": 45,
        "adapter_section": ADAPTER_SECTION_DEFAULT,
        "sprint2_block": SPRINT2_DEFAULT,
        "adapter_path": ADAPTERS_DIR / "maternal-seed256" / "final",
    },
}

DATASET_README = """\
---
license: apache-2.0
language:
  - ta
  - en
  - hi
  - gu
tags:
  - tamil
  - medical
  - triage
  - clinical-decision-support
  - low-resource
  - asha-worker
  - multilingual
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
  - text-generation
pretty_name: Marunthagam Tamil Triage Dataset
---

# Marunthagam — Tamil Triage Dataset

Multilingual training and evaluation data for the
[Marunthagam](https://github.com/mechramc/Marunthagam) project — an offline,
Tamil-first triage AI for ASHA workers in rural Tamil Nadu, built on
Gemma 4 E4B for the Gemma 4 Good Hackathon 2026.

## Layout

```
formatted/
  triage/
    train.jsonl          # 351 rows, post-Sprint-2 relabel + post-Sprint-3 derm-move
    val.jsonl            # 43 rows
    test.jsonl           # 45 rows (cleaned held-out)
    train_v1_pre_relabel.jsonl   # backup before Sprint 2 GREEN relabel
    val_v1_pre_relabel.jsonl
    test_v1_pre_relabel.jsonl
    train_v2_pre_derm_move.jsonl # backup before Sprint 3 derm contamination move
    val_v2_pre_derm_move.jsonl
    test_v2_pre_derm_move.jsonl
  derm/
    train.jsonl          # 287 rows (post-contamination-cleanup; original 328)
    val.jsonl            # 39 rows
    test.jsonl           # 35 rows (cleaned)
    *_v2_pre_derm_move.jsonl     # original (pre-cleanup) backups
  maternal/
    train.jsonl          # 353 rows
    val.jsonl            # 44 rows
    test.jsonl           # 45 rows
eval/
  adversarial_prompts.json     # 100 multilingual safety probes (Tamil/Hindi/Gujarati/English)
  safety_classifier_validation.jsonl  # 100 hand-labeled refusal/non-refusal examples
analysis/
  triage_green_relabel_LABELED.csv     # Sprint 2: 113-row clinician relabel of triage GREEN
  derm_contamination_candidates.csv    # Sprint 2: 86 derm cases flagged as wrong-specialist
```

## Format

Each `{train,val,test}.jsonl` row is a chat-format record:

```json
{
  "messages": [
    {"role": "user", "content": "<Tamil patient query>"},
    {"role": "assistant", "tool_calls": [{
      "function": {
        "name": "triage_classify",
        "arguments": {
          "verbal_symptoms": "...", "patient_age_group": "adult|child|...",
          "duration_days": 1, "vital_signs": null
        }
      }
    }]},
    {"role": "tool", "content": "<JSON of triage_result with level/confidence/...>"},
    {"role": "assistant", "content": "<plain Tamil next_steps>"}
  ]
}
```

## Construction

- **Source:** ChatDoctor patient-question corpus filtered to community-health-relevant cases (paediatric, ENT, dermatology, maternal/neonatal, common adult presentations) → translated to Tamil with Gemma 4 31B (Q4_K_M) on Mac Studio (MLX) → labeled with `triage_classify()` schema using Gemma 4 31B → 80/10/10 split per specialist.
- **Privacy:** No patient identifiers. The questions are public ChatDoctor data; translations are stored without any user-identifying metadata.

## Sprint 2 / Sprint 3 modifications

### Triage GREEN clinical relabel (Sprint 2)

A clinical rater reviewed 113 triage GREEN cases (90 train + 11 val + 12 test) and judged 20 (18%) as YELLOW or higher. Concrete findings:

- Cardiac-pattern queries (chest pain + radiation + autonomic features) labeled GREEN — under-triaged.
- Post-fall syncope, persistent post-trauma pain, new-onset palpitations all incorrectly GREEN.
- 30% disagreement on triage GREEN; **100% agreement on triage YELLOW** — the minority class carried the noise.

Backups at `*_v1_pre_relabel.jsonl`. Diff log in [the project repo](https://github.com/mechramc/Marunthagam/blob/main/eval/analysis/2026-05-07/triage_relabel_diff.json).

### Derm contamination move (Sprint 3)

Source-acquisition keyword regex routed 49 cases to derm-train when the chief
complaint was non-dermatologic (poison control, hepatology, pulmonology,
GI/surgery). Hand-reviewed and reassigned to triage-train. The derm-LoRA
re-trained on the cleaned data underperformed the original derm-LoRA in
head-to-head, suggesting low-resource specialist regimes can be more
sensitive to data quantity than to noise. **Production keeps the original
derm LoRA.** The contamination move is preserved as data hygiene.

Backups at `*_v2_pre_derm_move.jsonl`.

## Evaluation data

- `adversarial_prompts.json` — 100 prompts in 5 categories (surgery, prescription, mental_health_crisis, scope_violation, diagnosis_without_exam). Multiple languages — Tamil, Hindi, Gujarati, English mixed.
- `safety_classifier_validation.jsonl` — 100 hand-labeled refusal vs non-refusal examples used to validate the multilingual refusal classifier in Sprint 2.

## License

Apache 2.0. Translations and labels are derived works of public ChatDoctor data.

## Citation

If you use this dataset, please reference:

```
@misc{marunthagam2026,
  title = {Marunthagam: An offline Tamil-first triage AI for ASHA workers},
  author = {mechramc},
  year = {2026},
  howpublished = {Hugging Face Hub},
  url = {https://huggingface.co/datasets/mechramc/marunthagam-tamil-triage}
}
```
"""


def _ensure_repo(repo_id: str, repo_type: str, public: bool) -> None:
    create_repo(repo_id, repo_type=repo_type, private=not public, exist_ok=True)


def _upload_file(local: Path, repo_id: str, repo_path: str, repo_type: str, msg: str) -> None:
    if not local.exists():
        print(f"  skip (missing): {local}")
        return
    size_mb = local.stat().st_size / 1e6
    print(f"  upload {repo_path}  ({size_mb:.1f} MB) ...")
    upload_file(
        path_or_fileobj=str(local),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
    )


def upload_specialist_model(specialist: str, hf_user: str, public: bool) -> str:
    repo_id = f"{hf_user}/marunthagam-{specialist}-E4B-Q4_K_M"
    print(f"\n[model:{specialist}] {repo_id}")
    _ensure_repo(repo_id, "model", public)

    meta = SPECIALIST_META[specialist]
    readme = MODEL_README_TEMPLATE.format(
        specialist_title=specialist.capitalize(),
        adapter_section=meta["adapter_section"],
        sprint2_block=meta["sprint2_block"],
        epochs=meta["epochs"],
        eval_loss=meta["eval_loss"],
        train_rows=meta["train_rows"],
        test_rows=meta["test_rows"],
    )
    upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
        commit_message=f"docs: model card for {specialist}",
    )

    gguf_dir = MODELS_DIR / f"{specialist}-E4B-Q4_K_M_gguf"
    for name in ("gemma-4-e4b-it.Q4_K_M.gguf",
                 "gemma-4-e4b-it.BF16-mmproj.gguf",
                 "Modelfile"):
        _upload_file(gguf_dir / name, repo_id, name, "model",
                     f"weights: {specialist} {name}")

    adapter_path = meta["adapter_path"]
    if adapter_path.is_dir():
        print(f"  uploading adapter from {adapter_path}")
        for name in ("adapter_config.json", "adapter_model.safetensors",
                     "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja"):
            _upload_file(adapter_path / name, repo_id, f"adapter/{name}", "model",
                         f"adapter: {specialist} {name}")
    else:
        print(f"  no adapter dir at {adapter_path} — skipping adapter upload")

    return repo_id


def upload_dataset(hf_user: str, public: bool) -> str:
    repo_id = f"{hf_user}/marunthagam-tamil-triage"
    print(f"\n[dataset] {repo_id}")
    _ensure_repo(repo_id, "dataset", public)

    upload_file(
        path_or_fileobj=DATASET_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset",
        commit_message="docs: dataset card",
    )

    # Training splits
    for spec in SPECIALISTS:
        spec_dir = TRAIN_DATA / spec
        for f in sorted(spec_dir.glob("*.jsonl")):
            _upload_file(f, repo_id, f"formatted/{spec}/{f.name}", "dataset",
                         f"data: {spec}/{f.name}")

    # Eval data
    for f in (EVAL_DATA / "adversarial_prompts.json",
              EVAL_DATA / "safety_classifier_validation.jsonl"):
        _upload_file(f, repo_id, f"eval/{f.name}", "dataset",
                     f"eval: {f.name}")

    # Selected analysis CSVs (the user-completed clinical reviews)
    analysis_files = [
        ANALYSIS_DIR / "2026-05-07" / "triage_green_relabel_LABELED.csv",
        ANALYSIS_DIR / "2026-05-07" / "derm_contamination_candidates.csv",
        ANALYSIS_DIR / "2026-05-06" / "tamil_human_eval_template.csv",
    ]
    for f in analysis_files:
        if f.exists():
            _upload_file(f, repo_id, f"analysis/{f.name}", "dataset",
                         f"analysis: {f.name}")

    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Marunthagam to HF Hub")
    parser.add_argument("--user", default=None,
                        help="HF user/org (default: whoami)")
    parser.add_argument("--private", action="store_true",
                        help="Create private repos (default: public)")
    parser.add_argument("--models", action="store_true",
                        help="Models only (default: all)")
    parser.add_argument("--datasets", action="store_true",
                        help="Datasets only (default: all)")
    args = parser.parse_args()

    api = HfApi()
    hf_user = args.user or api.whoami().get("name")
    if not hf_user:
        raise RuntimeError("Could not determine HF user; pass --user explicitly.")
    public = not args.private
    do_models = args.models or not args.datasets
    do_datasets = args.datasets or not args.models
    print(f"Uploading as: {hf_user}  public={public}  models={do_models}  datasets={do_datasets}")

    repos: list[str] = []
    if do_models:
        for sp in SPECIALISTS:
            try:
                repos.append(("model", upload_specialist_model(sp, hf_user, public)))
            except Exception as exc:
                print(f"  ERROR uploading {sp}: {exc}")

    if do_datasets:
        try:
            repos.append(("dataset", upload_dataset(hf_user, public)))
        except Exception as exc:
            print(f"  ERROR uploading dataset: {exc}")

    print("\nDone. Repos:")
    for kind, r in repos:
        url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{r}"
        print(f"  {kind}: {url}")


if __name__ == "__main__":
    main()
