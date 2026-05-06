"""
Swap the sprint-1 triage GGUF on HuggingFace for the Sprint 2 B-retrained
GGUF.

The original `upload_all_to_hf.py` ran when the B-retrain export was still
in flight, so the public `mechramc/marunthagam-triage-E4B-Q4_K_M` repo
currently holds the sprint-1 GGUF (3-epoch plain SFT on the original
distribution). The B-retrained GGUF (6-epoch SFT on relabeled data) is
the production artifact that produced every held-out number in the
README. This script replaces the GGUF + Modelfile in place and refreshes
the model card to match.

What this does:
  1. Uploads the B-retrained Q4_K_M GGUF (overwrites the sprint-1 file)
  2. Uploads the matching Modelfile (overwrites the sprint-1 Modelfile)
  3. Re-writes README.md on the repo so the "What's in this repo right
     now" section reflects the swap. Adapter / mmproj / dataset linkages
     are preserved.

What this does NOT do:
  - Touch the adapter/ subdirectory (already the Sprint 2 B-retrained one)
  - Touch the BF16-mmproj.gguf file (base model unchanged)
  - Touch the derm or maternal repos
  - Touch the dataset repo

Idempotent: re-running just re-uploads the same bytes; HF deduplicates.

Usage:
    hf auth login   # if not already
    python training/scripts/swap_triage_gguf_on_hf.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "training" / "models"

# B-retrained GGUF lives at training/models/triage-B-E4B-Q4_K_M_gguf/
# gemma-4-e4b-it.Q4_K_M_gguf/{Q4_K_M.gguf, Modelfile, BF16-mmproj.gguf}
B_DIR = MODELS / "triage-B-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M_gguf"

HF_REPO = "mechramc/marunthagam-triage-E4B-Q4_K_M"

NEW_README = """\
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

# Marunthagam — Triage Specialist (Gemma 4 E4B Q4_K_M)

Tamil community-health triage specialist fine-tuned on Gemma 4 E4B with
Unsloth QLoRA, exported to Q4_K_M GGUF for offline phone deployment in
rural Tamil Nadu by ASHA workers.

Part of the **Marunthagam** project (Gemma 4 Good Hackathon 2026): a
three-tier offline health intelligence system. See the
[project README](https://github.com/mechramc/Marunthagam) for the KALAVAI
fusion architecture, dataset construction, the diagnostic methodology
that surfaces label-quality and morphology issues, and the
production-stack evaluation results.

## Files

- `gemma-4-e4b-it.Q4_K_M.gguf` — **Sprint 2 B-retrained** quantised
  model (~5 GB), Q4_K_M. Trained for 6 epochs of plain SFT on the
  post-relabel triage data. This is the artifact that produced every
  held-out number in the project README.
- `gemma-4-e4b-it.BF16-mmproj.gguf` — multimodal projector (~1 GB) —
  Gemma 4 multimodal requires this when image inputs are used. Base
  model unchanged from Sprint 1.
- `Modelfile` — Ollama Modelfile for `ollama create`
- `adapter/` — Sprint 2 B-retrained LoRA adapter (PEFT-compatible) for
  the HF+PEFT inference path

## Sprint 2 — what changed from Sprint 1

The original Sprint 1 triage LoRA was trained on the original triage
distribution before clinical relabeling. Sprint 1 diagnostic work
surfaced an 18% under-triage rate in the GREEN class — cardiac-pattern
queries, post-fall syncope, persistent chest discomfort, and other
adult-emergency cases were systematically labeled GREEN when they
should have been YELLOW (rater-clinician judgement against a project
lead with clinical background). 113 GREEN cases were reviewed; 20 were
re-labeled YELLOW. The Sprint 2 B-retrained model was trained for
6 epochs of plain SFT on the post-relabel data.

On the held-out test split (n=131, seed 42, T=0):

- Triage-rows F1: **0.6033** (vs Sprint 1 triage-rows F1 0.4972; +0.106)
- Higher RED-at-RED catch rate
- Same missed-as-GREEN rate (0/12)

The full production stack (B-retrained triage + sprint-1 derm +
sprint-1 maternal + v2.1 IMNCI rules + v2 multilingual safety
classifier) on the same held-out split:

- Weighted F1: **0.6491** (calibrated target ≥0.65 — 0.001 below)
- RED recall: **0.5833** (calibrated target ≥0.55)
- 0/12 missed-as-GREEN; 7/12 caught at full RED
- 100/100 adversarial safety refusals

## Inference paths

### llama.cpp / llama-cpp-python (recommended for production)

```python
from llama_cpp import Llama
llm = Llama(
    model_path="gemma-4-e4b-it.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
)
out = llm("Your prompt here", max_tokens=256, temperature=0.0)
```

### HF + PEFT (for fast experimentation)

```python
from unsloth import FastLanguageModel
from peft import PeftModel

base, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(base, "mechramc/marunthagam-triage-E4B-Q4_K_M",
                                  subfolder="adapter")
FastLanguageModel.for_inference(model)
```

### Ollama

```bash
ollama create marunthagam-triage -f Modelfile
ollama run marunthagam-triage "your prompt"
```

## Training

- Base: `unsloth/gemma-4-E4B-it` (4-bit)
- Method: Unsloth QLoRA, rank 32, alpha 64, lr 2e-4, 6 epochs plain SFT
- Hardware: RTX 5090 32GB
- Data: 351 train rows (post-relabel), 45 test rows (relabeled
  held-out); see [`mechramc/marunthagam-tamil-triage`](https://huggingface.co/datasets/mechramc/marunthagam-tamil-triage)

## Decision support, not replacement

Every triage output in production carries the mandatory Tamil
disclaimer **"இது மருத்துவ ஆலோசனை அல்ல"** ("This is not medical
advice"). The model's job is to help ASHA workers escalate appropriately
through India's existing tiered referral system — not to replace PHC
doctors. The IMNCI protocol engine sits below the LLM and can only
*escalate* triage urgency, never downgrade.

## License

Apache 2.0. See the [project repo](https://github.com/mechramc/Marunthagam)
for full attribution.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually upload, just print what would happen.")
    args = parser.parse_args()

    gguf_path = B_DIR / "gemma-4-e4b-it.Q4_K_M.gguf"
    modelfile_path = B_DIR / "Modelfile"

    if not gguf_path.exists():
        print(f"FATAL: B-retrained GGUF not found at {gguf_path}", file=sys.stderr)
        return 1
    if not modelfile_path.exists():
        print(f"FATAL: Modelfile not found at {modelfile_path}", file=sys.stderr)
        return 1

    gguf_size_gb = gguf_path.stat().st_size / 1024 / 1024 / 1024
    print(f"Source: {gguf_path}")
    print(f"  size: {gguf_size_gb:.2f} GB")
    print(f"Target: hf.co/{HF_REPO}")
    print()

    if args.dry_run:
        print("--dry-run: would upload Q4_K_M.gguf, Modelfile, README.md")
        return 0

    api = HfApi()
    print(f"[1/3] Uploading Q4_K_M.gguf ({gguf_size_gb:.2f} GB) — this is the long one...")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo="gemma-4-e4b-it.Q4_K_M.gguf",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Sprint 2 B-retrained Q4_K_M GGUF (6ep plain SFT on relabeled data)",
    )
    print("    done.")

    print(f"[2/3] Uploading Modelfile...")
    api.upload_file(
        path_or_fileobj=str(modelfile_path),
        path_in_repo="Modelfile",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Sprint 2 Modelfile",
    )
    print("    done.")

    print(f"[3/3] Refreshing README.md (model card)...")
    api.upload_file(
        path_or_fileobj=NEW_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Refresh card after Sprint 2 GGUF swap",
    )
    print("    done.")

    print()
    print("Swap complete. Verify at:")
    print(f"  https://huggingface.co/{HF_REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
