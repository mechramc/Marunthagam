"""
Upload the three specialist GGUFs to Hugging Face Hub under the user's account.

Each GGUF lives in <repo>/training/models/<specialist>-E4B-Q4_K_M_gguf/.
Uploads them to private-by-default repos:
    {hf_user}/marunthagam-{specialist}-E4B-Q4_K_M

Usage:
    python upload_to_hf.py
    python upload_to_hf.py --public      # publish
    python upload_to_hf.py --user mechramc
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

TRAINING_ROOT = Path(__file__).resolve().parents[1]
SPECIALISTS = ("triage", "derm", "maternal")

README_TEMPLATE = """\
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
---

# Marunthagam — {specialist_title} Specialist (Gemma 4 E4B Q4_K_M)

Tamil community-health triage specialist fine-tuned on Gemma 4 E4B with
Unsloth QLoRA, exported to Q4_K_M GGUF for offline phone deployment.

Part of the Marunthagam project (Gemma 4 Good Hackathon 2026): an offline,
Tamil-first triage AI for ASHA workers. See the
[full project README](https://github.com/mechramc/Marunthagam) for the
KALAVAI fusion architecture, dataset construction, and evaluation results.

## Files

- `gemma-4-e4b-it.Q4_K_M.gguf` — quantised model (~5 GB)
- `gemma-4-e4b-it.BF16-mmproj.gguf` — multimodal projector (Gemma 4 requirement)

## Training summary

- **Base:** `unsloth/gemma-4-E4B-it`
- **Method:** QLoRA, rank 32, α 64, dropout 0
- **Data:** ~{train_rows} Tamil patient Q&A pairs, auto-labeled via Gemma 4
  31B Q4_K_M; held-out test split (~{test_rows} rows)
- **Best seed:** {seed}
- **Final eval_loss:** {eval_loss}

## Eval (combined system, IMNCI protocol engine layered on)

| Metric | Target | Result |
|--------|--------|--------|
| Weighted F1 | > 0.80 | 0.8174 |
| RED recall | > 0.90 | 0.9231 |

## Disclaimer

> இது மருத்துவ ஆலோசனை அல்ல (this is not medical advice).

Always include the disclaimer in user-facing surfaces.
"""

SPECIALIST_META = {
    "triage": {"seed": 42, "eval_loss": 1.904, "train_rows": 351, "test_rows": 45},
    "derm":   {"seed": 42, "eval_loss": 2.018, "train_rows": 328, "test_rows": 41},
    "maternal": {"seed": 256, "eval_loss": 1.912, "train_rows": 353, "test_rows": 45},
}


def upload_specialist(specialist: str, hf_user: str, public: bool) -> str:
    repo_id = f"{hf_user}/marunthagam-{specialist}-E4B-Q4_K_M"
    local_dir = TRAINING_ROOT / "models" / f"{specialist}-E4B-Q4_K_M_gguf"
    if not local_dir.is_dir():
        raise FileNotFoundError(f"GGUF directory not found: {local_dir}")

    create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    print(f"\n[{specialist}] repo: https://huggingface.co/{repo_id}")

    meta = SPECIALIST_META[specialist]
    readme = README_TEMPLATE.format(
        specialist_title=specialist.capitalize(),
        seed=meta["seed"],
        eval_loss=meta["eval_loss"],
        train_rows=meta["train_rows"],
        test_rows=meta["test_rows"],
    )
    upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"docs: marunthagam {specialist} specialist card",
    )

    for name in (
        "gemma-4-e4b-it.Q4_K_M.gguf",
        "gemma-4-e4b-it.BF16-mmproj.gguf",
    ):
        src = local_dir / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        size_gb = src.stat().st_size / 1e9
        print(f"  upload {name} ({size_gb:.2f} GB) ...")
        upload_file(
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"weights: {specialist} {name}",
        )

    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Marunthagam GGUFs to HF Hub")
    parser.add_argument("--user", default=None, help="HF username (default: whoami)")
    parser.add_argument("--public", action="store_true", help="Create public repos")
    parser.add_argument(
        "--specialist",
        choices=SPECIALISTS,
        default=None,
        help="Upload only one specialist (default: all three)",
    )
    args = parser.parse_args()

    api = HfApi()
    hf_user = args.user or api.whoami().get("name")
    if not hf_user:
        raise RuntimeError("Could not determine HF user; pass --user explicitly.")
    print(f"Uploading as: {hf_user}  (public={args.public})")

    targets = (args.specialist,) if args.specialist else SPECIALISTS
    repos: list[str] = []
    for sp in targets:
        repos.append(upload_specialist(sp, hf_user=hf_user, public=args.public))

    print("\nDone. Repos:")
    for r in repos:
        print(f"  https://huggingface.co/{r}")


if __name__ == "__main__":
    main()
