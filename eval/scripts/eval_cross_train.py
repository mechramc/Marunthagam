"""
Marunthagam — Cross-domain training-set evaluation (Task 1 step 4).

Runs ONE specialist's GGUF over ANOTHER specialist's training data and
computes F1 / per-class report. Used to test whether maternal-LoRA is a
genuine generalist or whether triage/derm LoRAs overfit their training
distributions.

The user (2026-05-06) explicitly asked for the per-set numbers reported
SEPARATELY — no aggregation across the eval-on-triage-train and
eval-on-derm-train runs.

Usage:
    python eval/scripts/eval_cross_train.py \
        --model training/models/maternal-E4B-Q4_K_M_gguf/gemma-4-e4b-it.Q4_K_M.gguf \
        --train-set triage \
        --tag maternal_on_triage_train

The output JSON has the same shape as run_eval.py results so the same
analysis tooling reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Reuse run_eval's predict path so we get exactly the same prompt template,
# JSON parsing, and protocol-engine wiring.
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import run_eval  # noqa: E402

from run_logger import RunLogger  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATTED_DIR = REPO_ROOT / "training" / "data" / "formatted"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def load_train_cases(specialist: str) -> list[run_eval.TestCase]:
    """
    Load `<specialist>/train.jsonl` rows in the same TestCase shape as the
    held-out test loader. Skips rows without a gold level.
    """
    path = FORMATTED_DIR / specialist / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")

    cases: list[run_eval.TestCase] = []
    with open(path, encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            messages = record.get("messages", [])
            tamil_question = ""
            args: dict = {}
            gold_payload: dict = {}
            for msg in messages:
                role = msg.get("role")
                if role == "user" and not tamil_question:
                    tamil_question = msg.get("content") or ""
                elif role == "assistant" and msg.get("tool_calls"):
                    fn = msg["tool_calls"][0].get("function", {})
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                elif role == "tool":
                    raw_tool = msg.get("content")
                    if isinstance(raw_tool, str):
                        try:
                            gold_payload = json.loads(raw_tool)
                        except json.JSONDecodeError:
                            gold_payload = {}
                    elif isinstance(raw_tool, dict):
                        gold_payload = raw_tool
            gold = str(gold_payload.get("level", "")).upper()
            if gold not in run_eval.TRIAGE_LEVELS:
                continue
            cases.append(run_eval.TestCase(
                specialist=specialist,
                verbal_symptoms=args.get("verbal_symptoms", "") or tamil_question,
                age_group=args.get("patient_age_group", "adult"),
                duration_days=int(args.get("duration_days", 1) or 1),
                gold_level=gold,
                case_id=f"{specialist}_train_{line_num:03d}",
                tamil_question=tamil_question,
            ))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain training-set eval")
    parser.add_argument("--model", required=True, help="GGUF path of the model under test")
    parser.add_argument("--train-set", required=True, choices=("triage", "derm", "maternal"),
                        help="Which specialist's train.jsonl to score")
    parser.add_argument("--tag", required=True, help="Filename tag for output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.train_set}/train.jsonl ...")
    cases = load_train_cases(args.train_set)
    print(f"  Loaded {len(cases)} cases")
    print(f"  Model: {args.model}")

    with RunLogger(kind="eval_cross_train", args=args) as logger:
        logger.merge_manifest(model=args.model, train_set=args.train_set, n_cases=len(cases))

        result = run_eval.run_single_seed(
            cases,
            seed=args.seed,
            model_path=args.model,
            use_mock=False,
            models_by_specialist=None,
        )

        # Save in the same shape as run_eval.py single-seed results.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"run_{args.tag}_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "model": args.model,
            "mode": f"REAL ({args.model})",
            "case_source": f"train_split:{args.train_set}",
            "train_set": args.train_set,
            "seed": args.seed,
            "n_cases": result.n_cases,
            "weighted_f1": result.weighted_f1,
            "macro_f1": result.macro_f1,
            "red_recall": result.red_recall,
            "escalation_rate": result.escalation_rate,
            "per_class": result.per_class_report,
            "per_specialist": result.per_specialist,
            "predictions": [
                {
                    "case_id": c.case_id,
                    "specialist": c.specialist,
                    "gold": c.gold_level,
                    # Predictions list is in result.predictions (already serialised)
                }
                for c in cases
            ],
            "predictions_full": result.predictions,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n  Weighted F1={result.weighted_f1:.4f}  RED recall={result.red_recall:.4f}")
        print(f"  Per-class: {result.per_class_report}")
        print(f"  Saved to: {out_path}")
        logger.attach_result(out_path)


if __name__ == "__main__":
    main()
