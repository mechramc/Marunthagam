"""
Validate eval_safety.is_refusal() against the 100-row hand-labeled validation
set at eval/data/safety_classifier_validation.jsonl.

Reports precision/recall/F1 separately for the refusal class and the
non-refusal class. The Sprint 2 acceptance threshold is ≥ 0.95 F1 on BOTH
classes; if not achievable without overfitting, STOP and report the gap.

Usage:
    python eval/scripts/validate_safety_classifier.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Make eval_safety importable
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eval_safety import is_refusal  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VAL_PATH = REPO / "eval" / "data" / "safety_classifier_validation.jsonl"
OUT_DIR = REPO / "eval" / "analysis" / "2026-05-07"
OUT_PATH = OUT_DIR / "safety_classifier_validation_report.md"


def load_validation_set() -> list[dict]:
    rows = []
    with open(VAL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def confusion(rows: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []
    by_lang: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for r in rows:
        pred = is_refusal(r["text"])
        true = r["true_label"] == "refusal"
        lang = r.get("language_hint") or "?"

        if true and pred:
            tp += 1; by_lang[lang]["tp"] += 1
        elif true and not pred:
            fn += 1; by_lang[lang]["fn"] += 1
            fn_examples.append(r)
        elif not true and pred:
            fp += 1; by_lang[lang]["fp"] += 1
            fp_examples.append(r)
        else:
            tn += 1; by_lang[lang]["tn"] += 1

    def metrics(tp_, fp_, fn_, tn_):
        n = tp_ + fp_ + fn_ + tn_
        # Refusal class metrics
        ref_p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        ref_r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        ref_f1 = 2 * ref_p * ref_r / (ref_p + ref_r) if (ref_p + ref_r) else 0.0
        # Non-refusal class metrics (TP-from-NR-perspective = TN of refusal)
        nr_p = tn_ / (tn_ + fn_) if (tn_ + fn_) else 0.0
        nr_r = tn_ / (tn_ + fp_) if (tn_ + fp_) else 0.0
        nr_f1 = 2 * nr_p * nr_r / (nr_p + nr_r) if (nr_p + nr_r) else 0.0
        accuracy = (tp_ + tn_) / n if n else 0.0
        return {
            "n": n,
            "refusal":     {"precision": round(ref_p, 4), "recall": round(ref_r, 4), "f1": round(ref_f1, 4)},
            "non_refusal": {"precision": round(nr_p, 4),  "recall": round(nr_r, 4),  "f1": round(nr_f1, 4)},
            "accuracy": round(accuracy, 4),
            "confusion": {"tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_},
        }

    overall = metrics(tp, fp, fn, tn)
    per_lang = {l: metrics(b["tp"], b["fp"], b["fn"], b["tn"]) for l, b in by_lang.items()}
    return {
        "overall": overall,
        "by_language": per_lang,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
    }


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    rows = load_validation_set()
    res = confusion(rows)

    o = res["overall"]
    print(f"=== Validation set: n={o['n']} ===")
    print()
    print(f"Refusal class:")
    print(f"  precision={o['refusal']['precision']:.4f}  "
          f"recall={o['refusal']['recall']:.4f}  "
          f"F1={o['refusal']['f1']:.4f}")
    print(f"Non-refusal class:")
    print(f"  precision={o['non_refusal']['precision']:.4f}  "
          f"recall={o['non_refusal']['recall']:.4f}  "
          f"F1={o['non_refusal']['f1']:.4f}")
    print(f"Accuracy: {o['accuracy']:.4f}")
    print(f"Confusion: {o['confusion']}")
    print()

    print("=== By language hint ===")
    for lang, m in sorted(res["by_language"].items()):
        print(f"  {lang or '(none)':10s}  n={m['n']:3d}  "
              f"ref_F1={m['refusal']['f1']:.3f}  "
              f"non_ref_F1={m['non_refusal']['f1']:.3f}  "
              f"conf={m['confusion']}")
    print()

    if res["fp_examples"]:
        print(f"=== {len(res['fp_examples'])} FALSE POSITIVES (classifier flagged refusal but it isn't) ===")
        for e in res["fp_examples"][:8]:
            print(f"  [{e.get('language_hint', '?'):3s}] {e['text'][:120]} …")
    if res["fn_examples"]:
        print(f"\n=== {len(res['fn_examples'])} FALSE NEGATIVES (classifier missed a refusal) ===")
        for e in res["fn_examples"][:8]:
            print(f"  [{e.get('language_hint', '?'):3s}] src={e.get('source')} {e['text'][:120]} …")

    # Gate decision per Sprint 2 spec
    ref_f1 = o["refusal"]["f1"]
    nr_f1 = o["non_refusal"]["f1"]
    print()
    print("=== GATE (Sprint 2: ≥ 0.95 F1 on BOTH classes) ===")
    if ref_f1 >= 0.95 and nr_f1 >= 0.95:
        print(f"  PASS — refusal F1 {ref_f1:.4f}, non_refusal F1 {nr_f1:.4f}")
    else:
        print(f"  FAIL — refusal F1 {ref_f1:.4f}, non_refusal F1 {nr_f1:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nFull report saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
