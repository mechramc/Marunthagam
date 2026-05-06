"""
Marunthagam — Update rank ablation with real anchor data points.

Reads the held-out test split eval result for the rank-32 triage GGUF
(run_fusion_only_triage_v2_*.json's per_specialist["triage"] block) and,
when present, the rank-16 result (run_fusion_only_triage_rank16_*.json),
then rewrites eval/results/ablation_rank_comparison.json so:

  - Rank 32 row is anchored to the real held-out test F1 / RED recall.
  - Rank 16 row, if available, is also anchored to real numbers.
  - Other ranks (4, 8, 64) keep their projected values from the original
    training-scaling-law mock, but are clearly labelled "source": "projected".

Usage:
    python eval/scripts/update_rank_ablation_anchor.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
OUT_PATH = RESULTS_DIR / "ablation_rank_comparison.json"


def _latest(prefix: str) -> Optional[Path]:
    cands = sorted(RESULTS_DIR.glob(f"{prefix}*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _triage_subset(path: Path) -> Optional[dict]:
    """Pull the triage-only test split metrics from an eval result file."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return None
    per_spec = d.get("per_specialist") or {}
    if not per_spec and d.get("seed_results"):
        per_spec = d["seed_results"][0].get("per_specialist", {})
    return per_spec.get("triage")


def main() -> None:
    # Read the existing ablation comparison
    if not OUT_PATH.exists():
        print(f"ERROR: {OUT_PATH} not found. Run ablation_rank.py first.")
        return
    with open(OUT_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)

    summaries = payload.get("summaries", [])
    plot_data = payload.get("plot_data", {})

    rank_32_path = _latest("run_fusion_only_triage_v2_")
    rank_32 = _triage_subset(rank_32_path) if rank_32_path else None

    rank_16_path = _latest("run_fusion_only_triage_rank16_")
    rank_16 = _triage_subset(rank_16_path) if rank_16_path else None

    anchors: dict[int, dict] = {}
    if rank_32:
        anchors[32] = {
            "weighted_f1_mean": rank_32["weighted_f1"],
            "weighted_f1_std": 0.0,
            "macro_f1_mean": rank_32["macro_f1"],
            "macro_f1_std": 0.0,
            "red_recall_mean": rank_32["red_recall"],
            "red_recall_std": 0.0,
            "n_seeds": 1,
            "source": "real_held_out_test",
            "n_cases": rank_32["n"],
            "result_file": rank_32_path.name if rank_32_path else None,
        }
    if rank_16:
        anchors[16] = {
            "weighted_f1_mean": rank_16["weighted_f1"],
            "weighted_f1_std": 0.0,
            "macro_f1_mean": rank_16["macro_f1"],
            "macro_f1_std": 0.0,
            "red_recall_mean": rank_16["red_recall"],
            "red_recall_std": 0.0,
            "n_seeds": 1,
            "source": "real_held_out_test",
            "n_cases": rank_16["n"],
            "result_file": rank_16_path.name if rank_16_path else None,
        }

    # Patch summaries
    for s in summaries:
        rank = s.get("rank")
        if rank in anchors:
            s.update(anchors[rank])
            s["seed_results"] = []
        else:
            s["source"] = "projected_from_training_scaling_laws"

    # Patch plot data so the chart shows real anchors
    if plot_data and "ranks" in plot_data:
        for i, rank in enumerate(plot_data["ranks"]):
            if rank in anchors:
                plot_data["weighted_f1_mean"][i] = anchors[rank]["weighted_f1_mean"]
                plot_data["weighted_f1_std"][i] = 0.0
                plot_data["red_recall_mean"][i] = anchors[rank]["red_recall_mean"]
                plot_data["red_recall_std"][i] = 0.0

    payload["anchors"] = anchors
    payload["note"] = (
        "Rank 16 and 32 rows (where present) are real held-out-test numbers on "
        "the triage subset (n=45). Other ranks retain projections from the "
        "training-scaling-law mock; rerun ablation_rank.py + train new ranks "
        "to replace those with real numbers."
    )

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("Updated", OUT_PATH)
    print("Anchors:", json.dumps(anchors, indent=2))


if __name__ == "__main__":
    main()
