"""
Marunthagam — LoRA Rank Ablation Study.

Compares fine-tuning performance across LoRA ranks [4, 8, 16, 32, 64].
Loads pre-saved per-rank results from eval/results/ when available;
falls back to deterministic mock data so the script is immediately runnable.

Produces a rank vs F1 and rank vs RED-recall comparison table, and saves
plot-ready data to eval/results/ablation_rank_comparison.json.

Usage:
    python ablation_rank.py                # mock data if no saved results
    python ablation_rank.py --ranks 4,8,16,32,64
    python ablation_rank.py --seeds 42,137,256
"""

from __future__ import annotations

import argparse
import json
import sys

# Reconfigure stdout/stderr to UTF-8 so Tamil text and box-drawing characters
# render correctly on Windows (which defaults to CP1252 in some environments).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
TARGET_F1: float = 0.80
TARGET_RED_RECALL: float = 0.90
DEFAULT_RANKS: list[int] = [4, 8, 16, 32, 64]
DEFAULT_SEEDS: list[int] = [42, 137, 256]
TRIAGE_LEVELS: list[str] = ["GREEN", "YELLOW", "RED"]

# Filename convention for pre-saved ablation results:
# ablation_rank{rank}_seed{seed}.json
ABLATION_RESULT_PREFIX = "ablation_rank"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# ---------------------------------------------------------------------------
# Mock data model: F1 and RED recall as a function of LoRA rank.
# Based on expected LoRA scaling behaviour — diminishing returns above rank 16,
# with a realistic performance floor at rank 4.
# ---------------------------------------------------------------------------

# Base performance at each rank (before per-seed noise) — empirically motivated
_MOCK_BASE_F1: dict[int, float] = {
    4:  0.67,
    8:  0.74,
    16: 0.81,
    32: 0.84,
    64: 0.85,
}
_MOCK_BASE_RED_RECALL: dict[int, float] = {
    4:  0.71,
    8:  0.82,
    16: 0.91,
    32: 0.93,
    64: 0.94,
}
_MOCK_NOISE_STD: float = 0.015   # Seed-to-seed variation in mock data


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RankSeedResult:
    """Metrics from one (rank, seed) training run."""
    rank: int
    seed: int
    weighted_f1: float
    macro_f1: float
    red_recall: float
    n_cases: int
    source: str    # "loaded" | "mock"


@dataclass
class RankSummary:
    """Aggregated metrics for one rank across all seeds."""
    rank: int
    seeds: list[int]
    weighted_f1_mean: float
    weighted_f1_std: float
    macro_f1_mean: float
    macro_f1_std: float
    red_recall_mean: float
    red_recall_std: float
    n_seeds: int
    source: str    # "loaded" | "mock" | "mixed"
    seed_results: list[RankSeedResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading saved ablation results
# ---------------------------------------------------------------------------

def _ablation_result_path(rank: int, seed: int) -> Path:
    """Return the expected file path for a saved ablation result."""
    return RESULTS_DIR / f"{ABLATION_RESULT_PREFIX}{rank}_seed{seed}.json"


def _load_rank_seed_result(rank: int, seed: int) -> Optional[RankSeedResult]:
    """
    Load a pre-saved ablation result for (rank, seed).

    Returns None if the file does not exist or cannot be parsed.
    Expects keys: weighted_f1, macro_f1, red_recall, n_cases.
    """
    path = _ablation_result_path(rank, seed)
    if not path.exists():
        return None

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"  WARNING: Could not load {path}: {exc} — falling back to mock",
            file=sys.stderr,
        )
        return None

    required_keys = {"weighted_f1", "red_recall"}
    if not required_keys.issubset(data.keys()):
        print(
            f"  WARNING: {path} missing required keys {required_keys - data.keys()} "
            "— falling back to mock",
            file=sys.stderr,
        )
        return None

    return RankSeedResult(
        rank=rank,
        seed=seed,
        weighted_f1=float(data["weighted_f1"]),
        macro_f1=float(data.get("macro_f1", data["weighted_f1"])),
        red_recall=float(data["red_recall"]),
        n_cases=int(data.get("n_cases", 0)),
        source="loaded",
    )


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

def _mock_rank_seed_result(rank: int, seed: int) -> RankSeedResult:
    """
    Generate mock ablation result for (rank, seed).

    Uses a deterministic RNG seeded by (rank, seed) so results are
    reproducible across runs.
    """
    rng = np.random.RandomState(rank * 10000 + seed)  # noqa: NPY002

    base_f1 = _MOCK_BASE_F1.get(rank, 0.70)
    base_rr = _MOCK_BASE_RED_RECALL.get(rank, 0.75)

    weighted_f1 = float(np.clip(
        base_f1 + rng.normal(0, _MOCK_NOISE_STD), 0.0, 1.0
    ))
    macro_f1 = float(np.clip(
        weighted_f1 - rng.uniform(0.01, 0.03), 0.0, 1.0
    ))
    red_recall = float(np.clip(
        base_rr + rng.normal(0, _MOCK_NOISE_STD), 0.0, 1.0
    ))

    return RankSeedResult(
        rank=rank,
        seed=seed,
        weighted_f1=round(weighted_f1, 4),
        macro_f1=round(macro_f1, 4),
        red_recall=round(red_recall, 4),
        n_cases=30,
        source="mock",
    )


# ---------------------------------------------------------------------------
# Collecting results for all ranks
# ---------------------------------------------------------------------------

def collect_rank_results(ranks: list[int], seeds: list[int]) -> list[RankSummary]:
    """
    For each rank, collect results across all seeds.

    Prefers loaded results; falls back to mock per (rank, seed).
    """
    summaries: list[RankSummary] = []

    for rank in ranks:
        seed_results: list[RankSeedResult] = []
        sources: set[str] = set()

        for seed in seeds:
            loaded = _load_rank_seed_result(rank, seed)
            if loaded is not None:
                seed_results.append(loaded)
                sources.add("loaded")
            else:
                mock = _mock_rank_seed_result(rank, seed)
                seed_results.append(mock)
                sources.add("mock")

        wf1_arr = np.array([r.weighted_f1 for r in seed_results])
        mf1_arr = np.array([r.macro_f1 for r in seed_results])
        rr_arr = np.array([r.red_recall for r in seed_results])

        if len(sources) == 1:
            source_label = sources.pop()
        else:
            source_label = "mixed"

        summaries.append(RankSummary(
            rank=rank,
            seeds=seeds,
            weighted_f1_mean=round(float(np.mean(wf1_arr)), 4),
            weighted_f1_std=round(float(np.std(wf1_arr, ddof=0)), 4),
            macro_f1_mean=round(float(np.mean(mf1_arr)), 4),
            macro_f1_std=round(float(np.std(mf1_arr, ddof=0)), 4),
            red_recall_mean=round(float(np.mean(rr_arr)), 4),
            red_recall_std=round(float(np.std(rr_arr, ddof=0)), 4),
            n_seeds=len(seed_results),
            source=source_label,
            seed_results=seed_results,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------

def find_minimum_rank(summaries: list[RankSummary], target_f1: float) -> Optional[int]:
    """
    Return the minimum LoRA rank whose mean weighted F1 meets target_f1.

    Returns None if no rank meets the target.
    """
    for summary in sorted(summaries, key=lambda s: s.rank):
        if summary.weighted_f1_mean >= target_f1:
            return summary.rank
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _pass_fail(value: float, target: float) -> str:
    return "PASS" if value >= target else "FAIL"


def print_comparison_table(summaries: list[RankSummary]) -> None:
    """Print an ASCII rank comparison table to stdout."""
    sep = "─" * 80
    print(sep)
    print("  MARUNTHAGAM LORA RANK ABLATION — COMPARISON TABLE")
    print(sep)
    print(
        f"  {'Rank':>6}  {'Wt-F1 mean':>12}  {'±std':>6}  "
        f"{'RED-Recall mean':>16}  {'±std':>6}  "
        f"{'F1 target':>10}  {'RR target':>10}  {'Source':>8}"
    )
    print(
        f"  {'─'*6}  {'─'*12}  {'─'*6}  "
        f"{'─'*16}  {'─'*6}  "
        f"{'─'*10}  {'─'*10}  {'─'*8}"
    )
    for s in sorted(summaries, key=lambda x: x.rank):
        f1_result = _pass_fail(s.weighted_f1_mean, TARGET_F1)
        rr_result = _pass_fail(s.red_recall_mean, TARGET_RED_RECALL)
        print(
            f"  {s.rank:>6}  {s.weighted_f1_mean:>12.4f}  {s.weighted_f1_std:>6.4f}  "
            f"{s.red_recall_mean:>16.4f}  {s.red_recall_std:>6.4f}  "
            f"{f1_result:>10}  {rr_result:>10}  {s.source:>8}"
        )
    print(sep)
    print()

    # Break-even analysis
    min_rank_f1 = find_minimum_rank(summaries, TARGET_F1)
    if min_rank_f1 is not None:
        print(f"  Break-even for F1 > {TARGET_F1}: minimum rank = {min_rank_f1}")
    else:
        print(f"  Break-even for F1 > {TARGET_F1}: NO rank meets target — need more data or higher rank")

    # RED recall break-even
    min_rank_rr = find_minimum_rank(
        [
            RankSummary(
                rank=s.rank,
                seeds=s.seeds,
                weighted_f1_mean=s.red_recall_mean,   # reuse field for break-even lookup
                weighted_f1_std=s.red_recall_std,
                macro_f1_mean=s.macro_f1_mean,
                macro_f1_std=s.macro_f1_std,
                red_recall_mean=s.red_recall_mean,
                red_recall_std=s.red_recall_std,
                n_seeds=s.n_seeds,
                source=s.source,
            )
            for s in summaries
        ],
        TARGET_RED_RECALL,
    )
    if min_rank_rr is not None:
        print(f"  Break-even for RED recall > {TARGET_RED_RECALL}: minimum rank = {min_rank_rr}")
    else:
        print(f"  Break-even for RED recall > {TARGET_RED_RECALL}: NO rank meets target")

    print()

    # Per-rank seed detail
    print("  Per-seed breakdown:")
    print(f"  {'Rank':>6}  {'Seed':>8}  {'Wt-F1':>10}  {'RED-Recall':>12}  {'Source':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*8}")
    for s in sorted(summaries, key=lambda x: x.rank):
        for r in sorted(s.seed_results, key=lambda x: x.seed):
            print(
                f"  {r.rank:>6}  {r.seed:>8}  {r.weighted_f1:>10.4f}  "
                f"{r.red_recall:>12.4f}  {r.source:>8}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_comparison(summaries: list[RankSummary], timestamp: str) -> Path:
    """Save plot-ready comparison data to eval/results/ablation_rank_comparison.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ablation_rank_comparison.json"

    min_rank_f1 = find_minimum_rank(summaries, TARGET_F1)
    min_rank_rr_summary = find_minimum_rank(
        [
            RankSummary(
                rank=s.rank,
                seeds=s.seeds,
                weighted_f1_mean=s.red_recall_mean,
                weighted_f1_std=s.red_recall_std,
                macro_f1_mean=s.macro_f1_mean,
                macro_f1_std=s.macro_f1_std,
                red_recall_mean=s.red_recall_mean,
                red_recall_std=s.red_recall_std,
                n_seeds=s.n_seeds,
                source=s.source,
            )
            for s in summaries
        ],
        TARGET_RED_RECALL,
    )

    payload = {
        "timestamp": timestamp,
        "targets": {
            "weighted_f1": TARGET_F1,
            "red_recall": TARGET_RED_RECALL,
        },
        "break_even": {
            "min_rank_for_f1_target": min_rank_f1,
            "min_rank_for_red_recall_target": min_rank_rr_summary,
        },
        # Flat arrays for plotting (rank vs metric)
        "plot_data": {
            "ranks": [s.rank for s in sorted(summaries, key=lambda x: x.rank)],
            "weighted_f1_mean": [s.weighted_f1_mean for s in sorted(summaries, key=lambda x: x.rank)],
            "weighted_f1_std": [s.weighted_f1_std for s in sorted(summaries, key=lambda x: x.rank)],
            "red_recall_mean": [s.red_recall_mean for s in sorted(summaries, key=lambda x: x.rank)],
            "red_recall_std": [s.red_recall_std for s in sorted(summaries, key=lambda x: x.rank)],
        },
        "summaries": [
            {
                "rank": s.rank,
                "source": s.source,
                "n_seeds": s.n_seeds,
                "weighted_f1_mean": s.weighted_f1_mean,
                "weighted_f1_std": s.weighted_f1_std,
                "macro_f1_mean": s.macro_f1_mean,
                "macro_f1_std": s.macro_f1_std,
                "red_recall_mean": s.red_recall_mean,
                "red_recall_std": s.red_recall_std,
                "seed_results": [
                    {
                        "seed": r.seed,
                        "weighted_f1": r.weighted_f1,
                        "macro_f1": r.macro_f1,
                        "red_recall": r.red_recall,
                        "n_cases": r.n_cases,
                        "source": r.source,
                    }
                    for r in sorted(s.seed_results, key=lambda x: x.seed)
                ],
            }
            for s in sorted(summaries, key=lambda x: x.rank)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    parts = raw.split(",")
    values: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid integer value '{part}'."
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one value.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam LoRA rank ablation study. Compares performance across "
            "LoRA ranks, loading pre-saved results from eval/results/ when available "
            "and falling back to deterministic mock data otherwise. Saves a "
            "rank-comparison JSON to eval/results/ablation_rank_comparison.json."
        )
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RANKS),
        metavar="RANK_LIST",
        help=(
            f"Comma-separated LoRA ranks to compare. "
            f"Default: {','.join(str(r) for r in DEFAULT_RANKS)}."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        metavar="SEED_LIST",
        help=(
            f"Comma-separated seeds used for training runs. "
            f"Default: {','.join(str(s) for s in DEFAULT_SEEDS)}."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        ranks = _parse_int_list(args.ranks)
        seeds = _parse_int_list(args.seeds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return  # unreachable; satisfies type checker

    for rank in ranks:
        if rank < 1:
            parser.error(f"Rank must be >= 1, got {rank}.")
    for seed in seeds:
        if seed < 0:
            parser.error(f"Seed must be non-negative, got {seed}.")

    print(f"LoRA rank ablation: ranks={ranks}, seeds={seeds}")
    print(f"  Looking for pre-saved results in: {RESULTS_DIR}\n")

    summaries = collect_rank_results(ranks=ranks, seeds=seeds)

    print_comparison_table(summaries)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = save_comparison(summaries, timestamp)
    print(f"  Plot data saved to: {output_path}")


if __name__ == "__main__":
    main()
