"""
Marunthagam — Full Evaluation Suite Orchestrator.

Loads test cases from all three specialist fixture files (triage, derm, maternal),
calls triage_classify() via real llama.cpp subprocess or a deterministic mock,
computes classification metrics, and saves results to eval/results/.

Usage:
    python run_eval.py --mock
    python run_eval.py --model /path/to/model.gguf
    python run_eval.py --mock --seeds 42,137,256
    python run_eval.py --model /path/to/model.gguf --seeds 42,137,256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

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
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Module-level constants — no magic numbers
# ---------------------------------------------------------------------------
TARGET_F1: float = 0.80
TARGET_RED_RECALL: float = 0.90
TARGET_CHRF: float = 0.60
DISCLAIMER_TEXT: str = "இது மருத்துவ ஆலோசனை அல்ல"
TRIAGE_LEVELS: list[str] = ["GREEN", "YELLOW", "RED"]
SPECIALISTS: list[str] = ["triage", "derm", "maternal"]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "training" / "data" / "fixtures"
EVAL_DATA_DIR = REPO_ROOT / "eval" / "data"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Mock confidence values by level — deterministic and clinically plausible
_MOCK_CONFIDENCE: dict[str, float] = {
    "GREEN": 0.91,
    "YELLOW": 0.82,
    "RED": 0.95,
}

# Mock adds small per-seed noise to simulate real variation
_MOCK_NOISE_STD: float = 0.02


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation case loaded from a fixture or baseline file."""
    specialist: str                          # triage | derm | maternal
    verbal_symptoms: str
    age_group: str
    duration_days: int
    gold_level: str                          # GREEN | YELLOW | RED
    case_id: Optional[str] = None


@dataclass
class PredictedOutput:
    """Normalised output from triage_classify(), real or mock."""
    level: str
    confidence: float
    escalation_flag: bool
    reasoning_chain: str
    next_steps_tamil: str
    disclaimer: str = DISCLAIMER_TEXT


@dataclass
class SeedResult:
    """Metrics for a single seed run."""
    seed: int
    weighted_f1: float
    macro_f1: float
    red_recall: float
    green_f1: float
    yellow_f1: float
    red_f1: float
    escalation_rate: float
    n_cases: int
    per_class_report: dict[str, dict[str, float]]


@dataclass
class AggregatedResult:
    """Mean ± std across multiple seed runs."""
    seeds: list[int]
    weighted_f1_mean: float
    weighted_f1_std: float
    macro_f1_mean: float
    macro_f1_std: float
    red_recall_mean: float
    red_recall_std: float
    escalation_rate_mean: float
    escalation_rate_std: float
    seed_results: list[SeedResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _extract_gold_level(record: dict) -> str:
    """
    Extract gold triage level from a fixture record.

    Fixture records store the gold label in triage_result.level.
    Baseline records store it in gold_level directly.
    """
    if "triage_result" in record and "level" in record["triage_result"]:
        return record["triage_result"]["level"]
    if "gold_level" in record:
        return record["gold_level"]
    raise ValueError(f"Cannot determine gold level from record keys: {list(record.keys())}")


def load_fixture(specialist: str) -> list[TestCase]:
    """
    Load test cases from a specialist fixture JSONL file.

    Each line is a JSON object with function_call_args and triage_result.
    """
    fixture_path = FIXTURES_DIR / f"{specialist}_reviewed.jsonl"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    cases: list[TestCase] = []
    with open(fixture_path, encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {fixture_path}:{line_num}: {exc}"
                ) from exc

            args = record.get("function_call_args", {})
            gold_level = _extract_gold_level(record)

            cases.append(TestCase(
                specialist=specialist,
                verbal_symptoms=args.get("verbal_symptoms", record.get("symptom_description", "")),
                age_group=args.get("patient_age_group", record.get("age_group", "adult")),
                duration_days=int(args.get("duration_days", record.get("duration_days", 1))),
                gold_level=gold_level,
                case_id=f"{specialist}_{line_num:03d}",
            ))

    return cases


def load_baseline_examples() -> list[TestCase]:
    """Load the 20 baseline evaluation examples."""
    baseline_path = EVAL_DATA_DIR / "baseline_examples.json"
    if not baseline_path.exists():
        return []

    with open(baseline_path, encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    cases: list[TestCase] = []
    for idx, record in enumerate(records, start=1):
        cases.append(TestCase(
            specialist="triage",
            verbal_symptoms=record.get("symptom_description", ""),
            age_group=record.get("age_group", "adult"),
            duration_days=int(record.get("duration_days", 1)),
            gold_level=record["gold_level"],
            case_id=f"baseline_{idx:03d}",
        ))
    return cases


def load_all_cases() -> list[TestCase]:
    """Load all test cases: fixtures for each specialist + baseline examples."""
    all_cases: list[TestCase] = []
    for specialist in SPECIALISTS:
        try:
            cases = load_fixture(specialist)
            all_cases.extend(cases)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {specialist}", file=sys.stderr)

    baseline = load_baseline_examples()
    all_cases.extend(baseline)
    return all_cases


# ---------------------------------------------------------------------------
# Mock inference
# ---------------------------------------------------------------------------

def _mock_predict(case: TestCase, seed: int) -> PredictedOutput:
    """
    Deterministic mock of triage_classify().

    Returns the gold level with high confidence, plus small Gaussian noise
    seeded per (case_id, seed) for reproducible but non-identical results.
    Introduces a ~10% error rate for realism: one level below gold.
    """
    rng = random.Random(f"{case.case_id}-{seed}")
    noise_rng = random.Random(f"{case.case_id}-{seed}-noise")

    base_confidence = _MOCK_CONFIDENCE[case.gold_level]
    confidence = base_confidence + noise_rng.gauss(0, _MOCK_NOISE_STD)
    confidence = max(0.0, min(1.0, confidence))

    # 10% chance of one-level degradation (GREEN→YELLOW, YELLOW→RED, RED→RED)
    error_roll = rng.random()
    level = case.gold_level
    if error_roll < 0.10:
        idx = TRIAGE_LEVELS.index(case.gold_level)
        level = TRIAGE_LEVELS[min(idx + 1, len(TRIAGE_LEVELS) - 1)]
        confidence = max(0.0, confidence - 0.15)

    escalation_flag = confidence < 0.70 or level == "RED"

    return PredictedOutput(
        level=level,
        confidence=round(confidence, 4),
        escalation_flag=escalation_flag,
        reasoning_chain="[MOCK] பரிசோதனை முடிவு",
        next_steps_tamil="[MOCK] அடுத்த படி",
        disclaimer=DISCLAIMER_TEXT,
    )


# ---------------------------------------------------------------------------
# Real llama.cpp inference
# ---------------------------------------------------------------------------

_LLAMA_PROMPT_TEMPLATE = """\
You are a clinical triage assistant. Call triage_classify() for the following case.

Patient age group: {age_group}
Duration of symptoms: {duration_days} days
Symptoms: {verbal_symptoms}

Respond only with a valid JSON tool call.
"""


def _real_predict(case: TestCase, model_path: str) -> PredictedOutput:
    """
    Call triage_classify() by running llama.cpp as a subprocess.

    Parses the <tool_call>...</tool_call> JSON from the model's stdout.
    Raises RuntimeError if the model returns malformed output.
    """
    prompt = _LLAMA_PROMPT_TEMPLATE.format(
        age_group=case.age_group,
        duration_days=case.duration_days,
        verbal_symptoms=case.verbal_symptoms,
    )

    cmd = [
        "llama-cli",
        "--model", model_path,
        "--prompt", prompt,
        "--n-predict", "512",
        "--temp", "0.0",
        "--no-display-prompt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "llama-cli not found in PATH. Install llama.cpp or use --mock."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"llama.cpp timed out after 120s for case {case.case_id}"
        ) from exc

    raw_output = result.stdout

    # Parse tool call JSON from model output
    import re
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    match = tool_call_pattern.search(raw_output)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed tool_call JSON for case {case.case_id}: {exc}"
            ) from exc
    else:
        try:
            parsed = json.loads(raw_output.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"No parseable JSON in llama.cpp output for case {case.case_id}.\n"
                f"Raw output: {raw_output[:300]}"
            ) from exc

    # Support both flat output and nested result key
    output_data = parsed.get("result", parsed)
    level = str(output_data.get("level", "GREEN")).upper()
    if level not in TRIAGE_LEVELS:
        raise RuntimeError(
            f"Unexpected triage level '{level}' for case {case.case_id}"
        )

    confidence = float(output_data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return PredictedOutput(
        level=level,
        confidence=confidence,
        escalation_flag=bool(output_data.get("escalation_flag", confidence < 0.70)),
        reasoning_chain=str(output_data.get("reasoning_chain", "")),
        next_steps_tamil=str(output_data.get("next_steps_tamil", "")),
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    cases: list[TestCase],
    predictions: list[PredictedOutput],
    seed: int,
) -> SeedResult:
    """
    Compute per-class and aggregate classification metrics for one seed run.

    RED recall is treated as a critical safety metric.
    """
    gold_labels = [c.gold_level for c in cases]
    pred_labels = [p.level for p in predictions]
    escalation_flags = [p.escalation_flag for p in predictions]

    report_dict: dict[str, dict[str, float]] = classification_report(
        gold_labels,
        pred_labels,
        labels=TRIAGE_LEVELS,
        output_dict=True,
        zero_division=0,
    )  # type: ignore[assignment]

    weighted_f1 = f1_score(
        gold_labels, pred_labels, average="weighted", labels=TRIAGE_LEVELS, zero_division=0
    )
    macro_f1 = f1_score(
        gold_labels, pred_labels, average="macro", labels=TRIAGE_LEVELS, zero_division=0
    )

    red_stats = report_dict.get("RED", {})
    red_recall = float(red_stats.get("recall", 0.0))

    escalation_rate = sum(escalation_flags) / max(len(escalation_flags), 1)

    return SeedResult(
        seed=seed,
        weighted_f1=round(float(weighted_f1), 4),
        macro_f1=round(float(macro_f1), 4),
        red_recall=round(red_recall, 4),
        green_f1=round(float(report_dict.get("GREEN", {}).get("f1-score", 0.0)), 4),
        yellow_f1=round(float(report_dict.get("YELLOW", {}).get("f1-score", 0.0)), 4),
        red_f1=round(float(report_dict.get("RED", {}).get("f1-score", 0.0)), 4),
        escalation_rate=round(escalation_rate, 4),
        n_cases=len(cases),
        per_class_report={
            level: {
                "precision": round(float(report_dict.get(level, {}).get("precision", 0.0)), 4),
                "recall": round(float(report_dict.get(level, {}).get("recall", 0.0)), 4),
                "f1": round(float(report_dict.get(level, {}).get("f1-score", 0.0)), 4),
                "support": int(report_dict.get(level, {}).get("support", 0)),
            }
            for level in TRIAGE_LEVELS
        },
    )


def aggregate_seed_results(seed_results: list[SeedResult]) -> AggregatedResult:
    """Compute mean ± std across seed runs."""
    def _stats(values: list[float]) -> tuple[float, float]:
        arr = np.array(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    wf1_mean, wf1_std = _stats([r.weighted_f1 for r in seed_results])
    mf1_mean, mf1_std = _stats([r.macro_f1 for r in seed_results])
    rr_mean, rr_std = _stats([r.red_recall for r in seed_results])
    esc_mean, esc_std = _stats([r.escalation_rate for r in seed_results])

    return AggregatedResult(
        seeds=[r.seed for r in seed_results],
        weighted_f1_mean=round(wf1_mean, 4),
        weighted_f1_std=round(wf1_std, 4),
        macro_f1_mean=round(mf1_mean, 4),
        macro_f1_std=round(mf1_std, 4),
        red_recall_mean=round(rr_mean, 4),
        red_recall_std=round(rr_std, 4),
        escalation_rate_mean=round(esc_mean, 4),
        escalation_rate_std=round(esc_std, 4),
        seed_results=seed_results,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _pass_fail(value: float, target: float) -> str:
    return "PASS" if value >= target else "FAIL"


def print_summary_table(result: SeedResult | AggregatedResult) -> None:
    """Print a clean human-readable summary to stdout."""
    separator = "─" * 60
    print(separator)
    print("  MARUNTHAGAM EVAL RESULTS")
    print(separator)

    if isinstance(result, SeedResult):
        print(f"  Seed:            {result.seed}")
        print(f"  Total cases:     {result.n_cases}")
        print(f"  Weighted F1:     {result.weighted_f1:.4f}   target >{TARGET_F1}  "
              f"→ {_pass_fail(result.weighted_f1, TARGET_F1)}")
        print(f"  Macro F1:        {result.macro_f1:.4f}")
        print(f"  RED recall:      {result.red_recall:.4f}   target >{TARGET_RED_RECALL}  "
              f"→ {_pass_fail(result.red_recall, TARGET_RED_RECALL)}")
        print(f"  Escalation rate: {result.escalation_rate:.4f}")
        print()
        print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for level in TRIAGE_LEVELS:
            stats = result.per_class_report[level]
            print(f"  {level:<10} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
                  f"{stats['f1']:>10.4f} {stats['support']:>10d}")
    else:
        print(f"  Seeds:           {result.seeds}")
        print(f"  Weighted F1:     {result.weighted_f1_mean:.4f} ± {result.weighted_f1_std:.4f}"
              f"   target >{TARGET_F1}  "
              f"→ {_pass_fail(result.weighted_f1_mean, TARGET_F1)}")
        print(f"  Macro F1:        {result.macro_f1_mean:.4f} ± {result.macro_f1_std:.4f}")
        print(f"  RED recall:      {result.red_recall_mean:.4f} ± {result.red_recall_std:.4f}"
              f"   target >{TARGET_RED_RECALL}  "
              f"→ {_pass_fail(result.red_recall_mean, TARGET_RED_RECALL)}")
        print(f"  Escalation rate: {result.escalation_rate_mean:.4f} ± {result.escalation_rate_std:.4f}")

    print(separator)


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------

def run_single_seed(
    cases: list[TestCase],
    seed: int,
    model_path: Optional[str],
    use_mock: bool,
) -> SeedResult:
    """Run inference and compute metrics for one seed."""
    random.seed(seed)
    np.random.seed(seed)

    predictions: list[PredictedOutput] = []
    for case in cases:
        if use_mock:
            pred = _mock_predict(case, seed=seed)
        else:
            assert model_path is not None
            pred = _real_predict(case, model_path)
        predictions.append(pred)

    return compute_metrics(cases, predictions, seed=seed)


def run_eval(
    model_path: Optional[str],
    use_mock: bool,
    seeds: list[int],
) -> AggregatedResult | SeedResult:
    """
    Main eval entry point.

    Loads all test cases, runs inference for each seed, aggregates results,
    and saves to eval/results/.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test cases from {FIXTURES_DIR} and {EVAL_DATA_DIR} ...")
    cases = load_all_cases()
    if not cases:
        raise RuntimeError(
            "No test cases loaded. Check that fixture files exist in "
            f"{FIXTURES_DIR}"
        )
    print(f"  Loaded {len(cases)} cases across {len(SPECIALISTS)} specialists + baseline")

    mode = "MOCK" if use_mock else f"REAL ({model_path})"
    print(f"  Inference mode: {mode}")
    print(f"  Seeds: {seeds}\n")

    seed_results: list[SeedResult] = []
    for seed in seeds:
        print(f"  Running seed {seed} ...")
        start_time = time.monotonic()
        result = run_single_seed(cases, seed=seed, model_path=model_path, use_mock=use_mock)
        elapsed = time.monotonic() - start_time
        print(f"    Weighted F1={result.weighted_f1:.4f}  RED recall={result.red_recall:.4f}  "
              f"({elapsed:.1f}s)")
        seed_results.append(result)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{timestamp}.json"

    if len(seeds) > 1:
        aggregated = aggregate_seed_results(seed_results)
        print()
        print_summary_table(aggregated)
        payload = {
            "timestamp": timestamp,
            "model": model_path if model_path else "mock",
            "mode": mode,
            "seeds": seeds,
            "n_cases": len(cases),
            "targets": {
                "weighted_f1": TARGET_F1,
                "red_recall": TARGET_RED_RECALL,
                "chrf": TARGET_CHRF,
            },
            "aggregated": {
                "weighted_f1_mean": aggregated.weighted_f1_mean,
                "weighted_f1_std": aggregated.weighted_f1_std,
                "macro_f1_mean": aggregated.macro_f1_mean,
                "macro_f1_std": aggregated.macro_f1_std,
                "red_recall_mean": aggregated.red_recall_mean,
                "red_recall_std": aggregated.red_recall_std,
                "escalation_rate_mean": aggregated.escalation_rate_mean,
                "escalation_rate_std": aggregated.escalation_rate_std,
            },
            "seed_results": [
                {
                    "seed": r.seed,
                    "weighted_f1": r.weighted_f1,
                    "macro_f1": r.macro_f1,
                    "red_recall": r.red_recall,
                    "escalation_rate": r.escalation_rate,
                    "n_cases": r.n_cases,
                    "per_class": r.per_class_report,
                }
                for r in seed_results
            ],
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {output_path}")
        return aggregated

    # Single seed
    single = seed_results[0]
    print()
    print_summary_table(single)
    payload = {
        "timestamp": timestamp,
        "model": model_path if model_path else "mock",
        "mode": mode,
        "seed": single.seed,
        "n_cases": single.n_cases,
        "targets": {
            "weighted_f1": TARGET_F1,
            "red_recall": TARGET_RED_RECALL,
            "chrf": TARGET_CHRF,
        },
        "weighted_f1": single.weighted_f1,
        "macro_f1": single.macro_f1,
        "red_recall": single.red_recall,
        "escalation_rate": single.escalation_rate,
        "per_class": single.per_class_report,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")
    return single


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_seeds(raw: str) -> list[int]:
    """Parse comma-separated seed list, e.g. '42,137,256'."""
    parts = raw.split(",")
    seeds: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid seed value '{part}' — must be an integer."
            ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed must be provided.")
    return seeds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam full evaluation suite. Runs triage_classify() on all "
            "specialist fixture cases, computes P/R/F1 and RED recall, and saves "
            "results to eval/results/."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="GGUF_PATH",
        help="Path to the quantised GGUF model file for real inference via llama.cpp.",
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use the deterministic mock predictor instead of a real model. "
            "Simulates ~90%% accuracy with seed-specific noise — useful for "
            "testing the eval pipeline before model weights are available."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        metavar="SEED_LIST",
        help=(
            "Comma-separated list of random seeds, e.g. '42,137,256'. "
            "When multiple seeds are given, results are aggregated as mean ± std. "
            "Default: 42."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        seeds = _parse_seeds(args.seeds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    model_path: Optional[str] = args.model if not args.mock else None

    try:
        run_eval(
            model_path=model_path,
            use_mock=args.mock,
            seeds=seeds,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
