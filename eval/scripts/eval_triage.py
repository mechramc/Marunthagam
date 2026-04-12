"""
Marunthagam — Triage-Specific Detailed Evaluation.

Computes per-class P/R/F1/support, macro/weighted F1, RED recall with 95% CI
via bootstrap, and an ASCII confusion matrix.

Usage:
    python eval_triage.py --split test --mock
    python eval_triage.py --split all --model /path/to/model.gguf
    python eval_triage.py --split val --mock --seed 137
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys

# Reconfigure stdout/stderr to UTF-8 so Tamil text and box-drawing characters
# render correctly on Windows (which defaults to CP1252 in some environments).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
TARGET_F1: float = 0.80
TARGET_RED_RECALL: float = 0.90
TRIAGE_LEVELS: list[str] = ["GREEN", "YELLOW", "RED"]
DISCLAIMER_TEXT: str = "இது மருத்துவ ஆலோசனை அல்ல"

BOOTSTRAP_N_SAMPLES: int = 1000
CI_ALPHA: float = 0.05          # 95% confidence interval
SPLIT_RATIOS: dict[str, tuple[float, float]] = {
    "train": (0.0, 0.8),
    "val":   (0.8, 0.9),
    "test":  (0.9, 1.0),
    "all":   (0.0, 1.0),
}
VALID_SPLITS: list[str] = list(SPLIT_RATIOS.keys())

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "training" / "data" / "fixtures"
EVAL_DATA_DIR = REPO_ROOT / "eval" / "data"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Mock confidence values by level
_MOCK_CONFIDENCE: dict[str, float] = {
    "GREEN": 0.91,
    "YELLOW": 0.82,
    "RED": 0.95,
}
_MOCK_NOISE_STD: float = 0.02
_MOCK_ERROR_RATE: float = 0.10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TriageCase:
    """A single triage evaluation case."""
    case_id: str
    verbal_symptoms: str
    age_group: str
    duration_days: int
    gold_level: str


@dataclass
class TriagePrediction:
    """Model output for a single triage case."""
    level: str
    confidence: float
    escalation_flag: bool


@dataclass
class BootstrapCI:
    """95% confidence interval from bootstrap resampling."""
    point_estimate: float
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"{self.point_estimate:.4f} [CI: {self.lower:.4f}–{self.upper:.4f}]"


@dataclass
class ClassMetrics:
    """Precision, recall, F1, support for a single class."""
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TriageEvalResult:
    """Complete evaluation result for a triage run."""
    split: str
    seed: int
    n_cases: int
    per_class: dict[str, ClassMetrics]
    macro_f1: float
    weighted_f1: float
    red_recall: float
    red_recall_ci: BootstrapCI
    weighted_f1_ci: BootstrapCI
    escalation_rate: float
    confusion: list[list[int]]     # rows=gold, cols=pred, order=TRIAGE_LEVELS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_gold_level(record: dict) -> str:
    """Extract gold triage level from a fixture or baseline record."""
    if "triage_result" in record and "level" in record["triage_result"]:
        return record["triage_result"]["level"]
    if "gold_level" in record:
        return record["gold_level"]
    raise ValueError(
        f"Cannot determine gold level from record keys: {list(record.keys())}"
    )


def _load_triage_fixture() -> list[TriageCase]:
    """Load triage cases from triage_reviewed.jsonl."""
    fixture_path = FIXTURES_DIR / "triage_reviewed.jsonl"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Triage fixture not found: {fixture_path}")

    cases: list[TriageCase] = []
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

            cases.append(TriageCase(
                case_id=f"triage_fixture_{line_num:03d}",
                verbal_symptoms=args.get("verbal_symptoms", record.get("symptom_description", "")),
                age_group=args.get("patient_age_group", record.get("age_group", "adult")),
                duration_days=int(args.get("duration_days", record.get("duration_days", 1))),
                gold_level=gold_level,
            ))
    return cases


def _load_baseline_cases() -> list[TriageCase]:
    """Load triage cases from baseline_examples.json."""
    baseline_path = EVAL_DATA_DIR / "baseline_examples.json"
    if not baseline_path.exists():
        return []

    with open(baseline_path, encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)

    return [
        TriageCase(
            case_id=f"baseline_{idx:03d}",
            verbal_symptoms=r.get("symptom_description", ""),
            age_group=r.get("age_group", "adult"),
            duration_days=int(r.get("duration_days", 1)),
            gold_level=r["gold_level"],
        )
        for idx, r in enumerate(records, start=1)
    ]


def _apply_split(cases: list[TriageCase], split: str, seed: int) -> list[TriageCase]:
    """
    Apply an 80/10/10 train/val/test split using a seeded shuffle.

    'all' returns every case. Split is always reproducible given the same seed.
    """
    if split == "all":
        return cases

    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)

    start_ratio, end_ratio = SPLIT_RATIOS[split]
    n = len(shuffled)
    start_idx = int(n * start_ratio)
    end_idx = int(n * end_ratio)
    subset = shuffled[start_idx:end_idx]

    if not subset:
        raise ValueError(
            f"Split '{split}' produced 0 cases from {n} total. "
            "Try '--split all' or add more fixture data."
        )
    return subset


def load_triage_cases(split: str, seed: int) -> list[TriageCase]:
    """Load and split all triage cases (fixture + baseline)."""
    fixture_cases = _load_triage_fixture()
    baseline_cases = _load_baseline_cases()
    all_cases = fixture_cases + baseline_cases
    if not all_cases:
        raise RuntimeError(
            "No triage cases found. Ensure fixture and baseline files exist."
        )
    return _apply_split(all_cases, split=split, seed=seed)


# ---------------------------------------------------------------------------
# Mock inference
# ---------------------------------------------------------------------------

def _mock_predict(case: TriageCase, seed: int) -> TriagePrediction:
    """Deterministic mock with ~10% error rate and seed-specific noise."""
    rng = random.Random(f"{case.case_id}-{seed}")
    noise_rng = random.Random(f"{case.case_id}-{seed}-noise")

    base_confidence = _MOCK_CONFIDENCE[case.gold_level]
    confidence = base_confidence + noise_rng.gauss(0, _MOCK_NOISE_STD)
    confidence = max(0.0, min(1.0, confidence))

    level = case.gold_level
    if rng.random() < _MOCK_ERROR_RATE:
        idx = TRIAGE_LEVELS.index(case.gold_level)
        level = TRIAGE_LEVELS[min(idx + 1, len(TRIAGE_LEVELS) - 1)]
        confidence = max(0.0, confidence - 0.15)

    return TriagePrediction(
        level=level,
        confidence=round(confidence, 4),
        escalation_flag=confidence < 0.70 or level == "RED",
    )


# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------

def _real_predict(case: TriageCase, model_path: str) -> TriagePrediction:
    """
    Call triage_classify() via llama.cpp subprocess.

    Parses model output for <tool_call>...</tool_call> JSON.
    """
    import re
    import subprocess

    prompt = (
        f"Patient age: {case.age_group}. "
        f"Duration: {case.duration_days} days. "
        f"Symptoms: {case.verbal_symptoms}. "
        "Call triage_classify() and return only JSON."
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
            "llama-cli not found in PATH. Use --model mock or install llama.cpp."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"llama.cpp timed out for case {case.case_id}"
        ) from exc

    raw = result.stdout
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    match = tool_call_pattern.search(raw)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed tool_call JSON for case {case.case_id}: {exc}"
            ) from exc
    else:
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"No parseable JSON in output for case {case.case_id}. "
                f"Raw: {raw[:200]}"
            ) from exc

    output_data = parsed.get("result", parsed)
    level = str(output_data.get("level", "GREEN")).upper()
    if level not in TRIAGE_LEVELS:
        raise RuntimeError(f"Unexpected level '{level}' for case {case.case_id}")

    confidence = float(output_data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return TriagePrediction(
        level=level,
        confidence=confidence,
        escalation_flag=bool(output_data.get("escalation_flag", confidence < 0.70)),
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_recall(
    gold: list[str],
    pred: list[str],
    target_class: str,
    n_samples: int,
    seed: int,
) -> BootstrapCI:
    """
    Compute 95% CI for recall of a single class using bootstrap resampling.

    Uses the percentile method on 1000 bootstrap samples.
    """
    rng = np.random.RandomState(seed)  # noqa: NPY002 — explicit seed required
    gold_arr = np.array(gold)
    pred_arr = np.array(pred)
    n = len(gold_arr)

    recalls: list[float] = []
    for _ in range(n_samples):
        indices = rng.randint(0, n, size=n)
        g_boot = gold_arr[indices]
        p_boot = pred_arr[indices]
        tp = int(np.sum((g_boot == target_class) & (p_boot == target_class)))
        fn = int(np.sum((g_boot == target_class) & (p_boot != target_class)))
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)

    recalls_arr = np.array(recalls)
    lower = float(np.percentile(recalls_arr, 100 * CI_ALPHA / 2))
    upper = float(np.percentile(recalls_arr, 100 * (1 - CI_ALPHA / 2)))
    tp_full = int(np.sum((gold_arr == target_class) & (pred_arr == target_class)))
    fn_full = int(np.sum((gold_arr == target_class) & (pred_arr != target_class)))
    denom_full = tp_full + fn_full
    point = tp_full / denom_full if denom_full > 0 else 0.0

    return BootstrapCI(point_estimate=round(point, 4), lower=round(lower, 4), upper=round(upper, 4))


def _bootstrap_f1(
    gold: list[str],
    pred: list[str],
    average: str,
    n_samples: int,
    seed: int,
) -> BootstrapCI:
    """Compute 95% CI for macro/weighted F1 using bootstrap resampling."""
    rng = np.random.RandomState(seed)  # noqa: NPY002
    gold_arr = np.array(gold)
    pred_arr = np.array(pred)
    n = len(gold_arr)

    f1s: list[float] = []
    for _ in range(n_samples):
        indices = rng.randint(0, n, size=n)
        g_boot = gold_arr[indices].tolist()
        p_boot = pred_arr[indices].tolist()
        score = f1_score(g_boot, p_boot, average=average, labels=TRIAGE_LEVELS, zero_division=0)
        f1s.append(float(score))

    f1s_arr = np.array(f1s)
    lower = float(np.percentile(f1s_arr, 100 * CI_ALPHA / 2))
    upper = float(np.percentile(f1s_arr, 100 * (1 - CI_ALPHA / 2)))
    point = f1_score(gold, pred, average=average, labels=TRIAGE_LEVELS, zero_division=0)

    return BootstrapCI(point_estimate=round(float(point), 4), lower=round(lower, 4), upper=round(upper, 4))


# ---------------------------------------------------------------------------
# Metrics and display
# ---------------------------------------------------------------------------

def compute_triage_metrics(
    cases: list[TriageCase],
    predictions: list[TriagePrediction],
    split: str,
    seed: int,
) -> TriageEvalResult:
    """Compute the full triage evaluation metric suite."""
    gold = [c.gold_level for c in cases]
    pred = [p.level for p in predictions]
    escalation_flags = [p.escalation_flag for p in predictions]

    report_dict: dict[str, dict[str, float]] = classification_report(
        gold, pred, labels=TRIAGE_LEVELS, output_dict=True, zero_division=0
    )  # type: ignore[assignment]

    per_class: dict[str, ClassMetrics] = {
        level: ClassMetrics(
            precision=round(float(report_dict.get(level, {}).get("precision", 0.0)), 4),
            recall=round(float(report_dict.get(level, {}).get("recall", 0.0)), 4),
            f1=round(float(report_dict.get(level, {}).get("f1-score", 0.0)), 4),
            support=int(report_dict.get(level, {}).get("support", 0)),
        )
        for level in TRIAGE_LEVELS
    }

    macro_f1 = float(f1_score(gold, pred, average="macro", labels=TRIAGE_LEVELS, zero_division=0))
    weighted_f1 = float(f1_score(gold, pred, average="weighted", labels=TRIAGE_LEVELS, zero_division=0))
    red_recall = per_class["RED"].recall
    escalation_rate = sum(escalation_flags) / max(len(escalation_flags), 1)

    red_recall_ci = _bootstrap_recall(gold, pred, "RED", BOOTSTRAP_N_SAMPLES, seed=seed)
    weighted_f1_ci = _bootstrap_f1(gold, pred, "weighted", BOOTSTRAP_N_SAMPLES, seed=seed)

    cm = confusion_matrix(gold, pred, labels=TRIAGE_LEVELS)

    return TriageEvalResult(
        split=split,
        seed=seed,
        n_cases=len(cases),
        per_class=per_class,
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        red_recall=round(red_recall, 4),
        red_recall_ci=red_recall_ci,
        weighted_f1_ci=weighted_f1_ci,
        escalation_rate=round(escalation_rate, 4),
        confusion=cm.tolist(),
    )


def _pass_fail(value: float, target: float) -> str:
    return "PASS" if value >= target else "FAIL"


def print_triage_report(result: TriageEvalResult) -> None:
    """Print a detailed ASCII triage eval report to stdout."""
    sep = "─" * 64
    print(sep)
    print("  MARUNTHAGAM TRIAGE EVAL — DETAILED REPORT")
    print(sep)
    print(f"  Split:           {result.split}")
    print(f"  Seed:            {result.seed}")
    print(f"  Total cases:     {result.n_cases}")
    print()

    # Per-class table
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for level in TRIAGE_LEVELS:
        m = result.per_class[level]
        print(f"  {level:<10} {m.precision:>10.4f} {m.recall:>10.4f} "
              f"{m.f1:>10.4f} {m.support:>10d}")
    print()

    # Summary metrics
    print(f"  Macro F1:        {result.macro_f1:.4f}")
    print(f"  Weighted F1:     {result.weighted_f1_ci}   target >{TARGET_F1}  "
          f"→ {_pass_fail(result.weighted_f1, TARGET_F1)}")
    print(f"  RED recall:      {result.red_recall_ci}   target >{TARGET_RED_RECALL}  "
          f"→ {_pass_fail(result.red_recall, TARGET_RED_RECALL)}")
    print(f"  Escalation rate: {result.escalation_rate:.4f}")
    print()

    # ASCII confusion matrix
    print("  Confusion Matrix (rows = gold, cols = predicted):")
    col_width = 10
    header = f"  {'':10}" + "".join(f"{lv:>{col_width}}" for lv in TRIAGE_LEVELS)
    print(header)
    print(f"  {'─'*10}" + "─" * (col_width * len(TRIAGE_LEVELS)))
    for row_idx, row_label in enumerate(TRIAGE_LEVELS):
        row_str = f"  {row_label:<10}"
        for col_val in result.confusion[row_idx]:
            row_str += f"{col_val:>{col_width}}"
        print(row_str)
    print(sep)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(result: TriageEvalResult, timestamp: str) -> Path:
    """Save eval result JSON and confusion matrix CSV to eval/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"triage_eval_{result.split}_{timestamp}"
    json_path = RESULTS_DIR / f"{stem}.json"
    csv_path = RESULTS_DIR / f"{stem}_confusion.csv"

    payload = {
        "split": result.split,
        "seed": result.seed,
        "n_cases": result.n_cases,
        "targets": {"weighted_f1": TARGET_F1, "red_recall": TARGET_RED_RECALL},
        "per_class": {
            level: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
            }
            for level, m in result.per_class.items()
        },
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "weighted_f1_ci": {
            "point": result.weighted_f1_ci.point_estimate,
            "lower_95": result.weighted_f1_ci.lower,
            "upper_95": result.weighted_f1_ci.upper,
        },
        "red_recall": result.red_recall,
        "red_recall_ci": {
            "point": result.red_recall_ci.point_estimate,
            "lower_95": result.red_recall_ci.lower,
            "upper_95": result.red_recall_ci.upper,
        },
        "escalation_rate": result.escalation_rate,
        "confusion_matrix": {
            "labels": TRIAGE_LEVELS,
            "matrix": result.confusion,
        },
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # Save confusion matrix as CSV for spreadsheet analysis
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gold \\ pred"] + TRIAGE_LEVELS)
        for row_idx, row_label in enumerate(TRIAGE_LEVELS):
            writer.writerow([row_label] + result.confusion[row_idx])

    return json_path


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------

def run_triage_eval(
    split: str,
    seed: int,
    model_path: Optional[str],
    use_mock: bool,
) -> TriageEvalResult:
    """Load cases, run inference, compute metrics, print and save results."""
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading triage cases (split={split}, seed={seed}) ...")
    cases = load_triage_cases(split=split, seed=seed)
    print(f"  {len(cases)} cases loaded")

    predictions: list[TriagePrediction] = []
    for case in cases:
        if use_mock:
            pred = _mock_predict(case, seed=seed)
        else:
            assert model_path is not None
            pred = _real_predict(case, model_path)
        predictions.append(pred)

    result = compute_triage_metrics(cases, predictions, split=split, seed=seed)
    print_triage_report(result)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = save_results(result, timestamp)
    print(f"\n  Results saved to: {json_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam triage-specific eval. Computes per-class P/R/F1, "
            "macro/weighted F1, RED recall with 95% bootstrap CI, and an ASCII "
            "confusion matrix. Saves JSON + confusion CSV to eval/results/."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        metavar="GGUF_PATH",
        help="Path to the quantised GGUF model file for real inference.",
    )
    model_group.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use deterministic mock predictor (~90%% accuracy). "
            "Useful before model weights are available."
        ),
    )
    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="test",
        help=(
            "Data split to evaluate: test (last 10%%), val (middle 10%%), "
            "all (everything). Split is seeded and reproducible. Default: test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split shuffling and bootstrap CI. Default: 42.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path: Optional[str] = args.model if not args.mock else None

    try:
        run_triage_eval(
            split=args.split,
            seed=args.seed,
            model_path=model_path,
            use_mock=args.mock,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
