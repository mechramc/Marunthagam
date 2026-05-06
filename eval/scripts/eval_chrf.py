"""
Marunthagam — Tamil fluency evaluation (chrF++).

For each held-out test row we ask the routed specialist GGUF to produce a
short Tamil `next_steps_tamil` paragraph, then compute chrF++ against the
gold next-steps text using sacrebleu.

Target: chrF++ > 0.60 (CLAUDE.md).

Usage:
    python eval_chrf.py --models-dir training/models
    python eval_chrf.py --model path/to/triage.gguf
    python eval_chrf.py --models-dir training/models --max-rows 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make scripts/_llama_cpp_setup importable so the cu12 DLL dirs are registered.
_TRAINING_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training", "scripts")
)
if _TRAINING_SCRIPTS not in sys.path:
    sys.path.insert(0, _TRAINING_SCRIPTS)
import _llama_cpp_setup  # noqa: F401, E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from run_logger import RunLogger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMATTED_DIR = REPO_ROOT / "training" / "data" / "formatted"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

SPECIALISTS: list[str] = ["triage", "derm", "maternal"]
TARGET_CHRF: float = 0.60

_PROMPT_TEMPLATE = (
    "<|turn>user\n"
    "{tamil_question}\n\n"
    "ASHA worker triage prompt. Reply with a short, plain-Tamil paragraph "
    "(≤ 3 sentences) describing the next steps for this patient. "
    "Tamil only — no JSON, no English explanations.\n"
    "<turn|>\n"
    "<|turn>model\n"
)

_MAX_TOKENS = 192
_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_test_rows(specialist: str) -> list[dict]:
    """
    Returns a list of {"specialist", "tamil_question", "gold_next_steps"}.
    """
    path = FORMATTED_DIR / specialist / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            tamil_question = ""
            gold_next = ""
            saw_tool_call = False
            for msg in messages:
                role = msg.get("role")
                if role == "user" and not tamil_question:
                    tamil_question = msg.get("content") or ""
                elif role == "assistant" and msg.get("tool_calls"):
                    saw_tool_call = True
                elif role == "assistant" and saw_tool_call and not msg.get("tool_calls"):
                    if not gold_next and msg.get("content"):
                        gold_next = msg["content"].strip()
            if tamil_question and gold_next:
                rows.append({
                    "specialist": specialist,
                    "tamil_question": tamil_question,
                    "gold_next_steps": gold_next,
                })
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_LLM_CACHE: dict[str, "Llama"] = {}


def _get_llm(model_path: str) -> "Llama":
    abs_path = os.path.abspath(model_path)
    cached = _LLM_CACHE.get(abs_path)
    if cached is not None:
        return cached
    from llama_cpp import Llama
    llm = Llama(
        model_path=abs_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )
    _LLM_CACHE[abs_path] = llm
    return llm


def discover_specialist_models(models_dir: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    base = os.path.abspath(models_dir)
    for specialist in SPECIALISTS:
        candidate = os.path.join(
            base, f"{specialist}-E4B-Q4_K_M_gguf", "gemma-4-e4b-it.Q4_K_M.gguf",
        )
        if not os.path.exists(candidate):
            raise RuntimeError(
                f"Missing GGUF for specialist {specialist!r}: {candidate}"
            )
        resolved[specialist] = candidate
    return resolved


def generate_next_steps(
    tamil_question: str, model_path: str
) -> str:
    llm = _get_llm(model_path)
    prompt = _PROMPT_TEMPLATE.format(tamil_question=tamil_question)
    completion = llm(
        prompt,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        stop=["<turn|>", "<|turn>"],
    )
    return completion["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def chrf_plus_plus(hypothesis: str, reference: str) -> float:
    """Return chrF++ score (0..1) using sacrebleu's CHRF metric (word_order=2)."""
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)  # chrF++ = chrF with bigrams
    score = chrf.sentence_score(hypothesis, [reference]).score
    # sacrebleu returns 0–100 — normalise to 0–1.
    return score / 100.0


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------

def run_chrf_eval(
    models_by_specialist: dict[str, str],
    output_path: Path,
    max_rows_per_specialist: Optional[int],
    run_logger: Optional[RunLogger] = None,
) -> dict:
    overall_scores: list[float] = []
    by_specialist: dict[str, dict] = {}
    raw_records: list[dict] = []

    for specialist in SPECIALISTS:
        rows = load_test_rows(specialist)
        if max_rows_per_specialist:
            rows = rows[:max_rows_per_specialist]
        if not rows:
            print(f"  WARNING: no rows for {specialist}; skipping.")
            continue
        print(f"  Scoring {specialist}: {len(rows)} rows ...")
        scores: list[float] = []
        for idx, row in enumerate(rows):
            hypothesis = generate_next_steps(
                row["tamil_question"], models_by_specialist[specialist]
            )
            score = chrf_plus_plus(hypothesis, row["gold_next_steps"])
            scores.append(score)
            raw_records.append({
                "specialist": specialist,
                "row_index": idx,
                "tamil_question": row["tamil_question"][:300],
                "gold": row["gold_next_steps"][:500],
                "hypothesis": hypothesis[:500],
                "chrf_plus_plus": round(score, 4),
            })
            if run_logger is not None and idx % 10 == 0:
                run_logger.log_event(
                    "chrf_progress",
                    specialist=specialist,
                    row_index=idx,
                    rolling_mean=round(sum(scores) / len(scores), 4),
                )
        mean = sum(scores) / len(scores)
        by_specialist[specialist] = {
            "n_rows": len(scores),
            "chrf_mean": round(mean, 4),
            "chrf_min": round(min(scores), 4),
            "chrf_max": round(max(scores), 4),
        }
        overall_scores.extend(scores)
        print(f"    {specialist}: chrF++ mean = {mean:.4f}  (n={len(scores)})")

    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    status = "PASS" if overall_mean >= TARGET_CHRF else "FAIL"
    print(f"\n  Overall chrF++: {overall_mean:.4f}  target {TARGET_CHRF} → {status}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "models_by_specialist": models_by_specialist,
        "n_rows_total": len(overall_scores),
        "overall_chrf_plus_plus": round(overall_mean, 4),
        "target": TARGET_CHRF,
        "status": status,
        "by_specialist": by_specialist,
        "raw": raw_records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {output_path}")
    if run_logger is not None:
        run_logger.attach_result(output_path)
        run_logger.merge_manifest(
            overall_chrf=payload["overall_chrf_plus_plus"],
            by_specialist=by_specialist,
        )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Marunthagam Tamil fluency eval (chrF++) on the held-out test split."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--models-dir", metavar="MODELS_DIR",
                             help="Directory with per-specialist GGUFs (same layout as run_eval.py).")
    model_group.add_argument("--model", metavar="GGUF_PATH",
                             help="Single GGUF to use for all specialists.")
    parser.add_argument("--max-rows", type=int, default=None, metavar="N",
                        help="Cap rows per specialist (for smoke testing).")
    timestamp_default = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / f"chrf_eval_{timestamp_default}.json",
                        help="Output path.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.models_dir:
        try:
            models = discover_specialist_models(args.models_dir)
        except RuntimeError as exc:
            parser.error(str(exc))
    else:
        models = {s: args.model for s in SPECIALISTS}

    with RunLogger(kind="eval_chrf", args=args) as logger:
        logger.merge_manifest(models_by_specialist=models)
        try:
            run_chrf_eval(
                models_by_specialist=models,
                output_path=args.output,
                max_rows_per_specialist=args.max_rows,
                run_logger=logger,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            logger.merge_manifest(error_message=str(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
