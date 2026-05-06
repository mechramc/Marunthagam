"""
Marunthagam — CLI demo (Sprint 3 deliverable).

Runs the production triage stack on a Tamil patient query and emits the
structured triage_classify() output, with optional protocol-engine
overrides applied.

This is the script the README has always pointed to but never existed
in the repo. It exercises the same code paths the routed eval uses,
so the CLI demo and the held-out eval are not two different stacks.

Usage:
    # Mock — no model load, deterministic output for testing
    python inference/cli_demo.py --mock

    # Real — uses sprint-1 GGUFs + v2 IMNCI rules
    python inference/cli_demo.py \
        --models-dir training/models \
        --symptoms "குழந்தைக்கு மூன்று நாளாக காய்ச்சல், மூச்சுத் திணறல் இருக்கிறது" \
        --age child --duration 3

    # Just one specialist (e.g., for an out-of-band query)
    python inference/cli_demo.py \
        --model training/models/triage-E4B-Q4_K_M_gguf/gemma-4-e4b-it.Q4_K_M.gguf \
        --symptoms "..." --age adult --duration 1

If --symptoms is omitted, the script picks one of three preset Tamil
queries (one per triage level) and runs through them in sequence —
useful for the demo video.

The output is the production-stack triage_classify() schema:
    {
      "level": "GREEN" | "YELLOW" | "RED",
      "confidence": 0.0-1.0,
      "escalation_flag": true | false,
      "engine_overrides": [...],   // rule_ids that fired
      "disclaimer": "இது மருத்துவ ஆலோசனை அல்ல"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Make eval/scripts importable so we can reuse run_eval helpers and the
# same protocol-engine wiring the held-out eval uses.
REPO = Path(__file__).resolve().parents[1]
_EVAL_SCRIPTS = REPO / "eval" / "scripts"
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))
_TRAINING_SCRIPTS = REPO / "training" / "scripts"
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))
import _llama_cpp_setup  # noqa: F401, E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import run_eval  # noqa: E402

DISCLAIMER = "இது மருத்துவ ஆலோசனை அல்ல"

# Three preset Tamil queries — one per triage level — for demo runs.
PRESETS = [
    {
        "label": "GREEN preset",
        "symptoms": "மூக்கு ஒழுகுதல், லேசான இருமல், சளி",
        "tamil_question": "பெரியவருக்கு மூக்கு ஒழுகுதல் மற்றும் லேசான இருமல் இரண்டு நாளாக இருக்கிறது. என்ன செய்வது?",
        "age_group": "adult",
        "duration_days": 2,
        "specialist": "triage",
    },
    {
        "label": "YELLOW preset",
        "symptoms": "வயிற்று வலி, கல்லீரல் வீக்கம்",
        "tamil_question": "எனக்கு கடுமையான வயிற்று வலி மற்றும் கல்லீரலில் வீக்கம் இரண்டு வாரமாக இருக்கிறது. மருத்துவர் ஆலோசனை வேண்டுமா?",
        "age_group": "adult",
        "duration_days": 14,
        "specialist": "triage",
    },
    {
        "label": "RED preset (cardiac pattern)",
        "symptoms": "இடது மார்பில் கடுமையான வலி, இடது கையில் மரத்துப்போன உணர்வு",
        "tamil_question": "எனக்கு இடது மார்பில் கடுமையான வலி உள்ளது. வலி கழுத்தெலும்புக்கு பரவி, இடது கை மரத்துப்போய் உள்ளது. மூச்சு திணறுகிறது.",
        "age_group": "adult",
        "duration_days": 1,
        "specialist": "triage",
    },
]


def _print_result(out: dict) -> None:
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _mock_response(case: dict) -> dict:
    """Deterministic mock: emit the case's specialist-level + canonical structure."""
    # Match the 3 preset levels by content keywords (deterministic).
    s = case["symptoms"]
    if "மார்பில்" in s or "மரத்துப்போ" in s:
        level, conf = "RED", 0.92
    elif "வயிற்று வலி" in s or "வீக்கம்" in s:
        level, conf = "YELLOW", 0.84
    else:
        level, conf = "GREEN", 0.91
    return {
        "level": level,
        "confidence": conf,
        "escalation_flag": conf < 0.70 or level == "RED",
        "engine_overrides": [],
        "disclaimer": DISCLAIMER,
        "_mock": True,
    }


def _real_inference(case: dict, model_path: str) -> dict:
    """Run real GGUF inference + apply v2.1 protocol engine."""
    test_case = run_eval.TestCase(
        specialist=case.get("specialist", "triage"),
        verbal_symptoms=case["symptoms"],
        age_group=case["age_group"],
        duration_days=case["duration_days"],
        gold_level="GREEN",  # placeholder; not used by inference
        case_id="cli_demo",
        tamil_question=case.get("tamil_question", ""),
    )
    pred = run_eval._real_predict(test_case, model_path)
    return {
        "level": pred.level,
        "confidence": pred.confidence,
        "escalation_flag": pred.escalation_flag,
        "pre_engine_level": pred.pre_engine_level,
        "pre_engine_confidence": pred.pre_engine_confidence,
        "engine_overrides": pred.engine_overrides,
        "reasoning_chain": pred.reasoning_chain,
        "next_steps_tamil": pred.next_steps_tamil,
        "disclaimer": DISCLAIMER,
    }


def _resolve_model(case: dict, single_model: Optional[str],
                   models_by_specialist: Optional[dict[str, str]]) -> str:
    if models_by_specialist is not None:
        return models_by_specialist[case.get("specialist", "triage")]
    assert single_model is not None
    return single_model


def _print_header(case: dict) -> None:
    print("=" * 70)
    print(f"  {case.get('label', case.get('specialist', 'triage')).upper()}")
    print("=" * 70)
    print(f"  age_group: {case['age_group']}")
    print(f"  duration_days: {case['duration_days']}")
    print(f"  symptoms: {case['symptoms']}")
    if case.get("tamil_question"):
        print(f"  patient query (Tamil narrative):")
        print(f"    {case['tamil_question']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Marunthagam CLI demo — runs production triage stack."
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", help="Single GGUF path")
    model_group.add_argument("--models-dir",
                             help="Directory with per-specialist GGUFs (recommended)")
    model_group.add_argument("--mock", action="store_true",
                             help="Deterministic mock — no model load")

    parser.add_argument("--symptoms", help="verbal_symptoms (chief complaint)")
    parser.add_argument("--question",
                        help="Optional full Tamil patient narrative")
    parser.add_argument("--age", default="adult",
                        choices=["infant", "child", "adolescent", "adult", "elderly"])
    parser.add_argument("--duration", type=int, default=1,
                        help="duration_days")
    parser.add_argument("--specialist", default="triage",
                        choices=["triage", "derm", "maternal"])

    args = parser.parse_args()

    # Build the case(s) to run
    if args.symptoms:
        cases = [{
            "label": "user query",
            "symptoms": args.symptoms,
            "tamil_question": args.question or "",
            "age_group": args.age,
            "duration_days": args.duration,
            "specialist": args.specialist,
        }]
    else:
        cases = PRESETS
        print("(no --symptoms provided; running 3 preset Tamil queries)\n")

    # Resolve model(s)
    models_by_specialist: Optional[dict[str, str]] = None
    if args.models_dir:
        models_by_specialist = run_eval.discover_specialist_models(args.models_dir)
        print(f"Loaded models from {args.models_dir}:")
        for spec, path in models_by_specialist.items():
            print(f"  {spec}: {Path(path).name}")
        print()

    for case in cases:
        _print_header(case)
        t0 = time.monotonic()
        if args.mock:
            out = _mock_response(case)
        else:
            model_path = _resolve_model(case, args.model, models_by_specialist)
            out = _real_inference(case, model_path)
        elapsed = time.monotonic() - t0

        print(f"  inference time: {elapsed:.2f}s\n")
        _print_result(out)
        print()


if __name__ == "__main__":
    main()
