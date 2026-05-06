"""
Marunthagam — eval a saved LoRA adapter directly via HuggingFace+PEFT (no GGUF).

Used for fast post-training validation without the slow GGUF export +
base-model download cycle. Loads the same Unsloth 4-bit base the training
script uses (already cached locally) and applies the saved adapter.

Output JSON has the same shape as run_eval.py's single-seed result so the
analysis tooling reads it identically.

Usage:
    python eval/scripts/eval_hf_adapter.py \
        --adapter training/outputs/triage-relabel-seed42/final \
        --split-spec triage \
        --split train \
        --tag triage_relabel_on_relabeled_train

The --split-spec/--split flags select <spec>/<split>.jsonl from
training/data/formatted/ — same loader as eval_cross_train.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Reuse run_eval's helpers / data structures
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import run_eval  # noqa: E402

from run_logger import RunLogger  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
FORMATTED_DIR = REPO / "training" / "data" / "formatted"
RESULTS_DIR = REPO / "eval" / "results"

# Same prompt template as run_eval.py — primes a `{` so the model continues a JSON object.
_PROMPT = (
    "<|turn>user\n"
    "{user_message}\n\n"
    "Classify this case. Output ONE JSON object only, no other text:\n"
    '{{"level": "GREEN" | "YELLOW" | "RED", "confidence": 0.0-1.0, '
    '"escalation_flag": true | false}}<turn|>\n'
    "<|turn>model\n"
    "{{"
)


def load_split(spec: str, split: str) -> list[run_eval.TestCase]:
    """Re-use eval_cross_train's loader semantics for any split file."""
    path = FORMATTED_DIR / spec / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
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
                specialist=spec,
                verbal_symptoms=args.get("verbal_symptoms", "") or tamil_question,
                age_group=args.get("patient_age_group", "adult"),
                duration_days=int(args.get("duration_days", 1) or 1),
                gold_level=gold,
                case_id=f"{spec}_{split}_{line_num:03d}",
                tamil_question=tamil_question,
            ))
    return cases


# Cache loaded model so multiple eval scripts in one process don't re-load.
_HF_CACHE: dict[str, tuple] = {}


def get_hf_model(adapter_path: str):
    """Load the same 4-bit Unsloth base used during training, plus the adapter."""
    abs_adapter = os.path.abspath(adapter_path)
    cached = _HF_CACHE.get(abs_adapter)
    if cached is not None:
        return cached

    from unsloth import FastLanguageModel

    print(f"Loading model + adapter from {abs_adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=abs_adapter,    # PEFT adapter dir; Unsloth auto-loads base from adapter_config.json
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    _HF_CACHE[abs_adapter] = (model, text_tokenizer)
    return model, text_tokenizer


def hf_predict(case: run_eval.TestCase, model, tokenizer) -> run_eval.PredictedOutput:
    """Run one inference + parse the JSON, mirroring run_eval._real_predict structure
    (without engine application — caller layers engine.apply if desired)."""
    import torch

    user_msg = case.tamil_question.strip() or case.verbal_symptoms
    prompt = _PROMPT.format(user_message=user_msg)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = "{" + text  # we primed `{`; prepend so the parser sees a full object

    parsed: Optional[dict] = None
    blk = re.search(r"\{.*?\}", raw, re.DOTALL)
    if blk:
        try:
            parsed = json.loads(blk.group(0))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        return run_eval.PredictedOutput(
            level="GREEN", confidence=0.0, escalation_flag=True,
            reasoning_chain=f"[PARSE FAIL] {raw[:200]}", next_steps_tamil="",
            pre_engine_level="GREEN", pre_engine_confidence=0.0,
            pre_engine_escalation_flag=True,
        )

    output_data = parsed
    if isinstance(parsed.get("triage_result"), dict):
        output_data = parsed["triage_result"]
    elif isinstance(parsed.get("arguments"), dict):
        output_data = parsed["arguments"]

    level = str(output_data.get("level", "GREEN")).upper()
    if level not in run_eval.TRIAGE_LEVELS:
        return run_eval.PredictedOutput(
            level="GREEN", confidence=0.0, escalation_flag=True,
            reasoning_chain=f"[BAD LEVEL '{level}']", next_steps_tamil="",
            pre_engine_level="GREEN", pre_engine_confidence=0.0,
            pre_engine_escalation_flag=True,
        )
    try:
        conf = float(output_data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    esc = bool(output_data.get("escalation_flag", conf < 0.70))

    return run_eval.PredictedOutput(
        level=level, confidence=conf, escalation_flag=esc,
        reasoning_chain=str(output_data.get("reasoning_chain", "")),
        next_steps_tamil=str(output_data.get("next_steps_tamil", "")),
        pre_engine_level=level,
        pre_engine_confidence=conf,
        pre_engine_escalation_flag=esc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HF-PEFT adapter evaluation (no GGUF)")
    parser.add_argument("--adapter", required=True, help="Path to adapter dir (containing adapter_config.json)")
    parser.add_argument("--split-spec", required=True, choices=("triage", "derm", "maternal"))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--tag", required=True, help="Output filename tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on cases for quick smoke")
    args = parser.parse_args()

    print(f"Loading {args.split_spec}/{args.split}.jsonl ...")
    cases = load_split(args.split_spec, args.split)
    if args.limit:
        cases = cases[: args.limit]
    print(f"  Loaded {len(cases)} cases")

    with RunLogger(kind="eval_hf_adapter", args=args) as logger:
        logger.merge_manifest(adapter=args.adapter, n_cases=len(cases))

        model, tok = get_hf_model(args.adapter)

        predictions: list[run_eval.PredictedOutput] = []
        t0 = time.monotonic()
        for i, case in enumerate(cases):
            pred = hf_predict(case, model, tok)
            predictions.append(pred)
            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                print(f"    {i+1}/{len(cases)} done ({elapsed:.1f}s, {(i+1)/elapsed:.2f} cps)")

        result = run_eval.compute_metrics(cases, predictions, seed=args.seed)
        elapsed = time.monotonic() - t0
        print(f"  Done in {elapsed:.1f}s ({len(cases)/elapsed:.2f} cases/s)")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"run_{args.tag}_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "adapter": args.adapter,
            "mode": f"HF+PEFT (4bit)",
            "case_source": f"{args.split_spec}/{args.split}.jsonl",
            "split_spec": args.split_spec,
            "split": args.split,
            "seed": args.seed,
            "n_cases": result.n_cases,
            "weighted_f1": result.weighted_f1,
            "macro_f1": result.macro_f1,
            "red_recall": result.red_recall,
            "escalation_rate": result.escalation_rate,
            "per_class": result.per_class_report,
            "per_specialist": result.per_specialist,
            "predictions": result.predictions,
            "elapsed_s": round(elapsed, 2),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n  Weighted F1={result.weighted_f1:.4f}  RED recall={result.red_recall:.4f}")
        print(f"  Per-class: {result.per_class_report}")
        print(f"  Saved to: {out_path}")
        logger.attach_result(out_path)


if __name__ == "__main__":
    main()
