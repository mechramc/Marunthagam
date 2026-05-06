"""
Sprint 2 Task 6 — held-out re-eval with the production stack:
  - B-retrained triage LoRA (HF+PEFT adapter at training/outputs/triage-relabel-seed42-6ep/final/)
  - Sprint 1 derm GGUF (training/models/derm-E4B-Q4_K_M_gguf/...)
  - Sprint 1 maternal GGUF (training/models/maternal-E4B-Q4_K_M_gguf/...)
  - v2.1 IMNCI rules (loaded from inference/protocol_engine/data/protocol.db)
  - v2 safety classifier (eval/scripts/eval_safety.py — for the safety eval, not this)

Single-seed (42) per user direction (multi-seed std reporting is overkill given
the cross-variant differences are 5-15 F1 points apart, well above seed variance).

Runs all 4 routing configs:
  - routed:        case routed to its specialist (triage→B-HF, derm→GGUF, maternal→GGUF)
  - triage-only:   B-HF for ALL 131 cases
  - derm-only:     sprint 1 derm GGUF for ALL 131 cases
  - maternal-only: sprint 1 maternal GGUF for ALL 131 cases

Held-out test split: training/data/formatted/{spec}/test.jsonl (relabeled, n=131).

Output: eval/results/run_task6_<config>_<timestamp>.json (one per routing config).
Plus a summary table at eval/analysis/2026-05-07/task6_results.md.
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

# Make eval/scripts/ importable for run_eval helpers
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import run_eval  # noqa: E402

# Make training/scripts/ importable for the cu12 DLL bridge before llama_cpp
_TRAINING = os.path.abspath(os.path.join(_HERE, "..", "..", "training", "scripts"))
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
import _llama_cpp_setup  # noqa: F401, E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"

B_ADAPTER = REPO / "training" / "outputs" / "triage-relabel-seed42-6ep" / "final"
GGUF_DERM = REPO / "training" / "models" / "derm-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M.gguf"
GGUF_MATERNAL = REPO / "training" / "models" / "maternal-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M.gguf"
GGUF_TRIAGE_S1 = REPO / "training" / "models" / "triage-E4B-Q4_K_M_gguf" / "gemma-4-e4b-it.Q4_K_M.gguf"

# Cached model handles
_HF_MODEL = None
_HF_TOK = None
_GGUF_LLMS: dict[str, "Llama"] = {}


def load_hf_b():
    global _HF_MODEL, _HF_TOK
    if _HF_MODEL is None:
        from unsloth import FastLanguageModel
        print(f"[task6] loading B-retrained adapter (HF+PEFT) from {B_ADAPTER}")
        _HF_MODEL, processor = FastLanguageModel.from_pretrained(
            model_name=str(B_ADAPTER),
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_HF_MODEL)
        _HF_TOK = getattr(processor, "tokenizer", processor)
    return _HF_MODEL, _HF_TOK


def load_gguf(path: Path):
    key = str(path)
    if key not in _GGUF_LLMS:
        from llama_cpp import Llama
        print(f"[task6] loading GGUF {path.name}")
        _GGUF_LLMS[key] = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
            logits_all=False,  # we don't need logprobs for Task 6
        )
    return _GGUF_LLMS[key]


def hf_predict(case: run_eval.TestCase) -> run_eval.PredictedOutput:
    """Run B-retrained inference via HF+PEFT (no engine — caller applies)."""
    import torch
    model, tok = load_hf_b()
    user_msg = case.tamil_question.strip() or case.verbal_symptoms
    prompt = (
        "<|turn>user\n"
        f"{user_msg}\n\n"
        "Classify this case. Output ONE JSON object only, no other text:\n"
        '{"level": "GREEN" | "YELLOW" | "RED", "confidence": 0.0-1.0, '
        '"escalation_flag": true | false}<turn|>\n'
        "<|turn>model\n"
        "{"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            temperature=None, top_p=None, pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_to_predicted_output("{" + text)


def gguf_predict(case: run_eval.TestCase, gguf_path: Path) -> run_eval.PredictedOutput:
    """Run GGUF inference via llama-cpp-python."""
    llm = load_gguf(gguf_path)
    user_msg = case.tamil_question.strip() or case.verbal_symptoms
    prompt = run_eval._LLAMA_PROMPT_TEMPLATE.format(user_message=user_msg)
    completion = llm(
        prompt, max_tokens=128, temperature=0.0,
        stop=["<turn|>", "<|turn>", "\n\n"],
    )
    raw = "{" + completion["choices"][0]["text"]
    return _parse_to_predicted_output(raw)


def _parse_to_predicted_output(raw: str) -> run_eval.PredictedOutput:
    blk = re.search(r"\{.*?\}", raw, re.DOTALL)
    parsed: Optional[dict] = None
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


def apply_engine(case: run_eval.TestCase, pred: run_eval.PredictedOutput) -> run_eval.PredictedOutput:
    """Apply v2.1 engine to a model prediction, returning the updated PredictedOutput."""
    engine = run_eval._get_protocol_engine()
    if engine is None:
        return pred
    triage = run_eval.TriageResult(
        level=pred.level, confidence=pred.confidence,
        suspected_conditions=[], reasoning_chain=pred.reasoning_chain,
        next_steps_tamil=pred.next_steps_tamil,
        protocol_references=[], escalation_flag=pred.escalation_flag,
    )
    triage, overrides = engine.apply(
        triage,
        chief_complaint=(case.verbal_symptoms or "").strip(),
        narrative=(case.tamil_question or "").strip(),
        age_group=case.age_group,
        duration_days=case.duration_days,
    )
    pred.level = triage.level
    pred.escalation_flag = triage.escalation_flag
    pred.engine_overrides = [
        {
            "rule_id": o.rule_id,
            "original_level": o.original_level,
            "overridden_to": o.overridden_to,
        }
        for o in overrides
    ]
    return pred


def predict_for_config(case: run_eval.TestCase, config: str) -> run_eval.PredictedOutput:
    """
    Route a case to the right model based on the routing config:
      - routed:        case.specialist → B-HF for triage, GGUF for derm/maternal
      - triage-only:   B-HF for all
      - derm-only:     sprint 1 derm GGUF for all
      - maternal-only: sprint 1 maternal GGUF for all
    """
    if config == "routed":
        if case.specialist == "triage":
            return hf_predict(case)
        elif case.specialist == "derm":
            return gguf_predict(case, GGUF_DERM)
        elif case.specialist == "maternal":
            return gguf_predict(case, GGUF_MATERNAL)
    elif config == "triage-only":
        return hf_predict(case)
    elif config == "derm-only":
        return gguf_predict(case, GGUF_DERM)
    elif config == "maternal-only":
        return gguf_predict(case, GGUF_MATERNAL)
    raise ValueError(f"Unknown config: {config}")


def run_one_config(
    cases: list[run_eval.TestCase], config: str,
) -> tuple[run_eval.SeedResult, list[dict], float]:
    print(f"\n=== Task 6 config: {config} (n={len(cases)}) ===")
    t0 = time.monotonic()
    predictions: list[run_eval.PredictedOutput] = []
    for i, case in enumerate(cases):
        pred = predict_for_config(case, config)
        pred = apply_engine(case, pred)
        predictions.append(pred)
        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{config}] {i+1}/{len(cases)}  elapsed {elapsed:.1f}s "
                  f"({(i+1)/elapsed:.2f} cps)")
    elapsed = time.monotonic() - t0

    seed_result = run_eval.compute_metrics(cases, predictions, seed=42)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_task6_{config.replace('-', '_')}_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "task": "task6",
        "config": config,
        "production_stack": {
            "triage": "HF+PEFT triage-relabel-seed42-6ep" if config in ("routed", "triage-only") else None,
            "derm": "GGUF sprint1 derm" if config in ("routed", "derm-only") else None,
            "maternal": "GGUF sprint1 maternal" if config in ("routed", "maternal-only") else None,
            "engine_rules": "v2.1 (chief+co_signal Bucket A tightenings)",
            "safety_classifier": "v2 (~85 indicators across 4 languages)",
        },
        "n_cases": seed_result.n_cases,
        "weighted_f1": seed_result.weighted_f1,
        "macro_f1": seed_result.macro_f1,
        "red_recall": seed_result.red_recall,
        "escalation_rate": seed_result.escalation_rate,
        "per_class": seed_result.per_class_report,
        "per_specialist": seed_result.per_specialist,
        "predictions": seed_result.predictions,
        "elapsed_s": round(elapsed, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"  [{config}] wrote {out_path}")

    return seed_result, predictions, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 2 Task 6 held-out eval")
    parser.add_argument("--configs", default="routed,triage-only,derm-only,maternal-only",
                        help="Comma-separated list of configs to run")
    args = parser.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    cases = run_eval.load_all_test_split_cases()
    print(f"[task6] loaded {len(cases)} held-out test cases")

    summary: list[dict] = []
    for config in configs:
        result, _, elapsed = run_one_config(cases, config)
        per_class = result.per_class_report
        summary.append({
            "config": config,
            "weighted_f1": result.weighted_f1,
            "macro_f1": result.macro_f1,
            "red_recall": result.red_recall,
            "green_p": per_class["GREEN"]["precision"],
            "green_r": per_class["GREEN"]["recall"],
            "yellow_p": per_class["YELLOW"]["precision"],
            "yellow_r": per_class["YELLOW"]["recall"],
            "red_p": per_class["RED"]["precision"],
            "red_r": per_class["RED"]["recall"],
            "elapsed_s": round(elapsed, 1),
        })
    print("\n=== TASK 6 SUMMARY ===")
    print(f"{'config':<15s} F1     GR    YR    RR    GP    YP    RP    sec")
    for s in summary:
        print(f"{s['config']:<15s} {s['weighted_f1']:.4f}  "
              f"{s['green_r']:.3f}  {s['yellow_r']:.3f}  {s['red_r']:.3f}  "
              f"{s['green_p']:.3f}  {s['yellow_p']:.3f}  {s['red_p']:.3f}  "
              f"{s['elapsed_s']}")


if __name__ == "__main__":
    main()
