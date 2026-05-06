"""
Second 31B pass: turn Tamil-translated medical Q&A into fully labeled
reviewed-format JSONL ready for format_training_data.py.

Reads JSONL produced by translate_dataset.py (must contain tamil_question,
tamil_answer, and the original English question/answer). Adds:

  - function_call_args  (verbal_symptoms, patient_age_group, duration_days,
                         optional vital_signs)
  - triage_result       (level, confidence, suspected_conditions,
                         reasoning_chain, next_steps_tamil,
                         protocol_references, escalation_flag, disclaimer)
  - review_status       set to "auto_approved"

Usage:
    python label_triage.py --input data/reviewed/triage_pending.jsonl \
                           --output data/reviewed/triage_approved.jsonl \
                           --model models/gemma-4-31B-it-Q4_K_M.gguf
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _llama_cpp_setup  # noqa: F401  -- registers cu12 DLL dirs on Windows

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

DISCLAIMER = "இது மருத்துவ ஆலோசனை அல்ல"
VALID_LEVELS = {"GREEN", "YELLOW", "RED"}
VALID_AGE_GROUPS = {"infant", "child", "adolescent", "adult", "elderly"}

LABEL_PROMPT_TEMPLATE = """\
<start_of_turn>user
You are an expert Tamil medical triage assistant. Read this Tamil patient \
question and the Tamil doctor answer, then produce a structured triage \
classification matching the WHO IMNCI / Tamil Nadu state health protocols.

Return ONLY valid JSON with this exact schema (no extra text, no markdown):
{{
  "function_call_args": {{
    "verbal_symptoms": "<short Tamil string summarizing presenting symptoms>",
    "patient_age_group": "<one of: infant, child, adolescent, adult, elderly>",
    "duration_days": <integer days, best estimate from text, default 1>,
    "vital_signs": null
  }},
  "triage_result": {{
    "level": "<GREEN | YELLOW | RED>",
    "confidence": <float 0.5-0.95>,
    "suspected_conditions": [
      {{"condition": "<Tamil + (English) condition name>", "rank": 1}}
    ],
    "reasoning_chain": "<step-by-step Tamil reasoning, 2-4 sentences>",
    "next_steps_tamil": "<plain Tamil instructions for the ASHA worker>",
    "protocol_references": ["<protocol code like TN-001 or IMNCI-CHILD-FEVER>"],
    "escalation_flag": <true if confidence < 0.7 or RED, else false>
  }}
}}

Triage rules:
- GREEN  = mild self-limiting (cold, mild rash, minor cuts) — home care
- YELLOW = needs PHC visit within 24-48h (persistent fever, moderate symptoms)
- RED    = emergency / immediate escalation (severe pain, unconscious, \
heavy bleeding, severe dehydration, chest pain, difficulty breathing)

Tamil patient question:
{tamil_question}

Tamil doctor answer:
{tamil_answer}<end_of_turn>
<start_of_turn>model
"""

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str) -> dict | None:
    match = JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_label(parsed: dict) -> tuple[bool, str]:
    """Return (ok, reason). ok=False entries are dropped."""
    fc = parsed.get("function_call_args")
    tr = parsed.get("triage_result")
    if not isinstance(fc, dict) or not isinstance(tr, dict):
        return False, "missing function_call_args or triage_result"

    if not isinstance(fc.get("verbal_symptoms"), str) or not fc["verbal_symptoms"].strip():
        return False, "verbal_symptoms empty"
    if fc.get("patient_age_group") not in VALID_AGE_GROUPS:
        return False, f"bad age_group: {fc.get('patient_age_group')!r}"
    if not isinstance(fc.get("duration_days"), int) or fc["duration_days"] < 0:
        return False, f"bad duration_days: {fc.get('duration_days')!r}"

    level = tr.get("level")
    if level not in VALID_LEVELS:
        return False, f"bad level: {level!r}"
    confidence = tr.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return False, f"bad confidence: {confidence!r}"

    conditions = tr.get("suspected_conditions")
    if not isinstance(conditions, list) or not conditions:
        return False, "suspected_conditions empty"
    if not isinstance(tr.get("reasoning_chain"), str) or not tr["reasoning_chain"].strip():
        return False, "reasoning_chain empty"
    if not isinstance(tr.get("next_steps_tamil"), str) or not tr["next_steps_tamil"].strip():
        return False, "next_steps_tamil empty"
    if not isinstance(tr.get("protocol_references"), list):
        return False, "protocol_references not list"
    if not isinstance(tr.get("escalation_flag"), bool):
        return False, "escalation_flag not bool"

    return True, "ok"


def label_pair(llm, entry: dict) -> dict:
    prompt = LABEL_PROMPT_TEMPLATE.format(
        tamil_question=entry.get("tamil_question", ""),
        tamil_answer=entry.get("tamil_answer", ""),
    )
    output = llm(prompt, max_tokens=1024, temperature=0.2, stop=["<end_of_turn>"])
    raw = output["choices"][0]["text"].strip()
    parsed = extract_json(raw)

    if parsed is None:
        return {**entry, "review_status": "label_parse_error", "label_raw": raw[:300]}

    ok, reason = validate_label(parsed)
    if not ok:
        return {
            **entry,
            "review_status": "label_invalid",
            "label_invalid_reason": reason,
            "label_raw": raw[:300],
        }

    triage_result = dict(parsed["triage_result"])
    triage_result["disclaimer"] = DISCLAIMER

    return {
        **entry,
        "function_call_args": parsed["function_call_args"],
        "triage_result": triage_result,
        "review_status": "auto_approved",
    }


def label_dataset(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    n_gpu_layers: int,
    n_ctx: int,
    progress_every: int,
) -> dict[str, int]:
    if Llama is None:
        raise ImportError("llama-cpp-python not installed.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model GGUF not found: {model_path}")

    print(f"Loading 31B GGUF: {model_path.name}")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )

    counts = {"auto_approved": 0, "label_parse_error": 0, "label_invalid": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for idx, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("review_status") not in (None, "pending", "auto_approved"):
                # respect prior reviewer decisions (e.g. "rejected")
                continue

            labeled = label_pair(llm, entry)
            counts[labeled["review_status"]] = counts.get(labeled["review_status"], 0) + 1
            dst.write(json.dumps(labeled, ensure_ascii=False) + "\n")

            if idx % progress_every == 0:
                summary = ", ".join(f"{k}={v}" for k, v in counts.items())
                print(f"  labeled={idx}  {summary}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Label translated Tamil medical Q&A with structured triage")
    parser.add_argument("--input", required=True, help="JSONL from translate_dataset.py")
    parser.add_argument("--output", required=True, help="Output JSONL with structured labels")
    parser.add_argument(
        "--model",
        default="models/gemma-4-31B-it-Q4_K_M.gguf",
        help="Path to Gemma 4 31B GGUF",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    counts = label_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=Path(args.model),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        progress_every=args.progress_every,
    )
    total = sum(counts.values())
    print(f"\nDone. {total} rows processed: {counts}")


if __name__ == "__main__":
    main()
