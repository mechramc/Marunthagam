"""
Translate English medical Q&A pairs to Tamil using Gemma 4 31B.
Output is human-reviewable JSONL for the 3 Tamil medical reviewers.

Usage:
    python translate_dataset.py --source data/raw/mediqa_en.jsonl \
                                 --output data/reviewed/mediqa_ta_pending.jsonl \
                                 --model models/gemma-4-31B-it-Q4_K_M.gguf
"""
import argparse
import json
import re
from pathlib import Path

import _llama_cpp_setup  # noqa: F401  -- registers cu12 DLL dirs on Windows

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # Allow import without llama_cpp for testing


TRANSLATION_PROMPT_TEMPLATE = """\
<start_of_turn>user
Translate this medical Q&A pair to Tamil accurately.
Use Tamil medical terminology as used in Tamil Nadu government health communications.
Return ONLY valid JSON with these exact keys:
{{"tamil_question": "...", "tamil_answer": "...", "medical_terms_translated": ["term1", "term2"]}}

English Question: {question}
English Answer: {answer}<end_of_turn>
<start_of_turn>model
"""

JSON_EXTRACT_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


def load_source(source_path: str) -> list[dict]:
    """Load English Q&A pairs from JSONL (one JSON object per line)."""
    with open(source_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def translate_pair(llm, pair: dict) -> dict:
    """
    Translate a single Q&A pair to Tamil.
    Returns the pair enriched with Tamil fields and review_status.
    """
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        question=pair.get("question", ""),
        answer=pair.get("answer", ""),
    )
    output = llm(prompt, max_tokens=1024, temperature=0.1, stop=["<end_of_turn>"])
    raw_text = output["choices"][0]["text"].strip()

    # Try to extract JSON from the response
    match = JSON_EXTRACT_PATTERN.search(raw_text)
    if match:
        try:
            translated = json.loads(match.group())
            return {
                **pair,
                "tamil_question": translated.get("tamil_question", ""),
                "tamil_answer": translated.get("tamil_answer", ""),
                "medical_terms_translated": translated.get("medical_terms_translated", []),
                "review_status": "pending",
                "reviewer_notes": "",
            }
        except json.JSONDecodeError:
            pass

    # Translation failed — mark as error for human review
    return {
        **pair,
        "tamil_question": "",
        "tamil_answer": "",
        "medical_terms_translated": [],
        "review_status": "error",
        "reviewer_notes": f"Translation parse failed. Raw: {raw_text[:200]}",
    }


def translate_batch(
    llm,
    pairs: list[dict],
    batch_size: int = 10,
    start_idx: int = 0,
) -> list[dict]:
    """Translate pairs in batches, printing progress."""
    results = []
    for i, pair in enumerate(pairs, start=start_idx + 1):
        result = translate_pair(llm, pair)
        results.append(result)
        if i % batch_size == 0 or i == start_idx + len(pairs):
            status_counts = {
                "pending": sum(1 for r in results if r["review_status"] == "pending"),
                "error": sum(1 for r in results if r["review_status"] == "error"),
            }
            print(f"Translated {i} pairs — pending: {status_counts['pending']}, errors: {status_counts['error']}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English medical Q&A to Tamil")
    parser.add_argument("--source", required=True, help="Source JSONL (English Q&A pairs)")
    parser.add_argument("--output", required=True, help="Output JSONL (Tamil, pending review)")
    parser.add_argument(
        "--model",
        default="models/gemma-4-31B-it-Q4_K_M.gguf",
        help="Path to Gemma 4 31B GGUF model",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Progress reporting interval")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers for llama.cpp")
    args = parser.parse_args()

    if Llama is None:
        raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    pairs = load_source(str(source_path))
    print(f"Loaded {len(pairs)} pairs from {source_path}")

    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=4096,
        verbose=False,
    )

    results = translate_batch(llm, pairs, batch_size=args.batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    pending = sum(1 for r in results if r["review_status"] == "pending")
    errors = sum(1 for r in results if r["review_status"] == "error")
    print(f"\nDone. {len(results)} total — {pending} pending review, {errors} errors")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
