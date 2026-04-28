"""
Format reviewed Tamil medical Q&A pairs into Gemma 4 chat template
with triage_classify() function calling format.

Only processes entries with review_status == "approved".
Applies 80/10/10 train/val/test split.

Usage:
    python format_training_data.py \
        --reviewed data/reviewed/triage_approved.jsonl \
        --specialist triage \
        --output-dir data/formatted
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DISCLAIMER = "இது மருத்துவ ஆலோசனை அல்ல"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# Test ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.10


def load_approved(reviewed_path: str) -> list[dict]:
    """Load only approved entries from a reviewed JSONL file."""
    reviewed_file = Path(reviewed_path)
    if not reviewed_file.exists():
        raise FileNotFoundError(f"Reviewed JSONL not found: {reviewed_file}")

    approved = []
    with open(reviewed_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("review_status") == "approved":
                approved.append(entry)
    return approved


def format_as_chat(entry: dict) -> dict:
    """
    Convert a reviewed entry to Gemma 4 function-calling chat format.

    Produces a 4-turn conversation:
    1. user: Tamil symptom description
    2. assistant: function call (triage_classify)
    3. tool: triage result (with disclaimer enforced)
    4. assistant: next steps in Tamil
    """
    # Ensure disclaimer is always present in the tool response
    triage_result = dict(entry.get("triage_result", {}))
    triage_result["disclaimer"] = DISCLAIMER

    function_call = {
        "name": "triage_classify",
        "arguments": entry.get("function_call_args", {}),
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": entry["tamil_question"],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": function_call,
                    }
                ],
            },
            {
                "role": "tool",
                "name": "triage_classify",
                "content": json.dumps(triage_result, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": triage_result.get("next_steps_tamil", ""),
            },
        ]
    }


def split_data(
    examples: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split into 80/10/10 train/val/test.

    Attempts stratification by triage level (GREEN/YELLOW/RED).
    Falls back to random split if triage levels are unavailable.
    """
    random.seed(seed)

    # Group by triage level for stratification
    by_level: dict[str, list[dict]] = {}
    for ex in examples:
        try:
            triage_result = json.loads(ex["messages"][2]["content"])
            level = triage_result.get("level", "UNKNOWN")
        except (json.JSONDecodeError, KeyError, IndexError):
            level = "UNKNOWN"
        by_level.setdefault(level, []).append(ex)

    train, val, test = [], [], []
    for level_examples in by_level.values():
        random.shuffle(level_examples)
        n = len(level_examples)
        train_end = max(1, round(n * TRAIN_RATIO))
        val_end = train_end + max(1, round(n * VAL_RATIO))

        train.extend(level_examples[:train_end])
        val.extend(level_examples[train_end:val_end])
        test.extend(level_examples[val_end:])

    return train, val, test


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def format_reviewed_dataset(
    reviewed_path: str,
    specialist: str,
    output_dir: str,
    seed: int,
) -> tuple[Path, dict[str, int]]:
    """Format one reviewed dataset and return the output path plus split sizes."""
    approved = load_approved(reviewed_path)
    if not approved:
        raise ValueError(f"No approved entries found in {reviewed_path}")

    formatted = [format_as_chat(entry) for entry in approved]
    train, val, test = split_data(formatted, seed=seed)

    out_dir = Path(output_dir) / specialist
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    return out_dir, {"train": len(train), "val": len(val), "test": len(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Format reviewed data into Gemma 4 training format")
    parser.add_argument("--reviewed", required=True, help="Reviewed JSONL file path")
    parser.add_argument(
        "--specialist",
        required=True,
        choices=["triage", "derm", "maternal"],
        help="Which specialist LoRA this data is for",
    )
    parser.add_argument("--output-dir", default="data/formatted", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    approved = load_approved(args.reviewed)
    print(f"Loaded {len(approved)} approved entries for specialist: {args.specialist}")

    out_dir, split_sizes = format_reviewed_dataset(
        reviewed_path=args.reviewed,
        specialist=args.specialist,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(
        f"train: {split_sizes['train']} | "
        f"val: {split_sizes['val']} | "
        f"test: {split_sizes['test']}"
    )
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
