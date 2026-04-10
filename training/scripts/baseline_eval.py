"""
Evaluate stock Gemma 4 E4B on Tamil medical triage examples.
Run BEFORE any fine-tuning to establish the baseline performance gap.

Usage:
    python baseline_eval.py --config configs/baseline.yaml \
                             --examples ../eval/data/baseline_examples.json \
                             --output ../eval/results/baseline_results.json
"""

import json
import yaml
import argparse
from pathlib import Path

from llama_cpp import Llama


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(example: dict) -> str:
    return (
        "<start_of_turn>user\n"
        f"நோயாளி விவரம்: {example['symptom_description']}\n"
        f"வயது குழு: {example['age_group']}\n"
        f"நாட்கள்: {example['duration_days']}\n"
        "\n"
        "triage_classify செயல்பாட்டை அழைக்கவும்.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def run_inference(llm, example: dict) -> dict:
    prompt = build_prompt(example)
    response = llm(prompt, max_tokens=512, temperature=0.0)
    raw_text = response["choices"][0]["text"]

    parse_error = False
    try:
        json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        parse_error = True

    return {
        "input": example,
        "output": raw_text,
        "gold_level": example["gold_level"],
        "parse_error": parse_error,
    }


def run_baseline(config: dict, examples: list[dict]) -> list[dict]:
    llm = Llama(
        model_path=config["model_path"],
        n_gpu_layers=config["n_gpu_layers"],
        n_ctx=config["n_ctx"],
        verbose=False,
    )

    results = []
    for example in examples:
        result = run_inference(llm, example)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stock Gemma 4 E4B on Tamil triage examples."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--examples",
        default="../eval/data/baseline_examples.json",
        help="Path to JSON file with triage examples.",
    )
    parser.add_argument(
        "--output",
        default="../eval/results/baseline_results.json",
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.examples, "r", encoding="utf-8") as f:
        examples = json.load(f)

    results = run_baseline(config, examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
