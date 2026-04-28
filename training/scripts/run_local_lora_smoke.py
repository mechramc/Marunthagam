"""
Build a small triage split from fixture data, then run a 1-step local LoRA smoke test.

This keeps the local LoRA path self-contained: it prepares a non-empty validation
split before invoking train_lora.py's training function.

Usage:
    python scripts/run_local_lora_smoke.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from format_training_data import format_reviewed_dataset
from train_lora import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local Marunthagam triage LoRA smoke pipeline"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for formatting and LoRA training")
    parser.add_argument(
        "--config",
        default="configs/lora_triage_local.yaml",
        help="LoRA config path relative to the training directory",
    )
    args = parser.parse_args()

    training_root = Path(__file__).resolve().parents[1]
    reviewed_path = training_root / "data" / "fixtures" / "triage_reviewed.jsonl"
    local_data_dir = training_root / "data" / "local_formatted"
    config_path = training_root / args.config

    print("Building local triage split...")
    out_dir, split_sizes = format_reviewed_dataset(
        reviewed_path=str(reviewed_path),
        specialist="triage",
        output_dir=str(local_data_dir),
        seed=args.seed,
    )
    print(
        f"triage: train={split_sizes['train']} "
        f"val={split_sizes['val']} test={split_sizes['test']} -> {out_dir}"
    )

    print("\nRunning 1-step LoRA smoke training...")
    cfg = load_config(str(config_path))
    train(cfg, args.seed)


if __name__ == "__main__":
    main()
