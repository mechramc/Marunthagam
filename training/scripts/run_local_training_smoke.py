"""
Create local formatted fixture data and train the stub KALAVAI router.

This is the runnable local training path in the current checkout. It does not
fine-tune Gemma 4; it verifies the data formatting pipeline and router training
loop using the hand-crafted fixture datasets.

Usage:
    python scripts/run_local_training_smoke.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from format_training_data import format_reviewed_dataset
from train_router import (
    SPECIALISTS,
    embed_text,
    load_config,
    load_router_data,
    train_router,
)


FIXTURE_FILES = {
    "triage": "triage_reviewed.jsonl",
    "derm": "derm_reviewed.jsonl",
    "maternal": "maternal_reviewed.jsonl",
}


def build_fixture_splits(training_root: Path, output_dir: Path, seed: int) -> None:
    fixtures_dir = training_root / "data" / "fixtures"
    for specialist in SPECIALISTS:
        reviewed_path = fixtures_dir / FIXTURE_FILES[specialist]
        out_dir, split_sizes = format_reviewed_dataset(
            reviewed_path=str(reviewed_path),
            specialist=specialist,
            output_dir=str(output_dir),
            seed=seed,
        )
        print(
            f"{specialist}: train={split_sizes['train']} "
            f"val={split_sizes['val']} test={split_sizes['test']} -> {out_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local Marunthagam fixture formatting + router smoke pipeline"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for fixture split generation")
    parser.add_argument(
        "--router-config",
        default="configs/router_local.yaml",
        help="Router config path relative to the training directory",
    )
    args = parser.parse_args()

    training_root = Path(__file__).resolve().parents[1]
    router_config_path = training_root / args.router_config
    cfg = load_config(str(router_config_path))

    formatted_dir = training_root / cfg["data_dir"]
    output_dir = training_root / cfg["output_dir"]

    print("Building local fixture splits...")
    build_fixture_splits(training_root, formatted_dir, args.seed)

    print("\nTraining stub router...")
    texts, labels = load_router_data(str(formatted_dir))
    embeddings = embed_text(texts, dim=cfg["embedding_dim"])
    router = train_router(cfg, embeddings, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "router_weights.pt"
    torch.save(router.state_dict(), str(weights_path))

    print(f"\nSaved stub router weights to {weights_path}")
    print("Local smoke pipeline complete.")


if __name__ == "__main__":
    main()
