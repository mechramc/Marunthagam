"""
Train a specialist LoRA adapter on Gemma 4 E4B using Unsloth QLoRA.

Usage:
    python train_lora.py --config configs/lora_triage.yaml --seed 42
    python train_lora.py --config configs/lora_derm.yaml --seed 137
    python train_lora.py --config configs/lora_maternal.yaml --seed 256

Training produces: outputs/{specialist}-seed{seed}/final/
"""
from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path

# Guard imports for environments without GPU/Unsloth (allows syntax checking)
try:
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    _HAS_TRAINING_DEPS = True
except ImportError:
    _HAS_TRAINING_DEPS = False


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template(example: dict, tokenizer) -> dict:
    """Apply Gemma 4 chat template to convert a messages list to a single text string."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_dataset(data_dir: Path, specialist: str):
    """Load train and val splits for the given specialist."""
    specialist_dir = data_dir / specialist
    train_data = load_jsonl(specialist_dir / "train.jsonl")
    val_data = load_jsonl(specialist_dir / "val.jsonl")
    return train_data, val_data


def train(cfg: dict, seed: int) -> None:
    """Run QLoRA fine-tuning for one specialist + seed combination."""
    if not _HAS_TRAINING_DEPS:
        raise RuntimeError(
            "Training dependencies not installed. "
            "Run: pip install -r training/requirements.txt"
        )

    torch.manual_seed(seed)

    specialist = cfg["specialist"]
    output_dir = Path(f"outputs/{specialist}-seed{seed}")
    data_dir = Path("data/formatted")

    print(f"Training {specialist} specialist (seed={seed})")
    print(f"Output: {output_dir}/final")

    # Load base model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,          # Auto-detect (BF16 on Ampere+, FP16 otherwise)
        load_in_4bit=True,   # QLoRA 4-bit
    )

    # Apply LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Load and format datasets
    train_raw, val_raw = load_dataset(data_dir, specialist)
    train_ds = Dataset.from_list(train_raw).map(
        lambda ex: apply_chat_template(ex, tokenizer),
        remove_columns=list(train_raw[0].keys()),
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda ex: apply_chat_template(ex, tokenizer),
        remove_columns=list(val_raw[0].keys()),
    )

    wandb_enabled = cfg.get("use_wandb", False)
    run_name = f"marunthagam-{specialist}-seed{seed}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            warmup_steps=cfg.get("warmup_steps", 50),
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=seed,
            report_to="wandb" if wandb_enabled else "none",
            run_name=run_name if wandb_enabled else None,
        ),
        dataset_num_proc=4,
    )

    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved to {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Marunthagam specialist LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config (lora_triage.yaml etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (use 42, 137, 256)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.seed)


if __name__ == "__main__":
    main()
