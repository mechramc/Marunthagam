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

_TRAINING_STACK_ERROR: Exception | None = None
try:
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    _HAS_TRAINING_DEPS = True
except Exception as exc:
    _HAS_TRAINING_DEPS = False
    _TRAINING_STACK_ERROR = exc


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
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


def tokenize_formatted_text(example: dict, tokenizer, max_seq_length: int) -> dict:
    """Tokenize a formatted chat example for causal language modeling."""
    tokenized = tokenizer(
        text=example["text"],
        truncation=True,
        max_length=max_seq_length,
    )
    tokenized["labels"] = list(tokenized["input_ids"])
    return tokenized


def get_text_tokenizer(tokenizer_or_processor):
    """Use the underlying text tokenizer when the model loader returns a multimodal processor."""
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def load_dataset(data_dir: Path, specialist: str):
    """Load train and val splits for the given specialist."""
    specialist_dir = data_dir / specialist
    if not specialist_dir.exists():
        raise FileNotFoundError(
            f"Specialist data directory not found: {specialist_dir}. "
            "Run format_training_data.py first or point data_dir at an existing split."
        )
    train_data = load_jsonl(specialist_dir / "train.jsonl")
    val_data = load_jsonl(specialist_dir / "val.jsonl")
    if not train_data or not val_data:
        raise ValueError(
            f"Expected non-empty train and val splits in {specialist_dir}, "
            f"got train={len(train_data)} val={len(val_data)}."
        )
    return train_data, val_data


def train(cfg: dict, seed: int) -> None:
    """Run QLoRA fine-tuning for one specialist + seed combination."""
    if not _HAS_TRAINING_DEPS:
        raise RuntimeError(
            "Training dependencies are not usable in this environment. "
            f"Original error: {_TRAINING_STACK_ERROR!r}"
        )

    torch.manual_seed(seed)

    specialist = cfg["specialist"]
    configured_output_dir = cfg.get("output_dir", f"outputs/{specialist}-seed{seed}")
    output_dir = TRAINING_ROOT / configured_output_dir
    data_dir = TRAINING_ROOT / cfg.get("data_dir", "data/formatted")
    dataset_num_proc = cfg.get("dataset_num_proc", 4)
    logging_steps = cfg.get("logging_steps", 10)
    eval_strategy = cfg.get("eval_strategy", "epoch")
    save_strategy = cfg.get("save_strategy", "epoch")
    eval_steps = cfg.get("eval_steps")
    save_steps = cfg.get("save_steps")
    max_steps = cfg.get("max_steps", -1)
    load_best_model_at_end = cfg.get("load_best_model_at_end", True)

    # Fail on data issues before downloading the base model.
    train_raw, val_raw = load_dataset(data_dir, specialist)

    print(f"Training {specialist} specialist (seed={seed})")
    print(f"Output: {output_dir}/final")
    print(f"Data: {data_dir / specialist}")

    # Load base model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,          # Auto-detect (BF16 on Ampere+, FP16 otherwise)
        load_in_4bit=True,   # QLoRA 4-bit
    )
    text_tokenizer = get_text_tokenizer(tokenizer)

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
    formatted_train_ds = Dataset.from_list(train_raw).map(
        lambda ex: apply_chat_template(ex, text_tokenizer),
        remove_columns=list(train_raw[0].keys()),
    )
    formatted_val_ds = Dataset.from_list(val_raw).map(
        lambda ex: apply_chat_template(ex, text_tokenizer),
        remove_columns=list(val_raw[0].keys()),
    )
    train_ds = formatted_train_ds.map(
        lambda ex: tokenize_formatted_text(ex, text_tokenizer, cfg["max_seq_length"]),
        remove_columns=["text"],
    )
    val_ds = formatted_val_ds.map(
        lambda ex: tokenize_formatted_text(ex, text_tokenizer, cfg["max_seq_length"]),
        remove_columns=["text"],
    )

    wandb_enabled = cfg.get("use_wandb", False)
    run_name = f"marunthagam-{specialist}-seed{seed}"

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            warmup_steps=cfg.get("warmup_steps", 50),
            num_train_epochs=cfg["epochs"],
            max_steps=max_steps,
            dataset_kwargs={"skip_prepare_dataset": True},
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            load_best_model_at_end=load_best_model_at_end,
            remove_unused_columns=False,
            metric_for_best_model="eval_loss",
            seed=seed,
            report_to="wandb" if wandb_enabled else "none",
            run_name=run_name if wandb_enabled else None,
        ),
        dataset_num_proc=dataset_num_proc,
    )

    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    text_tokenizer.save_pretrained(str(final_path))
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
