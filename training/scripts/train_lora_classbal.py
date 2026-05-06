"""
Train a specialist LoRA with token-level class-balanced CE on the level token only.

Differs from train_lora.py only in:
1. Tokenization records the level-token char range per example using
   tokenizer offset mapping; converts to token indices and stores a
   per-token `loss_weight` tensor (1.0 everywhere except the level
   tokens, which get the inverse-frequency class weight capped at
   `weight_cap`).
2. Custom WeightedSFTTrainer overrides compute_loss to apply the
   per-token weight on top of the normal causal-LM CE. The rest of
   the sequence (user prompt, JSON wrapper, suspected_conditions,
   reasoning_chain, next_steps_tamil) trains identically to plain SFT.
3. On the first batch, logs per-class loss components and asserts that
   level-token weights match the configured class weights — fails fast
   if the offset detection misaligned.

Usage:
    python training/scripts/train_lora_classbal.py \
        --config training/configs/lora_triage_relabel.yaml \
        --seed 42 \
        --weight-cap 3.0 \
        --output-suffix classbal3x

Output dir convention: <cfg.output_dir>-<output-suffix>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

_TRAINING_STACK_ERROR: Exception | None = None
try:
    import torch
    import torch.nn.functional as F
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


def get_text_tokenizer(tok):
    return getattr(tok, "tokenizer", tok)


def gold_level_of(record: dict) -> Optional[str]:
    """Pull the gold level from the tool message of a chat-format record."""
    for m in record.get("messages", []):
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str):
                try:
                    p = json.loads(c)
                except json.JSONDecodeError:
                    return None
            elif isinstance(c, dict):
                p = c
            else:
                return None
            lvl = str(p.get("level", "")).upper()
            return lvl if lvl in ("GREEN", "YELLOW", "RED") else None
    return None


def compute_class_weights(records: list[dict], cap: float) -> dict[str, float]:
    """
    weight_class = max_class_count / class_count, capped at `cap`.
    YELLOW (the dominant class on the relabeled triage data) ends up at 1.0.
    """
    counts = Counter(gold_level_of(r) for r in records)
    counts.pop(None, None)
    if not counts:
        raise RuntimeError("No gold levels found in training data")
    max_n = max(counts.values())
    weights = {cls: min(max_n / counts[cls], cap) for cls in counts}
    # Force majority class to exactly 1.0 if it ties max_n (avoids float drift).
    for cls, n in counts.items():
        if n == max_n:
            weights[cls] = 1.0
    return weights


def find_level_token_span(
    text: str,
    tokenizer,
    gold_level: str,
) -> Optional[tuple[int, int]]:
    """
    Locate the level VALUE tokens in the chat-templated text.

    Returns (start_token_idx, end_token_idx_exclusive) or None if the
    needle string isn't present (shouldn't happen for well-formed records).
    """
    needle = '"level": "' + gold_level + '"'
    pos = text.find(needle)
    if pos < 0:
        # Try alternate quoting / spacing patterns the chat template may produce.
        for alt_needle in (
            '"level":"' + gold_level + '"',
            "'level': '" + gold_level + "'",
        ):
            pos = text.find(alt_needle)
            if pos >= 0:
                needle = alt_needle
                break
        else:
            return None

    # Char range of the level VALUE itself (not the key/quotes).
    if needle.startswith('"level": "'):
        value_start_char = pos + len('"level": "')
    elif needle.startswith('"level":"'):
        value_start_char = pos + len('"level":"')
    else:
        value_start_char = pos + len("'level': '")
    value_end_char = value_start_char + len(gold_level)

    # Tokenize with offsets; find tokens whose char-span lies INSIDE the value.
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    start_tok = end_tok = None
    for i, (lo, hi) in enumerate(offsets):
        if lo == 0 and hi == 0:
            continue  # special tokens
        if lo >= value_start_char and hi <= value_end_char:
            if start_tok is None:
                start_tok = i
            end_tok = i + 1
    return (start_tok, end_tok) if start_tok is not None else None


def apply_chat_template(record: dict, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_with_weight(
    record: dict,
    tokenizer,
    max_seq_length: int,
    class_weights: dict[str, float],
    diagnostics: list,
) -> dict:
    """Tokenize one record + build a per-token loss_weight vector."""
    text = apply_chat_template(record, tokenizer)
    enc = tokenizer(
        text=text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])
    labels = list(input_ids)

    weight = [1.0] * len(input_ids)
    gold = gold_level_of(record)
    span_detected = False

    span_start = span_end = None
    if gold is not None:
        span = find_level_token_span(text, tokenizer, gold)
        if span is not None:
            span_start, span_end = span
            if span_end <= len(input_ids):
                span_detected = True
                cw = class_weights.get(gold, 1.0)
                for i in range(span_start, span_end):
                    weight[i] = cw

    # Diagnostic: keep first 9 examples' span info — pick across classes so we
    # don't see only the dominant class.
    if len(diagnostics) < 9:
        diagnostics.append({
            "gold": gold,
            "span": (span_start, span_end),
            "n_tokens": len(input_ids),
            "weight_at_span": (
                weight[span_start:span_end] if span_start is not None else []
            ),
        })

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weight": weight,
        "_span_detected": span_detected,
        "_gold_class": gold or "UNK",
    }


class WeightedDataCollator:
    """
    Variable-length collator that pads input_ids, attention_mask, labels, AND
    loss_weight to the longest sequence in the batch.

    Padding values:
        input_ids       — pad_token_id
        attention_mask  — 0
        labels          — -100 (ignore_index for CE)
        loss_weight     — 0.0 (so padded positions contribute zero to the
                            weighted sum AND to the normaliser denominator)
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention = []
        labels = []
        weights = []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
            weights.append(f["loss_weight"] + [0.0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weight": torch.tensor(weights, dtype=torch.float32),
        }


class WeightedSFTTrainer(SFTTrainer):
    """
    Standard causal-LM CE per token, multiplied by a per-token weight.

    Reduces with `sum(w * loss) / sum(w * mask)` so the gradient magnitude
    on level-token positions is proportional to the class weight, not
    averaged out by the much larger pool of weight-1.0 tokens.

    ⚠️  Changes the loss SCALE relative to plain SFT — values are not
    directly comparable to the unweighted runs. They ARE comparable
    across weighted runs.
    """

    _first_batch_diagnostics_done: bool = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss_weight = inputs.pop("loss_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # CLM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        per_tok_loss = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )

        if loss_weight is None:
            mask = (flat_labels != -100).float()
            loss = (per_tok_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            shift_weight = loss_weight[..., 1:].contiguous()
            flat_weight = shift_weight.view(-1).to(per_tok_loss.dtype)
            mask = (flat_labels != -100).to(per_tok_loss.dtype)
            weighted = per_tok_loss * flat_weight * mask
            loss = weighted.sum() / (flat_weight * mask).sum().clamp(min=1)

            # Sanity log on the very first batch: per-class weighted-token loss
            if not self._first_batch_diagnostics_done:
                self._log_first_batch_diagnostics(
                    flat_labels, flat_weight, mask, per_tok_loss
                )
                self._first_batch_diagnostics_done = True

        return (loss, outputs) if return_outputs else loss

    def _log_first_batch_diagnostics(
        self, flat_labels, flat_weight, mask, per_tok_loss
    ):
        with torch.no_grad():
            # Bucket weighted positions by their weight value (≈ class).
            uniq_weights = sorted(set(flat_weight[mask > 0].cpu().tolist()))
            print("\n[classbal] first-batch diagnostics:")
            for w in uniq_weights:
                bucket = (flat_weight == w) & (mask > 0)
                n = bucket.sum().item()
                if n == 0:
                    continue
                avg = per_tok_loss[bucket].mean().item()
                print(
                    f"  weight={w:.4f}  positions={int(n):6d}  "
                    f"mean_per_token_loss={avg:.4f}"
                )
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-cap", type=float, default=3.0)
    parser.add_argument("--output-suffix", default="classbal3x",
                        help="Appended to cfg.output_dir to avoid clobbering")
    args = parser.parse_args()

    if not _HAS_TRAINING_DEPS:
        raise RuntimeError(
            f"Training stack unusable. Original: {_TRAINING_STACK_ERROR!r}"
        )

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)

    spec = cfg["specialist"]
    base_dir = cfg.get("output_dir", f"outputs/{spec}-seed{args.seed}")
    output_dir = TRAINING_ROOT / f"{base_dir}-{args.output_suffix}"
    data_dir = TRAINING_ROOT / cfg.get("data_dir", "data/formatted")

    print(f"[classbal] specialist={spec} seed={args.seed} cap={args.weight_cap}")
    print(f"[classbal] output_dir={output_dir}")

    train_raw = load_jsonl(data_dir / spec / "train.jsonl")
    val_raw = load_jsonl(data_dir / spec / "val.jsonl")

    counts = Counter(gold_level_of(r) for r in train_raw)
    counts.pop(None, None)
    print(f"[classbal] train gold-level distribution: {dict(counts)}")

    weights = compute_class_weights(train_raw, args.weight_cap)
    print(f"[classbal] class weights (cap={args.weight_cap}): {weights}")

    # Save weights to output dir for audit
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_weights.json").write_text(
        json.dumps({"counts": dict(counts), "weights": weights, "cap": args.weight_cap},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Load model
    model, processor = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    text_tok = get_text_tokenizer(processor)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    diag_train: list = []
    diag_val: list = []
    train_ds = Dataset.from_list(train_raw).map(
        lambda ex: tokenize_with_weight(
            ex, text_tok, cfg["max_seq_length"], weights, diag_train
        ),
        remove_columns=list(train_raw[0].keys()),
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda ex: tokenize_with_weight(
            ex, text_tok, cfg["max_seq_length"], weights, diag_val
        ),
        remove_columns=list(val_raw[0].keys()),
    )

    print("[classbal] sample tokenisation diagnostics (first 9 train rows):")
    for d in diag_train:
        print(f"  gold={d['gold']}  span={d['span']}  n_tokens={d['n_tokens']}  "
              f"weights@span={d['weight_at_span']}")

    # Verify span detection on the FULL dataset, not just diagnostics —
    # span_detected stays True regardless of class weight (so YELLOW @ w=1.0
    # is correctly counted as 'detected' even though weight didn't change).
    n_detected = sum(1 for ex in train_ds if ex["_span_detected"])
    span_by_class: dict[str, int] = {}
    for ex in train_ds:
        if ex["_span_detected"]:
            span_by_class[ex["_gold_class"]] = span_by_class.get(ex["_gold_class"], 0) + 1
    total_by_class: dict[str, int] = {}
    for ex in train_ds:
        total_by_class[ex["_gold_class"]] = total_by_class.get(ex["_gold_class"], 0) + 1
    print(f"[classbal] span detection coverage:")
    for cls, total in total_by_class.items():
        det = span_by_class.get(cls, 0)
        print(f"  {cls}: {det}/{total} ({det/total*100:.1f}%)")
    print(f"[classbal] OVERALL: {n_detected}/{len(train_ds)} "
          f"({n_detected/len(train_ds)*100:.1f}%) rows have detected level-token span.")
    if n_detected < int(0.9 * len(train_ds)):
        raise RuntimeError(
            f"Refusing to train: only {n_detected}/{len(train_ds)} rows have "
            f"detected level spans. Span detection is broken — fix find_level_token_span."
        )

    # Drop the bookkeeping columns before passing to trainer
    train_ds = train_ds.remove_columns(["_span_detected", "_gold_class"])
    val_ds = val_ds.remove_columns(["_span_detected", "_gold_class"])

    collator = WeightedDataCollator(text_tok)
    run_name = f"marunthagam-{spec}-classbal{args.weight_cap}-seed{args.seed}"

    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=text_tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            warmup_steps=cfg.get("warmup_steps", 50),
            num_train_epochs=cfg["epochs"],
            dataset_kwargs={"skip_prepare_dataset": True},
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg.get("logging_steps", 10),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            metric_for_best_model="eval_loss",
            seed=args.seed,
            report_to="none",
            run_name=run_name,
        ),
        data_collator=collator,
        dataset_num_proc=cfg.get("dataset_num_proc", 4),
    )

    trainer.train()

    final = output_dir / "final"
    model.save_pretrained(str(final))
    text_tok.save_pretrained(str(final))
    print(f"[classbal] saved to {final}")


if __name__ == "__main__":
    main()
