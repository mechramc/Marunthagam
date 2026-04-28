"""
Export fine-tuned Marunthagam LoRA + base model to GGUF for llama.cpp deployment.

Uses Unsloth's native GGUF export. Supports Q4_K_M (phone deployment),
Q8_0 (quality testing), and BF16 (full precision benchmark).

Usage:
    # Export triage specialist (best seed)
    python export_gguf.py \
        --checkpoint outputs/triage-seed42/final \
        --output models/triage-E4B-Q4_K_M.gguf

    # Export fused model
    python export_gguf.py \
        --checkpoint outputs/fused/final \
        --output models/marunthagam-fused-E4B-Q4_K_M.gguf \
        --quantization q4_k_m
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except ImportError:
    _HAS_UNSLOTH = False


QUANTIZATION_CHOICES = ("q4_k_m", "q8_0", "bf16")


def export_gguf(checkpoint_path: str, output_path: str, quantization: str) -> None:
    """Load a fine-tuned LoRA checkpoint and export to GGUF."""
    if not _HAS_UNSLOTH:
        raise ImportError(
            "Unsloth not installed. Run: pip install unsloth"
        )

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading checkpoint: {ckpt}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ckpt),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to GGUF ({quantization}): {out_path}")
    try:
        model.save_pretrained_gguf(
            str(out_path.with_suffix("")),  # Unsloth appends .gguf
            tokenizer,
            quantization_method=quantization,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "MODEL_ARCH.GEMMA4" in message or "GEMMA4" in message:
            raise RuntimeError(
                "GGUF conversion reached the local llama.cpp converter, but that converter "
                "does not support Gemma 4 yet. Update the local llama.cpp / gguf tooling "
                "that Unsloth uses, then retry the same export command."
            ) from exc
        raise
    print(f"Export complete: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Marunthagam LoRA checkpoint to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned LoRA checkpoint directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=QUANTIZATION_CHOICES,
        help="Quantization method (default: q4_k_m for phone deployment)",
    )
    args = parser.parse_args()
    export_gguf(args.checkpoint, args.output, args.quantization)


if __name__ == "__main__":
    main()
