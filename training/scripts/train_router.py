"""
Train the KALAVAI MoE router for Marunthagam.

The router is a single linear layer + softmax that routes input embeddings
to the correct specialist LoRA (triage=0, derm=1, maternal=2).

Based on KALAVAI cooperative LoRA fusion methodology (arXiv:2603.22755).

Usage:
    python train_router.py --config configs/router.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import yaml
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


SPECIALIST_MAP = {"triage": 0, "derm": 1, "maternal": 2}
SPECIALISTS = list(SPECIALIST_MAP.keys())


class KalavaiRouter(nn.Module):
    """
    KALAVAI MoE router: single linear layer + softmax.

    Input: embedding vector of dimension input_dim
    Output: probability distribution over num_specialists

    At inference: top-1 routing (or top-2 weighted merge for ambiguous inputs).
    This minimal architecture (< 1MB) embeds into the GGUF checkpoint.
    """

    def __init__(self, input_dim: int, num_specialists: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_specialists)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)


def load_router_data(data_dir: str) -> tuple[list[str], list[int]]:
    """
    Load validation examples from all three specialist val.jsonl files.
    Returns (texts, labels) where label is 0/1/2 for triage/derm/maternal.
    """
    texts: list[str] = []
    labels: list[int] = []

    for specialist, label in SPECIALIST_MAP.items():
        val_path = Path(data_dir) / specialist / "val.jsonl"
        if not val_path.exists():
            raise FileNotFoundError(
                f"Val data not found: {val_path}. "
                f"Run format_training_data.py for each specialist first."
            )
        with open(val_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                user_msg = next(
                    (m["content"] for m in ex["messages"] if m["role"] == "user"),
                    "",
                )
                texts.append(user_msg)
                labels.append(label)

    return texts, labels


def embed_text(texts: list[str], dim: int = 768, embedder: str = "stub") -> "np.ndarray":
    """
    Embed text for router training. Dispatches based on `embedder`:
      - "stub":     deterministic random vectors (pipeline-test only, learns nothing)
      - "e4b":      mean-pooled last hidden state of Gemma 4 E4B (real embeddings)
    """
    if embedder == "e4b":
        return _embed_text_e4b(texts)
    if embedder == "stub":
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), dim)).astype(np.float32)
    raise ValueError(f"Unknown embedder: {embedder!r}")


def _embed_text_e4b(
    texts: list[str],
    base_model: str = "unsloth/gemma-4-E4B-it",
    max_seq_length: int = 512,
    batch_size: int = 8,
) -> "np.ndarray":
    """
    Run Gemma 4 E4B (4-bit) forward passes and mean-pool the last hidden
    state over non-pad tokens. Returns float32 array shape (N, hidden_size).
    """
    # Local import keeps the script importable on machines without unsloth/torch.
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model.eval()
    device = next(model.parameters()).device
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    pieces: list["np.ndarray"] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = text_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = out.hidden_states[-1]  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pieces.append(pooled.float().cpu().numpy())

    return np.concatenate(pieces, axis=0).astype(np.float32)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_val_split(
    embeddings: "np.ndarray",
    labels: list[int],
    val_fraction: float = 0.20,
) -> tuple["np.ndarray", "np.ndarray", list[int], list[int]]:
    """Create a validation split that still works on small local fixture datasets."""
    total_examples = len(labels)
    num_classes = len(set(labels))
    if total_examples < 2:
        raise ValueError("Router training needs at least 2 examples.")

    min_examples_per_class = min(labels.count(label) for label in set(labels))
    can_stratify = min_examples_per_class >= 2 and total_examples > num_classes

    if can_stratify:
        val_size = max(math.ceil(total_examples * val_fraction), num_classes)
        val_size = min(val_size, total_examples - 1)
        stratify_labels = labels
    else:
        val_size = max(1, math.ceil(total_examples * val_fraction))
        val_size = min(val_size, total_examples - 1)
        stratify_labels = None

    return train_test_split(
        embeddings,
        labels,
        test_size=val_size,
        random_state=42,
        stratify=stratify_labels,
    )


def train_router(
    cfg: dict,
    embeddings: "np.ndarray",
    labels: list[int],
) -> "KalavaiRouter":
    """Train the router and return the trained model."""
    X_train, X_val, y_train, y_val = build_train_val_split(
        embeddings=embeddings,
        labels=labels,
        val_fraction=cfg.get("val_fraction", 0.20),
    )

    embedding_dim = cfg["embedding_dim"]
    router = KalavaiRouter(input_dim=embedding_dim, num_specialists=3)
    optimizer = torch.optim.Adam(router.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.from_numpy(X_val)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        router.train()
        optimizer.zero_grad()
        logits = router(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        router.eval()
        with torch.no_grad():
            val_probs = router(X_val_t)
            val_preds = val_probs.argmax(dim=1).numpy()

        val_acc = (val_preds == np.array(y_val)).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in router.state_dict().items()}

        if epoch % max(1, cfg["epochs"] // 10) == 0 or epoch == cfg["epochs"]:
            print(f"Epoch {epoch:3d}/{cfg['epochs']} — loss: {loss.item():.4f} — val_acc: {val_acc:.4f}")

    # Restore best checkpoint
    if best_state is not None:
        router.load_state_dict(best_state)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("\nClassification Report (validation set):")
    print(classification_report(y_val, val_preds, target_names=SPECIALISTS, zero_division=0))

    return router


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KALAVAI MoE router")
    parser.add_argument("--config", default="configs/router.yaml", help="Router config YAML")
    args = parser.parse_args()

    if not _HAS_DEPS:
        raise ImportError(
            "Training dependencies not installed. "
            "Run: pip install numpy torch scikit-learn"
        )

    cfg = load_config(args.config)
    embedder = cfg.get("embedder", "stub")
    print(f"Router embedding path: {embedder}")

    texts, labels = load_router_data(cfg["data_dir"])
    print(f"Loaded {len(texts)} examples ({dict(zip(SPECIALISTS, [labels.count(i) for i in range(3)]))})")

    embeddings = embed_text(texts, dim=cfg["embedding_dim"], embedder=embedder)
    print(f"Embeddings shape: {embeddings.shape}")

    router = train_router(cfg, embeddings, labels)

    out_path = Path(cfg["output_dir"]) / "router_weights.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), str(out_path))
    print(f"\nRouter weights saved to {out_path}")
    print(f"Router size: {sum(p.numel() for p in router.parameters()):,} parameters")
    print(f"Routing strategy: {cfg.get('routing_strategy', 'top1')}")


if __name__ == "__main__":
    main()
