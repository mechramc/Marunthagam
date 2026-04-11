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
import random
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


def embed_text(texts: list[str], dim: int = 768) -> "np.ndarray":
    """
    Embed text for router training.

    TODO: Replace stub with actual Gemma 4 E4B hidden state extraction.
    In production: run base model forward pass, extract last-layer CLS embedding.

    Current stub: random embeddings for pipeline development and testing.
    The router architecture is correct; swap embed_text when E4B embeddings are ready.
    """
    # Stub: random embeddings (replace with actual E4B embeddings)
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(texts), dim)).astype(np.float32)


def train_router(
    cfg: dict,
    embeddings: "np.ndarray",
    labels: list[int],
) -> "KalavaiRouter":
    """Train the router and return the trained model."""
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    texts, labels = load_router_data(cfg["data_dir"])
    print(f"Loaded {len(texts)} examples ({dict(zip(SPECIALISTS, [labels.count(i) for i in range(3)]))})")

    embeddings = embed_text(texts, dim=cfg["embedding_dim"])
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
