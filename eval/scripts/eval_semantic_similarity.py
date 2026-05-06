"""
Marunthagam — Tamil fluency replacement metric (Sprint 3 cheap win).

Replaces the chrF++ score (which gave 0.301 with semantically-valid outputs)
with embedding-based cosine similarity using a multilingual sentence
transformer. Tamil is supported by paraphrase-multilingual-mpnet-base-v2
(50+ languages).

The pipeline runs against the existing chrF eval artifact at
eval/results/chrf_eval_*.json (no model re-inference needed). Adds a
side-by-side column so the writeup can show "chrF++ said 0.30, semantic
similarity said X."

Usage:
    python eval/scripts/eval_semantic_similarity.py
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"
OUT_DIR = REPO / "eval" / "analysis" / "2026-05-07"


def find_latest_chrf() -> Path:
    matches = sorted(
        RESULTS_DIR.glob("chrf_eval_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No chrf_eval_*.json found in eval/results/")
    return matches[0]


def main() -> None:
    chrf_path = find_latest_chrf()
    print(f"Loading chrF eval from {chrf_path.name}")
    with open(chrf_path, encoding="utf-8") as fh:
        chrf_data = json.load(fh)
    rows = chrf_data["raw"]
    print(f"  {len(rows)} (specialist, hypothesis, gold) triples")

    print("\nLoading sentence-transformer (paraphrase-multilingual-mpnet-base-v2)...")
    print("  This is a 950MB model — first run downloads, subsequent runs cached.")
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    # Encode hypotheses + golds in batches
    print("\nEmbedding hypotheses...")
    hyps = [r["hypothesis"] for r in rows]
    golds = [r["gold"] for r in rows]
    hyp_emb = model.encode(hyps, show_progress_bar=False, convert_to_tensor=True,
                           normalize_embeddings=True)
    gold_emb = model.encode(golds, show_progress_bar=False, convert_to_tensor=True,
                            normalize_embeddings=True)

    # Pairwise cosine (normalized → dot product)
    sims = util.cos_sim(hyp_emb, gold_emb).diagonal().tolist()

    # Aggregate
    out_rows = []
    by_spec: dict = {}
    for r, s in zip(rows, sims):
        s = round(float(s), 4)
        out_rows.append({
            "specialist": r["specialist"],
            "row_index": r["row_index"],
            "chrf_plus_plus": r["chrf_plus_plus"],
            "semantic_cosine": s,
            "hypothesis_head": r["hypothesis"][:150],
            "gold_head": r["gold"][:150],
        })
        by_spec.setdefault(r["specialist"], []).append(s)

    print("\n=== Results ===")
    overall = sum(sims) / len(sims)
    print(f"Overall semantic cosine: {overall:.4f}  (chrF++ overall: {chrf_data['overall_chrf_plus_plus']:.4f})")
    print()
    print(f"{'specialist':<10s}  n  chrF++ mean   semantic mean   delta")
    chrf_by_spec = chrf_data.get("by_specialist", {})
    for spec in ("triage", "derm", "maternal"):
        if spec not in by_spec:
            continue
        mean_sem = sum(by_spec[spec]) / len(by_spec[spec])
        mean_chrf = chrf_by_spec.get(spec, {}).get("chrf_mean", 0.0)
        delta = mean_sem - mean_chrf
        print(f"{spec:<10s} {len(by_spec[spec]):3d}  {mean_chrf:.4f}        {mean_sem:.4f}        {delta:+.4f}")

    # Distribution
    print(f"\nSemantic similarity distribution:")
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    for lo, hi in bins:
        count = sum(1 for s in sims if lo <= s < hi)
        print(f"  [{lo:.2f}, {hi:.2f})  n={count:3d}  ({count / len(sims) * 100:.1f}%)")

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"semantic_eval_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "source_chrf_eval": chrf_path.name,
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "n_rows": len(out_rows),
        "overall_semantic_cosine": round(overall, 4),
        "overall_chrf_plus_plus": chrf_data["overall_chrf_plus_plus"],
        "by_specialist": {
            spec: {
                "n": len(by_spec[spec]),
                "semantic_mean": round(sum(by_spec[spec]) / len(by_spec[spec]), 4),
                "chrf_mean": chrf_by_spec.get(spec, {}).get("chrf_mean", 0.0),
            }
            for spec in by_spec
        },
        "raw": out_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path.name}")


if __name__ == "__main__":
    main()
