"""
Marunthagam — Results visualisation.

Reads everything in eval/results/, produces a deck of PNG figures in
eval/notebooks/figures/, and emits a Markdown report at
eval/notebooks/figures/SUMMARY.md that pulls the numbers together.

Usage:
    python eval/notebooks/plot_results.py
    python eval/notebooks/plot_results.py --no-show     # default; just write PNGs
    python eval/notebooks/plot_results.py --remaining   # also draw progress tracker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
FIGURES_DIR = REPO_ROOT / "eval" / "notebooks" / "figures"

# Triage class brand colors
COLOR_GREEN = "#2ca02c"
COLOR_YELLOW = "#e6b800"
COLOR_RED = "#d62728"
COLOR_NEUTRAL = "#1f77b4"
COLOR_FAIL = "#d62728"
COLOR_PASS = "#2ca02c"
COLOR_PENDING = "#cccccc"

TARGETS = {
    "weighted_f1": 0.80,
    "red_recall": 0.90,
    "chrf": 0.60,
    "refusal_rate": 1.00,
    "phone_ttft_s": 3.0,
    "phone_throughput_toks": 8.0,
    "ws_ttft_s": 1.0,
    "ws_throughput_toks": 30.0,
}


def _load_json(p: Path) -> Optional[dict]:
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"  WARNING: could not parse {p.name}: {exc}")
        return None


def _latest_matching(prefix: str, must_have_key: Optional[str] = None) -> Optional[Path]:
    """Pick the most recent eval/results/<prefix>*.json that has the expected schema."""
    candidates = sorted(
        RESULTS_DIR.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        data = _load_json(p)
        if data is None:
            continue
        if must_have_key and must_have_key not in data:
            continue
        return p
    return None


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_eval_overview(payload: dict, label: str, out_name: str) -> Optional[Path]:
    """Per-class P/R/F1 bars + headline F1/RED-recall vs targets."""
    seed_results = payload.get("seed_results")
    per_class: dict[str, dict[str, float]]
    if seed_results:
        per_class = seed_results[0]["per_class"]
        weighted_f1 = payload["aggregated"]["weighted_f1_mean"]
        red_recall = payload["aggregated"]["red_recall_mean"]
    else:
        per_class = payload.get("per_class") or {}
        weighted_f1 = payload.get("weighted_f1", 0.0)
        red_recall = payload.get("red_recall", 0.0)

    if not per_class:
        return None

    classes = ["GREEN", "YELLOW", "RED"]
    metrics = ["precision", "recall", "f1"]
    colors = [COLOR_GREEN, COLOR_YELLOW, COLOR_RED]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: P/R/F1 bars
    x = np.arange(len(metrics))
    width = 0.25
    ax = axes[0]
    for i, cls in enumerate(classes):
        vals = [per_class.get(cls, {}).get(m, 0.0) for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=cls, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class metrics — {label}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: headline targets
    ax = axes[1]
    headline_metrics = ["Weighted F1", "RED recall"]
    headline_vals = [weighted_f1, red_recall]
    headline_targets = [TARGETS["weighted_f1"], TARGETS["red_recall"]]
    bar_colors = [
        COLOR_PASS if v >= t else COLOR_FAIL
        for v, t in zip(headline_vals, headline_targets)
    ]
    bars = ax.bar(headline_metrics, headline_vals, color=bar_colors, edgecolor="black")
    for i, t in enumerate(headline_targets):
        ax.hlines(t, i - 0.4, i + 0.4, linestyle="--", color="black",
                  linewidth=1.5, label="Target" if i == 0 else None)
    for bar, val in zip(bars, headline_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Headline metrics vs targets — {label}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Marunthagam triage eval — {label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_name)


def plot_fixtures_vs_test_split() -> Optional[Path]:
    """Compare F1 / RED recall on fixtures (50 cases) vs held-out test (131 rows)."""
    fixtures = _latest_matching("run_", must_have_key="aggregated")
    # Find the test_split run specifically
    test_split: Optional[Path] = None
    for p in sorted(RESULTS_DIR.glob("run_test_split_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        test_split = p
        break
    # Find a fixtures-only aggregated run (not test_split)
    fixture_run: Optional[Path] = None
    for p in sorted(RESULTS_DIR.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if "test_split" in p.name:
            continue
        d = _load_json(p)
        if d and "aggregated" in d:
            fixture_run = p
            break
    if not fixture_run or not test_split:
        return None

    f_data = _load_json(fixture_run)
    t_data = _load_json(test_split)
    if not f_data or not t_data:
        return None

    metrics = ["weighted_f1_mean", "macro_f1_mean", "red_recall_mean"]
    metric_labels = ["Weighted F1", "Macro F1", "RED recall"]
    f_vals = [f_data["aggregated"][m] for m in metrics]
    t_vals = [t_data["aggregated"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width / 2, f_vals, width,
                   label=f"Fixtures (n={f_data.get('n_cases', '?')})",
                   color=COLOR_NEUTRAL)
    bars2 = ax.bar(x + width / 2, t_vals, width,
                   label=f"Held-out test (n={t_data.get('n_cases', '?')})",
                   color="#ff7f0e")
    for bars, vals in [(bars1, f_vals), (bars2, t_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
            )
    ax.axhline(TARGETS["weighted_f1"], linestyle="--", color="black", alpha=0.5, label="F1 target 0.80")
    ax.axhline(TARGETS["red_recall"], linestyle=":", color="black", alpha=0.5, label="RED recall target 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fixtures vs held-out test — generalisation gap")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "f1_fixtures_vs_test_split.png")


def plot_safety(safety_payload: dict) -> Path:
    by_cat = safety_payload.get("by_category", {})
    if not by_cat:
        # Older shape
        rate = safety_payload.get("refusal_rate", 0.0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Overall"], [rate], color=COLOR_PASS if rate >= 1.0 else COLOR_FAIL)
        ax.set_ylim(0, 1.05)
        ax.set_title("Safety refusal rate")
        return _save(fig, "safety_refusal.png")

    cats = list(by_cat.keys())
    rates = [by_cat[c]["rate"] for c in cats]
    totals = [by_cat[c]["total"] for c in cats]
    bar_colors = [COLOR_PASS if r >= 1.0 else COLOR_FAIL for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(cats, rates, color=bar_colors, edgecolor="black")
    for bar, val, total in zip(bars, rates, totals):
        ax.text(
            min(val + 0.02, 1.02), bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.0f}%  (n={total})",
            va="center", fontsize=10,
        )
    ax.axvline(TARGETS["refusal_rate"], linestyle="--", color="black",
               label="Target 100%")
    ax.set_xlim(0, 1.15)
    overall_rate = safety_payload.get("refusal_rate", 0.0)
    refused = safety_payload.get("refused", 0)
    total = safety_payload.get("total", 0)
    ax.set_title(
        f"Safety refusal rate by category — overall {overall_rate * 100:.1f}% "
        f"({refused}/{total})"
    )
    ax.set_xlabel("Refusal rate")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "safety_refusal_by_category.png")


def plot_latency(latency_payload: dict) -> Optional[Path]:
    benchmarks = latency_payload.get("benchmarks") or []
    if not benchmarks:
        # Legacy shape: results list under "results"
        results = latency_payload.get("results")
        if not results:
            return None
        benchmarks = [{"model_label": "model", "results": results}]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: TTFT vs prompt length
    ax = axes[0]
    for bench in benchmarks:
        lengths = [r["prompt_length"] for r in bench["results"]]
        ttft = [r["avg_ttft_s"] for r in bench["results"]]
        ax.plot(lengths, ttft, marker="o", label=bench["model_label"])
    ax.axhline(TARGETS["phone_ttft_s"], linestyle="--", color="orange",
               label=f"Phone target {TARGETS['phone_ttft_s']}s")
    ax.axhline(TARGETS["ws_ttft_s"], linestyle=":", color="black",
               label=f"Workstation target {TARGETS['ws_ttft_s']}s")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Avg TTFT (s)")
    ax.set_title("Time-to-first-token vs prompt length")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: throughput vs prompt length
    ax = axes[1]
    for bench in benchmarks:
        lengths = [r["prompt_length"] for r in bench["results"]]
        tput = [r["avg_throughput_toks"] for r in bench["results"]]
        ax.plot(lengths, tput, marker="o", label=bench["model_label"])
    ax.axhline(TARGETS["phone_throughput_toks"], linestyle="--", color="orange",
               label=f"Phone target {TARGETS['phone_throughput_toks']} tok/s")
    ax.axhline(TARGETS["ws_throughput_toks"], linestyle=":", color="black",
               label=f"Workstation target {TARGETS['ws_throughput_toks']} tok/s")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Decode throughput vs prompt length")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Latency benchmark — workstation (RTX 5090)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "latency_benchmark.png")


def plot_progress_tracker(remaining_items: list[dict]) -> Path:
    """
    Render a punch-list of remaining items with status colour.

    `remaining_items` is a list of {"id", "title", "status"} where status
    is one of: done, in_progress, pending, blocked.
    """
    fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(remaining_items) + 1)))
    status_colors = {
        "done": COLOR_PASS,
        "in_progress": "#1f77b4",
        "pending": COLOR_PENDING,
        "blocked": COLOR_FAIL,
    }

    for i, item in enumerate(remaining_items):
        y = len(remaining_items) - i - 1
        c = status_colors.get(item["status"], COLOR_PENDING)
        ax.barh(y, 1.0, color=c, edgecolor="black")
        text = f"{item['id']}  {item['title']}  [{item['status'].upper()}]"
        ax.text(0.01, y, text, va="center", fontsize=10, color="black")

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in status_colors.items()]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.set_title("Marunthagam — pending work tracker", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "progress_tracker.png")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _format_per_class(per_class: dict[str, dict[str, float]]) -> str:
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls in ["GREEN", "YELLOW", "RED"]:
        s = per_class.get(cls, {})
        lines.append(
            f"| {cls} | {s.get('precision', 0):.3f} | {s.get('recall', 0):.3f} | "
            f"{s.get('f1', 0):.3f} | {int(s.get('support', 0))} |"
        )
    return "\n".join(lines)


def write_summary(
    fixture_payload: Optional[dict],
    test_split_payload: Optional[dict],
    safety_payload: Optional[dict],
    latency_payload: Optional[dict],
) -> Path:
    parts: list[str] = []
    parts.append("# Marunthagam — Eval Results Summary\n")
    parts.append("Auto-generated by `eval/notebooks/plot_results.py`. Figures live alongside this file.\n")

    # Headline F1 table
    headline_rows: list[str] = ["| Eval | n | Weighted F1 | RED recall | Status |", "|---|---|---|---|---|"]
    if fixture_payload:
        agg = fixture_payload["aggregated"]
        wf1 = agg["weighted_f1_mean"]
        rr = agg["red_recall_mean"]
        st = "PASS" if (wf1 >= 0.80 and rr >= 0.90) else "FAIL"
        headline_rows.append(
            f"| Fixtures | {fixture_payload.get('n_cases', '?')} | "
            f"{wf1:.4f} ± {agg['weighted_f1_std']:.4f} | "
            f"{rr:.4f} ± {agg['red_recall_std']:.4f} | {st} |"
        )
    if test_split_payload:
        agg = test_split_payload["aggregated"]
        wf1 = agg["weighted_f1_mean"]
        rr = agg["red_recall_mean"]
        st = "PASS" if (wf1 >= 0.80 and rr >= 0.90) else "FAIL"
        headline_rows.append(
            f"| Held-out test split | {test_split_payload.get('n_cases', '?')} | "
            f"{wf1:.4f} ± {agg['weighted_f1_std']:.4f} | "
            f"{rr:.4f} ± {agg['red_recall_std']:.4f} | {st} |"
        )
    parts.append("## Triage classification")
    parts.append("\n".join(headline_rows) + "\n")

    if test_split_payload:
        per_class = test_split_payload["seed_results"][0]["per_class"]
        parts.append("### Per-class breakdown — held-out test split (seed 42)\n")
        parts.append(_format_per_class(per_class) + "\n")

    if fixture_payload:
        per_class = fixture_payload["seed_results"][0]["per_class"]
        parts.append("### Per-class breakdown — fixtures (seed 42)\n")
        parts.append(_format_per_class(per_class) + "\n")

    parts.append("![](f1_fixtures_vs_test_split.png)\n")
    parts.append("![](triage_eval_test_split.png)\n")
    parts.append("![](triage_eval_fixtures.png)\n")

    # Safety
    if safety_payload:
        parts.append("## Safety refusal\n")
        rate = safety_payload.get("refusal_rate", 0)
        n = safety_payload.get("total", 0)
        st = safety_payload.get("status", "?")
        parts.append(
            f"Overall: **{rate * 100:.1f}%** refusal "
            f"({safety_payload.get('refused', 0)}/{n}) — target 100% — **{st}**\n"
        )
        by_cat = safety_payload.get("by_category", {})
        if by_cat:
            parts.append("| Category | Refused / Total | Rate |", )
            parts.append("|---|---|---|")
            for cat, stats in sorted(by_cat.items()):
                parts.append(
                    f"| {cat} | {stats['refused']}/{stats['total']} | {stats['rate'] * 100:.1f}% |"
                )
        parts.append("\n![](safety_refusal_by_category.png)\n")

    # Latency
    if latency_payload:
        parts.append("## Latency (workstation, RTX 5090)\n")
        for bench in latency_payload.get("benchmarks", []):
            parts.append(f"### Model: `{bench['model_label']}`\n")
            parts.append("| Prompt len | Avg TTFT (s) | Median TTFT (s) | Avg tok/s | Phone PASS | Workstation PASS |")
            parts.append("|---|---|---|---|---|---|")
            for r in bench["results"]:
                parts.append(
                    f"| {r['prompt_length']} | {r['avg_ttft_s']:.3f} | "
                    f"{r['median_ttft_s']:.3f} | {r['avg_throughput_toks']:.2f} | "
                    f"{'PASS' if r['phone_pass'] else 'FAIL'} | "
                    f"{'PASS' if r['workstation_pass'] else 'FAIL'} |"
                )
            parts.append("")
        parts.append("![](latency_benchmark.png)\n")

    out = FIGURES_DIR / "SUMMARY.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"  Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Pending-work definition (kept inline so plotting is one self-contained script)
# ---------------------------------------------------------------------------

PENDING_ITEMS: list[dict[str, str]] = [
    {"id": "A8", "title": "Held-out test split eval", "status": "done"},
    {"id": "A1", "title": "Safety refusal (100 adversarial prompts)", "status": "done"},
    {"id": "A2", "title": "Workstation TTFT + throughput", "status": "in_progress"},
    {"id": "A3", "title": "KALAVAI fusion ablation (vs single specialist)", "status": "pending"},
    {"id": "A4", "title": "Per-domain specialist gain (vs base E4B)", "status": "pending"},
    {"id": "A5", "title": "Tamil fluency chrF++", "status": "pending"},
    {"id": "A7", "title": "LoRA rank ablation", "status": "pending"},
    {"id": "A6", "title": "Phone TTFT (Android)", "status": "pending"},
    {"id": "B1", "title": "Router triage-class collapse", "status": "pending"},
    {"id": "B2", "title": "Protocol engine in Android inference path", "status": "pending"},
    {"id": "C1", "title": "HuggingFace weights upload", "status": "blocked"},
    {"id": "C2", "title": "Demo video", "status": "pending"},
    {"id": "C3", "title": "README submission polish", "status": "pending"},
    {"id": "C4", "title": "Status doc refresh", "status": "pending"},
    {"id": "C5", "title": "Architecture doc refresh", "status": "pending"},
    {"id": "C6", "title": "Kaggle submission", "status": "pending"},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Marunthagam results plotter")
    parser.add_argument("--no-show", action="store_true", default=True,
                        help="Do not call plt.show(); only write PNGs (default).")
    parser.add_argument("--remaining", action="store_true",
                        help="Also draw the pending-work tracker.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from {RESULTS_DIR}")
    print(f"Writing figures to {FIGURES_DIR}\n")

    # Pick latest fixture run that is NOT a test_split run
    fixture_payload: Optional[dict] = None
    for p in sorted(RESULTS_DIR.glob("run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if "test_split" in p.name:
            continue
        d = _load_json(p)
        if d and "aggregated" in d:
            fixture_payload = d
            print(f"  Fixture run: {p.name}")
            break

    test_split_payload: Optional[dict] = None
    for p in sorted(RESULTS_DIR.glob("run_test_split_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        d = _load_json(p)
        if d and "aggregated" in d:
            test_split_payload = d
            print(f"  Test-split run: {p.name}")
            break

    safety_payload: Optional[dict] = None
    p = _latest_matching("safety_eval_", must_have_key="refusal_rate")
    if p:
        safety_payload = _load_json(p)
        print(f"  Safety run: {p.name}")

    latency_payload: Optional[dict] = None
    p = _latest_matching("latency_")
    if p:
        latency_payload = _load_json(p)
        print(f"  Latency run: {p.name}")

    print()

    # Generate plots
    if test_split_payload:
        plot_eval_overview(test_split_payload, "Held-out test split (n=131)",
                           "triage_eval_test_split.png")
    if fixture_payload:
        plot_eval_overview(fixture_payload, "Fixtures (n=50)",
                           "triage_eval_fixtures.png")
    if test_split_payload and fixture_payload:
        plot_fixtures_vs_test_split()
    if safety_payload:
        plot_safety(safety_payload)
    if latency_payload:
        plot_latency(latency_payload)
    if args.remaining:
        plot_progress_tracker(PENDING_ITEMS)
    plot_progress_tracker(PENDING_ITEMS)  # always include the tracker

    write_summary(fixture_payload, test_split_payload, safety_payload, latency_payload)

    print("\nDone.")


if __name__ == "__main__":
    main()
