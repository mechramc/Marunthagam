"""
Generate dashboard live-data JS module from real Marunthagam eval results.

Reads the Task 6 routed-config predictions (the production held-out result)
and synthesizes a HeatmapCell[] / TrendDay[] / Alert[] / Stats payload with
realistic Tamil Nadu geohash distribution + 7-day temporal spread.

Privacy invariant: no patient identifiers from Task 6 cross over. Only the
aggregate counts (per geohash, per day, per triage level) reach the
dashboard.

Output: dashboard/src/api/realData.js with the same exports as mockData.js
so the React components can swap in/out via a single import edit.

Usage:
    python dashboard/scripts/generate_live_data.py
"""
from __future__ import annotations

import io
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "eval" / "results"
OUT_PATH = REPO / "dashboard" / "src" / "api" / "realData.js"

# Plausible Tamil Nadu geohash cells. tf7/tf8/tf6 prefixes cover Tamil Nadu.
# Cluster names map to rough regions; the dashboard doesn't need real geo
# precision — these just have to render plausibly on the heatmap.
TN_CELLS = [
    # Chennai metropolitan area (urban dense, more cases)
    ("tf8n00", "Chennai-North"),
    ("tf8n01", "Chennai-Central"),
    ("tf8n02", "Chennai-South"),
    ("tf8n03", "Chennai-West"),
    ("tf8n04", "Chennai-Tambaram"),
    # Madurai region
    ("tf6q9d", "Madurai-East"),
    ("tf6q9e", "Madurai-Central"),
    ("tf6q9f", "Madurai-South"),
    # Coimbatore region
    ("tf6nzx", "Coimbatore-Central"),
    ("tf6nzz", "Coimbatore-South"),
    # Tiruchi
    ("tf7p2s", "Tiruchi-North"),
    ("tf7p2t", "Tiruchi-Central"),
    # Salem
    ("tf7nr0", "Salem-East"),
    ("tf7nr1", "Salem-West"),
    # Tirunelveli (rural-leaning, sparser)
    ("tf6jzm", "Tirunelveli-Cluster1"),
    ("tf6jzn", "Tirunelveli-Cluster2"),
    # Erode (rural)
    ("tf7mu6", "Erode-North"),
    ("tf7mu7", "Erode-South"),
    # Kanchipuram (semi-urban)
    ("tf8mfu", "Kanchipuram-East"),
    ("tf8mfv", "Kanchipuram-West"),
]


def find_routed_task6() -> Path:
    cands = sorted(
        RESULTS_DIR.glob("run_task6_routed_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError("No run_task6_routed_*.json found")
    return cands[0]


def main() -> None:
    src = find_routed_task6()
    print(f"Reading routed Task 6 result: {src.name}")
    data = json.loads(src.read_text(encoding="utf-8"))
    preds = data["predictions"]
    print(f"  {len(preds)} predictions")

    rng = random.Random(42)

    # --- 1. Distribute the n=131 predictions across cells with weights ---
    # Urban cells get more cases; rural cells fewer. Weights are arbitrary
    # but produce realistic-looking heatmap density variation.
    cell_weights = [3, 4, 3, 3, 2,   # Chennai
                    3, 3, 2,         # Madurai
                    3, 2,            # Coimbatore
                    2, 2,            # Tiruchi
                    2, 2,            # Salem
                    1, 1,            # Tirunelveli (rural)
                    1, 1,            # Erode (rural)
                    2, 2]            # Kanchipuram
    assert len(cell_weights) == len(TN_CELLS)

    # Sample a cell for each prediction
    cells_for_preds = rng.choices(
        [c[0] for c in TN_CELLS], weights=cell_weights, k=len(preds)
    )

    # --- 2. Spread across last 7 days, weighted toward "today" being partial ---
    # Today + 6 prior days. Predictions distributed roughly uniformly across
    # past 6 days, with today being only 50% complete.
    today = datetime.now(timezone(timedelta(hours=5, minutes=30)))  # IST
    today = today.replace(hour=18, minute=0, second=0, microsecond=0)
    day_offsets = []
    for _ in preds:
        # 6 days of prior data (full) + today (~50% of typical day)
        # Probability mass: 6 days × 1.0 + today × 0.5 = total 6.5, today share = 0.077
        r = rng.random()
        if r < 0.077:
            day_offsets.append(0)
        else:
            day_offsets.append(rng.randint(1, 6))

    # Time-of-day distribution: ASHA workers most active 7am-noon and 4-7pm
    def random_time_of_day():
        if rng.random() < 0.6:
            hour = rng.randint(7, 11)
        else:
            hour = rng.randint(16, 18)
        minute = rng.randint(0, 59)
        return hour, minute

    pred_records = []
    for pred, cell, off in zip(preds, cells_for_preds, day_offsets):
        h, m = random_time_of_day()
        ts = (today - timedelta(days=off)).replace(hour=h, minute=m)
        pred_records.append({
            "geohash": cell,
            "level": pred["pred"],
            "gold_level": pred["gold"],
            "specialist": pred["specialist"],
            "timestamp": ts.isoformat(timespec="seconds"),
            "date": ts.strftime("%Y-%m-%d"),
        })

    # --- 3. Aggregate into HeatmapCell[] ---
    by_cell = defaultdict(lambda: Counter())
    last_ts_by_cell: dict[str, str] = {}
    for r in pred_records:
        by_cell[r["geohash"]][r["level"]] += 1
        if r["timestamp"] > last_ts_by_cell.get(r["geohash"], ""):
            last_ts_by_cell[r["geohash"]] = r["timestamp"]
    heatmap = []
    for cell, _name in TN_CELLS:
        c = by_cell.get(cell, Counter())
        heatmap.append({
            "geohash": cell,
            "green_count": c.get("GREEN", 0),
            "yellow_count": c.get("YELLOW", 0),
            "red_count": c.get("RED", 0),
            "last_updated": last_ts_by_cell.get(
                cell,
                today.isoformat(timespec="seconds"),
            ),
        })

    # --- 4. Aggregate into TrendDay[] ---
    by_day = defaultdict(lambda: Counter())
    for r in pred_records:
        by_day[r["date"]][r["level"]] += 1
    # Build sorted list, oldest first
    dates_sorted = sorted(by_day.keys())
    trend = [
        {
            "date": d,
            "green": by_day[d].get("GREEN", 0),
            "yellow": by_day[d].get("YELLOW", 0),
            "red": by_day[d].get("RED", 0),
        }
        for d in dates_sorted
    ]

    # --- 5. Alerts: cells with red_count >= 2 in the last 24h ---
    cutoff = (today - timedelta(hours=24)).isoformat(timespec="seconds")
    cell_red_recent: dict[str, list[str]] = defaultdict(list)
    for r in pred_records:
        if r["level"] == "RED" and r["timestamp"] >= cutoff:
            cell_red_recent[r["geohash"]].append(r["timestamp"])
    alerts = []
    for cell, ts_list in cell_red_recent.items():
        if len(ts_list) < 2:
            continue
        ts_sorted = sorted(ts_list)
        # Trend: heuristic — compare to prior 24h count
        prior_window_start = (today - timedelta(hours=48)).isoformat(timespec="seconds")
        prior_count = sum(
            1 for r in pred_records
            if r["geohash"] == cell
            and r["level"] == "RED"
            and prior_window_start <= r["timestamp"] < cutoff
        )
        if len(ts_list) > prior_count:
            trend_dir = "up"
        elif len(ts_list) < prior_count:
            trend_dir = "down"
        else:
            trend_dir = "stable"
        alerts.append({
            "id": f"alert-{cell}",
            "geohash": cell,
            "red_count": len(ts_list),
            "trend": trend_dir,
            "first_seen": ts_sorted[0],
            "last_seen": ts_sorted[-1],
        })
    # Sort alerts by red_count desc, take top 10
    alerts.sort(key=lambda a: a["red_count"], reverse=True)
    alerts = alerts[:10]

    # --- 6. Stats: today vs yesterday ---
    today_str = today.strftime("%Y-%m-%d")
    yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    today_counts = by_day.get(today_str, Counter())
    yesterday_counts = by_day.get(yesterday_str, Counter())
    cells_today = {r["geohash"] for r in pred_records if r["date"] == today_str}
    cells_yesterday = {r["geohash"] for r in pred_records if r["date"] == yesterday_str}

    today_total = sum(today_counts.values())
    yesterday_total = sum(yesterday_counts.values())
    today_red = today_counts.get("RED", 0)
    yesterday_red = yesterday_counts.get("RED", 0)

    stats = {
        "total_cases_today": today_total,
        "red_cases_today": today_red,
        "active_cells": len(cells_today),
        "escalation_rate": round((today_red + today_counts.get("YELLOW", 0)) /
                                 max(today_total, 1), 4),
        "total_cases_yesterday": yesterday_total,
        "red_cases_yesterday": yesterday_red,
        "active_cells_yesterday": len(cells_yesterday),
        "escalation_rate_yesterday": round((yesterday_red + yesterday_counts.get("YELLOW", 0)) /
                                           max(yesterday_total, 1), 4),
    }

    # --- 7. Emit JS module ---
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = f"""\
/**
 * REAL eval-derived dashboard data (auto-generated — DO NOT hand-edit).
 *
 * Source: {src.name}  (production routed Task 6 held-out predictions)
 * Generator: dashboard/scripts/generate_live_data.py
 * Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}
 *
 * Aggregation pipeline:
 *   1. Read the n={len(preds)} routed Task 6 predictions
 *   2. Synthesize Tamil Nadu geohash + 7-day temporal distribution
 *   3. Aggregate per (geohash, level) and per (date, level)
 *   4. Compute alerts on cells with red_count >= 2 in last 24h
 *   5. Compute today vs yesterday stats
 *
 * Privacy: no patient identifiers cross over from the eval predictions —
 * only aggregate counts. Geohashes are synthesized at ~1km cell precision
 * matching the production schema.
 */

/** @returns {{HeatmapCell[]}} */
export function getMockHeatmapData() {{
  return {json.dumps(heatmap, ensure_ascii=False, indent=2)};
}}

/** @returns {{TrendDay[]}} */
export function getMockTrendData() {{
  return {json.dumps(trend, ensure_ascii=False, indent=2)};
}}

/** @returns {{Alert[]}} */
export function getMockAlerts() {{
  return {json.dumps(alerts, ensure_ascii=False, indent=2)};
}}

/** @returns {{Stats}} */
export function getMockStats() {{
  return {json.dumps(stats, ensure_ascii=False, indent=2)};
}}
"""
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(heatmap)} heatmap cells")
    print(f"  {len(trend)} trend days")
    print(f"  {len(alerts)} alerts")
    print(f"  stats: today={stats['total_cases_today']} (RED={stats['red_cases_today']}) "
          f"yesterday={stats['total_cases_yesterday']} (RED={stats['red_cases_yesterday']})")


if __name__ == "__main__":
    main()
