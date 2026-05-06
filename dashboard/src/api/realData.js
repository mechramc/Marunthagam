/**
 * REAL eval-derived dashboard data (auto-generated — DO NOT hand-edit).
 *
 * Source: run_task6_routed_20260506_184851.json  (production routed Task 6 held-out predictions)
 * Generator: dashboard/scripts/generate_live_data.py
 * Generated: 2026-05-06T20:35:57+00:00
 *
 * Aggregation pipeline:
 *   1. Read the n=131 routed Task 6 predictions
 *   2. Synthesize Tamil Nadu geohash + 7-day temporal distribution
 *   3. Aggregate per (geohash, level) and per (date, level)
 *   4. Compute alerts on cells with red_count >= 2 in last 24h
 *   5. Compute today vs yesterday stats
 *
 * Privacy: no patient identifiers cross over from the eval predictions —
 * only aggregate counts. Geohashes are synthesized at ~1km cell precision
 * matching the production schema.
 */

/** @returns {HeatmapCell[]} */
export function getMockHeatmapData() {
  return [
  {
    "geohash": "tf8n00",
    "green_count": 3,
    "yellow_count": 7,
    "red_count": 1,
    "last_updated": "2026-05-06T17:39:00+05:30"
  },
  {
    "geohash": "tf8n01",
    "green_count": 3,
    "yellow_count": 4,
    "red_count": 3,
    "last_updated": "2026-05-06T08:06:00+05:30"
  },
  {
    "geohash": "tf8n02",
    "green_count": 0,
    "yellow_count": 7,
    "red_count": 3,
    "last_updated": "2026-05-07T18:08:00+05:30"
  },
  {
    "geohash": "tf8n03",
    "green_count": 3,
    "yellow_count": 6,
    "red_count": 4,
    "last_updated": "2026-05-06T18:13:00+05:30"
  },
  {
    "geohash": "tf8n04",
    "green_count": 0,
    "yellow_count": 7,
    "red_count": 0,
    "last_updated": "2026-05-06T17:30:00+05:30"
  },
  {
    "geohash": "tf6q9d",
    "green_count": 1,
    "yellow_count": 5,
    "red_count": 1,
    "last_updated": "2026-05-06T16:49:00+05:30"
  },
  {
    "geohash": "tf6q9e",
    "green_count": 2,
    "yellow_count": 3,
    "red_count": 0,
    "last_updated": "2026-05-06T11:44:00+05:30"
  },
  {
    "geohash": "tf6q9f",
    "green_count": 1,
    "yellow_count": 2,
    "red_count": 0,
    "last_updated": "2026-05-05T16:04:00+05:30"
  },
  {
    "geohash": "tf6nzx",
    "green_count": 1,
    "yellow_count": 11,
    "red_count": 0,
    "last_updated": "2026-05-07T17:25:00+05:30"
  },
  {
    "geohash": "tf6nzz",
    "green_count": 2,
    "yellow_count": 4,
    "red_count": 0,
    "last_updated": "2026-05-06T10:24:00+05:30"
  },
  {
    "geohash": "tf7p2s",
    "green_count": 3,
    "yellow_count": 6,
    "red_count": 0,
    "last_updated": "2026-05-07T17:01:00+05:30"
  },
  {
    "geohash": "tf7p2t",
    "green_count": 1,
    "yellow_count": 2,
    "red_count": 1,
    "last_updated": "2026-05-06T11:38:00+05:30"
  },
  {
    "geohash": "tf7nr0",
    "green_count": 1,
    "yellow_count": 3,
    "red_count": 0,
    "last_updated": "2026-05-06T11:12:00+05:30"
  },
  {
    "geohash": "tf7nr1",
    "green_count": 1,
    "yellow_count": 4,
    "red_count": 1,
    "last_updated": "2026-05-07T10:58:00+05:30"
  },
  {
    "geohash": "tf6jzm",
    "green_count": 1,
    "yellow_count": 1,
    "red_count": 0,
    "last_updated": "2026-05-05T18:29:00+05:30"
  },
  {
    "geohash": "tf6jzn",
    "green_count": 1,
    "yellow_count": 2,
    "red_count": 1,
    "last_updated": "2026-05-06T17:08:00+05:30"
  },
  {
    "geohash": "tf7mu6",
    "green_count": 2,
    "yellow_count": 2,
    "red_count": 0,
    "last_updated": "2026-05-06T09:02:00+05:30"
  },
  {
    "geohash": "tf7mu7",
    "green_count": 0,
    "yellow_count": 2,
    "red_count": 0,
    "last_updated": "2026-05-02T18:07:00+05:30"
  },
  {
    "geohash": "tf8mfu",
    "green_count": 0,
    "yellow_count": 6,
    "red_count": 0,
    "last_updated": "2026-05-06T07:49:00+05:30"
  },
  {
    "geohash": "tf8mfv",
    "green_count": 2,
    "yellow_count": 4,
    "red_count": 0,
    "last_updated": "2026-05-07T18:51:00+05:30"
  }
];
}

/** @returns {TrendDay[]} */
export function getMockTrendData() {
  return [
  {
    "date": "2026-05-01",
    "green": 2,
    "yellow": 12,
    "red": 4
  },
  {
    "date": "2026-05-02",
    "green": 5,
    "yellow": 19,
    "red": 1
  },
  {
    "date": "2026-05-03",
    "green": 1,
    "yellow": 14,
    "red": 3
  },
  {
    "date": "2026-05-04",
    "green": 3,
    "yellow": 10,
    "red": 2
  },
  {
    "date": "2026-05-05",
    "green": 7,
    "yellow": 12,
    "red": 2
  },
  {
    "date": "2026-05-06",
    "green": 5,
    "yellow": 17,
    "red": 1
  },
  {
    "date": "2026-05-07",
    "green": 5,
    "yellow": 4,
    "red": 2
  }
];
}

/** @returns {Alert[]} */
export function getMockAlerts() {
  return [];
}

/** @returns {Stats} */
export function getMockStats() {
  return {
  "total_cases_today": 11,
  "red_cases_today": 2,
  "active_cells": 5,
  "escalation_rate": 0.5455,
  "total_cases_yesterday": 23,
  "red_cases_yesterday": 1,
  "active_cells_yesterday": 14,
  "escalation_rate_yesterday": 0.7826
};
}
