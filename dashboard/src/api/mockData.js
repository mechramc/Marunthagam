/**
 * Central mock data for Marunthagam District Health Dashboard.
 *
 * Privacy guarantee: NO patient-identifiable data here.
 * All records are aggregated counts per geohash cell (~1km resolution).
 * Tamil Nadu geohash prefixes: tf7, tf8.
 */

/**
 * @typedef {Object} HeatmapCell
 * @property {string} geohash     - Geohash string (~1km cell)
 * @property {number} green_count  - GREEN triage cases
 * @property {number} yellow_count - YELLOW triage cases
 * @property {number} red_count    - RED triage cases
 * @property {string} last_updated - ISO 8601 timestamp
 */

/**
 * @typedef {Object} TrendDay
 * @property {string} date   - ISO date string (YYYY-MM-DD)
 * @property {number} green  - GREEN cases that day
 * @property {number} yellow - YELLOW cases that day
 * @property {number} red    - RED cases that day
 */

/**
 * @typedef {Object} Alert
 * @property {string} id         - Unique alert ID
 * @property {string} geohash    - Geohash of cluster
 * @property {number} red_count  - RED cases in last 24h
 * @property {'up'|'down'|'stable'} trend - Case trend direction
 * @property {string} first_seen - ISO timestamp of first RED case
 * @property {string} last_seen  - ISO timestamp of most recent RED case
 */

/**
 * @typedef {Object} Stats
 * @property {number} total_cases_today   - All triage cases today
 * @property {number} red_cases_today     - RED cases today
 * @property {number} active_cells        - Geohash cells with activity
 * @property {number} escalation_rate     - Fraction escalated (0–1)
 * @property {number} total_cases_yesterday
 * @property {number} red_cases_yesterday
 * @property {number} active_cells_yesterday
 * @property {number} escalation_rate_yesterday
 */

const BASE_DATE = new Date('2026-04-11T08:00:00+05:30');

/** @returns {HeatmapCell[]} */
export function getMockHeatmapData() {
  return [
    { geohash: 'tf7npb', green_count: 18, yellow_count: 5,  red_count: 1,  last_updated: '2026-04-11T07:45:00+05:30' },
    { geohash: 'tf7npc', green_count: 12, yellow_count: 3,  red_count: 0,  last_updated: '2026-04-11T07:30:00+05:30' },
    { geohash: 'tf7npf', green_count: 8,  yellow_count: 7,  red_count: 3,  last_updated: '2026-04-11T07:50:00+05:30' },
    { geohash: 'tf7npg', green_count: 22, yellow_count: 2,  red_count: 0,  last_updated: '2026-04-11T06:55:00+05:30' },
    { geohash: 'tf7npu', green_count: 5,  yellow_count: 9,  red_count: 6,  last_updated: '2026-04-11T07:10:00+05:30' },
    { geohash: 'tf7npv', green_count: 30, yellow_count: 4,  red_count: 2,  last_updated: '2026-04-11T07:40:00+05:30' },
    { geohash: 'tf7npy', green_count: 14, yellow_count: 6,  red_count: 1,  last_updated: '2026-04-11T07:20:00+05:30' },
    { geohash: 'tf7npz', green_count: 9,  yellow_count: 11, red_count: 4,  last_updated: '2026-04-11T07:15:00+05:30' },
    { geohash: 'tf8n00', green_count: 25, yellow_count: 3,  red_count: 0,  last_updated: '2026-04-11T06:40:00+05:30' },
    { geohash: 'tf8n01', green_count: 17, yellow_count: 8,  red_count: 2,  last_updated: '2026-04-11T07:35:00+05:30' },
    { geohash: 'tf8n02', green_count: 11, yellow_count: 4,  red_count: 5,  last_updated: '2026-04-11T07:55:00+05:30' },
    { geohash: 'tf8n03', green_count: 6,  yellow_count: 2,  red_count: 0,  last_updated: '2026-04-11T06:30:00+05:30' },
    { geohash: 'tf8n04', green_count: 20, yellow_count: 5,  red_count: 1,  last_updated: '2026-04-11T07:00:00+05:30' },
    { geohash: 'tf8n05', green_count: 13, yellow_count: 10, red_count: 7,  last_updated: '2026-04-11T07:52:00+05:30' },
    { geohash: 'tf8n06', green_count: 3,  yellow_count: 1,  red_count: 0,  last_updated: '2026-04-11T05:50:00+05:30' },
    { geohash: 'tf8n07', green_count: 28, yellow_count: 6,  red_count: 3,  last_updated: '2026-04-11T07:45:00+05:30' },
    { geohash: 'tf8n08', green_count: 15, yellow_count: 9,  red_count: 2,  last_updated: '2026-04-11T07:25:00+05:30' },
    { geohash: 'tf8n09', green_count: 10, yellow_count: 3,  red_count: 0,  last_updated: '2026-04-11T06:15:00+05:30' },
    { geohash: 'tf8n0b', green_count: 7,  yellow_count: 12, red_count: 8,  last_updated: '2026-04-11T07:58:00+05:30' },
    { geohash: 'tf8n0c', green_count: 19, yellow_count: 4,  red_count: 1,  last_updated: '2026-04-11T07:05:00+05:30' },
  ];
}

/** @returns {TrendDay[]} */
export function getMockTrendData() {
  return [
    { date: '2026-04-05', green: 145, yellow: 42, red: 8  },
    { date: '2026-04-06', green: 132, yellow: 38, red: 11 },
    { date: '2026-04-07', green: 158, yellow: 51, red: 14 },
    { date: '2026-04-08', green: 171, yellow: 47, red: 9  },
    { date: '2026-04-09', green: 163, yellow: 55, red: 17 },
    { date: '2026-04-10', green: 148, yellow: 49, red: 12 },
    { date: '2026-04-11', green: 107, yellow: 60, red: 34 }, // today (partial)
  ];
}

/** @returns {Alert[]} */
export function getMockAlerts() {
  return [
    {
      id:         'alert-001',
      geohash:    'tf8n0b',
      red_count:  8,
      trend:      'up',
      first_seen: '2026-04-11T04:30:00+05:30',
      last_seen:  '2026-04-11T07:58:00+05:30',
    },
    {
      id:         'alert-002',
      geohash:    'tf8n05',
      red_count:  7,
      trend:      'stable',
      first_seen: '2026-04-10T21:15:00+05:30',
      last_seen:  '2026-04-11T07:52:00+05:30',
    },
    {
      id:         'alert-003',
      geohash:    'tf7npu',
      red_count:  6,
      trend:      'down',
      first_seen: '2026-04-10T18:00:00+05:30',
      last_seen:  '2026-04-11T07:10:00+05:30',
    },
  ];
}

/** @returns {Stats} */
export function getMockStats() {
  return {
    total_cases_today:        201,
    red_cases_today:           34,
    active_cells:              20,
    escalation_rate:         0.169, // 34/201

    total_cases_yesterday:    209,
    red_cases_yesterday:       12,
    active_cells_yesterday:    18,
    escalation_rate_yesterday: 0.057,
  };
}
