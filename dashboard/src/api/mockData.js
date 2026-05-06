/**
 * mockData.js — backward-compatible API shim.
 *
 * The dashboard's components import from `mockData.js` for historical reasons
 * (the early dev iteration shipped synthetic data only). As of Sprint 3, the
 * dashboard renders REAL data derived from production Task 6 held-out
 * predictions, aggregated through dashboard/scripts/generate_live_data.py.
 *
 * To regenerate the real data after a new eval run:
 *     python dashboard/scripts/generate_live_data.py
 *
 * To switch back to synthetic-only data (for testing or offline development),
 * change the import below to `./syntheticData.js`.
 */

export {
  getMockHeatmapData,
  getMockTrendData,
  getMockAlerts,
  getMockStats,
  getMockCases,
} from './realData.js';
