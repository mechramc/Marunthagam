import { useMemo } from 'react';
import { getMockCases, getMockHeatmapData } from '../../api/mockData.js';
import { TA } from '../../i18n/ta.js';
import '../PageLayout.css';
import './Clinic.css';

/**
 * Clinic catchment area — geohash cells served by the PHC's ASHA workers,
 * with per-cell case counts and last-seen timestamps. Doctor-side equivalent
 * of the district heatmap: scoped to the cells this PHC is responsible for.
 */
export default function Catchment() {
  const cases = useMemo(() => getMockCases(), []);
  const cells = useMemo(() => getMockHeatmapData(), []);

  // Build per-cell summary: case counts + last seen + assigned ASHA
  const rows = useMemo(() => {
    const byCell = new Map();
    for (const c of cases) {
      if (!byCell.has(c.geohash)) {
        byCell.set(c.geohash, {
          cell: c.geohash,
          asha: c.asha_worker,
          last: c.timestamp,
          green: 0, yellow: 0, red: 0,
        });
      }
      const r = byCell.get(c.geohash);
      r[c.level.toLowerCase()] += 1;
      if (c.timestamp > r.last) r.last = c.timestamp;
    }
    // Add cells from heatmap that have no cases (still in catchment)
    for (const cell of cells) {
      if (!byCell.has(cell.geohash)) {
        byCell.set(cell.geohash, {
          cell: cell.geohash,
          asha: '—',
          last: cell.last_updated,
          green: cell.green_count, yellow: cell.yellow_count, red: cell.red_count,
        });
      }
    }
    const out = Array.from(byCell.values());
    // Sort by RED desc, then total desc
    out.sort((a, b) => {
      if (b.red !== a.red) return b.red - a.red;
      return (b.green + b.yellow + b.red) - (a.green + a.yellow + a.red);
    });
    return out;
  }, [cases, cells]);

  return (
    <main className="page" aria-labelledby="catchment-heading">
      <header className="page__header">
        <h1 id="catchment-heading" className="page__title">{TA.CLINIC_CATCHMENT_TITLE_EN}</h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <p className="clinic__page-desc">{TA.CLINIC_CATCHMENT_DESC_EN}</p>

      <div className="clinic__queue-table-wrap">
        <table className="clinic__queue-table" aria-label="Catchment cells">
          <thead>
            <tr>
              <th>{TA.CLINIC_CATCHMENT_COL_CELL_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_ASHA_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_CASES_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_RED_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_YELLOW_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_GREEN_EN}</th>
              <th>{TA.CLINIC_CATCHMENT_COL_LAST_EN}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => {
              const total = r.red + r.yellow + r.green;
              return (
                <tr key={r.cell} className="clinic__queue-row">
                  <td><code className="clinic__geohash-mono">{r.cell}</code></td>
                  <td><span className="clinic__asha-name">{r.asha}</span></td>
                  <td><strong>{total}</strong></td>
                  <td>
                    {r.red > 0 ? (
                      <span className="clinic__level-pill clinic__level-pill--red">{r.red}</span>
                    ) : <span className="clinic__no-override">0</span>}
                  </td>
                  <td>
                    {r.yellow > 0 ? (
                      <span className="clinic__level-pill clinic__level-pill--yellow">{r.yellow}</span>
                    ) : <span className="clinic__no-override">0</span>}
                  </td>
                  <td>
                    {r.green > 0 ? (
                      <span className="clinic__level-pill clinic__level-pill--green">{r.green}</span>
                    ) : <span className="clinic__no-override">0</span>}
                  </td>
                  <td className="clinic__queue-cell-time">
                    {r.last ? new Date(r.last).toLocaleString('en-IN', {
                      hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short',
                    }) : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="page__disclaimer" lang="ta">{TA.DISCLAIMER}</p>
    </main>
  );
}
