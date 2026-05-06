import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { getMockCases } from '../../api/mockData.js';
import { TA } from '../../i18n/ta.js';
import '../PageLayout.css';
import './Clinic.css';

const LEVEL_RANK = { RED: 0, YELLOW: 1, GREEN: 2 };
const FILTERS = [
  { id: 'all',     label: TA.CLINIC_QUEUE_FILTER_ALL_EN },
  { id: 'red',     label: TA.CLINIC_QUEUE_FILTER_RED_EN },
  { id: 'yellow',  label: TA.CLINIC_QUEUE_FILTER_YELLOW_EN },
  { id: 'today',   label: TA.CLINIC_QUEUE_FILTER_TODAY_EN },
];

function formatRelative(ts) {
  if (!ts) return '—';
  const then = new Date(ts);
  const diffMs = Date.now() - then.getTime();
  const m = Math.round(diffMs / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

export default function CaseQueue() {
  const allCases = useMemo(() => getMockCases(), []);
  const [filter, setFilter] = useState('all');

  const filtered = useMemo(() => {
    let rows = [...allCases];
    if (filter === 'red')    rows = rows.filter(c => c.level === 'RED');
    if (filter === 'yellow') rows = rows.filter(c => c.level === 'YELLOW');
    if (filter === 'today') {
      const today = new Date().toISOString().slice(0, 10);
      rows = rows.filter(c => c.date === today);
    }
    // Sort: RED → YELLOW → GREEN, then most recent first within each
    rows.sort((a, b) => {
      const r = LEVEL_RANK[a.level] - LEVEL_RANK[b.level];
      if (r !== 0) return r;
      return b.timestamp.localeCompare(a.timestamp);
    });
    return rows;
  }, [allCases, filter]);

  const counts = useMemo(() => ({
    total:  allCases.length,
    red:    allCases.filter(c => c.level === 'RED').length,
    yellow: allCases.filter(c => c.level === 'YELLOW').length,
    green:  allCases.filter(c => c.level === 'GREEN').length,
  }), [allCases]);

  return (
    <main className="page" aria-labelledby="queue-heading">
      <header className="page__header">
        <h1 id="queue-heading" className="page__title" lang="ta">
          {TA.CLINIC_QUEUE_TITLE} <span className="page__title-en">· {TA.CLINIC_QUEUE_TITLE_EN}</span>
        </h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <p className="clinic__page-desc">{TA.CLINIC_QUEUE_DESC_EN}</p>

      <div className="clinic__counts">
        <span className="clinic__count clinic__count--red">RED <strong>{counts.red}</strong></span>
        <span className="clinic__count clinic__count--yellow">YELLOW <strong>{counts.yellow}</strong></span>
        <span className="clinic__count clinic__count--green">GREEN <strong>{counts.green}</strong></span>
        <span className="clinic__count clinic__count--total">Total <strong>{counts.total}</strong></span>
      </div>

      <div className="clinic__filters" role="tablist" aria-label="Queue filter">
        {FILTERS.map(f => (
          <button
            key={f.id}
            type="button"
            role="tab"
            aria-selected={filter === f.id}
            className={`clinic__filter${filter === f.id ? ' clinic__filter--active' : ''}`}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="clinic__queue-table-wrap">
        <table className="clinic__queue-table" aria-label="Case queue">
          <thead>
            <tr>
              <th>{TA.CLINIC_QUEUE_COL_RECEIVED_EN}</th>
              <th>{TA.CLINIC_QUEUE_COL_LEVEL_EN}</th>
              <th>{TA.CLINIC_QUEUE_COL_CHIEF_EN}</th>
              <th>{TA.CLINIC_QUEUE_COL_ASHA_EN}</th>
              <th>{TA.CLINIC_QUEUE_COL_CONF_EN}</th>
              <th>{TA.CLINIC_QUEUE_COL_OVERRIDES_EN}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(c => (
              <tr key={c.case_id} className="clinic__queue-row">
                <td className="clinic__queue-cell-received">
                  <span>{formatRelative(c.timestamp)}</span>
                  <span className="clinic__queue-cell-time">
                    {new Date(c.timestamp).toLocaleString('en-IN', {
                      hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short',
                    })}
                  </span>
                </td>
                <td>
                  <span className={`clinic__level-pill clinic__level-pill--${c.level.toLowerCase()}`}>
                    {c.level}
                  </span>
                </td>
                <td className="clinic__queue-cell-chief">
                  <Link to={`/clinic/case/${c.case_id}`} className="clinic__queue-link" lang="ta">
                    {truncate(c.chief_complaint_ta, 90) || <em>(no narrative captured)</em>}
                  </Link>
                </td>
                <td className="clinic__queue-cell-asha">
                  <span className="clinic__asha-name">{c.asha_worker}</span>
                  <span className="clinic__geohash-mono">{c.geohash}</span>
                </td>
                <td>
                  <span className="clinic__conf">
                    {(c.confidence * 100).toFixed(0)}%
                  </span>
                </td>
                <td className="clinic__queue-cell-overrides">
                  {c.engine_overrides && c.engine_overrides.length > 0 ? (
                    <span className="clinic__override-badge">
                      {c.engine_overrides.map(o => o.rule_id || o).join(', ')}
                    </span>
                  ) : (
                    <span className="clinic__no-override">—</span>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="clinic__queue-empty">No cases match this filter.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className="page__disclaimer" lang="ta">{TA.DISCLAIMER}</p>
    </main>
  );
}
