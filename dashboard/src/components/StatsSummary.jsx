import PropTypes from 'prop-types';
import { TA } from '../i18n/ta.js';
import './StatsSummary.css';

const COLORS = {
  GREEN:  '#2ecc71',
  YELLOW: '#f39c12',
  RED:    '#e74c3c',
  NEUTRAL:'#a0aec0',
};

/**
 * Formats a decimal fraction as a percentage string (e.g. 0.169 → "16.9%").
 * @param {number} value
 * @returns {string}
 */
function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Computes the delta between today's and yesterday's value.
 * @param {number} today
 * @param {number} yesterday
 * @returns {{ symbol: string, label: string, positive: boolean }}
 */
function computeDelta(today, yesterday) {
  if (yesterday === 0) return { symbol: '→', label: '—', positive: null };
  const diff = today - yesterday;
  const pct  = Math.abs((diff / yesterday) * 100).toFixed(1);
  return {
    symbol:   diff > 0 ? '↑' : diff < 0 ? '↓' : '→',
    label:    `${diff > 0 ? '+' : ''}${diff} (${pct}%)`,
    positive: diff > 0,
  };
}

/**
 * @typedef {Object} StatCardProps
 * @property {string}      labelTamil   - Tamil label text
 * @property {string}      labelEnglish - English label text
 * @property {string}      value        - Formatted primary value
 * @property {string}      [subValue]   - Optional secondary value (e.g. "16.9% of total")
 * @property {string}      deltaSymbol  - Trend arrow symbol
 * @property {string}      deltaLabel   - Numeric delta string
 * @property {boolean|null} deltaPositive - null = neutral, true = up, false = down
 * @property {string}      accentColor  - CSS color for accent bar
 */

/**
 * Individual stat card.
 * @param {StatCardProps} props
 */
function StatCard({ labelTamil, labelEnglish, value, subValue, deltaSymbol, deltaLabel, deltaPositive, accentColor }) {
  const deltaClass =
    deltaPositive === null ? 'delta neutral'
    : deltaPositive        ? 'delta up'
    :                        'delta down';

  return (
    <article
      className="stat-card"
      style={{ '--accent': accentColor }}
      aria-label={`${labelEnglish}: ${value}`}
    >
      <div className="stat-card__accent-bar" />
      <p className="stat-card__value">{value}</p>
      {subValue && <p className="stat-card__sub-value">{subValue}</p>}
      <p className="stat-card__label-tamil">{labelTamil}</p>
      <p className="stat-card__label-english">{labelEnglish}</p>
      <p className={deltaClass} aria-label={`${TA.STATS_VS_YESTERDAY_EN}: ${deltaLabel}`}>
        <span className="delta__symbol" aria-hidden="true">{deltaSymbol}</span>
        {' '}{deltaLabel} {TA.STATS_VS_YESTERDAY_EN}
      </p>
    </article>
  );
}

StatCard.propTypes = {
  labelTamil:    PropTypes.string.isRequired,
  labelEnglish:  PropTypes.string.isRequired,
  value:         PropTypes.string.isRequired,
  subValue:      PropTypes.string,
  deltaSymbol:   PropTypes.string.isRequired,
  deltaLabel:    PropTypes.string.isRequired,
  deltaPositive: PropTypes.bool,
  accentColor:   PropTypes.string.isRequired,
};

StatCard.defaultProps = {
  subValue:      undefined,
  deltaPositive: null,
};

/**
 * @typedef {Object} StatsSummaryProps
 * @property {Object} stats
 * @property {number} stats.total_cases_today
 * @property {number} stats.red_cases_today
 * @property {number} stats.active_cells
 * @property {number} stats.escalation_rate
 * @property {number} stats.total_cases_yesterday
 * @property {number} stats.red_cases_yesterday
 * @property {number} stats.active_cells_yesterday
 * @property {number} stats.escalation_rate_yesterday
 */

/**
 * Four-card summary strip at the top of the Overview page.
 * @param {StatsSummaryProps} props
 */
export default function StatsSummary({ stats }) {
  const totalDelta      = computeDelta(stats.total_cases_today,   stats.total_cases_yesterday);
  const redDelta        = computeDelta(stats.red_cases_today,     stats.red_cases_yesterday);
  const cellsDelta      = computeDelta(stats.active_cells,        stats.active_cells_yesterday);
  const escalationDelta = computeDelta(
    Math.round(stats.escalation_rate * 1000),
    Math.round(stats.escalation_rate_yesterday * 1000),
  );

  const redPercent = stats.total_cases_today > 0
    ? formatPercent(stats.red_cases_today / stats.total_cases_today)
    : '0%';

  return (
    <section className="stats-summary" aria-label="Summary statistics">
      <StatCard
        labelTamil={TA.STATS_TOTAL_CASES}
        labelEnglish={TA.STATS_TOTAL_CASES_EN}
        value={String(stats.total_cases_today)}
        deltaSymbol={totalDelta.symbol}
        deltaLabel={totalDelta.label}
        deltaPositive={totalDelta.positive}
        accentColor={COLORS.GREEN}
      />
      <StatCard
        labelTamil={TA.STATS_RED_CASES}
        labelEnglish={TA.STATS_RED_CASES_EN}
        value={String(stats.red_cases_today)}
        subValue={`${redPercent} of total`}
        deltaSymbol={redDelta.symbol}
        deltaLabel={redDelta.label}
        // More RED cases is bad (positive delta is negative outcome)
        deltaPositive={redDelta.positive === null ? null : !redDelta.positive}
        accentColor={COLORS.RED}
      />
      <StatCard
        labelTamil={TA.STATS_ACTIVE_CELLS}
        labelEnglish={TA.STATS_ACTIVE_CELLS_EN}
        value={String(stats.active_cells)}
        deltaSymbol={cellsDelta.symbol}
        deltaLabel={cellsDelta.label}
        deltaPositive={cellsDelta.positive}
        accentColor={COLORS.NEUTRAL}
      />
      <StatCard
        labelTamil={TA.STATS_ESCALATION_RATE}
        labelEnglish={TA.STATS_ESCALATION_RATE_EN}
        value={formatPercent(stats.escalation_rate)}
        deltaSymbol={escalationDelta.symbol}
        deltaLabel={escalationDelta.label}
        // Higher escalation rate is bad
        deltaPositive={escalationDelta.positive === null ? null : !escalationDelta.positive}
        accentColor={COLORS.YELLOW}
      />
    </section>
  );
}

StatsSummary.propTypes = {
  stats: PropTypes.shape({
    total_cases_today:        PropTypes.number.isRequired,
    red_cases_today:          PropTypes.number.isRequired,
    active_cells:             PropTypes.number.isRequired,
    escalation_rate:          PropTypes.number.isRequired,
    total_cases_yesterday:    PropTypes.number.isRequired,
    red_cases_yesterday:      PropTypes.number.isRequired,
    active_cells_yesterday:   PropTypes.number.isRequired,
    escalation_rate_yesterday: PropTypes.number.isRequired,
  }).isRequired,
};
