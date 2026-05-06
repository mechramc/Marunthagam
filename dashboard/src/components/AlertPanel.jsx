import PropTypes from 'prop-types';
import { TA } from '../i18n/ta.js';
import { getMockAlerts } from '../api/mockData.js';
import './AlertPanel.css';

const CRITICAL_THRESHOLD = 5; // ≥5 RED cases in 24h → critical badge

/**
 * Formats an ISO timestamp as a relative "time ago" string.
 * @param {string} isoString
 * @returns {string}
 */
function timeAgo(isoString) {
  const diffMs  = Date.now() - new Date(isoString).getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 60)  return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr  < 24)  return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

/**
 * Maps trend value to the appropriate Tamil/symbol label.
 * @param {'up'|'down'|'stable'} trend
 * @returns {string}
 */
function trendLabel(trend) {
  switch (trend) {
    case 'up':     return TA.ALERT_TREND_UP;
    case 'down':   return TA.ALERT_TREND_DOWN;
    case 'stable': return TA.ALERT_TREND_STABLE;
    default:       return '→';
  }
}

/**
 * CSS class for trend badge.
 * @param {'up'|'down'|'stable'} trend
 * @returns {string}
 */
function trendClass(trend) {
  switch (trend) {
    case 'up':     return 'trend-badge trend-badge--up';
    case 'down':   return 'trend-badge trend-badge--down';
    default:       return 'trend-badge trend-badge--stable';
  }
}

/**
 * @typedef {Object} AlertPanelProps
 * @property {Array<{
 *   id: string,
 *   geohash: string,
 *   red_count: number,
 *   trend: 'up'|'down'|'stable',
 *   first_seen: string,
 *   last_seen: string
 * }>} alerts
 */

/**
 * Displays active RED-level cluster alerts sorted by severity.
 * @param {AlertPanelProps} props
 */
export default function AlertPanel({ alerts }) {
  const sorted = [...alerts].sort((a, b) => b.red_count - a.red_count);

  return (
    <section className="alert-panel" aria-label={TA.ALERT_TITLE_EN}>
      <header className="alert-panel__header">
        <h2 className="alert-panel__title">{TA.ALERT_TITLE}</h2>
        <span className="alert-panel__title-en">{TA.ALERT_TITLE_EN}</span>
      </header>

      {sorted.length === 0 ? (
        <div className="alert-panel__empty" role="status">
          <p className="alert-panel__empty-tamil">{TA.ALERT_EMPTY}</p>
          <p className="alert-panel__empty-english">{TA.ALERT_EMPTY_EN}</p>
        </div>
      ) : (
        <ul className="alert-panel__list" role="list">
          {sorted.map((alert) => {
            const isCritical = alert.red_count >= CRITICAL_THRESHOLD;
            return (
              <li
                key={alert.id}
                className={`alert-item${isCritical ? ' alert-item--critical' : ''}`}
                aria-label={`Alert for geohash ${alert.geohash}: ${alert.red_count} RED cases`}
              >
                <div className="alert-item__left">
                  <span className="alert-item__geohash" aria-label={`Geohash ${alert.geohash}`}>
                    {alert.geohash}
                  </span>
                  {isCritical && (
                    <span className="critical-badge" role="status" aria-label={TA.ALERT_CRITICAL_BADGE_EN}>
                      {TA.ALERT_CRITICAL_BADGE}
                      <span className="critical-badge__en"> {TA.ALERT_CRITICAL_BADGE_EN}</span>
                    </span>
                  )}
                </div>

                <div className="alert-item__center">
                  <span className="alert-item__red-count" aria-label={`${alert.red_count} ${TA.ALERT_RED_COUNT_EN}`}>
                    <span className="alert-item__red-number">{alert.red_count}</span>
                    {' '}RED · 48h
                  </span>
                  {typeof alert.yellow_count === 'number' && alert.yellow_count > 0 && (
                    <span className="alert-item__yellow-count">
                      <span className="alert-item__yellow-number">{alert.yellow_count}</span>
                      {' '}YELLOW · 24h
                    </span>
                  )}
                  <span className={trendClass(alert.trend)} aria-label={`Trend: ${alert.trend}`}>
                    {trendLabel(alert.trend)}
                  </span>
                </div>

                <div className="alert-item__right">
                  <span className="alert-item__timing">
                    <span className="alert-item__timing-label">{TA.ALERT_FIRST_SEEN_EN}:</span>
                    {' '}{timeAgo(alert.first_seen)}
                  </span>
                  <span className="alert-item__timing">
                    <span className="alert-item__timing-label">{TA.ALERT_LATEST_EN}:</span>
                    {' '}{timeAgo(alert.last_seen)}
                  </span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

AlertPanel.propTypes = {
  alerts: PropTypes.arrayOf(
    PropTypes.shape({
      id:         PropTypes.string.isRequired,
      geohash:    PropTypes.string.isRequired,
      red_count:  PropTypes.number.isRequired,
      trend:      PropTypes.oneOf(['up', 'down', 'stable']).isRequired,
      first_seen: PropTypes.string.isRequired,
      last_seen:  PropTypes.string.isRequired,
    })
  ).isRequired,
};

// Named export for dev-mode standalone preview (exports the factory function)
export { getMockAlerts as getMockAlertsForPreview };
