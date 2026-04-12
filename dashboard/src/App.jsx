import { NavLink, Route, Routes, Navigate } from 'react-router-dom';
import Overview   from './pages/Overview.jsx';
import MapView    from './pages/MapView.jsx';
import AlertsView from './pages/AlertsView.jsx';
import TrendsView from './pages/TrendsView.jsx';
import { TA } from './i18n/ta.js';
import './App.css';

const NAV_ITEMS = [
  { to: '/',        label: TA.NAV_OVERVIEW, labelEn: 'Overview', icon: '⬛' },
  { to: '/map',     label: TA.NAV_MAP,      labelEn: 'Map',      icon: '🗺' },
  { to: '/alerts',  label: TA.NAV_ALERTS,   labelEn: 'Alerts',   icon: '🔴' },
  { to: '/trends',  label: TA.NAV_TRENDS,   labelEn: 'Trends',   icon: '📈' },
];

/**
 * Root application component.
 * Layout: fixed sidebar nav (left) + scrollable main content (right).
 */
export default function App() {
  return (
    <div className="app-shell">
      {/* ── Sidebar ── */}
      <nav className="sidebar" aria-label="Main navigation">
        <header className="sidebar__brand">
          <span className="sidebar__brand-tamil" lang="ta">{TA.APP_TITLE}</span>
          <span className="sidebar__brand-subtitle" lang="ta">{TA.DASHBOARD_TITLE}</span>
        </header>

        <ul className="sidebar__nav-list" role="list">
          {NAV_ITEMS.map(({ to, label, labelEn, icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `sidebar__nav-item${isActive ? ' sidebar__nav-item--active' : ''}`
                }
                aria-label={labelEn}
              >
                <span className="sidebar__nav-icon" aria-hidden="true">{icon}</span>
                <span className="sidebar__nav-label-tamil" lang="ta">{label}</span>
                <span className="sidebar__nav-label-en">{labelEn}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <footer className="sidebar__footer">
          <p className="sidebar__disclaimer" lang="ta">{TA.DISCLAIMER}</p>
        </footer>
      </nav>

      {/* ── Main content ── */}
      <div className="content-area">
        <Routes>
          <Route path="/"        element={<Overview />}   />
          <Route path="/map"     element={<MapView />}    />
          <Route path="/alerts"  element={<AlertsView />} />
          <Route path="/trends"  element={<TrendsView />} />
          {/* Catch-all: redirect unknown paths to overview */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  );
}
