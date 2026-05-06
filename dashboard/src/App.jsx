import { NavLink, Route, Routes, Navigate, useLocation, useNavigate } from 'react-router-dom';
import Overview     from './pages/Overview.jsx';
import MapView      from './pages/MapView.jsx';
import AlertsView   from './pages/AlertsView.jsx';
import TrendsView   from './pages/TrendsView.jsx';
import CaseQueue    from './pages/clinic/CaseQueue.jsx';
import CaseDetail   from './pages/clinic/CaseDetail.jsx';
import Catchment    from './pages/clinic/Catchment.jsx';
import { TA } from './i18n/ta.js';
import './App.css';

const DISTRICT_NAV = [
  { to: '/district',          label: TA.NAV_OVERVIEW, labelEn: 'Overview', icon: '⬛' },
  { to: '/district/map',      label: TA.NAV_MAP,      labelEn: 'Map',      icon: '🗺' },
  { to: '/district/alerts',   label: TA.NAV_ALERTS,   labelEn: 'Alerts',   icon: '🔴' },
  { to: '/district/trends',   label: TA.NAV_TRENDS,   labelEn: 'Trends',   icon: '📈' },
];

const CLINIC_NAV = [
  { to: '/clinic',           label: TA.NAV_CLINIC_QUEUE,      labelEn: TA.NAV_CLINIC_QUEUE_EN,      icon: '📋' },
  { to: '/clinic/catchment', label: TA.NAV_CLINIC_CATCHMENT,  labelEn: TA.NAV_CLINIC_CATCHMENT_EN,  icon: '🏘' },
];

/**
 * Root application component. Two top-level roles:
 *   - District health office (Tier 3) — overview / map / alerts / trends
 *   - PHC clinic doctor (Tier 2) — case queue + case detail + catchment
 *
 * Role is derived from the URL prefix (`/district/*` vs `/clinic/*`).
 */
export default function App() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const isClinic = pathname.startsWith('/clinic');
  const role = isClinic ? 'clinic' : 'district';
  const navItems  = isClinic ? CLINIC_NAV : DISTRICT_NAV;
  const brandSub  = isClinic ? TA.CLINIC_CONSOLE_TITLE : TA.DASHBOARD_TITLE;

  const switchRole = () => {
    navigate(isClinic ? '/district' : '/clinic');
  };

  return (
    <div className="app-shell" data-role={role}>
      {/* ── Sidebar ── */}
      <nav className="sidebar" aria-label="Main navigation">
        <header className="sidebar__brand">
          <span className="sidebar__brand-tamil" lang="ta">{TA.APP_TITLE}</span>
          <span className="sidebar__brand-subtitle" lang="ta">{brandSub}</span>
        </header>

        {/* Role switcher */}
        <div className="sidebar__role-switcher" role="radiogroup" aria-label="Role">
          <button
            type="button"
            className={`sidebar__role-btn${!isClinic ? ' sidebar__role-btn--active' : ''}`}
            onClick={() => navigate('/district')}
            aria-pressed={!isClinic}
          >
            <span className="sidebar__role-tamil" lang="ta">{TA.ROLE_DISTRICT}</span>
            <span className="sidebar__role-en">{TA.ROLE_DISTRICT_EN}</span>
          </button>
          <button
            type="button"
            className={`sidebar__role-btn${isClinic ? ' sidebar__role-btn--active' : ''}`}
            onClick={() => navigate('/clinic')}
            aria-pressed={isClinic}
          >
            <span className="sidebar__role-tamil" lang="ta">{TA.ROLE_CLINIC}</span>
            <span className="sidebar__role-en">{TA.ROLE_CLINIC_EN}</span>
          </button>
        </div>

        <ul className="sidebar__nav-list" role="list">
          {navItems.map(({ to, label, labelEn, icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/district' || to === '/clinic'}
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
          {/* District (Tier 3) */}
          <Route path="/district"        element={<Overview />}    />
          <Route path="/district/map"    element={<MapView />}     />
          <Route path="/district/alerts" element={<AlertsView />}  />
          <Route path="/district/trends" element={<TrendsView />}  />

          {/* Clinic (Tier 2) */}
          <Route path="/clinic"            element={<CaseQueue />}  />
          <Route path="/clinic/case/:id"   element={<CaseDetail />} />
          <Route path="/clinic/catchment"  element={<Catchment />}  />

          {/* Default + catch-all → district overview */}
          <Route path="/" element={<Navigate to="/district" replace />} />
          <Route path="*" element={<Navigate to="/district" replace />} />
        </Routes>
      </div>
    </div>
  );
}
