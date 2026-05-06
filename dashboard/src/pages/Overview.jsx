import { useMemo } from 'react';
import StatsSummary    from '../components/StatsSummary.jsx';
import AlertPanel      from '../components/AlertPanel.jsx';
import { getMockStats, getMockAlerts } from '../api/mockData.js';
import { TA } from '../i18n/ta.js';
import './PageLayout.css';

/**
 * Overview page — summary cards + top RED alerts.
 * Shows the district health officer a quick situation report.
 */
export default function Overview() {
  // In production these would come from an API call (axios / SWR / React Query).
  // For now we use mock data deterministically so the page renders offline.
  const stats  = useMemo(() => getMockStats(),  []);
  const alerts = useMemo(() => getMockAlerts(), []);

  return (
    <main className="page" aria-labelledby="overview-heading">
      <header className="page__header">
        <h1 id="overview-heading" className="page__title" lang="ta">
          {TA.PAGE_OVERVIEW_TITLE}
          <span className="page__title-en"> · {TA.ROLE_DISTRICT_EN}</span>
        </h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <StatsSummary stats={stats} />

      <section className="page__section" aria-label="Active RED alerts">
        <AlertPanel alerts={alerts} />
      </section>
    </main>
  );
}
