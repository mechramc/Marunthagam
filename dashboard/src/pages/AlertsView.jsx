import { useMemo } from 'react';
import AlertPanel from '../components/AlertPanel.jsx';
import { getMockAlerts } from '../api/mockData.js';
import { TA } from '../i18n/ta.js';
import './PageLayout.css';

/**
 * Alerts view page — full list of active RED cluster alerts.
 */
export default function AlertsView() {
  const alerts = useMemo(() => getMockAlerts(), []);

  return (
    <main className="page" aria-labelledby="alerts-heading">
      <header className="page__header">
        <h1 id="alerts-heading" className="page__title">
          {TA.PAGE_ALERTS_TITLE}
        </h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <section className="page__section" aria-label="Full RED alert list">
        <AlertPanel alerts={alerts} />
      </section>

      <p className="page__disclaimer" role="note">
        {TA.DISCLAIMER}
      </p>
    </main>
  );
}
