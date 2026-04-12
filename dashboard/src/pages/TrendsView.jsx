import { useMemo } from 'react';
import TriageTrendChart from '../components/TriageTrendChart.jsx';
import { getMockTrendData } from '../api/mockData.js';
import { TA } from '../i18n/ta.js';
import './PageLayout.css';

/**
 * Trends view page — 7-day triage trend line chart.
 */
export default function TrendsView() {
  const data = useMemo(() => getMockTrendData(), []);

  return (
    <main className="page" aria-labelledby="trends-heading">
      <header className="page__header">
        <h1 id="trends-heading" className="page__title">
          {TA.PAGE_TRENDS_TITLE}
        </h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <section className="page__section" aria-label="7-day triage trend chart">
        <TriageTrendChart data={data} />
      </section>
    </main>
  );
}
