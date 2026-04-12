import { useMemo } from 'react';
import TriageHeatmap from '../components/TriageHeatmap.jsx';
import { getMockHeatmapData } from '../api/mockData.js';
import { TA } from '../i18n/ta.js';
import './PageLayout.css';

/**
 * Map view page — full geohash heatmap for the district.
 */
export default function MapView() {
  const data = useMemo(() => getMockHeatmapData(), []);

  return (
    <main className="page" aria-labelledby="map-heading">
      <header className="page__header">
        <h1 id="map-heading" className="page__title">
          {TA.PAGE_MAP_TITLE}
        </h1>
        <span className="page__updated">
          {TA.DATA_UPDATED_EN}: {new Date().toLocaleTimeString('en-IN')}
        </span>
      </header>

      <section className="page__section" aria-label="Geohash triage heatmap">
        <TriageHeatmap data={data} />
      </section>
    </main>
  );
}
