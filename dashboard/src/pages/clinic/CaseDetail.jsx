import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getMockCases } from '../../api/mockData.js';
import { TA } from '../../i18n/ta.js';
import '../PageLayout.css';
import './Clinic.css';

/**
 * Clinic console — single-case detail view for the PHC doctor.
 *
 * Reads the case_id from the URL, joins against the per-case dataset
 * generated from the held-out Task 6 routed predictions, and renders:
 *   - Patient narrative (Tamil chief complaint as it arrived from ASHA)
 *   - ASHA worker triage decision (model output + post-engine final)
 *   - Engine overrides (which IMNCI rules fired)
 *   - Suggested doctor actions
 *
 * Doctor actions are placeholder buttons in the v1 UI — they would write
 * back to the local SQLite log in production. Showing the buttons makes
 * the screenshot speak for itself: this is decision support, the doctor
 * decides.
 */
export default function CaseDetail() {
  const { id } = useParams();
  const allCases = useMemo(() => getMockCases(), []);
  const c = useMemo(() => allCases.find(x => x.case_id === id), [allCases, id]);

  if (!c) {
    return (
      <main className="page" aria-labelledby="case-not-found">
        <header className="page__header">
          <h1 id="case-not-found" className="page__title">Case not found</h1>
          <Link to="/clinic" className="clinic__back-link">{TA.CLINIC_CASE_BACK_EN}</Link>
        </header>
        <p>The case <code>{id}</code> doesn't appear in the queue.</p>
      </main>
    );
  }

  const engineFlipped = c.pre_engine_level !== c.level;

  return (
    <main className="page clinic-case" aria-labelledby="case-heading">
      <header className="page__header">
        <div>
          <h1 id="case-heading" className="page__title" lang="ta">
            {TA.CLINIC_CASE_TITLE_EN}
            <span className="clinic__case-id">{c.case_id}</span>
          </h1>
          <p className="clinic__case-subhead">
            {TA.CLINIC_CASE_RECEIVED_EN} {new Date(c.timestamp).toLocaleString('en-IN')}
            {' '}{TA.CLINIC_CASE_FROM_EN}{' '}<strong>{c.asha_worker}</strong>
            {' · '}{TA.CLINIC_CASE_GEOHASH_EN}: <code>{c.geohash}</code>
            {' · '}{TA.CLINIC_CASE_SPECIALIST_EN}: <strong>{c.specialist}</strong>
          </p>
        </div>
        <Link to="/clinic" className="clinic__back-link">{TA.CLINIC_CASE_BACK_EN}</Link>
      </header>

      {/* Triage banner: post-engine level + confidence */}
      <section className="clinic__case-banner" aria-label="Final triage decision">
        <div className={`clinic__case-level clinic__case-level--${c.level.toLowerCase()}`}>
          <span className="clinic__case-level-label">{TA.CLINIC_CASE_POST_ENGINE_EN}</span>
          <span className="clinic__case-level-value">{c.level}</span>
        </div>
        <div className="clinic__case-banner-meta">
          <div className="clinic__case-meta-row">
            <span className="clinic__case-meta-label">{TA.CLINIC_CASE_CONFIDENCE_EN}</span>
            <span className="clinic__case-meta-value">
              {(c.confidence * 100).toFixed(1)}%
              {c.escalation_flag && <em className="clinic__esc-flag"> · escalation flag</em>}
            </span>
          </div>
          <div className="clinic__case-meta-row">
            <span className="clinic__case-meta-label">{TA.CLINIC_CASE_PRE_ENGINE_EN}</span>
            <span className="clinic__case-meta-value">
              <span className={`clinic__level-pill clinic__level-pill--${c.pre_engine_level.toLowerCase()}`}>
                {c.pre_engine_level}
              </span>
              {' '}
              <span className="clinic__case-meta-sub">
                ({(c.pre_engine_confidence * 100).toFixed(0)}%)
              </span>
              {engineFlipped && (
                <span className="clinic__engine-flipped">
                  {' '}→ engine escalated to <strong>{c.level}</strong>
                </span>
              )}
            </span>
          </div>
        </div>
      </section>

      {/* Patient narrative — Tamil */}
      <section className="clinic__case-section">
        <h2 className="clinic__case-section-title">
          {TA.CLINIC_CASE_PATIENT_NARRATIVE_EN}
          <span className="clinic__case-section-title-ta" lang="ta">
            · {TA.CLINIC_CASE_PATIENT_NARRATIVE_TA}
          </span>
        </h2>
        <blockquote className="clinic__narrative" lang="ta">
          {c.chief_complaint_ta || <em className="clinic__narrative-empty">No narrative captured for this case.</em>}
        </blockquote>
      </section>

      {/* Engine overrides — what IMNCI rules fired */}
      <section className="clinic__case-section">
        <h2 className="clinic__case-section-title">
          {TA.CLINIC_CASE_ENGINE_OVERRIDES_EN}
          <span className="clinic__case-section-title-ta" lang="ta">
            · {TA.CLINIC_CASE_ENGINE_OVERRIDES_TA}
          </span>
        </h2>
        {c.engine_overrides && c.engine_overrides.length > 0 ? (
          <ul className="clinic__overrides-list">
            {c.engine_overrides.map((o, i) => (
              <li key={i} className="clinic__overrides-item">
                <code className="clinic__overrides-rule">{o.rule_id || o}</code>
                {o.reason && <span className="clinic__overrides-reason">{o.reason}</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="clinic__no-overrides">{TA.CLINIC_CASE_NO_OVERRIDES_EN}</p>
        )}
      </section>

      {/* Doctor action buttons — placeholder, the screenshot tells the story */}
      <section className="clinic__case-section clinic__case-actions-section">
        <h2 className="clinic__case-section-title">
          {TA.CLINIC_CASE_DOCTOR_ACTION_EN}
          <span className="clinic__case-section-title-ta" lang="ta">
            · {TA.CLINIC_CASE_DOCTOR_ACTION_TA}
          </span>
        </h2>
        <div className="clinic__actions-grid">
          <button type="button" className="clinic__action clinic__action--confirm">
            <span className="clinic__action-en">{TA.CLINIC_CASE_ACTION_CONFIRM_EN}</span>
            <span className="clinic__action-ta" lang="ta">{TA.CLINIC_CASE_ACTION_CONFIRM_TA}</span>
          </button>
          <button type="button" className="clinic__action clinic__action--downgrade">
            <span className="clinic__action-en">{TA.CLINIC_CASE_ACTION_DOWNGRADE_EN}</span>
            <span className="clinic__action-ta" lang="ta">{TA.CLINIC_CASE_ACTION_DOWNGRADE_TA}</span>
          </button>
          <button type="button" className="clinic__action clinic__action--escalate">
            <span className="clinic__action-en">{TA.CLINIC_CASE_ACTION_ESCALATE_EN}</span>
            <span className="clinic__action-ta" lang="ta">{TA.CLINIC_CASE_ACTION_ESCALATE_TA}</span>
          </button>
        </div>
      </section>

      <p className="page__disclaimer" lang="ta">{TA.DISCLAIMER}</p>
    </main>
  );
}
