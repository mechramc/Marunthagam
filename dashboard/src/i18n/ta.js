/**
 * Tamil (ta) string constants for Marunthagam dashboards.
 * All Tamil UI strings must be defined here — never scattered inline.
 *
 * The same i18n table serves both Tier 2 (PHC clinic console — PHC doctor
 * audience) and Tier 3 (district health dashboard — block / district health
 * department audience). Role-specific strings are namespaced by prefix.
 */

export const TA = {
  // App-level
  APP_TITLE: 'மருந்தகம்',
  DASHBOARD_TITLE: 'மாவட்ட சுகாதார டாஷ்போர்டு',
  CLINIC_CONSOLE_TITLE: 'முதன்மை சுகாதார மையம் — மருத்துவர் பணியிடம்',
  DISCLAIMER: 'இது மருத்துவ ஆலோசனை அல்ல',

  // Role switcher
  ROLE_DISTRICT: 'மாவட்ட சுகாதார அலுவலகம்',
  ROLE_DISTRICT_EN: 'District Health Office',
  ROLE_CLINIC: 'PHC மருத்துவர்',
  ROLE_CLINIC_EN: 'PHC Doctor',
  ROLE_SWITCH: 'பார்வையை மாற்று',
  ROLE_SWITCH_EN: 'Switch view',

  // ── District (Tier 3) navigation ──
  NAV_OVERVIEW: 'கண்ணோட்டம்',
  NAV_MAP: 'வரைபடம்',
  NAV_ALERTS: 'விழிப்பூட்டல்கள்',
  NAV_TRENDS: 'போக்குகள்',

  // ── Clinic (Tier 2) navigation ──
  NAV_CLINIC_QUEUE: 'வழக்கு வரிசை',
  NAV_CLINIC_QUEUE_EN: 'Case Queue',
  NAV_CLINIC_CATCHMENT: 'சேவைப் பகுதி',
  NAV_CLINIC_CATCHMENT_EN: 'Catchment Area',
  NAV_CLINIC_PROTOCOLS: 'நடைமுறை குறிப்புகள்',
  NAV_CLINIC_PROTOCOLS_EN: 'Protocol Notes',

  // Triage levels
  LEVEL_GREEN: 'பச்சை',
  LEVEL_YELLOW: 'மஞ்சள்',
  LEVEL_RED: 'சிவப்பு',
  LEVEL_GREEN_EN: 'GREEN',
  LEVEL_YELLOW_EN: 'YELLOW',
  LEVEL_RED_EN: 'RED',

  // Stats summary cards (district)
  STATS_TOTAL_CASES: 'இன்றைய மொத்த வழக்குகள்',
  STATS_TOTAL_CASES_EN: 'Total Cases Today',
  STATS_RED_CASES: 'சிவப்பு வழக்குகள்',
  STATS_RED_CASES_EN: 'RED Cases',
  STATS_ACTIVE_CELLS: 'செயலில் உள்ள பகுதிகள்',
  STATS_ACTIVE_CELLS_EN: 'Active Geohash Cells',
  STATS_ESCALATION_RATE: 'அதிகரிப்பு விகிதம்',
  STATS_ESCALATION_RATE_EN: 'Escalation Rate',
  STATS_VS_YESTERDAY: 'நேற்றை விட',
  STATS_VS_YESTERDAY_EN: 'vs yesterday',

  // Heatmap
  HEATMAP_TITLE: 'புவியியல் முறை தகவல்கள்',
  HEATMAP_TITLE_EN: 'Geohash Triage Heatmap',
  HEATMAP_TOOLTIP_GEOHASH: 'புவியியல் குறியீடு',
  HEATMAP_TOOLTIP_LAST_UPDATED: 'கடைசியாக புதுப்பிக்கப்பட்டது',
  HEATMAP_NO_DATA: 'தரவு இல்லை',

  // Trend chart
  TREND_TITLE: '7-நாள் போக்கு',
  TREND_TITLE_EN: '7-Day Triage Trend',
  TREND_XAXIS: 'தேதி',
  TREND_YAXIS: 'வழக்குகளின் எண்ணிக்கை',
  TREND_YAXIS_EN: 'Case Count',

  // Alert panel
  ALERT_TITLE: 'சிவப்பு நிலை விழிப்பூட்டல்கள்',
  ALERT_TITLE_EN: 'RED Level Cluster Alerts',
  ALERT_EMPTY: 'கவலைப்படும் விழிப்பூட்டல்கள் இல்லை',
  ALERT_EMPTY_EN: 'No concerning alerts',
  ALERT_CRITICAL_BADGE: 'நெருக்கடி',
  ALERT_CRITICAL_BADGE_EN: 'Critical',
  ALERT_RED_COUNT: 'சிவப்பு வழக்குகள்',
  ALERT_RED_COUNT_EN: 'RED cases in 24h',
  ALERT_FIRST_SEEN: 'முதல் வழக்கு',
  ALERT_FIRST_SEEN_EN: 'First case',
  ALERT_TREND_UP: '↑ அதிகரித்து வருகிறது',
  ALERT_TREND_DOWN: '↓ குறைந்து வருகிறது',
  ALERT_TREND_STABLE: '→ நிலையாக உள்ளது',
  ALERT_LATEST_EN: 'Latest',

  // District page titles
  PAGE_OVERVIEW_TITLE: 'கண்ணோட்டம்',
  PAGE_MAP_TITLE: 'புவியியல் வரைபடம்',
  PAGE_ALERTS_TITLE: 'விழிப்பூட்டல்கள்',
  PAGE_TRENDS_TITLE: 'போக்குகள்',

  // Data freshness
  DATA_UPDATED: 'தரவு புதுப்பிக்கப்பட்டது',
  DATA_UPDATED_EN: 'Data updated',
  DATA_LOADING: 'ஏற்றுகிறது...',
  DATA_ERROR: 'தரவு ஏற்றுவதில் பிழை',

  // ── Clinic console (Tier 2) strings ──
  CLINIC_QUEUE_TITLE: 'வரும் வழக்குகள்',
  CLINIC_QUEUE_TITLE_EN: 'Incoming Cases',
  CLINIC_QUEUE_DESC_EN:
    'Cases escalated from ASHA workers in your catchment area. ' +
    'Review and confirm the triage decision before referral.',
  CLINIC_QUEUE_FILTER_ALL_EN: 'All',
  CLINIC_QUEUE_FILTER_RED_EN: 'RED only',
  CLINIC_QUEUE_FILTER_YELLOW_EN: 'YELLOW only',
  CLINIC_QUEUE_FILTER_TODAY_EN: 'Today only',
  CLINIC_QUEUE_COL_RECEIVED_EN: 'Received',
  CLINIC_QUEUE_COL_LEVEL_EN: 'ASHA triage',
  CLINIC_QUEUE_COL_CHIEF_EN: 'Chief complaint',
  CLINIC_QUEUE_COL_ASHA_EN: 'ASHA worker',
  CLINIC_QUEUE_COL_CONF_EN: 'Confidence',
  CLINIC_QUEUE_COL_OVERRIDES_EN: 'Engine overrides',

  CLINIC_CASE_TITLE_EN: 'Case detail',
  CLINIC_CASE_BACK_EN: '← Back to queue',
  CLINIC_CASE_PATIENT_NARRATIVE_EN: 'Patient narrative (Tamil)',
  CLINIC_CASE_PATIENT_NARRATIVE_TA: 'நோயாளியின் விளக்கம்',
  CLINIC_CASE_ASHA_TRIAGE_EN: 'ASHA worker triage decision',
  CLINIC_CASE_ASHA_TRIAGE_TA: 'ASHA தொழிலாளர் தீர்மானம்',
  CLINIC_CASE_MODEL_OUTPUT_EN: 'Model raw output',
  CLINIC_CASE_MODEL_OUTPUT_TA: 'மாதிரி வெளியீடு',
  CLINIC_CASE_ENGINE_OVERRIDES_EN: 'IMNCI rules fired',
  CLINIC_CASE_ENGINE_OVERRIDES_TA: 'IMNCI விதிகள்',
  CLINIC_CASE_NO_OVERRIDES_EN: 'No engine overrides — model output stands',
  CLINIC_CASE_DOCTOR_ACTION_EN: 'Doctor action',
  CLINIC_CASE_DOCTOR_ACTION_TA: 'மருத்துவரின் முடிவு',
  CLINIC_CASE_ACTION_CONFIRM_EN: 'Confirm triage and refer',
  CLINIC_CASE_ACTION_CONFIRM_TA: 'தீர்மானத்தை உறுதிசெய்',
  CLINIC_CASE_ACTION_DOWNGRADE_EN: 'Downgrade after exam',
  CLINIC_CASE_ACTION_DOWNGRADE_TA: 'பரிசோதனைக்குப் பின் குறை',
  CLINIC_CASE_ACTION_ESCALATE_EN: 'Escalate to district hospital',
  CLINIC_CASE_ACTION_ESCALATE_TA: 'மாவட்ட மருத்துவமனைக்கு அனுப்பு',
  CLINIC_CASE_RECEIVED_EN: 'Received',
  CLINIC_CASE_FROM_EN: 'from',
  CLINIC_CASE_GEOHASH_EN: 'Geohash',
  CLINIC_CASE_SPECIALIST_EN: 'Routed to',
  CLINIC_CASE_CONFIDENCE_EN: 'Model confidence',
  CLINIC_CASE_PRE_ENGINE_EN: 'Pre-engine prediction',
  CLINIC_CASE_POST_ENGINE_EN: 'Post-engine final',

  CLINIC_CATCHMENT_TITLE_EN: 'My Catchment Area',
  CLINIC_CATCHMENT_DESC_EN:
    'Geohash cells served by ASHA workers reporting to this PHC. ' +
    'Counts reflect today + the prior 6 days.',
  CLINIC_CATCHMENT_COL_CELL_EN: 'Cell',
  CLINIC_CATCHMENT_COL_CASES_EN: 'Cases (7d)',
  CLINIC_CATCHMENT_COL_RED_EN: 'RED',
  CLINIC_CATCHMENT_COL_YELLOW_EN: 'YELLOW',
  CLINIC_CATCHMENT_COL_GREEN_EN: 'GREEN',
  CLINIC_CATCHMENT_COL_LAST_EN: 'Last case',
  CLINIC_CATCHMENT_COL_ASHA_EN: 'ASHA worker',

  // Submission scope footer
  SUBMISSION_SCOPE_EN:
    'v1.0 hackathon submission — Tier 1 (model + protocol engine) ships; ' +
    'Tier 2 console is screenshot-ready UI over real eval data; ' +
    'Tier 3 dashboard renders aggregates derived from held-out predictions.',
};

export default TA;
