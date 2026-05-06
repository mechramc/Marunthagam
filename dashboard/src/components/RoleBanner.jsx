import './RoleBanner.css';

const ROLE_INFO = {
  district: {
    label: 'District Health Office',
    labelTa: 'மாவட்ட சுகாதார அலுவலகம்',
    blurb: 'Aggregated population view — geohash cells, 7-day trends, RED/YELLOW clusters. No patient identification possible.',
    blurbTa: 'மக்கள்தொகை கண்ணோட்டம் — பகுதிகள், போக்குகள், கூட்டங்கள். தனிப்பட்ட நோயாளி அடையாளம் இல்லை.',
    icon: '🏛',
  },
  clinic: {
    label: 'PHC Doctor — Clinic Console',
    labelTa: 'PHC மருத்துவர் — மருத்துவ பணியிடம்',
    blurb: 'Case-level access for the doctor on duty. Each row is one ASHA-escalated case with the patient narrative in Tamil.',
    blurbTa: 'பணியிலுள்ள மருத்துவருக்கு வழக்கு-மட்ட அணுகல். ஒவ்வொரு வழக்கும் ASHA தொழிலாளர் அனுப்பிய நோயாளி விளக்கத்துடன்.',
    icon: '🩺',
  },
};

/**
 * One-line role banner shown below every page header. Establishes context
 * for screenshots ("you are viewing as: X") so the audience for each view
 * is unambiguous.
 *
 * @param {{ role: 'district' | 'clinic' }} props
 */
export default function RoleBanner({ role }) {
  const info = ROLE_INFO[role] || ROLE_INFO.district;
  return (
    <aside className={`role-banner role-banner--${role}`} role="note">
      <span className="role-banner__icon" aria-hidden="true">{info.icon}</span>
      <div className="role-banner__text">
        <span className="role-banner__role">
          Viewing as: <strong>{info.label}</strong>
          <span className="role-banner__role-ta" lang="ta">  ·  {info.labelTa}</span>
        </span>
        <span className="role-banner__blurb">{info.blurb}</span>
      </div>
    </aside>
  );
}
