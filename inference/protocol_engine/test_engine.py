"""
Tests for the deterministic protocol grounding engine — v2 schema (2026-05-07).

All tests use temporary SQLite databases via pytest's tmp_path fixture.

v2 API change: engine.apply now takes `chief_complaint` and `narrative`
separately. Old tests passing a single `symptoms` string have been updated
to pass it as `chief_complaint` (matches old behaviour because the rules
that fired were already chief-complaint-y).
"""
import json
import sqlite3
import sys
from pathlib import Path
import pytest

# Add parent to path so we can import engine
sys.path.insert(0, str(Path(__file__).parent))
from engine import ProtocolEngine, TriageResult, DISCLAIMER


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def make_db(tmp_path: Path, rules: list[dict]) -> str:
    """
    Create a temp SQLite DB seeded with v2-schema rules.

    Each rule dict supports v2 fields plus legacy (condition_pattern as alias
    for chief_complaint_pattern). Empty defaults filled for new fields.
    """
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    for rule in rules:
        chief = rule.get("chief_complaint_pattern", rule.get("condition_pattern"))
        co = rule.get("required_co_signals", [])
        neg = rule.get("negative_scoping", [])
        conn.execute(
            """INSERT INTO protocol_rules
               (id, source, condition_pattern, required_co_signals,
                negative_scoping, age_group, duration_min_days,
                duration_max_days, minimum_triage_level, override_reason, active)
               VALUES (?,?,?,?,?,?,?,?,?,?,1)""",
            [
                rule["id"], rule["source"], chief,
                json.dumps(co, ensure_ascii=False),
                json.dumps(neg, ensure_ascii=False),
                rule.get("age_group"),
                rule.get("duration_min_days"),
                rule.get("duration_max_days"),
                rule["minimum_triage_level"],
                rule["override_reason"],
            ]
        )
    conn.commit()
    conn.close()
    return db_path


def make_result(level: str, confidence: float = 0.85) -> TriageResult:
    return TriageResult(
        level=level,
        confidence=confidence,
        suspected_conditions=[],
        reasoning_chain="test reasoning",
        next_steps_tamil="test steps",
    )


# ----------------------------- Existing v1 tests, ported to v2 ------------------------

class TestNoOverride:
    def test_no_override_when_llm_already_red(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "R1", "source": "TEST", "condition_pattern": "fever",
            "age_group": "infant", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "test"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("RED"), "fever", "", "infant", 1
            )
        assert result.level == "RED"
        assert len(overrides) == 0

    def test_no_rules_no_override(self, tmp_path):
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"), "mild cough", "", "adult", 1
            )
        assert result.level == "GREEN"
        assert len(overrides) == 0


class TestUpgrades:
    def test_upgrade_green_to_red_for_infant_fever(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "IMNCI-002", "source": "WHO_IMNCI",
            "condition_pattern": "fever|காய்ச்சல்",
            "age_group": "infant", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "IMNCI infant fever"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"), "காய்ச்சல் வந்தது", "", "infant", 1
            )
        assert result.level == "RED"
        assert any(o.rule_id == "IMNCI-002" for o in overrides)

    def test_upgrade_green_to_yellow_for_child_rash_fever(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "TN-001", "source": "TN_STATE",
            "condition_pattern": "fever.*rash|rash.*fever|காய்ச்சல்.*சிவந்த",
            "age_group": "child", "duration_min_days": 2,
            "minimum_triage_level": "YELLOW", "override_reason": "TN measles protocol"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"), "fever and rash on skin", "", "child", 3
            )
        assert result.level == "YELLOW"
        assert any(o.rule_id == "TN-001" for o in overrides)

    def test_multiple_rules_takes_highest_level(self, tmp_path):
        db = make_db(tmp_path, [
            {
                "id": "R-YELLOW", "source": "TEST", "condition_pattern": "fever",
                "age_group": "any", "duration_min_days": 0,
                "minimum_triage_level": "YELLOW", "override_reason": "yellow rule"
            },
            {
                "id": "R-RED", "source": "TEST", "condition_pattern": "convulsion",
                "age_group": "any", "duration_min_days": 0,
                "minimum_triage_level": "RED", "override_reason": "red rule"
            }
        ])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"), "fever and convulsion", "", "child", 1
            )
        assert result.level == "RED"
        assert len(overrides) == 2


class TestConfidenceFloor:
    def test_confidence_floor_escalates_green_to_yellow(self, tmp_path):
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", confidence=0.55), "cough", "", "adult", 1
            )
        assert result.level == "YELLOW"
        assert result.escalation_flag is True
        assert any(o.rule_id == "CONFIDENCE-FLOOR" for o in overrides)

    def test_confidence_floor_escalates_yellow_to_red(self, tmp_path):
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", confidence=0.60), "fever diarrhea", "", "child", 2
            )
        assert result.level == "RED"
        assert result.escalation_flag is True

    def test_high_confidence_red_no_escalation(self, tmp_path):
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("RED", confidence=0.95), "convulsion", "", "child", 1
            )
        assert result.level == "RED"
        assert not any(o.rule_id == "CONFIDENCE-FLOOR" for o in overrides)


class TestDisclaimer:
    def test_disclaimer_always_present(self, tmp_path):
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(
                make_result("GREEN"), "mild cold", "", "adult", 1
            )
        assert result.disclaimer == DISCLAIMER


# ------------------------- v2 schema-specific tests --------------------------

class TestChiefVsNarrativeMatching:
    """The key v2 invariant: condition_pattern is matched against chief
    complaint ONLY. A narrative-only mention does NOT fire the rule."""

    def test_narrative_mention_does_not_fire_rule(self, tmp_path):
        """Sprint 1 chemo+fever case: narrative mentioned 'fever' but chief
        complaint was GI bleeding. v1 would incorrectly fire IMNCI-002. v2
        must NOT fire."""
        db = make_db(tmp_path, [{
            "id": "IMNCI-002", "source": "WHO_IMNCI",
            "condition_pattern": "fever|காய்ச்சல்",
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "fever rule"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW"),
                chief_complaint="rectal bleeding and mucus",
                narrative="patient is on chemo, mentions previous fever resolved",
                age_group="adult", duration_days=7,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "IMNCI-002" for o in overrides)

    def test_chief_complaint_match_fires_rule(self, tmp_path):
        """Same rule, but fever IS the chief complaint — should fire."""
        db = make_db(tmp_path, [{
            "id": "IMNCI-002", "source": "WHO_IMNCI",
            "condition_pattern": "fever|காய்ச்சல்",
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "fever rule"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"),
                chief_complaint="high fever for 2 days",
                narrative="kid has had a high temperature",
                age_group="child", duration_days=2,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "IMNCI-002" for o in overrides)


class TestRequiredCoSignals:
    def test_co_signal_satisfied_fires(self, tmp_path):
        """Rule requires 'fever' chief + 'rash' co-signal in narrative — fires."""
        db = make_db(tmp_path, [{
            "id": "T1", "source": "TEST",
            "condition_pattern": "fever",
            "required_co_signals": ["rash|தோல் புள்ளி"],
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "YELLOW", "override_reason": "fever+rash"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"),
                chief_complaint="fever",
                narrative="also has a rash on the chest",
                age_group="child", duration_days=2,
            )
        assert result.level == "YELLOW"
        assert any(o.rule_id == "T1" for o in overrides)

    def test_co_signal_unsatisfied_does_not_fire(self, tmp_path):
        """Same rule, but rash NOT mentioned — does not fire."""
        db = make_db(tmp_path, [{
            "id": "T1", "source": "TEST",
            "condition_pattern": "fever",
            "required_co_signals": ["rash|தோல் புள்ளி"],
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "YELLOW", "override_reason": "fever+rash"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"),
                chief_complaint="fever",
                narrative="just a fever, nothing else",
                age_group="child", duration_days=2,
            )
        assert result.level == "GREEN"


class TestNegativeScoping:
    def test_negative_scope_suppresses_rule(self, tmp_path):
        """Jaundice rule, but narrative mentions 'known chronic hepatitis' —
        suppressed (this is not new-onset jaundice)."""
        db = make_db(tmp_path, [{
            "id": "JAUND", "source": "TEST",
            "condition_pattern": "jaundice|காமாலை",
            "negative_scoping": ["(known|chronic|prior)\\s*(liver|hepat)"],
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "new-onset jaundice"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW"),
                chief_complaint="jaundice",
                narrative="patient has known chronic hepatitis B",
                age_group="adult", duration_days=10,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "JAUND" for o in overrides)

    def test_negative_scope_does_not_suppress_when_pattern_absent(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "JAUND", "source": "TEST",
            "condition_pattern": "jaundice|காமாலை",
            "negative_scoping": ["(known|chronic|prior)\\s*(liver|hepat)"],
            "age_group": "any", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "new-onset jaundice"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW"),
                chief_complaint="jaundice",
                narrative="never had liver problems before",
                age_group="adult", duration_days=10,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "JAUND" for o in overrides)


class TestPipeAgeGroup:
    def test_adolescent_matches_adult_set(self, tmp_path):
        """age_group='adolescent|adult|elderly' matches a 14yo (adolescent)."""
        db = make_db(tmp_path, [{
            "id": "ADULT-X", "source": "TEST",
            "condition_pattern": "chest pain",
            "age_group": "adolescent|adult|elderly",
            "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "adult rule"
        }])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(
                make_result("GREEN"),
                chief_complaint="chest pain",
                narrative="",
                age_group="adolescent",
                duration_days=1,
            )
        assert result.level == "RED"

    def test_child_does_not_match_adult_set(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "ADULT-X", "source": "TEST",
            "condition_pattern": "chest pain",
            "age_group": "adolescent|adult|elderly",
            "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "adult rule"
        }])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(
                make_result("GREEN"),
                chief_complaint="chest pain",
                narrative="",
                age_group="child",
                duration_days=1,
            )
        assert result.level == "GREEN"


class TestDurationMaxDays:
    def test_duration_max_days_excludes_chronic(self, tmp_path):
        """Acute-only rule (duration_max_days=14) does NOT fire on 30-day case."""
        db = make_db(tmp_path, [{
            "id": "ACUTE", "source": "TEST",
            "condition_pattern": "jaundice",
            "age_group": "any", "duration_min_days": 0, "duration_max_days": 14,
            "minimum_triage_level": "RED", "override_reason": "acute jaundice"
        }])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(
                make_result("YELLOW"), "jaundice", "", "adult", 30
            )
        assert result.level == "YELLOW"

    def test_duration_max_days_fires_on_acute(self, tmp_path):
        db = make_db(tmp_path, [{
            "id": "ACUTE", "source": "TEST",
            "condition_pattern": "jaundice",
            "age_group": "any", "duration_min_days": 0, "duration_max_days": 14,
            "minimum_triage_level": "RED", "override_reason": "acute jaundice"
        }])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(
                make_result("YELLOW"), "jaundice", "", "adult", 5
            )
        assert result.level == "RED"


# ------------------------- 6 new adult-emergency rules: pos + neg per rule ---

class TestAdultCardiac001:
    """ADULT-CARDIAC-001 — chest pain + radiation + (dyspnea OR diaphoresis OR tachycardia)."""

    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-CARDIAC-001")
        return make_db(tmp_path, [rule])

    def test_pos_chest_pain_jaw_radiation_dyspnea_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", 0.9),
                chief_complaint="chest pressure radiating to jaw",
                narrative="also feeling short of breath and sweating",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-CARDIAC-001" for o in overrides)

    def test_neg_chest_pain_no_radiation_does_not_fire(self, db):
        """Chief is chest pain but neither radiation nor systemic co-signal — no fire."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", 0.9),
                chief_complaint="chest pain",
                narrative="just musculoskeletal soreness after exercise",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "GREEN"
        assert not any(o.rule_id == "ADULT-CARDIAC-001" for o in overrides)


class TestAdultAnaphylaxis001:
    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-ANAPHYLAXIS-001")
        return make_db(tmp_path, [rule])

    def test_pos_acute_tongue_swelling_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.8),
                chief_complaint="sudden tongue swelling and throat tightness",
                narrative="started 2 hours ago after eating peanuts",
                age_group="adult",
                duration_days=0,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-ANAPHYLAXIS-001" for o in overrides)

    def test_neg_chronic_tongue_swelling_does_not_fire(self, db):
        """duration_max_days=1; a 30-day-old swelling is excluded."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.8),
                chief_complaint="chronic tongue swelling",
                narrative="been like this for a month, scheduled with allergist",
                age_group="adult",
                duration_days=30,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "ADULT-ANAPHYLAXIS-001" for o in overrides)


class TestAdultHeadTrauma001:
    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-HEAD-TRAUMA-001")
        return make_db(tmp_path, [rule])

    def test_pos_head_injury_with_loc_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="head injury from fall",
                narrative="lost consciousness for a few seconds and has been confused since",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-HEAD-TRAUMA-001" for o in overrides)

    def test_neg_head_injury_no_loc_no_ams_does_not_fire(self, db):
        """Negative narrative MUST avoid the trigger keywords entirely. The
        regex doesn't distinguish 'no loss of consciousness' from 'loss of
        consciousness' — that's a known limitation. Clinically this is OK:
        the failure mode is over-escalation (false RED) not under-escalation
        (missed RED). For a benign head injury, the realistic narrative would
        not mention LOC at all."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="head injury from fall",
                narrative="bumped my head playing cricket, applied ice, eating dinner now",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "ADULT-HEAD-TRAUMA-001" for o in overrides)


class TestAdultRespiratory001:
    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-RESPIRATORY-001")
        return make_db(tmp_path, [rule])

    def test_pos_severe_wheezing_speech_difficulty_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="severe wheezing",
                narrative="cannot speak in complete sentences, lips look blue",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-RESPIRATORY-001" for o in overrides)

    def test_neg_mild_wheezing_no_distress_does_not_fire(self, db):
        """Same caveat as the head-trauma negative test: the narrative must
        not contain trigger keywords AT ALL (the regex doesn't handle
        'no cyanosis'). The realistic benign-wheezing narrative is silence
        on red-flag features."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="mild wheezing",
                narrative="responds well to inhaler, eating and sleeping normally",
                age_group="adult",
                duration_days=2,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "ADULT-RESPIRATORY-001" for o in overrides)


class TestAnimalBiteRespiratory001:
    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ANIMAL-BITE-RESPIRATORY-001")
        return make_db(tmp_path, [rule])

    def test_pos_dog_bite_with_dyspnea_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.8),
                chief_complaint="dog bite on arm",
                narrative="now short of breath and feels like throat is swelling",
                age_group="adult",
                duration_days=0,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ANIMAL-BITE-RESPIRATORY-001" for o in overrides)

    def test_neg_dog_bite_without_respiratory_does_not_fire(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.8),
                chief_complaint="dog bite",
                narrative="superficial scratch, breathing normally",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "ANIMAL-BITE-RESPIRATORY-001" for o in overrides)


class TestNewOnsetJaundice001:
    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "NEW-ONSET-JAUNDICE-001")
        return make_db(tmp_path, [rule])

    def test_pos_acute_yellow_skin_no_prior_dx_fires(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="yellow skin and sclera",
                narrative="started 5 days ago, no prior diagnosis",
                age_group="adult",
                duration_days=5,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "NEW-ONSET-JAUNDICE-001" for o in overrides)

    def test_neg_chronic_jaundice_with_known_liver_disease_does_not_fire(self, db):
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="jaundice",
                narrative="known chronic hepatitis B, on lifelong tenofovir",
                age_group="adult",
                duration_days=5,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "NEW-ONSET-JAUNDICE-001" for o in overrides)

    def test_neg_pediatric_jaundice_does_not_fire(self, db):
        """Pediatric jaundice is governed by other (existing) rules; this rule
        is adolescent+ only."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="yellow skin",
                narrative="6-month-old infant, started this week",
                age_group="infant",
                duration_days=3,
            )
        assert result.level == "YELLOW"
        assert not any(o.rule_id == "NEW-ONSET-JAUNDICE-001" for o in overrides)


# ------------- v2.1 Bucket A regex tightenings (2026-05-07) ------------------
#
# Targeted morphology fixes for three Tamil case-inflected forms found
# in held-out test cases that the v2.0 chief regex didn't handle:
#   - locative -இல் on மார்பு (chest)            → ADULT-CARDIAC-001
#   - instrumental -இனால் on நாய் (dog)          → ANIMAL-BITE-RESPIRATORY-001
#   - sandhi compound -த்திணறல் on மூச்சு (breath) → ADULT-RESPIRATORY-001
#
# Each fix has a positive test (rule fires on the previously-missed case)
# and a negative test (rule does not over-fire on a benign case that
# contains the same morpheme in a non-emergency context).


class TestCardiacLocativeFix:
    """ADULT-CARDIAC-001 v2.1 fix: locative `மார்பில்` + intervening words."""

    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-CARDIAC-001")
        return make_db(tmp_path, [rule])

    def test_pos_tamil_locative_chest_pain_fires(self, db):
        """triage_test_039 case: locative மார்பில் + adjective + வலி + arm numbness."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.9),
                chief_complaint="இடது மார்பில் கடுமையான வலி மற்றும் இடது கையில் மரத்துப்போன உணர்வு",
                narrative="ER visit, chest pain radiating to jaw, dyspnea",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-CARDIAC-001" for o in overrides)

    def test_neg_chest_skin_red_spot_does_not_fire(self, db):
        """`மார்பில் சிவப்பு புள்ளி` (red spot on chest) — locative form present
        but no pain/tightness/discomfort word. Negative-test guards the
        morphology fix from over-firing on dermatologic chest mentions."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", 0.85),
                chief_complaint="மார்பில் சிவப்பு புள்ளி, அரிப்பு",
                narrative="redness on chest skin, no pain",
                age_group="adult",
                duration_days=3,
            )
        assert result.level == "GREEN"
        assert not any(o.rule_id == "ADULT-CARDIAC-001" for o in overrides)


class TestAnimalBiteInstrumentalFix:
    """ANIMAL-BITE-RESPIRATORY-001 v2.1 fix: instrumental `நாயினால்` + passive `கடிக்கப்பட்`."""

    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ANIMAL-BITE-RESPIRATORY-001")
        return make_db(tmp_path, [rule])

    def test_pos_tamil_instrumental_dog_bite_with_dyspnea_fires(self, db):
        """Instrumental form நாயினால் + bite passive + respiratory co-signal."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="நாயினால் கடிக்கப்பட்டது, கடித்த இடத்தில் வலி",
                narrative="மூச்சு திண, தொண்டை வீக்கம் (dyspnea, throat swelling)",
                age_group="adult",
                duration_days=0,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ANIMAL-BITE-RESPIRATORY-001" for o in overrides)

    def test_neg_dog_raised_no_bite_does_not_fire(self, db):
        """`நாயினால் வளர்க்கப்பட்ட` (raised by a dog — fanciful) — uses
        instrumental form but no `கடி` keyword. Negative-test guards the
        morphology fix from over-firing on non-bite mentions of நாய்."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", 0.9),
                chief_complaint="நாயினால் வளர்க்கப்பட்ட குழந்தை, தடிப்பு",
                narrative="child raised around a dog, has rash",
                age_group="child",
                duration_days=14,
            )
        assert result.level == "GREEN"
        assert not any(o.rule_id == "ANIMAL-BITE-RESPIRATORY-001" for o in overrides)


class TestRespiratoryCompoundFix:
    """ADULT-RESPIRATORY-001 v2.1 fix: sandhi compound `மூச்சுத்திணறல்`."""

    @pytest.fixture
    def db(self, tmp_path):
        rules_file = Path(__file__).parent / "rules" / "imnci_rules_v2.json"
        all_rules = json.loads(rules_file.read_text(encoding="utf-8"))
        rule = next(r for r in all_rules if r.get("id") == "ADULT-RESPIRATORY-001")
        return make_db(tmp_path, [rule])

    def test_pos_tamil_compound_dyspnea_with_severe_distress_fires(self, db):
        """Compound form மூச்சுத்திணறல் + severe-distress co-signal."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", 0.85),
                chief_complaint="மூச்சுத்திணறல் கடுமையாக",
                narrative="பேச முடியவில்லை, நீல உதடு (cannot speak, blue lips)",
                age_group="adult",
                duration_days=1,
            )
        assert result.level == "RED"
        assert any(o.rule_id == "ADULT-RESPIRATORY-001" for o in overrides)

    def test_neg_breathing_exercise_does_not_fire(self, db):
        """`மூச்சுப் பயிற்சி` (breathing exercise) — uses மூச்சு + sandhi
        but with non-respiratory-distress noun. Negative-test guards the
        morphology fix from matching unrelated breath-prefix compounds."""
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", 0.9),
                chief_complaint="மூச்சுப் பயிற்சி பற்றி கேள்வி",
                narrative="question about breathing exercises for stress relief",
                age_group="adult",
                duration_days=30,
            )
        assert result.level == "GREEN"
        assert not any(o.rule_id == "ADULT-RESPIRATORY-001" for o in overrides)
