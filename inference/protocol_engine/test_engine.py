"""
Tests for the deterministic protocol grounding engine.
All tests use temporary SQLite databases via pytest's tmp_path fixture.
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
    """Create a temp SQLite DB seeded with given rules."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    for rule in rules:
        conn.execute(
            """INSERT INTO protocol_rules
               (id, source, condition_pattern, age_group, duration_min_days,
                minimum_triage_level, override_reason, active)
               VALUES (?,?,?,?,?,?,?,1)""",
            [rule["id"], rule["source"], rule.get("condition_pattern"),
             rule.get("age_group"), rule.get("duration_min_days"),
             rule["minimum_triage_level"], rule["override_reason"]]
        )
    conn.commit()
    conn.close()
    return db_path


def make_result(level: str, confidence: float = 0.85) -> TriageResult:
    """Create a minimal TriageResult for testing."""
    return TriageResult(
        level=level,
        confidence=confidence,
        suspected_conditions=[],
        reasoning_chain="test reasoning",
        next_steps_tamil="test steps",
    )


class TestNoOverride:
    def test_no_override_when_llm_already_red(self, tmp_path):
        """LLM says RED and rule says RED — no upgrade needed, no overrides."""
        db = make_db(tmp_path, [{
            "id": "R1", "source": "TEST", "condition_pattern": "fever",
            "age_group": "infant", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "test"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(make_result("RED"), "fever", "infant", 1)
        assert result.level == "RED"
        assert len(overrides) == 0

    def test_no_rules_no_override(self, tmp_path):
        """Empty DB — result passes through unchanged."""
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(make_result("GREEN"), "mild cough", "adult", 1)
        assert result.level == "GREEN"
        assert len(overrides) == 0


class TestUpgrades:
    def test_upgrade_green_to_red_for_infant_fever(self, tmp_path):
        """IMNCI-002: any fever in infant → RED, even if LLM says GREEN."""
        db = make_db(tmp_path, [{
            "id": "IMNCI-002", "source": "WHO_IMNCI", "condition_pattern": "fever|காய்ச்சல்",
            "age_group": "infant", "duration_min_days": 0,
            "minimum_triage_level": "RED", "override_reason": "IMNCI infant fever"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(make_result("GREEN"), "காய்ச்சல் வந்தது", "infant", 1)
        assert result.level == "RED"
        assert len(overrides) == 1
        assert overrides[0].rule_id == "IMNCI-002"
        assert overrides[0].original_level == "GREEN"
        assert overrides[0].overridden_to == "RED"

    def test_upgrade_green_to_yellow_for_child_rash_fever(self, tmp_path):
        """TN-001: fever+rash in child after 2 days → YELLOW minimum."""
        db = make_db(tmp_path, [{
            "id": "TN-001", "source": "TN_STATE",
            "condition_pattern": "fever.*rash|rash.*fever|காய்ச்சல்.*சிவந்த",
            "age_group": "child", "duration_min_days": 2,
            "minimum_triage_level": "YELLOW", "override_reason": "TN measles protocol"
        }])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN"), "fever and rash on skin", "child", 3
            )
        assert result.level == "YELLOW"
        assert len(overrides) == 1
        assert overrides[0].rule_id == "TN-001"

    def test_multiple_rules_takes_highest_level(self, tmp_path):
        """Two rules fire: one says YELLOW, one says RED → final is RED."""
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
                make_result("GREEN"), "fever and convulsion", "child", 1
            )
        assert result.level == "RED"
        assert len(overrides) == 2


class TestConfidenceFloor:
    def test_confidence_floor_escalates_green_to_yellow(self, tmp_path):
        """Confidence 0.55 < 0.7 → GREEN escalates to YELLOW, escalation_flag=True."""
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("GREEN", confidence=0.55), "cough", "adult", 1
            )
        assert result.level == "YELLOW"
        assert result.escalation_flag is True
        assert any(o.rule_id == "CONFIDENCE-FLOOR" for o in overrides)

    def test_confidence_floor_escalates_yellow_to_red(self, tmp_path):
        """Confidence 0.60 < 0.7 → YELLOW escalates to RED."""
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("YELLOW", confidence=0.60), "fever diarrhea", "child", 2
            )
        assert result.level == "RED"
        assert result.escalation_flag is True

    def test_high_confidence_red_no_escalation(self, tmp_path):
        """RED with confidence 0.95 → stays RED, no confidence floor override."""
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, overrides = engine.apply(
                make_result("RED", confidence=0.95), "convulsion", "child", 1
            )
        assert result.level == "RED"
        assert not any(o.rule_id == "CONFIDENCE-FLOOR" for o in overrides)


class TestDisclaimer:
    def test_disclaimer_always_present(self, tmp_path):
        """Disclaimer is always set to Tamil string regardless of rules applied."""
        db = make_db(tmp_path, [])
        with ProtocolEngine(db) as engine:
            result, _ = engine.apply(make_result("GREEN"), "mild cold", "adult", 1)
        assert result.disclaimer == "இது மருத்துவ ஆலோசனை அல்ல"
