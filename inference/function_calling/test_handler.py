"""
Tests for the triage_classify() function calling handler.
"""
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from handler import extract_function_call, parse_triage_input, parse_triage_output
from schemas import DISCLAIMER_TEXT


class TestExtractFunctionCall:
    def test_extract_from_tool_call_tags(self):
        raw = '<tool_call>{"name": "triage_classify", "arguments": {"x": 1}}</tool_call>'
        result = extract_function_call(raw)
        assert result is not None
        assert result["name"] == "triage_classify"

    def test_extract_fallback_to_plain_json(self):
        raw = '{"name": "triage_classify", "level": "GREEN"}'
        result = extract_function_call(raw)
        assert result is not None
        assert result["name"] == "triage_classify"

    def test_extract_returns_none_for_garbage(self):
        result = extract_function_call("This is not JSON at all")
        assert result is None

    def test_extract_returns_none_for_empty_string(self):
        result = extract_function_call("")
        assert result is None


class TestParseTriageInput:
    def test_valid_input_parses(self):
        result = parse_triage_input({
            "verbal_symptoms": "காய்ச்சல் மற்றும் இருமல்",
            "patient_age_group": "child",
            "duration_days": 3,
        })
        assert result.patient_age_group.value == "child"
        assert result.duration_days == 3

    def test_invalid_age_group_raises(self):
        with pytest.raises(ValidationError):
            parse_triage_input({
                "verbal_symptoms": "fever",
                "patient_age_group": "baby",  # invalid
                "duration_days": 1,
            })

    def test_symptoms_too_long_raises(self):
        with pytest.raises(ValidationError):
            parse_triage_input({
                "verbal_symptoms": "x" * 2001,  # over 2000 char limit
                "patient_age_group": "adult",
                "duration_days": 1,
            })


class TestParseTriageOutput:
    def _valid_output(self, **overrides) -> dict:
        base = {
            "level": "GREEN",
            "confidence": 0.85,
            "suspected_conditions": [],
            "reasoning_chain": "சாதாரண சளி",
            "next_steps_tamil": "ஓய்வு எடுக்கவும்",
            "escalation_flag": False,
        }
        base.update(overrides)
        return base

    def test_disclaimer_enforced_even_if_wrong(self):
        result = parse_triage_output(self._valid_output(disclaimer="wrong"))
        assert result.disclaimer == DISCLAIMER_TEXT

    def test_disclaimer_set_when_missing(self):
        d = self._valid_output()
        d.pop("disclaimer", None)
        result = parse_triage_output(d)
        assert result.disclaimer == DISCLAIMER_TEXT

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            parse_triage_output(self._valid_output(confidence=1.5))

    def test_max_three_conditions(self):
        with pytest.raises(ValidationError):
            parse_triage_output(self._valid_output(suspected_conditions=[
                {"condition": "A", "rank": 1},
                {"condition": "B", "rank": 2},
                {"condition": "C", "rank": 3},
                {"condition": "D", "rank": 3},  # 4th — over limit
            ]))

    def test_valid_full_output_parses(self):
        result = parse_triage_output(self._valid_output(
            suspected_conditions=[{"condition": "Common cold", "rank": 1}],
            protocol_references=["TN-004"],
        ))
        assert result.level.value == "GREEN"
        assert result.confidence == 0.85
        assert len(result.suspected_conditions) == 1
        assert result.protocol_references == ["TN-004"]
