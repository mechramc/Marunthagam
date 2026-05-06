"""
Deterministic protocol grounding engine for Marunthagam — v2 (2026-05-07).

v2 schema decision: chief complaint as the primary regex target, narrative
as positive-co-signal AND negative-scoping context only.

  condition_pattern        — matched against verbal_symptoms (chief complaint) ONLY.
                             Old v1 behavior was full-narrative match; that caused
                             false positives like IMNCI-002 firing on a chemo case
                             that *mentioned* the word fever.
  required_co_signals      — list of patterns (JSON-encoded in DB); ALL must match
                             somewhere in chief OR narrative for the rule to fire.
                             Use to express AND-combinations like "chest pain AND
                             (radiation OR dyspnea)".
  negative_scoping         — list of patterns; rule is suppressed if ANY match
                             anywhere in chief OR narrative. Use for "rule fires
                             UNLESS narrative explicitly negates."
  age_group                — 'any' or pipe-separated set: 'adolescent|adult|elderly'.
  duration_min_days        — earliest day the rule applies (inclusive).
  duration_max_days        — latest day the rule applies (inclusive); NULL = no upper bound.

Adolescent edge case (10-17): treated as non-pediatric for ADULT-* rules but
NOT matched by 'child' or 'infant'. So pediatric IMNCI-* rules don't fire on
adolescents and adult-* rules don't fire on children — clean partition.

Engine ONLY upgrades triage urgency; it never downgrades.
A confidence < 0.7 always escalates one level and sets escalation_flag = True.
"""
from __future__ import annotations

import json as _json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


TRIAGE_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}
TRIAGE_LEVELS = ["GREEN", "YELLOW", "RED"]
CONFIDENCE_THRESHOLD = 0.7
DISCLAIMER = "இது மருத்துவ ஆலோசனை அல்ல"


@dataclass
class TriageResult:
    level: str
    confidence: float
    suspected_conditions: list
    reasoning_chain: str
    next_steps_tamil: str
    protocol_references: list = field(default_factory=list)
    escalation_flag: bool = False
    disclaimer: str = DISCLAIMER


@dataclass
class ProtocolOverride:
    rule_id: str
    original_level: str
    overridden_to: str
    reason: str


def _decode_pattern_list(raw: Optional[str]) -> list[str]:
    """Decode a JSON-encoded list of patterns from the DB. Returns [] for null/empty."""
    if not raw:
        return []
    try:
        v = _json.loads(raw)
        if isinstance(v, list):
            return [str(p) for p in v if p]
    except _json.JSONDecodeError:
        return []
    return []


def _age_matches(rule_age: Optional[str], patient_age: str) -> bool:
    """
    Return True iff the patient's age_group is a member of the rule's
    age_group set. The rule's age_group is either 'any' (always matches)
    or a pipe-separated set: 'adolescent|adult|elderly'.

    A NULL/empty rule_age is treated as 'any' to be lenient.
    """
    if not rule_age or rule_age.lower() == "any":
        return True
    allowed = {a.strip().lower() for a in rule_age.split("|") if a.strip()}
    return patient_age.lower() in allowed


class ProtocolEngine:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def apply(
        self,
        result: TriageResult,
        chief_complaint: str,
        narrative: str = "",
        age_group: str = "any",
        duration_days: int = 0,
    ) -> tuple[TriageResult, list[ProtocolOverride]]:
        """
        Apply v2 rule schema to a TriageResult.

        Args:
            result: LLM triage output.
            chief_complaint: structured chief complaint (verbal_symptoms field).
                             Primary regex target for condition_pattern.
            narrative: free-text patient narrative (tamil_question field).
                       Used ONLY for required_co_signals and negative_scoping.
                       Default empty so callers from before v2 still work.
            age_group: patient age band (infant|child|adolescent|adult|elderly).
            duration_days: symptom duration.

        Result level is NEVER downgraded — only upgraded.
        """
        overrides: list[ProtocolOverride] = []
        current_level = result.level

        rules = self.conn.execute(
            "SELECT * FROM protocol_rules WHERE active = 1"
        ).fetchall()

        for rule in rules:
            if not self._matches_rule(rule, chief_complaint, narrative, age_group, duration_days):
                continue

            rule_level = rule["minimum_triage_level"]
            if TRIAGE_ORDER[rule_level] > TRIAGE_ORDER[current_level]:
                overrides.append(ProtocolOverride(
                    rule_id=rule["id"],
                    original_level=current_level,
                    overridden_to=rule_level,
                    reason=rule["override_reason"],
                ))
                current_level = rule_level
                if rule["id"] not in result.protocol_references:
                    result.protocol_references.append(rule["id"])

        # Confidence floor: escalate one level if confidence is below threshold.
        if result.confidence < CONFIDENCE_THRESHOLD:
            current_idx = TRIAGE_ORDER[current_level]
            if current_idx < TRIAGE_ORDER["RED"]:
                escalated_to = TRIAGE_LEVELS[current_idx + 1]
                overrides.append(ProtocolOverride(
                    rule_id="CONFIDENCE-FLOOR",
                    original_level=current_level,
                    overridden_to=escalated_to,
                    reason=(
                        f"Confidence {result.confidence:.2f} < {CONFIDENCE_THRESHOLD} "
                        f"— escalate per safety protocol"
                    ),
                ))
                current_level = escalated_to
                result.escalation_flag = True

        result.level = current_level
        result.disclaimer = DISCLAIMER
        return result, overrides

    def _matches_rule(
        self,
        rule: sqlite3.Row,
        chief_complaint: str,
        narrative: str,
        age_group: str,
        duration_days: int,
    ) -> bool:
        """v2 rule match. See module docstring for field semantics."""
        # 1. condition_pattern matches the chief complaint ONLY.
        condition_pattern = rule["condition_pattern"]
        if condition_pattern:
            try:
                if not re.search(condition_pattern, chief_complaint, re.IGNORECASE):
                    return False
            except re.error:
                return False

        # The full text (chief + narrative) is what co-signals and negative
        # scoping check against.
        full_text = f"{chief_complaint}\n{narrative}" if narrative else chief_complaint

        # 2. required_co_signals: ALL patterns must match somewhere in full text.
        try:
            co_signals = _decode_pattern_list(rule["required_co_signals"])
        except (IndexError, KeyError):
            co_signals = []
        for pat in co_signals:
            try:
                if not re.search(pat, full_text, re.IGNORECASE):
                    return False
            except re.error:
                return False

        # 3. negative_scoping: ANY match suppresses the rule.
        try:
            neg_scope = _decode_pattern_list(rule["negative_scoping"])
        except (IndexError, KeyError):
            neg_scope = []
        for pat in neg_scope:
            try:
                if re.search(pat, full_text, re.IGNORECASE):
                    return False
            except re.error:
                continue  # broken regex never suppresses

        # 4. Age constraint.
        if not _age_matches(rule["age_group"], age_group):
            return False

        # 5. Duration window.
        duration_min = rule["duration_min_days"]
        if duration_min is not None and duration_days < duration_min:
            return False
        try:
            duration_max = rule["duration_max_days"]
        except (IndexError, KeyError):
            duration_max = None
        if duration_max is not None and duration_days > duration_max:
            return False

        return True

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ProtocolEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()
