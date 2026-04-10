"""
Deterministic protocol grounding engine for Marunthagam.

Applies WHO IMNCI and Tamil Nadu state health rules on top of LLM triage output.
The engine ONLY upgrades triage urgency — it never downgrades.
A confidence < 0.7 always escalates one level and sets escalation_flag = True.

This is the safety floor: the system is never purely generative.
"""
import re
import sqlite3
from dataclasses import dataclass, field


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


class ProtocolEngine:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def apply(
        self,
        result: TriageResult,
        symptoms: str,
        age_group: str,
        duration_days: int,
    ) -> tuple[TriageResult, list[ProtocolOverride]]:
        """
        Apply deterministic protocol rules to the LLM triage output.

        Returns the (possibly upgraded) result and a list of overrides applied.
        The result level is NEVER downgraded — only upgraded.
        """
        overrides: list[ProtocolOverride] = []
        current_level = result.level

        rules = self.conn.execute(
            "SELECT * FROM protocol_rules WHERE active = 1"
        ).fetchall()

        for rule in rules:
            if not self._matches_rule(rule, symptoms, age_group, duration_days):
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

        # Confidence floor: escalate one level if confidence is below threshold
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
        symptoms: str,
        age_group: str,
        duration_days: int,
    ) -> bool:
        """Check if a rule applies to the current patient presentation."""
        condition_pattern = rule["condition_pattern"]
        if condition_pattern:
            try:
                if not re.search(condition_pattern, symptoms, re.IGNORECASE):
                    return False
            except re.error:
                return False

        rule_age = rule["age_group"]
        if rule_age and rule_age != "any" and rule_age != age_group:
            return False

        duration_min = rule["duration_min_days"]
        if duration_min is not None and duration_days < duration_min:
            return False

        return True

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ProtocolEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()
