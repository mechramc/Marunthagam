"""
Pydantic v2 schemas for Marunthagam triage_classify() function calling.

Every triage output is validated against these schemas before being
displayed to the ASHA worker or logged. This prevents hallucinated
or malformed outputs from reaching users.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


DISCLAIMER_TEXT = "இது மருத்துவ ஆலோசனை அல்ல"


class AgeGroup(str, Enum):
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    ELDERLY = "elderly"


class TriageLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class VitalSigns(BaseModel):
    """Optional vital signs — recorded if ASHA worker has a thermometer/pulse oximeter."""
    temperature: Optional[float] = Field(None, ge=30.0, le=45.0, description="Body temperature in Celsius")
    pulse: Optional[int] = Field(None, ge=20, le=300, description="Pulse rate in bpm")
    respiratory_rate: Optional[int] = Field(None, ge=5, le=80, description="Respiratory rate per minute")


class TriageClassifyInput(BaseModel):
    """Input to the triage_classify() function call."""
    verbal_symptoms: str = Field(min_length=1, max_length=2000, description="Tamil symptom description from ASHA worker")
    image_findings: Optional[str] = Field(None, max_length=2000, description="Model's interpretation of the clinical image")
    patient_age_group: AgeGroup = Field(description="Patient age group")
    duration_days: int = Field(ge=0, le=3650, description="Number of days symptoms have persisted")
    vital_signs: Optional[VitalSigns] = Field(None, description="Optional vital signs if available")


class SuspectedCondition(BaseModel):
    """A single suspected diagnosis with rank."""
    condition: str = Field(min_length=1, description="Suspected condition name")
    rank: int = Field(ge=1, le=3, description="Rank among suspected conditions (1 = most likely)")


class TriageClassifyOutput(BaseModel):
    """
    Structured output of the triage_classify() function.

    The disclaimer field is ALWAYS set to the Tamil disclaimer string,
    regardless of what the model produces.
    """
    level: TriageLevel = Field(description="Triage urgency: GREEN (self-care), YELLOW (PHC within 48h), RED (emergency now)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's calibrated confidence 0.0-1.0")
    suspected_conditions: list[SuspectedCondition] = Field(max_length=3, description="Up to 3 ranked suspected conditions")
    reasoning_chain: str = Field(min_length=1, description="Step-by-step clinical reasoning in Tamil")
    next_steps_tamil: str = Field(min_length=1, description="Plain Tamil instructions for ASHA worker and patient")
    protocol_references: list[str] = Field(default_factory=list, description="WHO/IMNCI/TN protocol codes referenced")
    escalation_flag: bool = Field(description="True if confidence < 0.7 or conflicting signals")
    disclaimer: str = Field(default=DISCLAIMER_TEXT, description="Mandatory medical disclaimer in Tamil")

    @field_validator("disclaimer", mode="before")
    @classmethod
    def enforce_disclaimer(cls, v: str) -> str:
        """Always return the Tamil disclaimer regardless of model output."""
        return DISCLAIMER_TEXT
