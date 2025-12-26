"""Pydantic output models for validating LLM responses.

These models are used with Pydantic AI TextOutput extractors to validate
responses after free-form generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EvidenceOutput(BaseModel):
    """Evidence for a single PHQ-8 item."""

    evidence: str = Field(description="Direct quote from transcript")
    reason: str = Field(description="Reasoning for score assignment")
    score: int | None = Field(description="PHQ-8 score (0-3) or None for N/A")

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            if value.strip().upper() == "N/A":
                return None
            try:
                value = int(value)
            except ValueError as e:
                raise ValueError("score must be 0-3 or N/A") from e
        if isinstance(value, float) and value == int(value):
            value = int(value)
        if isinstance(value, int) and 0 <= value <= 3:
            return value
        raise ValueError("score must be 0-3 or N/A")


class QuantitativeOutput(BaseModel):
    """Complete quantitative assessment output."""

    PHQ8_NoInterest: EvidenceOutput
    PHQ8_Depressed: EvidenceOutput
    PHQ8_Sleep: EvidenceOutput
    PHQ8_Tired: EvidenceOutput
    PHQ8_Appetite: EvidenceOutput
    PHQ8_Failure: EvidenceOutput
    PHQ8_Concentrating: EvidenceOutput
    PHQ8_Moving: EvidenceOutput


class QualitativeOutput(BaseModel):
    """Qualitative assessment output."""

    assessment: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    exact_quotes: list[str] = Field(default_factory=list)


class JudgeMetricOutput(BaseModel):
    """Judge output for a single metric."""

    score: int = Field(ge=1, le=5)
    explanation: str


class JudgeOutput(BaseModel):
    """Judge output for all four metrics."""

    coherence: JudgeMetricOutput
    completeness: JudgeMetricOutput
    specificity: JudgeMetricOutput
    accuracy: JudgeMetricOutput


class MetaReviewOutput(BaseModel):
    """Meta-review agent output."""

    severity: int = Field(ge=0, le=4)
    explanation: str
