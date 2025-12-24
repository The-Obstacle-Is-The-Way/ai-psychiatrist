"""Domain enumerations for AI Psychiatrist.

This module defines core enumerations used throughout the domain layer:
- PHQ8Item: The 8 depression symptom items (DSM-5 criteria)
- PHQ8Score: Frequency scores (0-3) for PHQ-8 items
- SeverityLevel: Depression severity categories (paper Section 2.1)
- EvaluationMetric: Qualitative assessment metrics (paper Appendix B)
- AssessmentMode: Quantitative assessment modes (zero-shot vs few-shot)
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class PHQ8Item(StrEnum):
    """PHQ-8 assessment items (DSM-5 criteria).

    The PHQ-8 assesses eight depression symptoms over the past two weeks.
    Values match the legacy as-is repo format for parity audits.
    """

    NO_INTEREST = "NoInterest"
    """Little interest or pleasure in doing things (anhedonia)."""

    DEPRESSED = "Depressed"
    """Feeling down, depressed, or hopeless."""

    SLEEP = "Sleep"
    """Trouble falling/staying asleep, or sleeping too much."""

    TIRED = "Tired"
    """Feeling tired or having little energy."""

    APPETITE = "Appetite"
    """Poor appetite or overeating."""

    FAILURE = "Failure"
    """Feeling bad about yourself — or that you are a failure."""

    CONCENTRATING = "Concentrating"
    """Trouble concentrating on things."""

    MOVING = "Moving"
    """Moving/speaking slowly, or being fidgety/restless."""

    @classmethod
    def all_items(cls) -> list[PHQ8Item]:
        """Return all PHQ-8 items in order.

        Returns:
            Ordered list of all 8 PHQ-8 items.
        """
        return list(cls)


class PHQ8Score(IntEnum):
    """PHQ-8 item score (frequency over past 2 weeks).

    Each PHQ-8 item is scored 0-3 based on how often the symptom
    occurred during the past two weeks.
    """

    NOT_AT_ALL = 0
    """Not at all (0-1 days)."""

    SEVERAL_DAYS = 1
    """Several days (2-6 days)."""

    MORE_THAN_HALF = 2
    """More than half the days (7-11 days)."""

    NEARLY_EVERY_DAY = 3
    """Nearly every day (12-14 days)."""

    @classmethod
    def from_int(cls, value: int) -> PHQ8Score:
        """Create score from integer, clamping to valid range.

        Args:
            value: Integer value to convert (will be clamped to 0-3).

        Returns:
            PHQ8Score corresponding to the clamped value.
        """
        clamped = max(0, min(3, value))
        return cls(clamped)


class SeverityLevel(IntEnum):
    """Depression severity based on PHQ-8 total score.

    Severity categories per paper Section 2.1:
    - Minimal (0-4): No significant depressive symptoms
    - Mild (5-9): Mild depressive symptoms
    - Moderate (10-14): Moderate symptoms, MDD threshold
    - Moderately Severe (15-19): Moderately severe symptoms
    - Severe (20-24): Severe symptoms
    """

    MINIMAL = 0
    """No significant symptoms (total 0-4)."""

    MILD = 1
    """Mild symptoms (total 5-9)."""

    MODERATE = 2
    """Moderate symptoms (total 10-14). MDD threshold."""

    MOD_SEVERE = 3
    """Moderately severe symptoms (total 15-19)."""

    SEVERE = 4
    """Severe symptoms (total 20-24)."""

    @classmethod
    def from_total_score(cls, total: int) -> SeverityLevel:
        """Determine severity from total PHQ-8 score.

        Uses paper thresholds:
        - 0-4: Minimal
        - 5-9: Mild
        - 10-14: Moderate
        - 15-19: Moderately Severe
        - 20-24: Severe

        Args:
            total: Total PHQ-8 score (0-24, clamped if outside range).

        Returns:
            SeverityLevel corresponding to the total score.
        """
        if total <= 4:
            return cls.MINIMAL
        if total <= 9:
            return cls.MILD
        if total <= 14:
            return cls.MODERATE
        if total <= 19:
            return cls.MOD_SEVERE
        return cls.SEVERE

    @property
    def is_mdd(self) -> bool:
        """Check if severity indicates Major Depressive Disorder.

        MDD threshold is PHQ-8 total score >= 10, which corresponds
        to MODERATE severity or higher.

        Returns:
            True if severity is MODERATE or higher.
        """
        return self >= SeverityLevel.MODERATE


class EvaluationMetric(StrEnum):
    """Qualitative assessment evaluation metrics.

    Metrics used by the judge agent to evaluate qualitative assessments
    (paper Appendix B). Each metric is scored 1-5 on a Likert scale.
    """

    COHERENCE = "coherence"
    """Logical consistency of the assessment."""

    COMPLETENESS = "completeness"
    """Coverage of all relevant symptoms and frequencies."""

    SPECIFICITY = "specificity"
    """Avoidance of vague or generic statements."""

    ACCURACY = "accuracy"
    """Alignment with PHQ-8/DSM-5 criteria."""

    @classmethod
    def all_metrics(cls) -> list[EvaluationMetric]:
        """Return all evaluation metrics.

        Returns:
            List of all 4 evaluation metrics.
        """
        return list(cls)


class NAReason(StrEnum):
    """Reason for N/A (unable to assess) score.

    Used for debugging extraction failures and comparing
    backfill-enabled vs backfill-disabled runs.
    """

    NO_MENTION = "no_mention"
    """Neither LLM nor keywords found any evidence."""

    LLM_ONLY_MISSED = "llm_only_missed"
    """LLM missed evidence that keywords would have found (backfill disabled)."""

    KEYWORDS_INSUFFICIENT = "keywords_insufficient"
    """Keywords matched but still insufficient for scoring."""

    SCORE_NA_WITH_EVIDENCE = "score_na_with_evidence"
    """Evidence exists (LLM and/or keyword) but scorer returned N/A (abstained)."""


class AssessmentMode(StrEnum):
    """Quantitative assessment mode.

    Determines whether the quantitative assessment agent uses
    zero-shot or embedding-based few-shot prompting.
    """

    ZERO_SHOT = "zero_shot"
    """No reference examples provided to the LLM."""

    FEW_SHOT = "few_shot"
    """Embedding-based retrieval of reference examples."""
