"""Tests for domain enumerations.

Tests verify PHQ-8 items, scores, severity levels, and evaluation metrics
match the paper specifications (Section 2.1, 2.3, Appendix B).
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.domain.enums import (
    AssessmentMode,
    EvaluationMetric,
    PHQ8Item,
    PHQ8Score,
    SeverityLevel,
)


class TestPHQ8Item:
    """Tests for PHQ8Item enum (DSM-IV criteria, 8 items)."""

    def test_has_exactly_eight_items(self) -> None:
        """PHQ-8 must have exactly 8 items per DSM-IV criteria."""
        assert len(PHQ8Item) == 8

    def test_all_items_returns_list_of_eight(self) -> None:
        """all_items() should return ordered list of all 8 items."""
        items = PHQ8Item.all_items()
        assert len(items) == 8
        assert all(isinstance(item, PHQ8Item) for item in items)

    def test_item_values_match_legacy_format(self) -> None:
        """Item values must match legacy item names used in keys (PHQ8_{value})."""
        expected_values = [
            "NoInterest",
            "Depressed",
            "Sleep",
            "Tired",
            "Appetite",
            "Failure",
            "Concentrating",
            "Moving",
        ]
        actual_values = [item.value for item in PHQ8Item.all_items()]
        assert actual_values == expected_values

    def test_no_interest_is_anhedonia(self) -> None:
        """NO_INTEREST represents anhedonia (little interest or pleasure)."""
        assert PHQ8Item.NO_INTEREST.value == "NoInterest"

    def test_depressed_is_mood(self) -> None:
        """DEPRESSED represents depressed mood."""
        assert PHQ8Item.DEPRESSED.value == "Depressed"

    def test_sleep_is_sleep_problems(self) -> None:
        """SLEEP represents sleep disturbances."""
        assert PHQ8Item.SLEEP.value == "Sleep"

    def test_tired_is_fatigue(self) -> None:
        """TIRED represents fatigue/low energy."""
        assert PHQ8Item.TIRED.value == "Tired"

    def test_appetite_is_eating_changes(self) -> None:
        """APPETITE represents appetite/weight changes."""
        assert PHQ8Item.APPETITE.value == "Appetite"

    def test_failure_is_negative_self_perception(self) -> None:
        """FAILURE represents negative self-perception (worthlessness)."""
        assert PHQ8Item.FAILURE.value == "Failure"

    def test_concentrating_is_focus_problems(self) -> None:
        """CONCENTRATING represents concentration difficulties."""
        assert PHQ8Item.CONCENTRATING.value == "Concentrating"

    def test_moving_is_psychomotor_changes(self) -> None:
        """MOVING represents psychomotor changes (agitation/retardation)."""
        assert PHQ8Item.MOVING.value == "Moving"

    def test_is_string_enum(self) -> None:
        """PHQ8Item should be usable as a string."""
        assert str(PHQ8Item.NO_INTEREST) == "NoInterest"
        assert f"PHQ8_{PHQ8Item.SLEEP}" == "PHQ8_Sleep"


class TestPHQ8Score:
    """Tests for PHQ8Score enum (frequency over past 2 weeks)."""

    def test_has_four_levels(self) -> None:
        """PHQ-8 scores have 4 levels (0-3)."""
        assert len(PHQ8Score) == 4

    def test_not_at_all_is_zero(self) -> None:
        """NOT_AT_ALL (0-1 days) equals 0."""
        assert PHQ8Score.NOT_AT_ALL == 0
        assert int(PHQ8Score.NOT_AT_ALL) == 0

    def test_several_days_is_one(self) -> None:
        """SEVERAL_DAYS (2-6 days) equals 1."""
        assert PHQ8Score.SEVERAL_DAYS == 1

    def test_more_than_half_is_two(self) -> None:
        """MORE_THAN_HALF (7-11 days) equals 2."""
        assert PHQ8Score.MORE_THAN_HALF == 2

    def test_nearly_every_day_is_three(self) -> None:
        """NEARLY_EVERY_DAY (12-14 days) equals 3."""
        assert PHQ8Score.NEARLY_EVERY_DAY == 3

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, PHQ8Score.NOT_AT_ALL),
            (1, PHQ8Score.SEVERAL_DAYS),
            (2, PHQ8Score.MORE_THAN_HALF),
            (3, PHQ8Score.NEARLY_EVERY_DAY),
        ],
    )
    def test_from_int_valid_values(self, value: int, expected: PHQ8Score) -> None:
        """from_int() should convert valid integers to scores."""
        assert PHQ8Score.from_int(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (-1, PHQ8Score.NOT_AT_ALL),
            (-100, PHQ8Score.NOT_AT_ALL),
        ],
    )
    def test_from_int_clamps_negative_to_zero(self, value: int, expected: PHQ8Score) -> None:
        """from_int() should clamp negative values to 0."""
        assert PHQ8Score.from_int(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (4, PHQ8Score.NEARLY_EVERY_DAY),
            (100, PHQ8Score.NEARLY_EVERY_DAY),
        ],
    )
    def test_from_int_clamps_high_to_three(self, value: int, expected: PHQ8Score) -> None:
        """from_int() should clamp values above 3 to 3."""
        assert PHQ8Score.from_int(value) == expected


class TestSeverityLevel:
    """Tests for SeverityLevel enum (paper Section 2.1 thresholds)."""

    def test_has_five_levels(self) -> None:
        """Severity has 5 levels per paper."""
        assert len(SeverityLevel) == 5

    def test_minimal_is_zero(self) -> None:
        """MINIMAL (no significant symptoms) equals 0."""
        assert SeverityLevel.MINIMAL == 0

    def test_mild_is_one(self) -> None:
        """MILD equals 1."""
        assert SeverityLevel.MILD == 1

    def test_moderate_is_two(self) -> None:
        """MODERATE equals 2."""
        assert SeverityLevel.MODERATE == 2

    def test_mod_severe_is_three(self) -> None:
        """MOD_SEVERE equals 3."""
        assert SeverityLevel.MOD_SEVERE == 3

    def test_severe_is_four(self) -> None:
        """SEVERE equals 4."""
        assert SeverityLevel.SEVERE == 4

    @pytest.mark.parametrize(
        ("total", "expected"),
        [
            (0, SeverityLevel.MINIMAL),
            (1, SeverityLevel.MINIMAL),
            (2, SeverityLevel.MINIMAL),
            (3, SeverityLevel.MINIMAL),
            (4, SeverityLevel.MINIMAL),
            (5, SeverityLevel.MILD),
            (6, SeverityLevel.MILD),
            (7, SeverityLevel.MILD),
            (8, SeverityLevel.MILD),
            (9, SeverityLevel.MILD),
            (10, SeverityLevel.MODERATE),
            (11, SeverityLevel.MODERATE),
            (12, SeverityLevel.MODERATE),
            (13, SeverityLevel.MODERATE),
            (14, SeverityLevel.MODERATE),
            (15, SeverityLevel.MOD_SEVERE),
            (16, SeverityLevel.MOD_SEVERE),
            (17, SeverityLevel.MOD_SEVERE),
            (18, SeverityLevel.MOD_SEVERE),
            (19, SeverityLevel.MOD_SEVERE),
            (20, SeverityLevel.SEVERE),
            (21, SeverityLevel.SEVERE),
            (22, SeverityLevel.SEVERE),
            (23, SeverityLevel.SEVERE),
            (24, SeverityLevel.SEVERE),
        ],
    )
    def test_from_total_score_paper_thresholds(self, total: int, expected: SeverityLevel) -> None:
        """from_total_score() must match paper thresholds exactly."""
        assert SeverityLevel.from_total_score(total) == expected

    def test_from_total_score_above_max_is_severe(self) -> None:
        """Scores above 24 should still return SEVERE."""
        assert SeverityLevel.from_total_score(25) == SeverityLevel.SEVERE
        assert SeverityLevel.from_total_score(100) == SeverityLevel.SEVERE

    def test_from_total_score_negative_is_minimal(self) -> None:
        """Negative scores should return MINIMAL."""
        assert SeverityLevel.from_total_score(-1) == SeverityLevel.MINIMAL

    @pytest.mark.parametrize(
        ("severity", "is_mdd"),
        [
            (SeverityLevel.MINIMAL, False),
            (SeverityLevel.MILD, False),
            (SeverityLevel.MODERATE, True),
            (SeverityLevel.MOD_SEVERE, True),
            (SeverityLevel.SEVERE, True),
        ],
    )
    def test_is_mdd_threshold_at_moderate(self, severity: SeverityLevel, is_mdd: bool) -> None:
        """is_mdd should be True for MODERATE and above (>=10 total)."""
        assert severity.is_mdd == is_mdd

    def test_mdd_threshold_is_ten(self) -> None:
        """MDD threshold must be at total score 10 (MODERATE)."""
        assert SeverityLevel.from_total_score(9).is_mdd is False
        assert SeverityLevel.from_total_score(10).is_mdd is True


class TestEvaluationMetric:
    """Tests for EvaluationMetric enum (paper Appendix B)."""

    def test_has_four_metrics(self) -> None:
        """Should have exactly 4 evaluation metrics."""
        assert len(EvaluationMetric) == 4

    def test_all_metrics_returns_list_of_four(self) -> None:
        """all_metrics() should return all 4 metrics."""
        metrics = EvaluationMetric.all_metrics()
        assert len(metrics) == 4
        assert all(isinstance(m, EvaluationMetric) for m in metrics)

    def test_coherence_value(self) -> None:
        """COHERENCE represents logical consistency."""
        assert EvaluationMetric.COHERENCE.value == "coherence"

    def test_completeness_value(self) -> None:
        """COMPLETENESS represents coverage of symptoms."""
        assert EvaluationMetric.COMPLETENESS.value == "completeness"

    def test_specificity_value(self) -> None:
        """SPECIFICITY represents avoidance of vague statements."""
        assert EvaluationMetric.SPECIFICITY.value == "specificity"

    def test_accuracy_value(self) -> None:
        """ACCURACY represents alignment with PHQ-8/DSM-5."""
        assert EvaluationMetric.ACCURACY.value == "accuracy"

    def test_is_string_enum(self) -> None:
        """EvaluationMetric should be usable as a string."""
        assert str(EvaluationMetric.COHERENCE) == "coherence"


class TestAssessmentMode:
    """Tests for AssessmentMode enum."""

    def test_has_two_modes(self) -> None:
        """Should have exactly 2 assessment modes."""
        assert len(AssessmentMode) == 2

    def test_zero_shot_value(self) -> None:
        """ZERO_SHOT represents no reference examples."""
        assert AssessmentMode.ZERO_SHOT.value == "zero_shot"

    def test_few_shot_value(self) -> None:
        """FEW_SHOT represents embedding-based references."""
        assert AssessmentMode.FEW_SHOT.value == "few_shot"

    def test_is_string_enum(self) -> None:
        """AssessmentMode should be usable as a string."""
        assert str(AssessmentMode.ZERO_SHOT) == "zero_shot"
