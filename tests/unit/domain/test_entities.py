"""Tests for domain entities.

Entities have identity (UUID) and mutable state. They represent
the core business concepts of the AI Psychiatrist system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from ai_psychiatrist.domain.entities import (
    FullAssessment,
    MetaReview,
    PHQ8Assessment,
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import (
    AssessmentMode,
    EvaluationMetric,
    PHQ8Item,
    SeverityLevel,
)
from ai_psychiatrist.domain.value_objects import (
    EvaluationScore,
    ItemAssessment,
)


class TestTranscript:
    """Tests for Transcript entity."""

    def test_create_valid_transcript(self) -> None:
        """Should create transcript with valid data."""
        transcript = Transcript(participant_id=123, text="Hello world")
        assert transcript.participant_id == 123
        assert transcript.text == "Hello world"
        assert isinstance(transcript.id, UUID)

    def test_auto_generates_id(self) -> None:
        """Each transcript should get a unique UUID."""
        t1 = Transcript(participant_id=1, text="text 1")
        t2 = Transcript(participant_id=2, text="text 2")
        assert t1.id != t2.id

    def test_auto_generates_created_at(self) -> None:
        """created_at should default to current UTC time."""
        before = datetime.now(UTC)
        transcript = Transcript(participant_id=1, text="test")
        after = datetime.now(UTC)
        assert before <= transcript.created_at <= after

    def test_custom_id_and_timestamp(self) -> None:
        """Should accept custom ID and timestamp."""
        custom_id = UUID("12345678-1234-1234-1234-123456789abc")
        custom_time = datetime(2024, 1, 1, tzinfo=UTC)
        transcript = Transcript(
            participant_id=1,
            text="test",
            id=custom_id,
            created_at=custom_time,
        )
        assert transcript.id == custom_id
        assert transcript.created_at == custom_time

    def test_reject_empty_text(self) -> None:
        """Should reject empty transcript text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Transcript(participant_id=123, text="   ")

    def test_reject_invalid_participant_id(self) -> None:
        """Should reject non-positive participant ID."""
        with pytest.raises(ValueError, match="must be positive"):
            Transcript(participant_id=0, text="Hello")

        with pytest.raises(ValueError, match="must be positive"):
            Transcript(participant_id=-1, text="Hello")

    def test_word_count(self) -> None:
        """word_count should count words correctly."""
        transcript = Transcript(participant_id=1, text="one two three four")
        assert transcript.word_count == 4

    def test_line_count(self) -> None:
        """line_count should count lines correctly."""
        transcript = Transcript(
            participant_id=1,
            text="line one\nline two\nline three",
        )
        assert transcript.line_count == 3

    def test_line_count_handles_blank_lines(self) -> None:
        """line_count should count lines including blanks after stripping edges."""
        transcript = Transcript(
            participant_id=1,
            text="line one\n\nline two\n",
        )
        assert transcript.line_count == 3


class TestPHQ8Assessment:
    """Tests for PHQ8Assessment entity."""

    @pytest.fixture
    def complete_items(self) -> dict[PHQ8Item, ItemAssessment]:
        """Create complete item assessments with score=1 each."""
        return {
            item: ItemAssessment(
                item=item,
                evidence="Test evidence",
                reason="Test reason",
                score=1,
            )
            for item in PHQ8Item.all_items()
        }

    @pytest.fixture
    def all_na_items(self) -> dict[PHQ8Item, ItemAssessment]:
        """Create complete items with all N/A scores."""
        return {
            item: ItemAssessment(
                item=item,
                evidence="No evidence",
                reason="Topic not discussed",
                score=None,
            )
            for item in PHQ8Item.all_items()
        }

    def test_create_valid_assessment(self, complete_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """Should create assessment with all 8 items."""
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert len(assessment.items) == 8
        assert assessment.mode == AssessmentMode.ZERO_SHOT
        assert assessment.participant_id == 123
        assert isinstance(assessment.id, UUID)

    def test_reject_missing_items(self) -> None:
        """Should reject assessment with missing items."""
        partial_items = {
            PHQ8Item.NO_INTEREST: ItemAssessment(
                item=PHQ8Item.NO_INTEREST,
                evidence="Test",
                reason="Test",
                score=1,
            )
        }
        with pytest.raises(ValueError, match="Missing PHQ-8 items"):
            PHQ8Assessment(
                items=partial_items,
                mode=AssessmentMode.ZERO_SHOT,
                participant_id=123,
            )

    def test_total_score_all_ones(self, complete_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """total_score should sum all item scores."""
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.total_score == 8  # 8 items * 1

    def test_total_score_mixed_scores(self) -> None:
        """total_score should handle mixed scores."""
        items = {}
        for i, item in enumerate(PHQ8Item.all_items()):
            items[item] = ItemAssessment(
                item=item,
                evidence="Test",
                reason="Test",
                score=i % 4,  # 0, 1, 2, 3, 0, 1, 2, 3
            )
        assessment = PHQ8Assessment(
            items=items,
            mode=AssessmentMode.FEW_SHOT,
            participant_id=1,
        )
        assert assessment.total_score == 12  # 0+1+2+3+0+1+2+3

    def test_total_score_with_na_values(self, all_na_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """N/A scores should not contribute to total."""
        assessment = PHQ8Assessment(
            items=all_na_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.total_score == 0
        assert assessment.na_count == 8

    def test_severity_minimal(self) -> None:
        """severity should be MINIMAL for total 0-4."""
        items = {
            item: ItemAssessment(item=item, evidence="", reason="", score=0)
            for item in PHQ8Item.all_items()
        }
        assessment = PHQ8Assessment(items=items, mode=AssessmentMode.ZERO_SHOT, participant_id=1)
        assert assessment.severity == SeverityLevel.MINIMAL

    def test_severity_mild(self, complete_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """severity should be MILD for total 5-9."""
        # Total = 8 (8 items * 1) -> MILD
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.severity == SeverityLevel.MILD

    def test_severity_severe(self) -> None:
        """severity should be SEVERE for total 20-24."""
        items = {
            item: ItemAssessment(item=item, evidence="", reason="", score=3)
            for item in PHQ8Item.all_items()
        }
        assessment = PHQ8Assessment(items=items, mode=AssessmentMode.ZERO_SHOT, participant_id=1)
        assert assessment.severity == SeverityLevel.SEVERE  # total=24

    def test_available_count(self, complete_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """available_count should count non-N/A items."""
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=1,
        )
        assert assessment.available_count == 8
        assert assessment.na_count == 0

    def test_available_count_mixed(self) -> None:
        """available_count should handle mixed N/A and scored items."""
        items = {}
        for i, item in enumerate(PHQ8Item.all_items()):
            items[item] = ItemAssessment(
                item=item,
                evidence="Test",
                reason="Test",
                score=1 if i < 5 else None,  # 5 scored, 3 N/A
            )
        assessment = PHQ8Assessment(items=items, mode=AssessmentMode.ZERO_SHOT, participant_id=1)
        assert assessment.available_count == 5
        assert assessment.na_count == 3

    def test_get_item(self, complete_items: dict[PHQ8Item, ItemAssessment]) -> None:
        """get_item should return specific item assessment."""
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=1,
        )
        item_assessment = assessment.get_item(PHQ8Item.SLEEP)
        assert item_assessment.item == PHQ8Item.SLEEP


class TestQualitativeAssessment:
    """Tests for QualitativeAssessment entity."""

    def test_create_valid_assessment(self) -> None:
        """Should create assessment with all fields."""
        assessment = QualitativeAssessment(
            overall="Patient shows signs of moderate depression",
            phq8_symptoms="Sleep disturbances, low energy",
            social_factors="Social isolation, recent divorce",
            biological_factors="Family history of depression",
            risk_factors="Recent job loss, financial stress",
            participant_id=123,
        )
        assert "moderate depression" in assessment.overall
        assert assessment.participant_id == 123
        assert isinstance(assessment.id, UUID)

    @pytest.mark.parametrize("invalid_id", [0, -1, -100])
    def test_reject_non_positive_participant_id(self, invalid_id: int) -> None:
        """Should reject participant_id <= 0."""
        with pytest.raises(ValueError, match="must be positive"):
            QualitativeAssessment(
                overall="test",
                phq8_symptoms="test",
                social_factors="test",
                biological_factors="test",
                risk_factors="test",
                participant_id=invalid_id,
            )

    def test_supporting_quotes_default_empty(self) -> None:
        """supporting_quotes should default to empty list."""
        assessment = QualitativeAssessment(
            overall="test",
            phq8_symptoms="test",
            social_factors="test",
            biological_factors="test",
            risk_factors="test",
            participant_id=1,
        )
        assert assessment.supporting_quotes == []

    def test_supporting_quotes_provided(self) -> None:
        """Should store provided supporting quotes."""
        quotes = ["Quote 1", "Quote 2"]
        assessment = QualitativeAssessment(
            overall="test",
            phq8_symptoms="test",
            social_factors="test",
            biological_factors="test",
            risk_factors="test",
            participant_id=1,
            supporting_quotes=quotes,
        )
        assert assessment.supporting_quotes == quotes

    def test_full_text_property(self) -> None:
        """full_text should format all sections."""
        assessment = QualitativeAssessment(
            overall="Overall summary",
            phq8_symptoms="PHQ-8 symptoms",
            social_factors="Social factors",
            biological_factors="Biological factors",
            risk_factors="Risk factors",
            participant_id=1,
        )
        full_text = assessment.full_text
        assert "Overall Assessment:" in full_text
        assert "Overall summary" in full_text
        assert "PHQ-8 Symptoms:" in full_text
        assert "Social Factors:" in full_text
        assert "Biological Factors:" in full_text
        assert "Risk Factors:" in full_text


class TestQualitativeEvaluation:
    """Tests for QualitativeEvaluation entity."""

    @pytest.fixture
    def complete_scores(self) -> dict[EvaluationMetric, EvaluationScore]:
        """Create complete evaluation scores with score=4 each."""
        return {
            metric: EvaluationScore(
                metric=metric,
                score=4,
                explanation="Good",
            )
            for metric in EvaluationMetric.all_metrics()
        }

    @pytest.fixture
    def assessment_id(self) -> UUID:
        """Create a test assessment ID."""
        return UUID("12345678-1234-1234-1234-123456789abc")

    def test_create_valid_evaluation(
        self,
        complete_scores: dict[EvaluationMetric, EvaluationScore],
        assessment_id: UUID,
    ) -> None:
        """Should create evaluation with all metrics."""
        evaluation = QualitativeEvaluation(
            scores=complete_scores,
            assessment_id=assessment_id,
        )
        assert len(evaluation.scores) == 4
        assert evaluation.assessment_id == assessment_id
        assert evaluation.iteration == 0
        assert isinstance(evaluation.id, UUID)

    def test_reject_missing_metrics(self, assessment_id: UUID) -> None:
        """Should reject evaluation with missing metrics."""
        partial_scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=4, explanation="Good"
            )
        }
        with pytest.raises(ValueError, match="Missing evaluation metrics"):
            QualitativeEvaluation(
                scores=partial_scores,
                assessment_id=assessment_id,
            )

    def test_average_score(
        self,
        complete_scores: dict[EvaluationMetric, EvaluationScore],
        assessment_id: UUID,
    ) -> None:
        """average_score should calculate mean of all scores."""
        evaluation = QualitativeEvaluation(
            scores=complete_scores,
            assessment_id=assessment_id,
        )
        assert evaluation.average_score == 4.0

    def test_average_score_mixed(self, assessment_id: UUID) -> None:
        """average_score should handle mixed scores."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=5, explanation=""
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=3, explanation=""
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=4, explanation=""
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=4, explanation=""
            ),
        }
        evaluation = QualitativeEvaluation(scores=scores, assessment_id=assessment_id)
        # (5 + 3 + 4 + 4) / 4 = 4.0
        assert evaluation.average_score == 4.0

    def test_low_scores_detection(self, assessment_id: UUID) -> None:
        """low_scores should return metrics with score <= 3."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=5, explanation=""
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=2, explanation=""
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=3, explanation=""
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=1, explanation=""
            ),
        }
        evaluation = QualitativeEvaluation(scores=scores, assessment_id=assessment_id)
        low = evaluation.low_scores
        assert EvaluationMetric.COMPLETENESS in low
        assert EvaluationMetric.ACCURACY in low
        assert EvaluationMetric.COHERENCE not in low
        assert EvaluationMetric.SPECIFICITY in low

    def test_low_scores_for_threshold(self, assessment_id: UUID) -> None:
        """low_scores_for_threshold should respect provided cutoff."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=4, explanation=""
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=3, explanation=""
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=2, explanation=""
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=5, explanation=""
            ),
        }
        evaluation = QualitativeEvaluation(scores=scores, assessment_id=assessment_id)

        assert evaluation.low_scores_for_threshold(2) == [EvaluationMetric.SPECIFICITY]
        assert set(evaluation.low_scores_for_threshold(3)) == {
            EvaluationMetric.COMPLETENESS,
            EvaluationMetric.SPECIFICITY,
        }

    def test_needs_improvement(self, assessment_id: UUID) -> None:
        """needs_improvement should be True when any score is low."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=5, explanation=""
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=2, explanation=""
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=5, explanation=""
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=5, explanation=""
            ),
        }
        evaluation = QualitativeEvaluation(scores=scores, assessment_id=assessment_id)
        assert evaluation.needs_improvement is True

    def test_no_improvement_needed(
        self,
        complete_scores: dict[EvaluationMetric, EvaluationScore],
        assessment_id: UUID,
    ) -> None:
        """needs_improvement should be False when all scores are acceptable."""
        evaluation = QualitativeEvaluation(scores=complete_scores, assessment_id=assessment_id)
        assert evaluation.needs_improvement is False

    def test_all_acceptable(
        self,
        complete_scores: dict[EvaluationMetric, EvaluationScore],
        assessment_id: UUID,
    ) -> None:
        """all_acceptable should be True when all scores >= 4."""
        evaluation = QualitativeEvaluation(scores=complete_scores, assessment_id=assessment_id)
        assert evaluation.all_acceptable is True

    def test_not_all_acceptable(self, assessment_id: UUID) -> None:
        """all_acceptable should be False when any score < 4."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=5, explanation=""
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=3, explanation=""
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=5, explanation=""
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=5, explanation=""
            ),
        }
        evaluation = QualitativeEvaluation(scores=scores, assessment_id=assessment_id)
        assert evaluation.all_acceptable is False

    def test_get_score(
        self,
        complete_scores: dict[EvaluationMetric, EvaluationScore],
        assessment_id: UUID,
    ) -> None:
        """get_score should return specific metric score."""
        evaluation = QualitativeEvaluation(scores=complete_scores, assessment_id=assessment_id)
        score = evaluation.get_score(EvaluationMetric.COHERENCE)
        assert score.metric == EvaluationMetric.COHERENCE


class TestMetaReview:
    """Tests for MetaReview entity."""

    def test_create_valid_meta_review(self) -> None:
        """Should create meta review with all fields."""
        meta_review = MetaReview(
            severity=SeverityLevel.MODERATE,
            explanation="Based on quantitative and qualitative assessments...",
            quantitative_assessment_id=UUID("11111111-1111-1111-1111-111111111111"),
            qualitative_assessment_id=UUID("22222222-2222-2222-2222-222222222222"),
            participant_id=123,
        )
        assert meta_review.severity == SeverityLevel.MODERATE
        assert "quantitative and qualitative" in meta_review.explanation
        assert meta_review.participant_id == 123
        assert isinstance(meta_review.id, UUID)

    def test_is_mdd_property(self) -> None:
        """is_mdd should delegate to severity.is_mdd."""
        # Test MDD positive (moderate or higher)
        meta_moderate = MetaReview(
            severity=SeverityLevel.MODERATE,
            explanation="test",
            quantitative_assessment_id=UUID("11111111-1111-1111-1111-111111111111"),
            qualitative_assessment_id=UUID("22222222-2222-2222-2222-222222222222"),
            participant_id=1,
        )
        assert meta_moderate.is_mdd is True

        # Test MDD negative (mild)
        meta_mild = MetaReview(
            severity=SeverityLevel.MILD,
            explanation="test",
            quantitative_assessment_id=UUID("11111111-1111-1111-1111-111111111111"),
            qualitative_assessment_id=UUID("22222222-2222-2222-2222-222222222222"),
            participant_id=1,
        )
        assert meta_mild.is_mdd is False


class TestFullAssessment:
    """Tests for FullAssessment entity."""

    @pytest.fixture
    def transcript(self) -> Transcript:
        """Create a test transcript."""
        return Transcript(participant_id=123, text="Test transcript content")

    @pytest.fixture
    def phq8_assessment(self) -> PHQ8Assessment:
        """Create a test PHQ8Assessment."""
        items = {
            item: ItemAssessment(item=item, evidence="test", reason="test", score=1)
            for item in PHQ8Item.all_items()
        }
        return PHQ8Assessment(
            items=items,
            mode=AssessmentMode.FEW_SHOT,
            participant_id=123,
        )

    @pytest.fixture
    def qualitative_assessment(self) -> QualitativeAssessment:
        """Create a test QualitativeAssessment."""
        return QualitativeAssessment(
            overall="test",
            phq8_symptoms="test",
            social_factors="test",
            biological_factors="test",
            risk_factors="test",
            participant_id=123,
        )

    @pytest.fixture
    def qualitative_evaluation(
        self, qualitative_assessment: QualitativeAssessment
    ) -> QualitativeEvaluation:
        """Create a test QualitativeEvaluation."""
        scores = {
            metric: EvaluationScore(metric=metric, score=4, explanation="")
            for metric in EvaluationMetric.all_metrics()
        }
        return QualitativeEvaluation(
            scores=scores,
            assessment_id=qualitative_assessment.id,
        )

    @pytest.fixture
    def meta_review(
        self,
        phq8_assessment: PHQ8Assessment,
        qualitative_assessment: QualitativeAssessment,
    ) -> MetaReview:
        """Create a test MetaReview."""
        return MetaReview(
            severity=SeverityLevel.MILD,
            explanation="test",
            quantitative_assessment_id=phq8_assessment.id,
            qualitative_assessment_id=qualitative_assessment.id,
            participant_id=123,
        )

    def test_create_valid_full_assessment(
        self,
        transcript: Transcript,
        phq8_assessment: PHQ8Assessment,
        qualitative_assessment: QualitativeAssessment,
        qualitative_evaluation: QualitativeEvaluation,
        meta_review: MetaReview,
    ) -> None:
        """Should create full assessment with all components."""
        full = FullAssessment(
            transcript=transcript,
            quantitative=phq8_assessment,
            qualitative=qualitative_assessment,
            qualitative_evaluation=qualitative_evaluation,
            meta_review=meta_review,
        )
        assert full.transcript is transcript
        assert full.quantitative is phq8_assessment
        assert full.qualitative is qualitative_assessment
        assert full.qualitative_evaluation is qualitative_evaluation
        assert full.meta_review is meta_review
        assert isinstance(full.id, UUID)

    def test_participant_id_from_transcript(
        self,
        transcript: Transcript,
        phq8_assessment: PHQ8Assessment,
        qualitative_assessment: QualitativeAssessment,
        qualitative_evaluation: QualitativeEvaluation,
        meta_review: MetaReview,
    ) -> None:
        """participant_id should come from transcript."""
        full = FullAssessment(
            transcript=transcript,
            quantitative=phq8_assessment,
            qualitative=qualitative_assessment,
            qualitative_evaluation=qualitative_evaluation,
            meta_review=meta_review,
        )
        assert full.participant_id == transcript.participant_id

    def test_final_severity_from_meta_review(
        self,
        transcript: Transcript,
        phq8_assessment: PHQ8Assessment,
        qualitative_assessment: QualitativeAssessment,
        qualitative_evaluation: QualitativeEvaluation,
        meta_review: MetaReview,
    ) -> None:
        """final_severity should come from meta_review."""
        full = FullAssessment(
            transcript=transcript,
            quantitative=phq8_assessment,
            qualitative=qualitative_assessment,
            qualitative_evaluation=qualitative_evaluation,
            meta_review=meta_review,
        )
        assert full.final_severity == meta_review.severity
