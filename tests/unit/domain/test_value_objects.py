"""Tests for domain value objects.

Value objects are immutable (frozen) dataclasses that represent
domain concepts without identity.
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.domain.enums import EvaluationMetric, NAReason, PHQ8Item
from ai_psychiatrist.domain.value_objects import (
    EmbeddedChunk,
    EvaluationScore,
    Evidence,
    ItemAssessment,
    SimilarityMatch,
    TranscriptChunk,
)

pytestmark = pytest.mark.unit


class TestTranscriptChunk:
    """Tests for TranscriptChunk value object."""

    def test_create_valid_chunk(self) -> None:
        """Should create chunk with valid data."""
        chunk = TranscriptChunk(
            text="I have trouble sleeping at night.",
            participant_id=123,
            line_start=10,
            line_end=15,
        )
        assert chunk.text == "I have trouble sleeping at night."
        assert chunk.participant_id == 123
        assert chunk.line_start == 10
        assert chunk.line_end == 15

    def test_default_line_numbers(self) -> None:
        """Line numbers should default to 0."""
        chunk = TranscriptChunk(
            text="Some text",
            participant_id=1,
        )
        assert chunk.line_start == 0
        assert chunk.line_end == 0

    def test_reject_empty_text(self) -> None:
        """Should reject empty or whitespace-only text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TranscriptChunk(text="", participant_id=1)

        with pytest.raises(ValueError, match="cannot be empty"):
            TranscriptChunk(text="   ", participant_id=1)

        with pytest.raises(ValueError, match="cannot be empty"):
            TranscriptChunk(text="\n\t", participant_id=1)

    def test_reject_non_positive_participant_id(self) -> None:
        """Should reject participant_id <= 0."""
        with pytest.raises(ValueError, match="must be positive"):
            TranscriptChunk(text="valid text", participant_id=0)

        with pytest.raises(ValueError, match="must be positive"):
            TranscriptChunk(text="valid text", participant_id=-1)

    def test_word_count(self) -> None:
        """word_count should return correct count."""
        chunk = TranscriptChunk(
            text="one two three four five",
            participant_id=1,
        )
        assert chunk.word_count == 5

    def test_word_count_with_extra_whitespace(self) -> None:
        """word_count should handle multiple spaces."""
        chunk = TranscriptChunk(
            text="one  two   three",
            participant_id=1,
        )
        assert chunk.word_count == 3

    def test_is_immutable(self) -> None:
        """TranscriptChunk should be immutable (frozen)."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        with pytest.raises(AttributeError):
            chunk.text = "new text"  # type: ignore[misc]

    def test_uses_slots(self) -> None:
        """TranscriptChunk should use slots for memory efficiency."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        assert hasattr(chunk, "__slots__") or not hasattr(chunk, "__dict__")


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk value object."""

    def test_create_valid_embedded_chunk(self) -> None:
        """Should create embedded chunk with chunk and embedding."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        embedding = (0.1, 0.2, 0.3, 0.4)
        embedded = EmbeddedChunk(chunk=chunk, embedding=embedding)

        assert embedded.chunk is chunk
        assert embedded.embedding == embedding

    def test_dimension_calculated_from_embedding(self) -> None:
        """dimension should be set from embedding length."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        embedding = tuple(range(768))
        embedded = EmbeddedChunk(chunk=chunk, embedding=embedding)

        assert embedded.dimension == 768

    def test_dimension_small_embedding(self) -> None:
        """dimension should work with small embeddings."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        embedding = (1.0, 2.0)
        embedded = EmbeddedChunk(chunk=chunk, embedding=embedding)

        assert embedded.dimension == 2

    def test_participant_id_delegated_to_chunk(self) -> None:
        """participant_id should return chunk's participant_id."""
        chunk = TranscriptChunk(text="test", participant_id=42)
        embedded = EmbeddedChunk(chunk=chunk, embedding=(1.0,))

        assert embedded.participant_id == 42

    def test_is_immutable(self) -> None:
        """EmbeddedChunk should be immutable (frozen)."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        embedded = EmbeddedChunk(chunk=chunk, embedding=(1.0,))
        with pytest.raises(AttributeError):
            embedded.embedding = (2.0,)  # type: ignore[misc]


class TestEvidence:
    """Tests for Evidence value object."""

    def test_create_valid_evidence(self) -> None:
        """Should create evidence with quotes and item."""
        evidence = Evidence(
            quotes=("I don't enjoy things anymore", "Nothing brings me joy"),
            item=PHQ8Item.NO_INTEREST,
            source_participant_id=123,
        )
        assert len(evidence.quotes) == 2
        assert evidence.item == PHQ8Item.NO_INTEREST
        assert evidence.source_participant_id == 123

    def test_source_participant_id_optional(self) -> None:
        """source_participant_id should be optional."""
        evidence = Evidence(
            quotes=("test quote",),
            item=PHQ8Item.SLEEP,
        )
        assert evidence.source_participant_id is None

    def test_empty_factory(self) -> None:
        """empty() should create evidence with no quotes."""
        evidence = Evidence.empty(PHQ8Item.TIRED)
        assert evidence.quotes == ()
        assert evidence.item == PHQ8Item.TIRED
        assert evidence.source_participant_id is None

    def test_has_evidence_with_quotes(self) -> None:
        """has_evidence should be True when quotes exist."""
        evidence = Evidence(quotes=("some quote",), item=PHQ8Item.APPETITE)
        assert evidence.has_evidence is True

    def test_has_evidence_without_quotes(self) -> None:
        """has_evidence should be False when no quotes."""
        evidence = Evidence.empty(PHQ8Item.FAILURE)
        assert evidence.has_evidence is False

    def test_is_immutable(self) -> None:
        """Evidence should be immutable (frozen)."""
        evidence = Evidence(quotes=("test",), item=PHQ8Item.MOVING)
        with pytest.raises(AttributeError):
            evidence.item = PHQ8Item.SLEEP  # type: ignore[misc]


class TestItemAssessment:
    """Tests for ItemAssessment value object."""

    def test_create_valid_assessment(self) -> None:
        """Should create assessment with all fields."""
        assessment = ItemAssessment(
            item=PHQ8Item.DEPRESSED,
            evidence="Patient said 'I feel down most days'",
            reason="Direct statement of depressed mood",
            score=2,
        )
        assert assessment.item == PHQ8Item.DEPRESSED
        assert "feel down" in assessment.evidence
        assert assessment.reason == "Direct statement of depressed mood"
        assert assessment.score == 2

    def test_score_can_be_none_for_na(self) -> None:
        """score can be None to represent N/A."""
        assessment = ItemAssessment(
            item=PHQ8Item.APPETITE,
            evidence="No discussion of appetite",
            reason="Topic not covered in interview",
            score=None,
        )
        assert assessment.score is None

    @pytest.mark.parametrize("valid_score", [0, 1, 2, 3])
    def test_valid_scores_accepted(self, valid_score: int) -> None:
        """PHQ-8 scores 0-3 should be accepted."""
        assessment = ItemAssessment(
            item=PHQ8Item.SLEEP,
            evidence="test",
            reason="test",
            score=valid_score,
        )
        assert assessment.score == valid_score

    @pytest.mark.parametrize("invalid_score", [-1, 4, 5, 100, -100])
    def test_invalid_scores_rejected(self, invalid_score: int) -> None:
        """Scores outside 0-3 should be rejected."""
        with pytest.raises(ValueError, match="Score must be 0-3"):
            ItemAssessment(
                item=PHQ8Item.SLEEP,
                evidence="test",
                reason="test",
                score=invalid_score,
            )

    def test_is_available_with_score(self) -> None:
        """is_available should be True when score is not None."""
        assessment = ItemAssessment(
            item=PHQ8Item.SLEEP,
            evidence="test",
            reason="test",
            score=1,
        )
        assert assessment.is_available is True

    def test_is_available_without_score(self) -> None:
        """is_available should be False when score is None (N/A)."""
        assessment = ItemAssessment(
            item=PHQ8Item.SLEEP,
            evidence="test",
            reason="test",
            score=None,
        )
        assert assessment.is_available is False

    def test_score_value_with_score(self) -> None:
        """score_value should return score when available."""
        assessment = ItemAssessment(
            item=PHQ8Item.TIRED,
            evidence="test",
            reason="test",
            score=3,
        )
        assert assessment.score_value == 3

    def test_score_value_defaults_to_zero_for_na(self) -> None:
        """score_value should return 0 when score is None."""
        assessment = ItemAssessment(
            item=PHQ8Item.TIRED,
            evidence="test",
            reason="test",
            score=None,
        )
        assert assessment.score_value == 0

    def test_is_immutable(self) -> None:
        """ItemAssessment should be immutable (frozen)."""
        assessment = ItemAssessment(
            item=PHQ8Item.CONCENTRATING,
            evidence="test",
            reason="test",
            score=1,
        )
        with pytest.raises(AttributeError):
            assessment.score = 2  # type: ignore[misc]

    def test_item_assessment_extended_fields(self) -> None:
        """ItemAssessment should support NA reason and evidence tracking."""
        item = ItemAssessment(
            item=PHQ8Item.TIRED,
            score=None,
            evidence="",
            reason="No evidence found",
            na_reason=NAReason.NO_MENTION,
            evidence_source=None,
            llm_evidence_count=0,
            keyword_evidence_count=0,
        )
        assert item.na_reason == NAReason.NO_MENTION
        assert item.evidence_source is None
        assert item.llm_evidence_count == 0
        assert item.keyword_evidence_count == 0


class TestEvaluationScore:
    """Tests for EvaluationScore value object."""

    def test_create_valid_score(self) -> None:
        """Should create score with valid data."""
        score = EvaluationScore(
            metric=EvaluationMetric.COHERENCE,
            score=4,
            explanation="Assessment is logically consistent",
        )
        assert score.metric == EvaluationMetric.COHERENCE
        assert score.score == 4
        assert "logically consistent" in score.explanation

    @pytest.mark.parametrize("valid_score", [1, 2, 3, 4, 5])
    def test_valid_scores_accepted(self, valid_score: int) -> None:
        """Scores 1-5 should be accepted."""
        score = EvaluationScore(
            metric=EvaluationMetric.ACCURACY,
            score=valid_score,
            explanation="test",
        )
        assert score.score == valid_score

    @pytest.mark.parametrize("invalid_score", [0, -1, 6, 10, -100])
    def test_invalid_scores_rejected(self, invalid_score: int) -> None:
        """Scores outside 1-5 should be rejected."""
        with pytest.raises(ValueError, match="Score must be 1-5"):
            EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS,
                score=invalid_score,
                explanation="test",
            )

    @pytest.mark.parametrize("low_score", [1, 2, 3])
    def test_is_low_for_scores_at_or_below_three(self, low_score: int) -> None:
        """is_low should be True for scores <= 3."""
        score = EvaluationScore(
            metric=EvaluationMetric.SPECIFICITY,
            score=low_score,
            explanation="test",
        )
        assert score.is_low is True

    @pytest.mark.parametrize("not_low_score", [4, 5])
    def test_is_low_false_for_scores_above_three(self, not_low_score: int) -> None:
        """is_low should be False for scores > 3."""
        score = EvaluationScore(
            metric=EvaluationMetric.SPECIFICITY,
            score=not_low_score,
            explanation="test",
        )
        assert score.is_low is False

    @pytest.mark.parametrize("acceptable_score", [4, 5])
    def test_is_acceptable_for_scores_four_and_five(self, acceptable_score: int) -> None:
        """is_acceptable should be True for scores >= 4."""
        score = EvaluationScore(
            metric=EvaluationMetric.ACCURACY,
            score=acceptable_score,
            explanation="test",
        )
        assert score.is_acceptable is True

    @pytest.mark.parametrize("not_acceptable_score", [1, 2, 3])
    def test_is_acceptable_false_below_four(self, not_acceptable_score: int) -> None:
        """is_acceptable should be False for scores < 4."""
        score = EvaluationScore(
            metric=EvaluationMetric.ACCURACY,
            score=not_acceptable_score,
            explanation="test",
        )
        assert score.is_acceptable is False

    def test_is_immutable(self) -> None:
        """EvaluationScore should be immutable (frozen)."""
        score = EvaluationScore(
            metric=EvaluationMetric.COHERENCE,
            score=5,
            explanation="test",
        )
        with pytest.raises(AttributeError):
            score.score = 3  # type: ignore[misc]


class TestSimilarityMatch:
    """Tests for SimilarityMatch value object."""

    def test_create_valid_match(self) -> None:
        """Should create match with valid data."""
        chunk = TranscriptChunk(text="test text", participant_id=1)
        match = SimilarityMatch(
            chunk=chunk,
            similarity=0.85,
            reference_score=2,
        )
        assert match.chunk is chunk
        assert match.similarity == 0.85
        assert match.reference_score == 2

    def test_reference_score_optional(self) -> None:
        """reference_score should be optional."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        match = SimilarityMatch(chunk=chunk, similarity=0.5)
        assert match.reference_score is None

    @pytest.mark.parametrize("valid_sim", [0.0, 0.5, 1.0, 0.001, 0.999])
    def test_valid_similarities_accepted(self, valid_sim: float) -> None:
        """Similarities 0-1 should be accepted."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        match = SimilarityMatch(chunk=chunk, similarity=valid_sim)
        assert match.similarity == valid_sim

    @pytest.mark.parametrize("invalid_sim", [-0.1, -1.0, 1.1, 2.0, 100.0])
    def test_invalid_similarities_rejected(self, invalid_sim: float) -> None:
        """Similarities outside 0-1 should be rejected."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        with pytest.raises(ValueError, match="Similarity must be 0-1"):
            SimilarityMatch(chunk=chunk, similarity=invalid_sim)

    def test_is_immutable(self) -> None:
        """SimilarityMatch should be immutable (frozen)."""
        chunk = TranscriptChunk(text="test", participant_id=1)
        match = SimilarityMatch(chunk=chunk, similarity=0.5)
        with pytest.raises(AttributeError):
            match.similarity = 0.9  # type: ignore[misc]
