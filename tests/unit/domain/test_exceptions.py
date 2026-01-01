"""Tests for domain exceptions.

Tests verify exception hierarchy and message formatting.
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.domain.enums import EvaluationMetric, PHQ8Item
from ai_psychiatrist.domain.exceptions import (
    AssessmentError,
    DomainError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmptyTranscriptError,
    EvaluationError,
    InsufficientEvidenceError,
    LLMError,
    LLMResponseParseError,
    LLMTimeoutError,
    LowScoreError,
    MaxIterationsError,
    PHQ8ItemError,
    TranscriptError,
    ValidationError,
)

pytestmark = pytest.mark.unit


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_domain_error_is_base(self) -> None:
        """DomainError should be the base exception."""
        assert issubclass(DomainError, Exception)

    def test_validation_error_inherits_domain(self) -> None:
        """ValidationError should inherit from DomainError."""
        assert issubclass(ValidationError, DomainError)

    def test_transcript_error_inherits_domain(self) -> None:
        """TranscriptError should inherit from DomainError."""
        assert issubclass(TranscriptError, DomainError)

    def test_empty_transcript_inherits_transcript(self) -> None:
        """EmptyTranscriptError should inherit from TranscriptError."""
        assert issubclass(EmptyTranscriptError, TranscriptError)

    def test_assessment_error_inherits_domain(self) -> None:
        """AssessmentError should inherit from DomainError."""
        assert issubclass(AssessmentError, DomainError)

    def test_phq8_item_error_inherits_assessment(self) -> None:
        """PHQ8ItemError should inherit from AssessmentError."""
        assert issubclass(PHQ8ItemError, AssessmentError)

    def test_insufficient_evidence_inherits_assessment(self) -> None:
        """InsufficientEvidenceError should inherit from AssessmentError."""
        assert issubclass(InsufficientEvidenceError, AssessmentError)

    def test_evaluation_error_inherits_domain(self) -> None:
        """EvaluationError should inherit from DomainError."""
        assert issubclass(EvaluationError, DomainError)

    def test_low_score_error_inherits_evaluation(self) -> None:
        """LowScoreError should inherit from EvaluationError."""
        assert issubclass(LowScoreError, EvaluationError)

    def test_max_iterations_inherits_evaluation(self) -> None:
        """MaxIterationsError should inherit from EvaluationError."""
        assert issubclass(MaxIterationsError, EvaluationError)

    def test_llm_error_inherits_domain(self) -> None:
        """LLMError should inherit from DomainError."""
        assert issubclass(LLMError, DomainError)

    def test_llm_parse_error_inherits_llm(self) -> None:
        """LLMResponseParseError should inherit from LLMError."""
        assert issubclass(LLMResponseParseError, LLMError)

    def test_llm_timeout_inherits_llm(self) -> None:
        """LLMTimeoutError should inherit from LLMError."""
        assert issubclass(LLMTimeoutError, LLMError)

    def test_embedding_error_inherits_domain(self) -> None:
        """EmbeddingError should inherit from DomainError."""
        assert issubclass(EmbeddingError, DomainError)

    def test_dimension_mismatch_inherits_embedding(self) -> None:
        """EmbeddingDimensionMismatchError should inherit from EmbeddingError."""
        assert issubclass(EmbeddingDimensionMismatchError, EmbeddingError)


class TestDomainError:
    """Tests for DomainError base class."""

    def test_can_raise_with_message(self) -> None:
        """DomainError should be raisable with a message."""
        with pytest.raises(DomainError, match="test message"):
            raise DomainError("test message")


class TestValidationError:
    """Tests for ValidationError."""

    def test_can_raise_with_message(self) -> None:
        """ValidationError should be raisable with a message."""
        with pytest.raises(ValidationError, match="invalid data"):
            raise ValidationError("invalid data")


class TestTranscriptError:
    """Tests for TranscriptError and subclasses."""

    def test_can_raise_with_message(self) -> None:
        """TranscriptError should be raisable with a message."""
        with pytest.raises(TranscriptError, match="transcript issue"):
            raise TranscriptError("transcript issue")

    def test_empty_transcript_error(self) -> None:
        """EmptyTranscriptError should be raisable."""
        with pytest.raises(EmptyTranscriptError, match="empty"):
            raise EmptyTranscriptError("empty")


class TestPHQ8ItemError:
    """Tests for PHQ8ItemError."""

    def test_stores_item(self) -> None:
        """PHQ8ItemError should store the PHQ8Item."""
        error = PHQ8ItemError(PHQ8Item.SLEEP, "score out of range")
        assert error.item == PHQ8Item.SLEEP

    def test_formats_message_with_item(self) -> None:
        """Message should include item value."""
        error = PHQ8ItemError(PHQ8Item.DEPRESSED, "invalid score")
        assert "PHQ-8 Depressed" in str(error)
        assert "invalid score" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(PHQ8ItemError, match="PHQ-8 NoInterest: test"):
            raise PHQ8ItemError(PHQ8Item.NO_INTEREST, "test")


class TestInsufficientEvidenceError:
    """Tests for InsufficientEvidenceError."""

    def test_stores_item(self) -> None:
        """InsufficientEvidenceError should store the PHQ8Item."""
        error = InsufficientEvidenceError(PHQ8Item.APPETITE)
        assert error.item == PHQ8Item.APPETITE

    def test_formats_message_with_item(self) -> None:
        """Message should describe insufficient evidence for item."""
        error = InsufficientEvidenceError(PHQ8Item.MOVING)
        assert "Insufficient evidence" in str(error)
        assert "Moving" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(InsufficientEvidenceError, match="Insufficient evidence"):
            raise InsufficientEvidenceError(PHQ8Item.TIRED)


class TestLowScoreError:
    """Tests for LowScoreError."""

    def test_stores_metric_and_score(self) -> None:
        """LowScoreError should store metric and score."""
        error = LowScoreError(EvaluationMetric.COHERENCE, 2)
        assert error.metric == EvaluationMetric.COHERENCE
        assert error.score == 2

    def test_formats_message(self) -> None:
        """Message should include metric and score."""
        error = LowScoreError(EvaluationMetric.COMPLETENESS, 1)
        assert "completeness" in str(error)
        assert "1/5" in str(error)
        assert "below acceptable threshold" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(LowScoreError, match="specificity scored 2/5"):
            raise LowScoreError(EvaluationMetric.SPECIFICITY, 2)


class TestMaxIterationsError:
    """Tests for MaxIterationsError."""

    def test_stores_iterations(self) -> None:
        """MaxIterationsError should store iteration count."""
        error = MaxIterationsError(10)
        assert error.iterations == 10

    def test_formats_message(self) -> None:
        """Message should include iteration count."""
        error = MaxIterationsError(5)
        assert "5" in str(error)
        assert "Max iterations" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(MaxIterationsError, match=r"Max iterations \(10\)"):
            raise MaxIterationsError(10)


class TestLLMResponseParseError:
    """Tests for LLMResponseParseError."""

    def test_stores_response_and_error(self) -> None:
        """LLMResponseParseError should store raw response and parse error."""
        error = LLMResponseParseError("<invalid>xml</invalid>", "missing closing tag")
        assert error.raw_response == "<invalid>xml</invalid>"
        assert error.parse_error == "missing closing tag"

    def test_formats_message(self) -> None:
        """Message should include parse error."""
        error = LLMResponseParseError("bad response", "JSON decode error")
        assert "Failed to parse LLM response" in str(error)
        assert "JSON decode error" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(LLMResponseParseError, match="Failed to parse"):
            raise LLMResponseParseError("response", "error")


class TestLLMTimeoutError:
    """Tests for LLMTimeoutError."""

    def test_stores_timeout(self) -> None:
        """LLMTimeoutError should store timeout duration."""
        error = LLMTimeoutError(30)
        assert error.timeout_seconds == 30

    def test_formats_message(self) -> None:
        """Message should include timeout duration."""
        error = LLMTimeoutError(60)
        assert "60s" in str(error)
        assert "timed out" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(LLMTimeoutError, match="timed out after 120s"):
            raise LLMTimeoutError(120)


class TestEmbeddingDimensionMismatchError:
    """Tests for EmbeddingDimensionMismatchError."""

    def test_stores_dimensions(self) -> None:
        """EmbeddingDimensionMismatchError should store expected and actual."""
        error = EmbeddingDimensionMismatchError(4096, 768)
        assert error.expected == 4096
        assert error.actual == 768

    def test_formats_message(self) -> None:
        """Message should include both dimensions."""
        error = EmbeddingDimensionMismatchError(1024, 512)
        assert "expected 1024" in str(error)
        assert "got 512" in str(error)
        assert "dimension mismatch" in str(error)

    def test_raises_with_correct_message(self) -> None:
        """Should be catchable with formatted message."""
        with pytest.raises(EmbeddingDimensionMismatchError, match="expected 256, got 128"):
            raise EmbeddingDimensionMismatchError(256, 128)


class TestExceptionsCatchable:
    """Integration tests verifying exceptions can be caught properly."""

    def test_catch_domain_error_catches_all_subtypes(self) -> None:
        """Catching DomainError should catch all domain exceptions."""
        exceptions_to_test = [
            ValidationError("test"),
            TranscriptError("test"),
            EmptyTranscriptError("test"),
            AssessmentError("test"),
            PHQ8ItemError(PHQ8Item.SLEEP, "test"),
            InsufficientEvidenceError(PHQ8Item.TIRED),
            EvaluationError("test"),
            LowScoreError(EvaluationMetric.ACCURACY, 1),
            MaxIterationsError(10),
            LLMError("test"),
            LLMResponseParseError("raw", "error"),
            LLMTimeoutError(30),
            EmbeddingError("test"),
            EmbeddingDimensionMismatchError(100, 50),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except DomainError as caught:
                assert caught is exc
