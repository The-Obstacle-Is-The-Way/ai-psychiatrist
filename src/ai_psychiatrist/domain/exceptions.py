"""Domain-specific exceptions for AI Psychiatrist.

This module defines a hierarchical exception system for domain errors:

    DomainError (base)
    ├── ValidationError
    ├── TranscriptError
    │   └── EmptyTranscriptError
    ├── AssessmentError
    │   ├── PHQ8ItemError
    │   └── InsufficientEvidenceError
    ├── EvaluationError
    │   ├── LowScoreError
    │   └── MaxIterationsError
    ├── LLMError
    │   ├── LLMResponseParseError
    │   └── LLMTimeoutError
    └── EmbeddingError
        └── EmbeddingDimensionMismatchError
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_psychiatrist.domain.enums import EvaluationMetric, PHQ8Item


class DomainError(Exception):
    """Base class for domain errors.

    All domain-specific exceptions should inherit from this class
    to allow for catching all domain errors with a single except clause.
    """


class ValidationError(DomainError):
    """Raised when domain validation fails.

    Used for general validation errors that don't fit into more
    specific exception categories.
    """


class TranscriptError(DomainError):
    """Errors related to transcript processing.

    Base class for transcript-specific errors such as empty
    or malformed transcript data.
    """


class EmptyTranscriptError(TranscriptError):
    """Raised when transcript is empty or invalid.

    Indicates that a transcript was provided but contains no
    usable content (empty string, whitespace only, etc.).
    """


class AssessmentError(DomainError):
    """Errors during assessment.

    Base class for errors that occur during PHQ-8 assessment
    processing, including item-specific errors.
    """


class PHQ8ItemError(AssessmentError):
    """Errors related to PHQ-8 items.

    Used when an error is specific to a particular PHQ-8 item,
    such as invalid scores or parsing errors.
    """

    def __init__(self, item: PHQ8Item, message: str) -> None:
        """Initialize with the affected item and error message.

        Args:
            item: The PHQ8Item that caused the error.
            message: Description of what went wrong.
        """
        self.item = item
        super().__init__(f"PHQ-8 {item.value}: {message}")


class InsufficientEvidenceError(AssessmentError):
    """Raised when insufficient evidence for assessment.

    Indicates that the transcript does not contain enough
    information to assess a particular PHQ-8 item.
    """

    def __init__(self, item: PHQ8Item) -> None:
        """Initialize with the item lacking evidence.

        Args:
            item: The PHQ8Item that lacks sufficient evidence.
        """
        self.item = item
        super().__init__(f"Insufficient evidence for {item.value}")


class EvaluationError(DomainError):
    """Errors during evaluation.

    Base class for errors that occur during qualitative
    assessment evaluation by the judge agent.
    """


class LowScoreError(EvaluationError):
    """Raised when evaluation score is too low.

    Indicates that an evaluation metric scored below the
    acceptable threshold (typically < 4 on the 1-5 scale).
    """

    def __init__(self, metric: EvaluationMetric, score: int) -> None:
        """Initialize with the metric and its score.

        Args:
            metric: The EvaluationMetric that scored low.
            score: The actual score (1-5) received.
        """
        self.metric = metric
        self.score = score
        super().__init__(f"{metric.value} scored {score}/5, below acceptable threshold")


class MaxIterationsError(EvaluationError):
    """Raised when max feedback iterations reached.

    Indicates that the feedback loop between the qualitative
    assessment agent and judge agent has exceeded the maximum
    number of iterations without achieving acceptable scores.
    """

    def __init__(self, iterations: int) -> None:
        """Initialize with the iteration count.

        Args:
            iterations: The maximum number of iterations attempted.
        """
        self.iterations = iterations
        super().__init__(f"Max iterations ({iterations}) reached without acceptable scores")


class LLMError(DomainError):
    """Errors from LLM interactions.

    Base class for errors that occur during communication
    with the language model backend.
    """


class LLMResponseParseError(LLMError):
    """Raised when LLM response cannot be parsed.

    Indicates that the LLM returned a response that could not
    be parsed according to the expected format (XML, JSON, etc.).
    """

    def __init__(self, raw_response: str, parse_error: str) -> None:
        """Initialize with the raw response and parse error.

        Args:
            raw_response: The raw text returned by the LLM.
            parse_error: Description of the parsing failure.
        """
        self.raw_response = raw_response
        self.parse_error = parse_error
        super().__init__(f"Failed to parse LLM response: {parse_error}")


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out.

    Indicates that the LLM did not respond within the
    configured timeout period.
    """

    def __init__(self, timeout_seconds: int) -> None:
        """Initialize with the timeout duration.

        Args:
            timeout_seconds: The timeout duration in seconds.
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(f"LLM request timed out after {timeout_seconds}s")


class EmbeddingError(DomainError):
    """Errors during embedding operations.

    Base class for errors that occur during transcript
    embedding or similarity search.
    """


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match.

    Indicates that two embeddings being compared have
    different dimensions, which prevents similarity calculation.
    """

    def __init__(self, expected: int, actual: int) -> None:
        """Initialize with expected and actual dimensions.

        Args:
            expected: The expected embedding dimension.
            actual: The actual embedding dimension received.
        """
        self.expected = expected
        self.actual = actual
        super().__init__(f"Embedding dimension mismatch: expected {expected}, got {actual}")
