"""Immutable value objects for AI Psychiatrist domain.

Value objects are immutable (frozen) dataclasses that represent domain
concepts without identity. They are equal if all their attributes are equal.

All value objects use:
- frozen=True: Makes instances immutable
- slots=True: Optimizes memory usage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_psychiatrist.domain.enums import EvaluationMetric, PHQ8Item


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """A segment of an interview transcript.

    Represents a contiguous portion of text from an interview,
    used for embedding-based retrieval in few-shot prompting.
    """

    text: str
    """The text content of the chunk."""

    participant_id: int
    """Identifier of the interview participant."""

    line_start: int = 0
    """Starting line number in the original transcript (0-indexed)."""

    line_end: int = 0
    """Ending line number in the original transcript (0-indexed)."""

    def __post_init__(self) -> None:
        """Validate chunk data.

        Raises:
            ValueError: If text is empty or participant_id is non-positive.
        """
        if not self.text.strip():
            raise ValueError("Transcript chunk cannot be empty")
        if self.participant_id <= 0:
            raise ValueError("Participant ID must be positive")

    @property
    def word_count(self) -> int:
        """Count words in chunk.

        Returns:
            Number of whitespace-separated tokens in the text.
        """
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A transcript chunk with its embedding vector.

    Associates a TranscriptChunk with its vector embedding for
    similarity search during few-shot retrieval.
    """

    chunk: TranscriptChunk
    """The underlying transcript chunk."""

    embedding: tuple[float, ...]
    """The embedding vector (immutable tuple for hashability)."""

    dimension: int = field(init=False)
    """The dimensionality of the embedding (calculated from embedding)."""

    def __post_init__(self) -> None:
        """Set dimension from embedding length.

        Note: Uses object.__setattr__ because the dataclass is frozen.
        """
        object.__setattr__(self, "dimension", len(self.embedding))

    @property
    def participant_id(self) -> int:
        """Get participant ID from the underlying chunk.

        Returns:
            The participant_id from the wrapped TranscriptChunk.
        """
        return self.chunk.participant_id


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence supporting a PHQ-8 item score.

    Collects direct quotes from the transcript that support
    the assessment of a specific PHQ-8 symptom.
    """

    quotes: tuple[str, ...]
    """Direct quotes from the transcript supporting this item."""

    item: PHQ8Item
    """The PHQ-8 item this evidence relates to."""

    source_participant_id: int | None = None
    """Participant ID if evidence is from a reference transcript."""

    @classmethod
    def empty(cls, item: PHQ8Item) -> Evidence:
        """Create empty evidence for an item.

        Factory method for when no evidence is available.

        Args:
            item: The PHQ8Item to create empty evidence for.

        Returns:
            An Evidence instance with no quotes.
        """
        return cls(quotes=(), item=item)

    @property
    def has_evidence(self) -> bool:
        """Check if any evidence exists.

        Returns:
            True if at least one quote is present.
        """
        return len(self.quotes) > 0


@dataclass(frozen=True, slots=True)
class ItemAssessment:
    """Assessment result for a single PHQ-8 item.

    Contains the LLM's assessment of one PHQ-8 symptom,
    including the evidence found, reasoning, and score.
    """

    item: PHQ8Item
    """The PHQ-8 item being assessed."""

    evidence: str
    """Evidence from the transcript (direct quotes or summary)."""

    reason: str
    """Reasoning for the assigned score."""

    score: int | None
    """Score (0-3) or None for N/A (insufficient evidence)."""

    @property
    def is_available(self) -> bool:
        """Check if score is available (not N/A).

        Returns:
            True if a numeric score was assigned.
        """
        return self.score is not None

    @property
    def score_value(self) -> int:
        """Get score value, defaulting to 0 for N/A.

        Used when calculating total scores where N/A
        should not contribute to the sum.

        Returns:
            The score if available, otherwise 0.
        """
        return self.score if self.score is not None else 0


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    """Score for a single evaluation metric.

    Represents the judge agent's assessment of one quality
    metric for a qualitative assessment.
    """

    metric: EvaluationMetric
    """The evaluation metric being scored."""

    score: int
    """Score on 1-5 Likert scale (higher is better)."""

    explanation: str
    """Explanation for the assigned score."""

    def __post_init__(self) -> None:
        """Validate score is within range.

        Raises:
            ValueError: If score is not 1-5.
        """
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be 1-5, got {self.score}")

    @property
    def is_low(self) -> bool:
        """Check if score is considered low (needs improvement).

        Per paper, scores <= 2 trigger the feedback loop.

        Returns:
            True if score is 1 or 2.
        """
        return self.score <= 2

    @property
    def is_acceptable(self) -> bool:
        """Check if score is acceptable.

        Per paper, scores >= 4 are considered acceptable.

        Returns:
            True if score is 4 or 5.
        """
        return self.score >= 4


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """A similarity match from embedding search.

    Represents a reference chunk found during few-shot
    retrieval, along with its similarity score.
    """

    chunk: TranscriptChunk
    """The matched transcript chunk."""

    similarity: float
    """Cosine similarity score (0-1, higher is more similar)."""

    reference_score: int | None = None
    """The PHQ-8 score from the reference transcript, if available."""

    def __post_init__(self) -> None:
        """Validate similarity is within range.

        Raises:
            ValueError: If similarity is not in [0, 1].
        """
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError(f"Similarity must be 0-1, got {self.similarity}")
