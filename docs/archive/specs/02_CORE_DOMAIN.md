# Spec 02: Core Domain

## Objective

Define the core domain entities, value objects, and business rules that form the heart of the AI Psychiatrist system. These are pure Python objects with no external dependencies.

## Paper Reference

- **Section 2.1**: PHQ-8 scoring system (0-3 scale, 8 items)
- **Section 2.3**: Assessment outputs and severity categories
- **Table 1**: Performance metrics definitions

## As-Is Domain Representation (Repo)

The current repo does **not** yet implement a dedicated domain layer under `src/ai_psychiatrist/`.
Instead, domain concepts appear as plain dicts/strings in the agents and scripts:

- **PHQ-8 item keys (as-is):** `PHQ8_NoInterest`, `PHQ8_Depressed`, `PHQ8_Sleep`, `PHQ8_Tired`, `PHQ8_Appetite`, `PHQ8_Failure`, `PHQ8_Concentrating`, `PHQ8_Moving` (see `agents/quantitative_assessor_f.py`).
- **Item score type (as-is):** `int` 0–3 or the string `"N/A"` (multiple agents/scripts).
- **Quantitative severity (as-is):** computed by `agents/quantitative_assessor_f.py::_compute_total_and_severity()` and returned as a string in `result["_severity"]`:
  `minimal`, `mild`, `moderate`, `mod_severe`, `severe`.
- **Meta-review severity output (as-is):** model returns `<severity>` as an integer 0–4 in XML (see `agents/meta_reviewer.py`).

This spec defines the **target** typed domain layer while preserving these as-is conventions for parity audits.

## Deliverables

1. `src/ai_psychiatrist/domain/entities.py` - Core business entities
2. `src/ai_psychiatrist/domain/value_objects.py` - Immutable value types
3. `src/ai_psychiatrist/domain/enums.py` - Domain enumerations
4. `src/ai_psychiatrist/domain/exceptions.py` - Domain-specific exceptions
5. `tests/unit/domain/` - Comprehensive domain tests

## Implementation

### 1. Enumerations (domain/enums.py)

```python
"""Domain enumerations for AI Psychiatrist."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class PHQ8Item(StrEnum):
    """PHQ-8 assessment items (DSM-5 criteria)."""

    NO_INTEREST = "NoInterest"      # Little interest or pleasure (anhedonia)
    DEPRESSED = "Depressed"         # Feeling down, depressed, hopeless
    SLEEP = "Sleep"                 # Sleep problems
    TIRED = "Tired"                 # Fatigue, little energy
    APPETITE = "Appetite"           # Appetite/weight changes
    FAILURE = "Failure"             # Negative self-perception
    CONCENTRATING = "Concentrating" # Concentration problems
    MOVING = "Moving"               # Psychomotor changes

    @classmethod
    def all_items(cls) -> list[PHQ8Item]:
        """Return all PHQ-8 items in order."""
        return list(cls)


class PHQ8Score(IntEnum):
    """PHQ-8 item score (frequency over past 2 weeks)."""

    NOT_AT_ALL = 0      # 0-1 days
    SEVERAL_DAYS = 1    # 2-6 days
    MORE_THAN_HALF = 2  # 7-11 days
    NEARLY_EVERY_DAY = 3  # 12-14 days

    @classmethod
    def from_int(cls, value: int) -> PHQ8Score:
        """Create score from integer, clamping to valid range."""
        return cls(max(0, min(3, value)))


class SeverityLevel(IntEnum):
    """Depression severity based on PHQ-8 total score."""

    MINIMAL = 0     # Total 0-4: No significant symptoms
    MILD = 1        # Total 5-9: Mild symptoms
    MODERATE = 2    # Total 10-14: Moderate symptoms
    MOD_SEVERE = 3  # Total 15-19: Moderately severe
    SEVERE = 4      # Total 20-24: Severe symptoms

    @classmethod
    def from_total_score(cls, total: int) -> SeverityLevel:
        """Determine severity from total PHQ-8 score."""
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
        """Check if severity indicates Major Depressive Disorder (>=10)."""
        return self >= SeverityLevel.MODERATE


class EvaluationMetric(StrEnum):
    """Qualitative assessment evaluation metrics."""

    COHERENCE = "coherence"         # Logical consistency
    COMPLETENESS = "completeness"   # Coverage of symptoms
    SPECIFICITY = "specificity"     # Avoidance of vague statements
    ACCURACY = "accuracy"           # Alignment with PHQ-8/DSM-5

    @classmethod
    def all_metrics(cls) -> list[EvaluationMetric]:
        """Return all evaluation metrics."""
        return list(cls)


class AssessmentMode(StrEnum):
    """Quantitative assessment mode."""

    ZERO_SHOT = "zero_shot"  # No reference examples
    FEW_SHOT = "few_shot"    # Embedding-based references
```

### 2. Value Objects (domain/value_objects.py)

```python
"""Immutable value objects for AI Psychiatrist domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_psychiatrist.domain.enums import EvaluationMetric, PHQ8Item


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """A segment of an interview transcript."""

    text: str
    participant_id: int
    line_start: int = 0
    line_end: int = 0

    def __post_init__(self) -> None:
        """Validate chunk data."""
        if not self.text.strip():
            raise ValueError("Transcript chunk cannot be empty")
        if self.participant_id <= 0:
            raise ValueError("Participant ID must be positive")

    @property
    def word_count(self) -> int:
        """Count words in chunk."""
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A transcript chunk with its embedding vector."""

    chunk: TranscriptChunk
    embedding: tuple[float, ...]
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        """Set dimension from embedding."""
        object.__setattr__(self, "dimension", len(self.embedding))

    @property
    def participant_id(self) -> int:
        """Get participant ID from chunk."""
        return self.chunk.participant_id


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence supporting a PHQ-8 item score."""

    quotes: tuple[str, ...]
    item: PHQ8Item
    source_participant_id: int | None = None

    @classmethod
    def empty(cls, item: PHQ8Item) -> Evidence:
        """Create empty evidence for an item."""
        return cls(quotes=(), item=item)

    @property
    def has_evidence(self) -> bool:
        """Check if any evidence exists."""
        return len(self.quotes) > 0


@dataclass(frozen=True, slots=True)
class ItemAssessment:
    """Assessment result for a single PHQ-8 item."""

    item: PHQ8Item
    evidence: str
    reason: str
    score: int | None  # None means N/A

    @property
    def is_available(self) -> bool:
        """Check if score is available (not N/A)."""
        return self.score is not None

    @property
    def score_value(self) -> int:
        """Get score value, defaulting to 0 for N/A."""
        return self.score if self.score is not None else 0


@dataclass(frozen=True, slots=True)
class EvaluationScore:
    """Score for a single evaluation metric."""

    metric: EvaluationMetric
    score: int  # 1-5 scale
    explanation: str

    def __post_init__(self) -> None:
        """Validate score range."""
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be 1-5, got {self.score}")

    @property
    def is_low(self) -> bool:
        """Check if score is considered low (needs improvement)."""
        return self.score <= 3

    @property
    def is_acceptable(self) -> bool:
        """Check if score is acceptable (>= 4)."""
        return self.score >= 4


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """A similarity match from embedding search.

    Note: The domain constrains similarity to [0, 1]. When cosine similarity is used
    for retrieval, implementations should transform raw cosine similarity in [-1, 1]
    to this range via: (1 + cos) / 2.
    """

    chunk: TranscriptChunk
    similarity: float
    reference_score: int | None = None

    def __post_init__(self) -> None:
        """Validate similarity range."""
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError(f"Similarity must be 0-1, got {self.similarity}")
```

### 3. Entities (domain/entities.py)

```python
"""Core domain entities for AI Psychiatrist."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

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

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class Transcript:
    """An interview transcript with metadata."""

    participant_id: int
    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        """Validate transcript."""
        if not self.text.strip():
            raise ValueError("Transcript text cannot be empty")
        if self.participant_id <= 0:
            raise ValueError("Participant ID must be positive")

    @property
    def word_count(self) -> int:
        """Count words in transcript."""
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        """Count lines in transcript."""
        return len(self.text.strip().splitlines())


@dataclass
class PHQ8Assessment:
    """Complete PHQ-8 assessment with all 8 items."""

    items: Mapping[PHQ8Item, ItemAssessment]
    mode: AssessmentMode
    participant_id: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        """Validate all items present."""
        missing = set(PHQ8Item.all_items()) - set(self.items.keys())
        if missing:
            raise ValueError(f"Missing PHQ-8 items: {missing}")

    @property
    def total_score(self) -> int:
        """Calculate total PHQ-8 score (0-24)."""
        return sum(item.score_value for item in self.items.values())

    @property
    def severity(self) -> SeverityLevel:
        """Determine severity from total score."""
        return SeverityLevel.from_total_score(self.total_score)

    @property
    def available_count(self) -> int:
        """Count items with available (non-N/A) scores."""
        return sum(1 for item in self.items.values() if item.is_available)

    @property
    def na_count(self) -> int:
        """Count items with N/A scores."""
        return 8 - self.available_count

    def get_item(self, item: PHQ8Item) -> ItemAssessment:
        """Get assessment for specific item."""
        return self.items[item]


@dataclass
class QualitativeAssessment:
    """Qualitative assessment output."""

    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    supporting_quotes: list[str] = field(default_factory=list)
    participant_id: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    @property
    def full_text(self) -> str:
        """Get full assessment as formatted text."""
        return f"""Overall Assessment:
{self.overall}

PHQ-8 Symptoms:
{self.phq8_symptoms}

Social Factors:
{self.social_factors}

Biological Factors:
{self.biological_factors}

Risk Factors:
{self.risk_factors}
"""


@dataclass
class QualitativeEvaluation:
    """Evaluation of a qualitative assessment."""

    scores: Mapping[EvaluationMetric, EvaluationScore]
    assessment_id: UUID
    iteration: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        """Validate all metrics present."""
        missing = set(EvaluationMetric.all_metrics()) - set(self.scores.keys())
        if missing:
            raise ValueError(f"Missing evaluation metrics: {missing}")

    @property
    def average_score(self) -> float:
        """Calculate average score across all metrics."""
        return sum(s.score for s in self.scores.values()) / len(self.scores)

    @property
    def low_scores(self) -> list[EvaluationMetric]:
        """Get list of metrics with low scores (<=3)."""
        return [m for m, s in self.scores.items() if s.is_low]

    @property
    def needs_improvement(self) -> bool:
        """Check if any metric needs improvement."""
        return len(self.low_scores) > 0

    @property
    def all_acceptable(self) -> bool:
        """Check if all metrics are acceptable (>=4)."""
        return all(s.is_acceptable for s in self.scores.values())

    def get_score(self, metric: EvaluationMetric) -> EvaluationScore:
        """Get score for specific metric."""
        return self.scores[metric]


@dataclass
class MetaReview:
    """Integrated meta-review combining all assessments."""

    severity: SeverityLevel
    explanation: str
    quantitative_assessment_id: UUID
    qualitative_assessment_id: UUID
    participant_id: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    @property
    def is_mdd(self) -> bool:
        """Check if indicates Major Depressive Disorder."""
        return self.severity.is_mdd


@dataclass
class FullAssessment:
    """Complete assessment result combining all components."""

    transcript: Transcript
    quantitative: PHQ8Assessment
    qualitative: QualitativeAssessment
    qualitative_evaluation: QualitativeEvaluation
    meta_review: MetaReview
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: UUID = field(default_factory=uuid4)

    @property
    def participant_id(self) -> int:
        """Get participant ID."""
        return self.transcript.participant_id

    @property
    def final_severity(self) -> SeverityLevel:
        """Get final severity from meta-review."""
        return self.meta_review.severity
```

### 4. Domain Exceptions (domain/exceptions.py)

```python
"""Domain-specific exceptions for AI Psychiatrist."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_psychiatrist.domain.enums import EvaluationMetric, PHQ8Item


class DomainError(Exception):
    """Base class for domain errors."""

    pass


class ValidationError(DomainError):
    """Raised when domain validation fails."""

    pass


class TranscriptError(DomainError):
    """Errors related to transcript processing."""

    pass


class EmptyTranscriptError(TranscriptError):
    """Raised when transcript is empty or invalid."""

    pass


class AssessmentError(DomainError):
    """Errors during assessment."""

    pass


class PHQ8ItemError(AssessmentError):
    """Errors related to PHQ-8 items."""

    def __init__(self, item: PHQ8Item, message: str) -> None:
        self.item = item
        super().__init__(f"PHQ-8 {item.value}: {message}")


class InsufficientEvidenceError(AssessmentError):
    """Raised when insufficient evidence for assessment."""

    def __init__(self, item: PHQ8Item) -> None:
        self.item = item
        super().__init__(f"Insufficient evidence for {item.value}")


class EvaluationError(DomainError):
    """Errors during evaluation."""

    pass


class LowScoreError(EvaluationError):
    """Raised when evaluation score is too low."""

    def __init__(self, metric: EvaluationMetric, score: int) -> None:
        self.metric = metric
        self.score = score
        super().__init__(f"{metric.value} scored {score}/5, below acceptable threshold")


class MaxIterationsError(EvaluationError):
    """Raised when max feedback iterations reached."""

    def __init__(self, iterations: int) -> None:
        self.iterations = iterations
        super().__init__(f"Max iterations ({iterations}) reached without acceptable scores")


class LLMError(DomainError):
    """Errors from LLM interactions."""

    pass


class LLMResponseParseError(LLMError):
    """Raised when LLM response cannot be parsed."""

    def __init__(self, raw_response: str, parse_error: str) -> None:
        self.raw_response = raw_response
        self.parse_error = parse_error
        super().__init__(f"Failed to parse LLM response: {parse_error}")


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"LLM request timed out after {timeout_seconds}s")


class EmbeddingError(DomainError):
    """Errors during embedding operations."""

    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Embedding dimension mismatch: expected {expected}, got {actual}")
```

### 5. Tests (tests/unit/domain/test_entities.py)

```python
"""Tests for domain entities."""

from __future__ import annotations

import pytest
from uuid import UUID

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


class TestPHQ8Item:
    """Tests for PHQ8Item enum."""

    def test_all_items_count(self) -> None:
        """Should have exactly 8 items."""
        assert len(PHQ8Item.all_items()) == 8

    def test_item_values(self) -> None:
        """Item values should match expected strings."""
        assert PHQ8Item.NO_INTEREST.value == "NoInterest"
        assert PHQ8Item.DEPRESSED.value == "Depressed"


class TestSeverityLevel:
    """Tests for SeverityLevel enum."""

    @pytest.mark.parametrize(
        ("total", "expected"),
        [
            (0, SeverityLevel.MINIMAL),
            (4, SeverityLevel.MINIMAL),
            (5, SeverityLevel.MILD),
            (9, SeverityLevel.MILD),
            (10, SeverityLevel.MODERATE),
            (14, SeverityLevel.MODERATE),
            (15, SeverityLevel.MOD_SEVERE),
            (19, SeverityLevel.MOD_SEVERE),
            (20, SeverityLevel.SEVERE),
            (24, SeverityLevel.SEVERE),
        ],
    )
    def test_from_total_score(self, total: int, expected: SeverityLevel) -> None:
        """Should correctly categorize total scores."""
        assert SeverityLevel.from_total_score(total) == expected

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
    def test_is_mdd(self, severity: SeverityLevel, is_mdd: bool) -> None:
        """Should correctly identify MDD threshold (>=10)."""
        assert severity.is_mdd == is_mdd


class TestTranscript:
    """Tests for Transcript entity."""

    def test_create_valid_transcript(self) -> None:
        """Should create transcript with valid data."""
        transcript = Transcript(participant_id=123, text="Hello world")
        assert transcript.participant_id == 123
        assert transcript.text == "Hello world"
        assert isinstance(transcript.id, UUID)

    def test_reject_empty_text(self) -> None:
        """Should reject empty transcript text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Transcript(participant_id=123, text="   ")

    def test_reject_invalid_participant_id(self) -> None:
        """Should reject non-positive participant ID."""
        with pytest.raises(ValueError, match="must be positive"):
            Transcript(participant_id=0, text="Hello")

    def test_word_count(self) -> None:
        """Should count words correctly."""
        transcript = Transcript(participant_id=1, text="one two three four")
        assert transcript.word_count == 4


class TestPHQ8Assessment:
    """Tests for PHQ8Assessment entity."""

    @pytest.fixture
    def complete_items(self) -> dict[PHQ8Item, ItemAssessment]:
        """Create complete item assessments."""
        return {
            item: ItemAssessment(
                item=item,
                evidence="Test evidence",
                reason="Test reason",
                score=1,
            )
            for item in PHQ8Item.all_items()
        }

    def test_create_valid_assessment(
        self, complete_items: dict[PHQ8Item, ItemAssessment]
    ) -> None:
        """Should create assessment with all items."""
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.total_score == 8  # 8 items * 1 each
        assert assessment.available_count == 8

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

    def test_severity_calculation(
        self, complete_items: dict[PHQ8Item, ItemAssessment]
    ) -> None:
        """Should calculate severity from total score."""
        # Total = 8 (8 items * 1) -> MILD
        assessment = PHQ8Assessment(
            items=complete_items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.severity == SeverityLevel.MILD

    def test_na_scores_not_counted(self) -> None:
        """N/A scores should not contribute to total."""
        items = {
            item: ItemAssessment(
                item=item,
                evidence="Test",
                reason="No evidence",
                score=None,  # N/A
            )
            for item in PHQ8Item.all_items()
        }
        assessment = PHQ8Assessment(
            items=items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )
        assert assessment.total_score == 0
        assert assessment.na_count == 8


class TestQualitativeEvaluation:
    """Tests for QualitativeEvaluation entity."""

    @pytest.fixture
    def complete_scores(self) -> dict[EvaluationMetric, EvaluationScore]:
        """Create complete evaluation scores."""
        return {
            metric: EvaluationScore(
                metric=metric,
                score=4,
                explanation="Good",
            )
            for metric in EvaluationMetric.all_metrics()
        }

    def test_average_score(
        self, complete_scores: dict[EvaluationMetric, EvaluationScore]
    ) -> None:
        """Should calculate average score correctly."""
        evaluation = QualitativeEvaluation(
            scores=complete_scores,
            assessment_id=UUID("12345678-1234-1234-1234-123456789abc"),
        )
        assert evaluation.average_score == 4.0

    def test_low_scores_detection(self) -> None:
        """Should detect low scores (<= 3)."""
        scores = {
            EvaluationMetric.COHERENCE: EvaluationScore(
                metric=EvaluationMetric.COHERENCE, score=5, explanation="Great"
            ),
            EvaluationMetric.COMPLETENESS: EvaluationScore(
                metric=EvaluationMetric.COMPLETENESS, score=2, explanation="Low"
            ),
            EvaluationMetric.SPECIFICITY: EvaluationScore(
                metric=EvaluationMetric.SPECIFICITY, score=3, explanation="Low"
            ),
            EvaluationMetric.ACCURACY: EvaluationScore(
                metric=EvaluationMetric.ACCURACY, score=1, explanation="Very low"
            ),
        }
        evaluation = QualitativeEvaluation(
            scores=scores,
            assessment_id=UUID("12345678-1234-1234-1234-123456789abc"),
        )
        low = evaluation.low_scores
        assert EvaluationMetric.COMPLETENESS in low
        assert EvaluationMetric.SPECIFICITY in low
        assert EvaluationMetric.ACCURACY in low
        assert EvaluationMetric.COHERENCE not in low
```

## Acceptance Criteria

- [ ] All 8 PHQ-8 items represented in `PHQ8Item` enum
- [ ] Severity levels match paper thresholds (0-4, 5-9, 10-14, 15-19, 20-24)
- [ ] MDD threshold correctly set at 10
- [ ] All evaluation metrics represented (coherence, completeness, specificity, accuracy)
- [ ] Entities are immutable where appropriate (value objects)
- [ ] Full test coverage for domain logic
- [ ] Type hints throughout
- [ ] No external dependencies in domain layer

## Dependencies

- **Spec 01**: Project structure

## Specs That Depend on This

- **Spec 03**: Configuration (uses domain types)
- **Spec 05-11**: All agent and service specs use domain entities
