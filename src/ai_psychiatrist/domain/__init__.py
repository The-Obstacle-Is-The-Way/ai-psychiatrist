"""Domain models and entities for PHQ-8 assessment.

This module provides the core domain layer for the AI Psychiatrist system,
containing pure Python objects with no external dependencies.

Modules:
    enums: Domain enumerations (PHQ8Item, SeverityLevel, etc.)
    value_objects: Immutable value types (TranscriptChunk, ItemAssessment, etc.)
    entities: Core business entities (Transcript, PHQ8Assessment, etc.)
    exceptions: Domain-specific exceptions

Example:
    >>> from ai_psychiatrist.domain import PHQ8Item, SeverityLevel
    >>> severity = SeverityLevel.from_total_score(12)
    >>> severity.is_mdd
    True
"""

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
    PHQ8Score,
    SeverityLevel,
)
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
from ai_psychiatrist.domain.value_objects import (
    EmbeddedChunk,
    EvaluationScore,
    Evidence,
    ItemAssessment,
    SimilarityMatch,
    TranscriptChunk,
)

__all__ = [
    "AssessmentError",
    "AssessmentMode",
    "DomainError",
    "EmbeddedChunk",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmptyTranscriptError",
    "EvaluationError",
    "EvaluationMetric",
    "EvaluationScore",
    "Evidence",
    "FullAssessment",
    "InsufficientEvidenceError",
    "ItemAssessment",
    "LLMError",
    "LLMResponseParseError",
    "LLMTimeoutError",
    "LowScoreError",
    "MaxIterationsError",
    "MetaReview",
    "PHQ8Assessment",
    "PHQ8Item",
    "PHQ8ItemError",
    "PHQ8Score",
    "QualitativeAssessment",
    "QualitativeEvaluation",
    "SeverityLevel",
    "SimilarityMatch",
    "Transcript",
    "TranscriptChunk",
    "TranscriptError",
    "ValidationError",
]
