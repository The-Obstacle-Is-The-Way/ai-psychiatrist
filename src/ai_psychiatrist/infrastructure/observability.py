"""Failure pattern observability (Spec 056).

This module provides a small failure registry that can be attached to runs to
track failure rates by category without leaking sensitive transcript content.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class FailureCategory(str, Enum):
    """Top-level failure categories."""

    # Stage 1: Transcript
    TRANSCRIPT_NOT_FOUND = "transcript_not_found"
    TRANSCRIPT_PARSE_ERROR = "transcript_parse_error"
    TRANSCRIPT_EMPTY = "transcript_empty"

    # Stage 2: Evidence Extraction
    EVIDENCE_JSON_PARSE = "evidence_json_parse"
    EVIDENCE_SCHEMA_INVALID = "evidence_schema_invalid"
    EVIDENCE_HALLUCINATION = "evidence_hallucination"
    EVIDENCE_LLM_TIMEOUT = "evidence_llm_timeout"

    # Stage 3: Embeddings
    EMBEDDING_NAN = "embedding_nan"
    EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"
    EMBEDDING_ZERO_VECTOR = "embedding_zero_vector"
    EMBEDDING_TIMEOUT = "embedding_timeout"

    # Stage 4: Reference Store
    REFERENCE_ARTIFACT_MISSING = "reference_artifact_missing"
    REFERENCE_ARTIFACT_CORRUPT = "reference_artifact_corrupt"
    REFERENCE_TAG_MISMATCH = "reference_tag_mismatch"

    # Stage 5: Scoring
    SCORING_JSON_PARSE = "scoring_json_parse"
    SCORING_SCHEMA_INVALID = "scoring_schema_invalid"
    SCORING_LLM_TIMEOUT = "scoring_llm_timeout"
    SCORING_PYDANTIC_RETRY_EXHAUSTED = "scoring_pydantic_retry_exhausted"

    # Stage 7: Evaluation
    GROUND_TRUTH_MISSING = "ground_truth_missing"
    GROUND_TRUTH_INVALID = "ground_truth_invalid"

    # Other
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Failure severity levels."""

    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class Failure:
    """Single failure event."""

    category: FailureCategory
    severity: FailureSeverity
    message: str
    participant_id: int | None = None
    phq8_item: str | None = None
    stage: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "participant_id": self.participant_id,
            "phq8_item": self.phq8_item,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "context": self.context,
        }


@dataclass
class FailureRegistry:
    """Collects and persists failure events for a run."""

    run_id: str
    failures: list[Failure] = field(default_factory=list)
    _start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def record(
        self,
        category: FailureCategory,
        severity: FailureSeverity,
        message: str,
        *,
        participant_id: int | None = None,
        phq8_item: str | None = None,
        stage: str | None = None,
        **context: Any,
    ) -> None:
        failure = Failure(
            category=category,
            severity=severity,
            message=message,
            participant_id=participant_id,
            phq8_item=phq8_item,
            stage=stage,
            context=context,
        )
        self.failures.append(failure)

        log_method = {
            FailureSeverity.FATAL: logger.error,
            FailureSeverity.ERROR: logger.error,
            FailureSeverity.WARNING: logger.warning,
            FailureSeverity.INFO: logger.info,
        }.get(severity, logger.warning)

        log_method(
            f"failure_{category.value}",
            message=message,
            participant_id=participant_id,
            phq8_item=phq8_item,
            stage=stage,
            **context,
        )

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_participant: dict[int, int] = {}
        by_stage: dict[str, int] = {}

        for failure in self.failures:
            by_category[failure.category.value] = by_category.get(failure.category.value, 0) + 1
            by_severity[failure.severity.value] = by_severity.get(failure.severity.value, 0) + 1
            if failure.participant_id is not None:
                by_participant[failure.participant_id] = (
                    by_participant.get(failure.participant_id, 0) + 1
                )
            if failure.stage:
                by_stage[failure.stage] = by_stage.get(failure.stage, 0) + 1

        return {
            "run_id": self.run_id,
            "start_time": self._start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "total_failures": len(self.failures),
            "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
            "by_severity": by_severity,
            "by_participant": dict(sorted(by_participant.items(), key=lambda x: -x[1])[:10]),
            "by_stage": by_stage,
            "fatal_count": by_severity.get(FailureSeverity.FATAL.value, 0),
            "error_count": by_severity.get(FailureSeverity.ERROR.value, 0),
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"failures_{self.run_id}.json"
        data = {
            "summary": self.summary(),
            "failures": [failure.to_dict() for failure in self.failures],
        }
        output_path.write_text(json.dumps(data, indent=2))
        return output_path

    def print_summary(self) -> None:
        summary = self.summary()

        print("\n" + "=" * 60)
        print("FAILURE SUMMARY")
        print("=" * 60)
        print(f"Run ID: {summary['run_id']}")
        print(f"Total failures: {summary['total_failures']}")
        print(f"  Fatal: {summary['fatal_count']}")
        print(f"  Error: {summary['error_count']}")

        by_category = summary.get("by_category", {})
        if by_category:
            print("\nBy Category:")
            for category, count in by_category.items():
                print(f"  {category}: {count}")

        by_stage = summary.get("by_stage", {})
        if by_stage:
            print("\nBy Stage:")
            for stage, count in by_stage.items():
                print(f"  {stage}: {count}")

        by_participant = summary.get("by_participant", {})
        if by_participant:
            print("\nMost Failing Participants:")
            for participant_id, count in list(by_participant.items())[:5]:
                print(f"  Participant {participant_id}: {count} failures")

        print("=" * 60 + "\n")


_registry_var: contextvars.ContextVar[FailureRegistry | None] = contextvars.ContextVar(
    "ai_psychiatrist_failure_registry",
    default=None,
)


def get_failure_registry() -> FailureRegistry:
    registry = _registry_var.get()
    if registry is None:
        raise RuntimeError("Failure registry not initialized. Call init_failure_registry() first.")
    return registry


def init_failure_registry(run_id: str) -> FailureRegistry:
    registry = FailureRegistry(run_id=run_id)
    _registry_var.set(registry)
    return registry


def record_failure(
    category: FailureCategory,
    severity: FailureSeverity,
    message: str,
    **kwargs: Any,
) -> None:
    try:
        registry = get_failure_registry()
        registry.record(category, severity, message, **kwargs)
    except RuntimeError:
        logger.warning(
            "failure_registry_not_initialized",
            category=category.value,
            message=message,
            **kwargs,
        )
