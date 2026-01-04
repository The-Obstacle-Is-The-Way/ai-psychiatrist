# Spec 056: Failure Pattern Observability

**Status**: Implemented (PR #92, 2026-01-03)
**Priority**: Medium
**Complexity**: Medium
**Related**: PIPELINE-BRITTLENESS.md, ANALYSIS-026

---

## SSOT (Implemented)

- Code: `src/ai_psychiatrist/infrastructure/observability.py` (`FailureRegistry`, `record_failure()`)
- Wire-up: `scripts/reproduce_results.py` (registry init + per-participant recording + `failures_{run_id}.json`)
- Tests: `tests/unit/infrastructure/test_observability.py`

## Problem Statement

We don't systematically track failure patterns across runs. When something goes wrong, we discover it through:

1. Manual log inspection
2. Unexplained metric changes
3. User reports

We need structured observability to:
1. Know failure rates by category
2. Identify patterns (e.g., "participant 373 always fails on evidence extraction")
3. Track improvement over time
4. Debug issues faster

---

## Previous Behavior (Fixed)

Logging is ad-hoc:
- Some failures logged with `logger.warning()` or `logger.error()`
- No consistent taxonomy or structure
- No aggregation across runs
- No per-participant failure tracking

---

## Implemented Solution

Implement a **Failure Registry** that:
1. Captures all failures with consistent structure
2. Aggregates by failure type
3. Persists to JSON for cross-run analysis
4. Integrates with existing logging

### Privacy / Licensing Constraint (Non-Negotiable)

DAIC-WOZ transcripts are licensed and must not leak into logs or artifacts. The failure registry MUST:
- Never store raw transcript text.
- Never store raw LLM responses or evidence quote strings.
- Only store counts, lengths, stable hashes, model ids, error codes, and stack-trace-free messages.

Examples of allowed context fields:
- `response_hash`, `response_len`
- `transcript_hash`, `transcript_len`
- `text_hash`, `text_len` (for embeddings)
- `exception_type`, `http_status`

---

## Implementation

### Failure Taxonomy

```python
# New: src/ai_psychiatrist/infrastructure/observability.py

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import json


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

    # Stage 6: Aggregation
    AGGREGATION_MISSING_ITEMS = "aggregation_missing_items"

    # Stage 7: Evaluation
    GROUND_TRUTH_MISSING = "ground_truth_missing"
    GROUND_TRUTH_INVALID = "ground_truth_invalid"

    # Other
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Failure severity levels."""

    FATAL = "fatal"      # Participant cannot be processed
    ERROR = "error"      # Significant issue, partial results possible
    WARNING = "warning"  # Minor issue, results may be degraded
    INFO = "info"        # Informational, no impact on results


@dataclass
class Failure:
    """Single failure event."""

    category: FailureCategory
    severity: FailureSeverity
    message: str
    participant_id: int | None = None
    phq8_item: str | None = None  # e.g., "PHQ8_Sleep"
    stage: str | None = None  # e.g., "evidence_extraction"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    context: dict[str, Any] = field(default_factory=dict)
    """Additional privacy-safe context (never raw transcript/LLM text)."""

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
```

### Failure Registry

```python
# Continue in observability.py

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
        """Record a failure event."""
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

        # Also log for immediate visibility
        from ai_psychiatrist.infrastructure.logging import get_logger
        logger = get_logger("failure_registry")

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
        """Generate summary statistics."""
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_participant: dict[int, int] = {}
        by_stage: dict[str, int] = {}

        for f in self.failures:
            by_category[f.category.value] = by_category.get(f.category.value, 0) + 1
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1
            if f.participant_id is not None:
                by_participant[f.participant_id] = by_participant.get(f.participant_id, 0) + 1
            if f.stage:
                by_stage[f.stage] = by_stage.get(f.stage, 0) + 1

        return {
            "run_id": self.run_id,
            "start_time": self._start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "total_failures": len(self.failures),
            "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
            "by_severity": by_severity,
            "by_participant": dict(sorted(by_participant.items(), key=lambda x: -x[1])[:10]),  # Top 10
            "by_stage": by_stage,
            "fatal_count": by_severity.get("fatal", 0),
            "error_count": by_severity.get("error", 0),
        }

    def save(self, output_dir: Path) -> Path:
        """Save failures to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"failures_{self.run_id}.json"
        output_path = output_dir / filename

        data = {
            "summary": self.summary(),
            "failures": [f.to_dict() for f in self.failures],
        }

        output_path.write_text(json.dumps(data, indent=2))
        return output_path

    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        summary = self.summary()

        print("\n" + "=" * 60)
        print("FAILURE SUMMARY")
        print("=" * 60)
        print(f"Run ID: {summary['run_id']}")
        print(f"Total failures: {summary['total_failures']}")
        print(f"  Fatal: {summary['fatal_count']}")
        print(f"  Error: {summary['error_count']}")

        if summary['by_category']:
            print("\nBy Category:")
            for cat, count in summary['by_category'].items():
                print(f"  {cat}: {count}")

        if summary['by_stage']:
            print("\nBy Stage:")
            for stage, count in summary['by_stage'].items():
                print(f"  {stage}: {count}")

        if summary['by_participant']:
            print("\nMost Failing Participants:")
            for pid, count in list(summary['by_participant'].items())[:5]:
                print(f"  Participant {pid}: {count} failures")

        print("=" * 60 + "\n")


# Global registry instance (created per run)
_current_registry: FailureRegistry | None = None


def get_failure_registry() -> FailureRegistry:
    """Get the current failure registry."""
    global _current_registry
    if _current_registry is None:
        raise RuntimeError("Failure registry not initialized. Call init_failure_registry() first.")
    return _current_registry


def init_failure_registry(run_id: str) -> FailureRegistry:
    """Initialize a new failure registry for a run."""
    global _current_registry
    _current_registry = FailureRegistry(run_id=run_id)
    return _current_registry


def record_failure(
    category: FailureCategory,
    severity: FailureSeverity,
    message: str,
    **kwargs: Any,
) -> None:
    """Convenience function to record a failure to the global registry."""
    try:
        registry = get_failure_registry()
        registry.record(category, severity, message, **kwargs)
    except RuntimeError:
        # Registry not initialized - log directly instead
        from ai_psychiatrist.infrastructure.logging import get_logger
        logger = get_logger("failure_registry")
        logger.warning(
            "failure_registry_not_initialized",
            category=category.value,
            message=message,
            **kwargs,
        )
```

### Integration Points

#### 1) Per-participant failure capture (required)

The most robust/low-coupling integration point is the existing per-participant exception handler in
`scripts/reproduce_results.py:evaluate_participant()`. Record failures there so every participant failure
is captured even if it originates deep in the call stack.

```python
# scripts/reproduce_results.py (inside evaluate_participant)

from ai_psychiatrist.infrastructure.observability import (
    record_failure,
    FailureCategory,
    FailureSeverity,
)

def classify_failure(exc: Exception) -> tuple[FailureCategory, FailureSeverity, dict[str, object]]:
    # Minimal, privacy-safe classification by exception type.
    name = type(exc).__name__
    if name == "UnexpectedModelBehavior":
        return (FailureCategory.SCORING_PYDANTIC_RETRY_EXHAUSTED, FailureSeverity.FATAL, {})
    if name in {"EmbeddingDimensionMismatchError", "EmbeddingArtifactMismatchError"}:
        return (FailureCategory.EMBEDDING_DIMENSION_MISMATCH, FailureSeverity.FATAL, {})
    if name in {"EmbeddingValidationError"}:
        return (FailureCategory.EMBEDDING_NAN, FailureSeverity.FATAL, {})
    return (FailureCategory.UNKNOWN, FailureSeverity.ERROR, {"exception_type": name})

# ...
except Exception as e:
    category, severity, ctx = classify_failure(e)
    record_failure(
        category,
        severity,
        str(e),
        participant_id=participant_id,
        stage="evaluate_participant",
        **ctx,
    )
    return EvaluationResult(
        participant_id=participant_id,
        mode=mode,
        duration_seconds=duration,
        success=False,
        error=str(e),
    )
```

#### 2) Evidence extraction parse failures (optional, adds hash/length)

If you want extra observability for deterministic JSON failures, also record them at the source in
`src/ai_psychiatrist/agents/quantitative.py:_extract_evidence()` where the sanitized JSON string is available:

- `response_hash` / `response_len` (never raw output)
- `exception_type`

#### 3) Run initialization + persistence (required)

```python
# scripts/reproduce_results.py (inside main_async; RunMetadata is already captured)

from ai_psychiatrist.infrastructure.observability import init_failure_registry

# Initialize at start of run (SSOT run id)
failure_registry = init_failure_registry(run_metadata.run_id)

# ... run evaluation ...

# At end of run
failure_registry.print_summary()
failure_path = failure_registry.save(Path("data/outputs"))
print(f"Failures saved to: {failure_path}")
```

---

## Output Format

### Summary (printed to console)

```
============================================================
FAILURE SUMMARY
============================================================
Run ID: 19b42478
Total failures: 12
  Fatal: 3
  Error: 7

By Category:
  evidence_json_parse: 2
  scoring_pydantic_retry_exhausted: 1
  evidence_hallucination: 7
  embedding_nan: 2

By Stage:
  evidence_extraction: 9
  scoring: 1
  embedding_generation: 2

Most Failing Participants:
  Participant 373: 3 failures
  Participant 444: 2 failures
  Participant 318: 1 failures
============================================================
```

### Full JSON (saved to file)

```json
{
  "summary": {
    "run_id": "19b42478",
    "start_time": "2026-01-03T14:30:22.123456+00:00",
    "end_time": "2026-01-03T16:45:33.789012+00:00",
    "total_failures": 12,
    "by_category": {
      "evidence_hallucination": 7,
      "evidence_json_parse": 2,
      "embedding_nan": 2,
      "scoring_pydantic_retry_exhausted": 1
    },
    "by_severity": {
      "fatal": 3,
      "error": 7,
      "warning": 2
    },
    "by_participant": {
      "373": 3,
      "444": 2,
      "318": 1
    },
    "by_stage": {
      "evidence_extraction": 9,
      "scoring": 1,
      "embedding_generation": 2
    },
    "fatal_count": 3,
    "error_count": 7
  },
  "failures": [
    {
      "category": "evidence_json_parse",
      "severity": "fatal",
      "message": "Evidence JSON parse failed: Expecting ',' delimiter",
      "participant_id": 373,
      "phq8_item": null,
      "stage": "evidence_extraction",
      "timestamp": "2026-01-03T14:35:12.456789+00:00",
      "context": {
        "response_hash": "4f1c2b6a19d0",
        "response_len": 1289,
        "exception_type": "JSONDecodeError"
      }
    }
  ]
}
```

---

## Testing

```python
# tests/unit/infrastructure/test_observability.py

import pytest
from ai_psychiatrist.infrastructure.observability import (
    FailureRegistry,
    FailureCategory,
    FailureSeverity,
    init_failure_registry,
    record_failure,
)


def test_record_failure():
    registry = FailureRegistry(run_id="test_run")
    registry.record(
        FailureCategory.EVIDENCE_JSON_PARSE,
        FailureSeverity.FATAL,
        "Test failure",
        participant_id=300,
        stage="evidence_extraction",
    )

    assert len(registry.failures) == 1
    assert registry.failures[0].category == FailureCategory.EVIDENCE_JSON_PARSE


def test_summary_aggregation():
    registry = FailureRegistry(run_id="test")

    # Add multiple failures
    for i in range(3):
        registry.record(
            FailureCategory.EVIDENCE_JSON_PARSE,
            FailureSeverity.FATAL,
            f"Failure {i}",
            participant_id=300,
        )

    registry.record(
        FailureCategory.EMBEDDING_NAN,
        FailureSeverity.ERROR,
        "NaN failure",
        participant_id=301,
    )

    summary = registry.summary()
    assert summary["total_failures"] == 4
    assert summary["by_category"]["evidence_json_parse"] == 3
    assert summary["by_category"]["embedding_nan"] == 1
    assert summary["by_participant"][300] == 3


def test_save_and_load(tmp_path):
    registry = FailureRegistry(run_id="test")
    registry.record(
        FailureCategory.SCORING_LLM_TIMEOUT,
        FailureSeverity.FATAL,
        "Timeout",
    )

    output_path = registry.save(tmp_path)
    assert output_path.exists()

    import json
    data = json.loads(output_path.read_text())
    assert data["summary"]["total_failures"] == 1
    assert len(data["failures"]) == 1


def test_global_registry():
    init_failure_registry("global_test")

    record_failure(
        FailureCategory.GROUND_TRUTH_MISSING,
        FailureSeverity.FATAL,
        "Missing ground truth",
        participant_id=999,
    )

    from ai_psychiatrist.infrastructure.observability import get_failure_registry
    registry = get_failure_registry()
    assert len(registry.failures) == 1
```

---

## Rollout Plan

1. **Phase 1**: Implement FailureRegistry and taxonomy
2. **Phase 2**: Integrate with evidence extraction failures
3. **Phase 3**: Integrate with embedding failures
4. **Phase 4**: Integrate with scoring failures
5. **Phase 5**: Add to reproduction script with console summary

---

## Success Criteria

1. All failure types have a defined category
2. Every fatal/error failure is recorded
3. Summary printed at end of each run
4. JSON file persisted for cross-run analysis
5. No performance regression (registry is append-only)

---

## Future Enhancements

1. **Dashboard**: Web UI to visualize failure trends
2. **Alerting**: Notify when failure rate exceeds threshold
3. **Cross-run comparison**: Compare failure rates across runs
4. **Root cause analysis**: Auto-detect correlated failures
