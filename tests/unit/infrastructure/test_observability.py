"""Unit tests for failure pattern observability (Spec 056)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ai_psychiatrist.infrastructure.observability import (
    FailureCategory,
    FailureRegistry,
    FailureSeverity,
    get_failure_registry,
    init_failure_registry,
    record_failure,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_record_failure() -> None:
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


def test_summary_aggregation() -> None:
    registry = FailureRegistry(run_id="test")

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


def test_save_and_load(tmp_path: Path) -> None:
    registry = FailureRegistry(run_id="test")
    registry.record(
        FailureCategory.SCORING_LLM_TIMEOUT,
        FailureSeverity.FATAL,
        "Timeout",
    )

    output_path = registry.save(tmp_path)
    assert output_path.exists()

    data = json.loads(output_path.read_text())
    assert data["summary"]["total_failures"] == 1
    assert len(data["failures"]) == 1


def test_global_registry() -> None:
    init_failure_registry("global_test")

    record_failure(
        FailureCategory.GROUND_TRUTH_MISSING,
        FailureSeverity.FATAL,
        "Missing ground truth",
        participant_id=999,
    )

    registry = get_failure_registry()
    assert len(registry.failures) == 1
