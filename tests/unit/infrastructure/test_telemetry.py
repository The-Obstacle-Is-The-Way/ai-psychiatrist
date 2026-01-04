"""Unit tests for retry telemetry (Spec 060)."""

from __future__ import annotations

import pytest

from ai_psychiatrist.infrastructure.llm.responses import parse_llm_json
from ai_psychiatrist.infrastructure.telemetry import (
    TelemetryCategory,
    TelemetryRegistry,
    get_telemetry_registry,
    init_telemetry_registry,
    record_telemetry,
)

pytestmark = pytest.mark.unit


def test_record_telemetry_noop_when_uninitialized() -> None:
    """record_telemetry() must not raise when registry is not initialized."""
    record_telemetry(TelemetryCategory.JSON_REPAIR_FALLBACK, event="test")
    with pytest.raises(RuntimeError, match="Telemetry registry not initialized"):
        get_telemetry_registry()


def test_registry_summary_counts() -> None:
    registry = TelemetryRegistry(run_id="test")
    registry.record(TelemetryCategory.PYDANTIC_RETRY, extractor="extract_quantitative")
    registry.record(TelemetryCategory.PYDANTIC_RETRY, extractor="extract_quantitative")
    registry.record(TelemetryCategory.JSON_PYTHON_LITERAL_FALLBACK)

    summary = registry.summary()
    assert summary["total_events"] == 3
    assert summary["by_category"]["pydantic_retry"] == 2
    assert summary["by_category"]["json_python_literal_fallback"] == 1
    assert summary["by_extractor"]["extract_quantitative"] == 2


def test_parse_llm_json_emits_python_literal_telemetry() -> None:
    """Successful python-literal fallback must emit telemetry when enabled."""
    init_telemetry_registry("run123")
    result = parse_llm_json("{'score': 2, 'ok': True}")
    assert result == {"score": 2, "ok": True}

    registry = get_telemetry_registry()
    categories = [event.category for event in registry.events]
    assert TelemetryCategory.JSON_PYTHON_LITERAL_FALLBACK in categories
