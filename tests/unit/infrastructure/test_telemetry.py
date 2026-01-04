"""Unit tests for retry telemetry (Spec 060)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path


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


def test_registry_event_cap_tracks_dropped_events() -> None:
    registry = TelemetryRegistry(run_id="test", max_events=2)
    registry.record(TelemetryCategory.PYDANTIC_RETRY, extractor="extract_quantitative")
    registry.record(TelemetryCategory.PYDANTIC_RETRY, extractor="extract_quantitative")
    registry.record(TelemetryCategory.JSON_REPAIR_FALLBACK)

    assert len(registry.events) == 2
    assert registry.dropped_events == 1
    assert registry.summary()["dropped_events"] == 1


def test_registry_save_writes_summary_and_events(tmp_path: Path) -> None:
    """save() should write a stable JSON file containing summary + events."""
    registry = TelemetryRegistry(run_id="abc123")
    registry.record(TelemetryCategory.PYDANTIC_RETRY, extractor="extract_quantitative")

    out = registry.save(tmp_path)
    assert out.exists()

    data = json.loads(out.read_text())
    assert set(data.keys()) == {"summary", "events"}
    assert data["summary"]["run_id"] == "abc123"
    assert data["summary"]["total_events"] == 1
    assert data["events"][0]["category"] == TelemetryCategory.PYDANTIC_RETRY.value
