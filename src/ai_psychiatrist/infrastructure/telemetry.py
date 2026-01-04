"""Retry and JSON parsing telemetry (Spec 060).

This module provides a privacy-safe telemetry registry intended to answer:

- How often structured-output validation triggers retries (PydanticAI ModelRetry).
- How often JSON repair paths are used (fixups / fallbacks).

Telemetry is strictly additive: it must not change runtime behavior or evaluation outputs.
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

_SUMMARY_WIDTH: Final[int] = 60
_DEFAULT_MAX_EVENTS: Final[int] = 5000
_TOP_N_EXTRACTORS: Final[int] = 10


class TelemetryCategory(str, Enum):
    """Telemetry event categories."""

    # JSON parsing / repair path visibility
    JSON_FIXUPS_APPLIED = "json_fixups_applied"
    JSON_PYTHON_LITERAL_FALLBACK = "json_python_literal_fallback"
    JSON_REPAIR_FALLBACK = "json_repair_fallback"

    # Structured output retries
    PYDANTIC_RETRY = "pydantic_retry"


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Single telemetry event."""

    category: TelemetryCategory
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extractor: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "timestamp": self.timestamp,
            "extractor": self.extractor,
            "context": self.context,
        }


@dataclass
class TelemetryRegistry:
    """Collects telemetry events for a run."""

    run_id: str
    events: list[TelemetryEvent] = field(default_factory=list)
    max_events: int = _DEFAULT_MAX_EVENTS
    dropped_events: int = 0
    _start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def record(
        self,
        category: TelemetryCategory,
        *,
        extractor: str | None = None,
        **context: Any,
    ) -> None:
        if len(self.events) < self.max_events:
            self.events.append(
                TelemetryEvent(category=category, extractor=extractor, context=context)
            )
            return
        self.dropped_events += 1

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        by_extractor: dict[str, int] = {}

        for event in self.events:
            by_category[event.category.value] = by_category.get(event.category.value, 0) + 1
            if event.extractor:
                by_extractor[event.extractor] = by_extractor.get(event.extractor, 0) + 1

        return {
            "run_id": self.run_id,
            "start_time": self._start_time,
            "end_time": datetime.now(UTC).isoformat(),
            "total_events": len(self.events),
            "dropped_events": self.dropped_events,
            "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
            "by_extractor": dict(sorted(by_extractor.items(), key=lambda x: -x[1])),
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"telemetry_{self.run_id}.json"
        data = {
            "summary": self.summary(),
            "events": [event.to_dict() for event in self.events],
        }
        output_path.write_text(json.dumps(data, indent=2))
        return output_path

    def print_summary(self) -> None:
        summary = self.summary()
        print("\n" + "=" * _SUMMARY_WIDTH)
        print("TELEMETRY SUMMARY")
        print("=" * _SUMMARY_WIDTH)
        print(f"Run ID: {summary['run_id']}")
        print(f"Total events: {summary['total_events']}")
        if summary.get("dropped_events"):
            print(f"Dropped events (cap={self.max_events}): {summary['dropped_events']}")
        by_category = summary.get("by_category", {})
        if by_category:
            print("\nBy Category:")
            for category, count in by_category.items():
                print(f"  {category}: {count}")
        by_extractor = summary.get("by_extractor", {})
        if by_extractor:
            print("\nBy Extractor:")
            for extractor, count in list(by_extractor.items())[:_TOP_N_EXTRACTORS]:
                print(f"  {extractor}: {count}")
        print("=" * _SUMMARY_WIDTH + "\n")


_registry_var: contextvars.ContextVar[TelemetryRegistry | None] = contextvars.ContextVar(
    "ai_psychiatrist_telemetry_registry",
    default=None,
)


def get_telemetry_registry() -> TelemetryRegistry:
    registry = _registry_var.get()
    if registry is None:
        raise RuntimeError(
            "Telemetry registry not initialized. Call init_telemetry_registry() first."
        )
    return registry


def init_telemetry_registry(run_id: str) -> TelemetryRegistry:
    registry = TelemetryRegistry(run_id=run_id)
    _registry_var.set(registry)
    return registry


def clear_telemetry_registry() -> None:
    """Clear the telemetry registry for the current context.

    This is primarily intended for unit tests to avoid test-order dependence.
    """

    _registry_var.set(None)


def record_telemetry(category: TelemetryCategory, **kwargs: Any) -> None:
    """Record telemetry if initialized; no-op otherwise."""
    try:
        registry = get_telemetry_registry()
        registry.record(category, **kwargs)
    except RuntimeError:
        # Do not forward arbitrary kwargs to logs: structlog reserves keys like "event",
        # and kwargs may include sensitive values in future callers. Keep this minimal.
        logger.debug("telemetry_registry_not_initialized", category=category.value)
