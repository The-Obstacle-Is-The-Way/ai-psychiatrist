"""Integration tests for PydanticAI retry behavior + telemetry wiring.

These tests exercise the real PydanticAI retry loop (ModelRetry → retry prompt)
and verify that our privacy-safe telemetry is emitted for:
1) structured output retries (Spec 060)
2) json-repair fallback parsing (Spec 059)

We run the agent in an isolated contextvars Context to avoid leaking a telemetry
registry into other tests that intentionally assert the "uninitialized" behavior.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING

import pytest
from pydantic_ai import Agent, TextOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from ai_psychiatrist.agents.extractors import extract_quantitative
from ai_psychiatrist.infrastructure.telemetry import (
    TelemetryCategory,
    init_telemetry_registry,
)

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from ai_psychiatrist.agents.output_models import QuantitativeOutput


@pytest.mark.asyncio
async def test_pydantic_ai_retries_emit_telemetry_without_leaking_registry() -> None:
    """Retries must emit telemetry once per attempt and not pollute other tests."""
    model = TestModel(custom_output_text="no answer tags here")
    agent: Agent[None, QuantitativeOutput] = Agent(
        model=model,
        output_type=TextOutput(extract_quantitative),
        retries=2,
        system_prompt="test",
    )

    ctx = contextvars.copy_context()
    registry = ctx.run(init_telemetry_registry, "run_test")
    task = ctx.run(asyncio.create_task, agent.run("prompt"))

    with pytest.raises(UnexpectedModelBehavior, match=r"Exceeded maximum retries \(2\)"):
        await task

    assert len(registry.events) == 3  # initial attempt + 2 retries
    assert all(event.category == TelemetryCategory.PYDANTIC_RETRY for event in registry.events)
    assert all(event.extractor == "extract_quantitative" for event in registry.events)
    assert all(event.context.get("reason") == "missing_structure" for event in registry.events)


@pytest.mark.asyncio
async def test_pydantic_ai_json_repair_fallback_allows_success_without_retry() -> None:
    """json-repair should recover malformed output inside <answer> without triggering retries."""
    # Valid QuantitativeOutput payload, but with a stray backslash + trailing text appended.
    # This breaks both json.loads(...) and ast.literal_eval(...) but is recovered by json-repair.
    response = """
<answer>
{
  "PHQ8_NoInterest": {"score": 0},
  "PHQ8_Depressed": {"score": 0},
  "PHQ8_Sleep": {"score": 0},
  "PHQ8_Tired": {"score": 0},
  "PHQ8_Appetite": {"score": "N/A"},
  "PHQ8_Failure": {"score": 0},
  "PHQ8_Concentrating": {"score": 0},
  "PHQ8_Moving": {"score": "N/A"}
}\\ trailing-junk
</answer>
"""
    model = TestModel(custom_output_text=response)
    agent: Agent[None, QuantitativeOutput] = Agent(
        model=model,
        output_type=TextOutput(extract_quantitative),
        retries=0,
        system_prompt="test",
    )

    ctx = contextvars.copy_context()
    registry = ctx.run(init_telemetry_registry, "run_test")
    task = ctx.run(asyncio.create_task, agent.run("prompt"))
    result = await task

    assert result.output.PHQ8_Appetite.score is None
    assert result.output.PHQ8_Moving.score is None

    categories = [event.category for event in registry.events]
    assert TelemetryCategory.JSON_REPAIR_FALLBACK in categories
    assert TelemetryCategory.PYDANTIC_RETRY not in categories
