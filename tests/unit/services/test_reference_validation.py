"""Tests for reference validation (Spec 38 fail-fast behavior)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.services.reference_validation import (
    LLMReferenceValidator,
    ReferenceValidationRequest,
)


@pytest.mark.unit
class TestLLMReferenceValidator:
    @pytest.mark.asyncio
    async def test_exceptions_propagate(self) -> None:
        """When enabled, reference validation should crash on errors (no silent fallback)."""
        client = MagicMock()
        client.simple_chat = AsyncMock(side_effect=RuntimeError("boom"))

        validator = LLMReferenceValidator(client=client, model="mock")
        request = ReferenceValidationRequest(
            item=PHQ8Item.SLEEP,
            evidence_text="evidence",
            reference_text="reference",
            reference_score=2,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await validator.validate(request)

    @pytest.mark.asyncio
    async def test_invalid_json_crashes(self) -> None:
        """Invalid JSON must raise LLMResponseParseError (no silent 'unsure')."""
        client = MagicMock()
        client.simple_chat = AsyncMock(return_value="not json")

        validator = LLMReferenceValidator(client=client, model="mock")
        request = ReferenceValidationRequest(
            item=PHQ8Item.SLEEP,
            evidence_text="evidence",
            reference_text="reference",
            reference_score=2,
        )

        with pytest.raises(LLMResponseParseError):
            await validator.validate(request)
