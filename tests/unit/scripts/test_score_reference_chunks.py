"""Unit tests for scripts/score_reference_chunks.py parsing behavior."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from scripts.score_reference_chunks import score_chunk, validate_cli_args

from ai_psychiatrist.services.chunk_scoring import PHQ8_ITEM_KEY_SET

pytestmark = pytest.mark.unit


class _FakeChatClient:
    def __init__(self, response: str) -> None:
        self._response = response

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.0,
        format: str | None = None,
    ) -> str:
        _ = (user_prompt, system_prompt, model, temperature, format)
        return self._response


def _valid_payload() -> dict[str, int | None]:
    payload: dict[str, int | None] = dict.fromkeys(PHQ8_ITEM_KEY_SET)
    payload["PHQ8_Sleep"] = 2
    return payload


@pytest.mark.asyncio
async def test_score_chunk_parses_raw_json() -> None:
    payload = _valid_payload()
    client = _FakeChatClient(json.dumps(payload))

    result = await score_chunk(client=client, chunk_text="chunk", model="m", temperature=0.0)

    assert result == payload


@pytest.mark.asyncio
async def test_score_chunk_parses_fenced_json() -> None:
    payload = _valid_payload()
    response = f"```json\n{json.dumps(payload)}\n```"
    client = _FakeChatClient(response)

    result = await score_chunk(client=client, chunk_text="chunk", model="m", temperature=0.0)

    assert result == payload


@pytest.mark.asyncio
async def test_score_chunk_parses_json_with_leading_prose_and_fence() -> None:
    payload = _valid_payload()
    response = f"Sure, here's the JSON:\n```json\n{json.dumps(payload)}\n```"
    client = _FakeChatClient(response)

    result = await score_chunk(client=client, chunk_text="chunk", model="m", temperature=0.0)

    assert result == payload


@pytest.mark.asyncio
async def test_score_chunk_returns_none_on_invalid_schema() -> None:
    payload = _valid_payload()
    payload.pop("PHQ8_Sleep")
    client = _FakeChatClient(json.dumps(payload))

    result = await score_chunk(client=client, chunk_text="chunk", model="m", temperature=0.0)

    assert result is None


@pytest.mark.asyncio
async def test_score_chunk_returns_none_on_invalid_value() -> None:
    payload = _valid_payload()
    payload["PHQ8_Sleep"] = 4  # invalid range
    client = _FakeChatClient(json.dumps(payload))

    result = await score_chunk(client=client, chunk_text="chunk", model="m", temperature=0.0)

    assert result is None


def test_validate_cli_args_rejects_limit_zero() -> None:
    with pytest.raises(ValueError, match="--limit"):
        validate_cli_args(Namespace(limit=0))


@pytest.mark.asyncio
async def test_score_chunk_logs_are_privacy_safe_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never log raw chunk text or response previews (privacy-safe observability)."""
    chunk_text = "CHUNK_SECRET_TEXT_SHOULD_NOT_APPEAR"
    response = "RESPONSE_SECRET_TEXT_SHOULD_NOT_APPEAR (not json)"
    client = _FakeChatClient(response)

    logger_mock = MagicMock()
    monkeypatch.setattr("scripts.score_reference_chunks.logger", logger_mock)

    result = await score_chunk(client=client, chunk_text=chunk_text, model="m", temperature=0.0)

    assert result is None

    warning_calls = [
        call
        for call in logger_mock.warning.call_args_list
        if call.args and call.args[0] == "Scoring failed: invalid JSON response"
    ]
    assert len(warning_calls) == 1

    call = warning_calls[0]
    assert call.kwargs.get("chunk_preview") is None
    assert call.kwargs.get("response_preview") is None
    assert chunk_text not in str(call.kwargs)
    assert response not in str(call.kwargs)
    assert isinstance(call.kwargs.get("chunk_hash"), str)
    assert call.kwargs.get("chunk_chars") == len(chunk_text)
    assert isinstance(call.kwargs.get("response_hash"), str)
    assert call.kwargs.get("response_chars") == len(response)
