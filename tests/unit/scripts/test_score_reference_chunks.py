"""Unit tests for scripts/score_reference_chunks.py parsing behavior."""

from __future__ import annotations

import json

import pytest
from scripts.score_reference_chunks import score_chunk

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
