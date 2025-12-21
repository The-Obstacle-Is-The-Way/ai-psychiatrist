"""Fixtures for real Ollama end-to-end tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ai_psychiatrist.config import Settings
    from ai_psychiatrist.infrastructure.llm import OllamaClient


def _require_ollama_opt_in() -> None:
    if os.environ.get("AI_PSYCHIATRIST_OLLAMA_TESTS") != "1":
        pytest.skip("Set AI_PSYCHIATRIST_OLLAMA_TESTS=1 to run live Ollama tests")


@pytest.fixture(scope="session")
def app_settings() -> Settings:
    """Return application settings loaded from env / .env (real mode only)."""
    _require_ollama_opt_in()
    from ai_psychiatrist.config import get_settings  # noqa: PLC0415

    return get_settings()


@pytest.fixture
async def ollama_client(app_settings: Settings) -> AsyncIterator[OllamaClient]:
    """Return a live OllamaClient connected to a reachable Ollama server.

    Fails with actionable guidance if Ollama is unreachable or required models are missing.
    """
    _require_ollama_opt_in()

    from ai_psychiatrist.domain.exceptions import LLMError  # noqa: PLC0415
    from ai_psychiatrist.infrastructure.llm import OllamaClient  # noqa: PLC0415

    client = OllamaClient(app_settings.ollama)

    try:
        await client.ping()
    except LLMError as e:
        await client.close()
        pytest.fail(
            "Ollama is not reachable. Start it with `ollama serve` or "
            f"`brew services start ollama`. Error: {e}"
        )

    models = await client.list_models()
    available = {m.get("name") for m in models if isinstance(m, dict)}

    required = {
        app_settings.model.qualitative_model,
        app_settings.model.judge_model,
        app_settings.model.meta_review_model,
        app_settings.model.quantitative_model,
        app_settings.model.embedding_model,
    }

    missing = {m for m in required if m and m not in available}
    if missing:
        await client.close()
        pull_cmds = "\n".join(f"ollama pull {m}" for m in sorted(missing))
        pytest.fail(
            "Missing required Ollama models for e2e tests.\n"
            f"Missing: {sorted(missing)}\n"
            "Pull them with:\n"
            f"{pull_cmds}"
        )

    yield client

    await client.close()
