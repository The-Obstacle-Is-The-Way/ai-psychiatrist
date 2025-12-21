from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from ai_psychiatrist.infrastructure.llm.responses import extract_xml_tags

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings
    from ai_psychiatrist.infrastructure.llm import OllamaClient


@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.slow
class TestOllamaSmoke:
    async def test_simple_chat_returns_expected_xml_answer_tag(
        self,
        ollama_client: OllamaClient,
        app_settings: Settings,
    ) -> None:
        response = await ollama_client.simple_chat(
            user_prompt="Reply with exactly: <answer>OK</answer>",
            system_prompt="You are a deterministic test harness. Follow the user instruction exactly.",
            model=app_settings.model.qualitative_model,
            temperature=0.0,
        )

        tags = extract_xml_tags(response, ["answer"])
        assert tags["answer"].strip() == "OK"

    async def test_simple_embed_dimension_and_l2_norm(
        self,
        ollama_client: OllamaClient,
        app_settings: Settings,
    ) -> None:
        embedding = await ollama_client.simple_embed(
            text="hello world (embedding smoke test)",
            model=app_settings.model.embedding_model,
            dimension=app_settings.embedding.dimension,
        )

        assert len(embedding) == app_settings.embedding.dimension
        norm = math.sqrt(sum(x * x for x in embedding))
        assert norm == pytest.approx(1.0, abs=1e-3)
