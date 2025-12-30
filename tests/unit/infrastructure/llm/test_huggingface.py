"""Tests for HuggingFace backend wiring.

These tests validate:
- Canonical model alias resolution for HuggingFace vs Ollama
- LLM client factory backend selection
- Deterministic behavior when HuggingFace optional deps are missing
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ai_psychiatrist.config import (
    BackendSettings,
    HuggingFaceSettings,
    LLMBackend,
    ModelSettings,
    Settings,
)
from ai_psychiatrist.infrastructure.llm import huggingface as hf_mod
from ai_psychiatrist.infrastructure.llm.factory import create_llm_client
from ai_psychiatrist.infrastructure.llm.huggingface import (
    HuggingFaceClient,
    MissingHuggingFaceDependenciesError,
)
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    EmbeddingBatchRequest,
    EmbeddingRequest,
)


class TestModelAliases:
    """Tests for canonical model alias mapping."""

    def test_resolves_gemma3_huggingface(self) -> None:
        """Should map canonical Gemma name to HF repo ID."""
        resolved = resolve_model_name("gemma3:27b", LLMBackend.HUGGINGFACE)
        assert resolved == "google/gemma-3-27b-it"

    def test_resolves_qwen_embedding_huggingface(self) -> None:
        """Should map canonical Qwen embedding name to HF repo ID."""
        resolved = resolve_model_name("qwen3-embedding:8b", LLMBackend.HUGGINGFACE)
        assert resolved == "Qwen/Qwen3-Embedding-8B"

    def test_allows_backend_native_identifier(self) -> None:
        """Unknown model identifiers should pass through unchanged."""
        assert (
            resolve_model_name("some-org/some-model", LLMBackend.HUGGINGFACE)
            == "some-org/some-model"
        )

    def test_rejects_medgemma_on_ollama(self) -> None:
        """MedGemma should be unavailable in the official Ollama library."""
        with pytest.raises(ValueError, match="not available"):
            resolve_model_name("medgemma:27b", LLMBackend.OLLAMA)


class TestLLMFactory:
    """Tests for LLM client factory selection."""

    def test_creates_ollama_client(self) -> None:
        """Should return OllamaClient when backend=ollama."""
        settings = Settings(backend=BackendSettings(backend=LLMBackend.OLLAMA))
        client = create_llm_client(settings)
        assert isinstance(client, OllamaClient)

    def test_creates_huggingface_client(self) -> None:
        """Should return HuggingFaceClient when backend=huggingface."""
        settings = Settings(backend=BackendSettings(backend=LLMBackend.HUGGINGFACE))
        client = create_llm_client(settings)
        assert isinstance(client, HuggingFaceClient)


class TestHuggingFaceClientMissingDeps:
    """Tests for behavior when HF optional deps are not installed."""

    @pytest.mark.asyncio
    async def test_chat_raises_when_deps_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise MissingHuggingFaceDependenciesError when deps unavailable."""

        def _missing(_name: str) -> Any:
            raise ModuleNotFoundError(_name)

        importlib_mod = cast("Any", hf_mod).importlib
        monkeypatch.setattr(importlib_mod, "import_module", _missing)

        client = HuggingFaceClient(
            backend_settings=BackendSettings(backend=LLMBackend.HUGGINGFACE),
            model_settings=ModelSettings(),
            huggingface_settings=HuggingFaceSettings(),
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gemma3:27b",
            timeout_seconds=1,
        )

        with pytest.raises(MissingHuggingFaceDependenciesError, match="optional dependencies"):
            await client.chat(request)

    @pytest.mark.asyncio
    async def test_embed_raises_when_deps_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise MissingHuggingFaceDependenciesError when deps unavailable."""

        def _missing(_name: str) -> Any:
            raise ModuleNotFoundError(_name)

        importlib_mod = cast("Any", hf_mod).importlib
        monkeypatch.setattr(importlib_mod, "import_module", _missing)

        client = HuggingFaceClient(
            backend_settings=BackendSettings(backend=LLMBackend.HUGGINGFACE),
            model_settings=ModelSettings(),
            huggingface_settings=HuggingFaceSettings(),
        )

        request = EmbeddingRequest(
            text="hello",
            model="qwen3-embedding:8b",
            timeout_seconds=1,
        )

        with pytest.raises(MissingHuggingFaceDependenciesError, match="optional dependencies"):
            await client.embed(request)

    @pytest.mark.asyncio
    async def test_embed_batch_raises_when_deps_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should raise MissingHuggingFaceDependenciesError when deps unavailable."""

        def _missing(_name: str) -> Any:
            raise ModuleNotFoundError(_name)

        importlib_mod = cast("Any", hf_mod).importlib
        monkeypatch.setattr(importlib_mod, "import_module", _missing)

        client = HuggingFaceClient(
            backend_settings=BackendSettings(backend=LLMBackend.HUGGINGFACE),
            model_settings=ModelSettings(),
            huggingface_settings=HuggingFaceSettings(),
        )

        request = EmbeddingBatchRequest(
            texts=["hello"],
            model="qwen3-embedding:8b",
            timeout_seconds=1,
        )

        with pytest.raises(MissingHuggingFaceDependenciesError, match="optional dependencies"):
            await client.embed_batch(request)
