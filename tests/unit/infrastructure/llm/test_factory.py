"""Tests for LLM client factory."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from ai_psychiatrist.config import (
    EmbeddingBackend,
    LLMBackend,
    Settings,
)
from ai_psychiatrist.infrastructure.llm import huggingface as hf_mod
from ai_psychiatrist.infrastructure.llm.factory import (
    create_embedding_client,
    create_llm_client,
)
from ai_psychiatrist.infrastructure.llm.huggingface import (
    HuggingFaceClient,
    MissingHuggingFaceDependenciesError,
)
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient

pytestmark = pytest.mark.unit


def _patch_hf_deps_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing heavyweight HF deps during factory unit tests."""
    monkeypatch.setattr(hf_mod, "_load_transformers_deps", lambda: object())


def _patch_hf_deps_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate missing HF optional dependencies (e.g., torch not installed)."""

    def _missing(_name: str) -> Any:
        raise ModuleNotFoundError(_name)

    importlib_mod = cast("Any", hf_mod).importlib
    monkeypatch.setattr(importlib_mod, "import_module", _missing)


@pytest.fixture
def settings() -> Settings:
    """Create basic settings fixture."""
    return Settings()


class TestCreateLLMClient:
    """Tests for existing create_llm_client."""

    def test_create_ollama_client(self, settings: Settings) -> None:
        """Should create OllamaClient when backend is Ollama."""
        settings.backend.backend = LLMBackend.OLLAMA
        client = create_llm_client(settings)
        assert isinstance(client, OllamaClient)

    def test_create_hf_client(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should create HuggingFaceClient when backend is HF."""
        settings.backend.backend = LLMBackend.HUGGINGFACE
        _patch_hf_deps_available(monkeypatch)
        client = create_llm_client(settings)
        assert isinstance(client, HuggingFaceClient)

    def test_create_hf_client_fails_fast_when_deps_missing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fail fast (before a long run) when torch/transformers deps are missing."""
        settings.backend.backend = LLMBackend.HUGGINGFACE
        _patch_hf_deps_missing(monkeypatch)

        with pytest.raises(MissingHuggingFaceDependenciesError, match="optional dependencies"):
            create_llm_client(settings)


class TestCreateEmbeddingClient:
    """Tests for create_embedding_client."""

    def test_create_ollama_client(self, settings: Settings) -> None:
        """Should create OllamaClient when backend is Ollama."""
        settings.embedding_config.backend = EmbeddingBackend.OLLAMA
        client = create_embedding_client(settings)
        assert isinstance(client, OllamaClient)

    def test_create_hf_client(self, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should create HuggingFaceClient when backend is HF."""
        settings.embedding_config.backend = EmbeddingBackend.HUGGINGFACE
        _patch_hf_deps_available(monkeypatch)
        # Mock HF import to verify it is used
        with patch("ai_psychiatrist.infrastructure.llm.huggingface.HuggingFaceClient") as MockHF:
            client = create_embedding_client(settings)
            assert client == MockHF.return_value
            # Verify constructor args
            MockHF.assert_called_once_with(
                backend_settings=settings.backend,
                model_settings=settings.model,
                huggingface_settings=settings.huggingface,
            )

    def test_create_hf_client_fails_fast_when_deps_missing(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fail fast when HF deps are missing (Run10 failure mode)."""
        settings.embedding_config.backend = EmbeddingBackend.HUGGINGFACE
        _patch_hf_deps_missing(monkeypatch)

        with pytest.raises(MissingHuggingFaceDependenciesError, match="optional dependencies"):
            create_embedding_client(settings)

    def test_lazy_import_failure(self, settings: Settings) -> None:
        """Should raise ImportError with helpful message if HF deps missing."""
        settings.embedding_config.backend = EmbeddingBackend.HUGGINGFACE

        # Simulate ImportError when importing HuggingFaceClient
        with patch.dict("sys.modules", {"ai_psychiatrist.infrastructure.llm.huggingface": None}):
            # We need to ensure the module isn't already imported/cached
            with pytest.raises(ImportError) as exc:
                create_embedding_client(settings)

            assert "pip install 'ai-psychiatrist[hf]'" in str(exc.value)

    def test_unknown_backend(self, settings: Settings) -> None:
        """Should raise ValueError for unknown backend."""
        # Using a mock enum value to simulate unknown
        settings.embedding_config.backend = "unknown"  # type: ignore
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            create_embedding_client(settings)
