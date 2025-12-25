"""Tests for LLM client factory."""

from unittest.mock import patch

import pytest

from ai_psychiatrist.config import (
    EmbeddingBackend,
    LLMBackend,
    Settings,
)
from ai_psychiatrist.infrastructure.llm.factory import (
    create_embedding_client,
    create_llm_client,
)
from ai_psychiatrist.infrastructure.llm.huggingface import HuggingFaceClient
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient


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

    def test_create_hf_client(self, settings: Settings) -> None:
        """Should create HuggingFaceClient when backend is HF."""
        settings.backend.backend = LLMBackend.HUGGINGFACE
        client = create_llm_client(settings)
        assert isinstance(client, HuggingFaceClient)


class TestCreateEmbeddingClient:
    """Tests for create_embedding_client."""

    def test_create_ollama_client(self, settings: Settings) -> None:
        """Should create OllamaClient when backend is Ollama."""
        settings.embedding_backend.backend = EmbeddingBackend.OLLAMA
        client = create_embedding_client(settings)
        assert isinstance(client, OllamaClient)

    def test_create_hf_client(self, settings: Settings) -> None:
        """Should create HuggingFaceClient when backend is HF."""
        settings.embedding_backend.backend = EmbeddingBackend.HUGGINGFACE
        # Mock HF import to verify it is used
        with patch("ai_psychiatrist.infrastructure.llm.huggingface.HuggingFaceClient") as MockHF:
            client = create_embedding_client(settings)
            assert client == MockHF.return_value
            # Verify constructor args
            MockHF.assert_called_once_with(
                backend_settings=settings.backend,
                model_settings=settings.model,
            )

    def test_lazy_import_failure(self, settings: Settings) -> None:
        """Should raise ImportError with helpful message if HF deps missing."""
        settings.embedding_backend.backend = EmbeddingBackend.HUGGINGFACE

        # Simulate ImportError when importing HuggingFaceClient
        with patch.dict("sys.modules", {"ai_psychiatrist.infrastructure.llm.huggingface": None}):
            # We need to ensure the module isn't already imported/cached
            with pytest.raises(ImportError) as exc:
                # Note: This test is tricky because HuggingFaceClient is already imported
                # at top level in factory.py currently (which we need to fix).
                # For now, this test expects the behavior AFTER we fix the code.
                create_embedding_client(settings)

            assert "pip install 'ai-psychiatrist[hf]'" in str(exc.value)

    def test_unknown_backend(self, settings: Settings) -> None:
        """Should raise ValueError for unknown backend."""
        # Using a mock enum value to simulate unknown
        settings.embedding_backend.backend = "unknown"  # type: ignore
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            create_embedding_client(settings)
