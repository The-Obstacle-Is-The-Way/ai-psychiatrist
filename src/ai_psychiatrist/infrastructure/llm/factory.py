"""LLM client factory.

Creates a concrete LLM client based on configuration. This keeps backend selection
out of business logic and enables swapping implementations via the Strategy pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_psychiatrist.config import EmbeddingBackend, LLMBackend, Settings
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import (
        EmbeddingClient,
        LLMClient,
    )


def create_llm_client(settings: Settings) -> LLMClient:
    """Create an LLM client based on settings.backend.backend."""
    backend = settings.backend.backend
    if backend == LLMBackend.OLLAMA:
        return OllamaClient(settings.ollama, model_settings=settings.model)
    if backend == LLMBackend.HUGGINGFACE:
        try:
            from ai_psychiatrist.infrastructure.llm.huggingface import (  # noqa: PLC0415
                HuggingFaceClient,
            )
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires optional dependencies. "
                "Install with `make dev` (repo) or `uv sync --extra hf`, "
                "or `pip install 'ai-psychiatrist[hf]'`."
            ) from e

        # Fail fast: selecting HuggingFace backend implies these deps must exist.
        # This avoids wasting an entire run only to error on the first embed/chat call.
        from ai_psychiatrist.infrastructure.llm import huggingface as hf_mod  # noqa: PLC0415

        hf_mod._load_transformers_deps()

        return HuggingFaceClient(
            backend_settings=settings.backend,
            model_settings=settings.model,
            huggingface_settings=settings.huggingface,
        )

    msg = f"Unsupported LLM backend: {backend}"
    raise ValueError(msg)


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create embedding client based on EMBEDDING_BACKEND.

    Separate from create_llm_client() to allow different backends
    for chat vs embeddings.
    """
    backend = settings.embedding_config.backend

    if backend == EmbeddingBackend.OLLAMA:
        return OllamaClient(settings.ollama, model_settings=settings.model)

    if backend == EmbeddingBackend.HUGGINGFACE:
        # Lazy import to avoid requiring HF deps when using Ollama
        try:
            from ai_psychiatrist.infrastructure.llm.huggingface import (  # noqa: PLC0415
                HuggingFaceClient,
            )
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires optional dependencies. "
                "Install with `make dev` (repo) or `uv sync --extra hf`, "
                "or `pip install 'ai-psychiatrist[hf]'`."
            ) from e

        # Fail fast: using HF embeddings requires torch/transformers installed.
        from ai_psychiatrist.infrastructure.llm import huggingface as hf_mod  # noqa: PLC0415

        hf_mod._load_transformers_deps()

        # HuggingFaceClient takes (backend_settings, model_settings)
        return HuggingFaceClient(
            backend_settings=settings.backend,
            model_settings=settings.model,
            huggingface_settings=settings.huggingface,
        )

    raise ValueError(f"Unknown embedding backend: {backend}")
