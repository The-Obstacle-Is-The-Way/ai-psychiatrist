"""LLM client factory.

Creates a concrete LLM client based on configuration. This keeps backend selection
out of business logic and enables swapping implementations via the Strategy pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_psychiatrist.config import LLMBackend, Settings
from ai_psychiatrist.infrastructure.llm.huggingface import HuggingFaceClient
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Create an LLM client based on settings.backend.backend."""
    backend = settings.backend.backend
    if backend == LLMBackend.OLLAMA:
        return OllamaClient(settings.ollama)
    if backend == LLMBackend.HUGGINGFACE:
        return HuggingFaceClient(
            backend_settings=settings.backend,
            model_settings=settings.model,
        )

    msg = f"Unsupported LLM backend: {backend}"
    raise ValueError(msg)
