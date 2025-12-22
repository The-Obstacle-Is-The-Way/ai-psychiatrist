"""Model alias resolution for different LLM backends.

The codebase uses canonical model names (e.g., ``gemma3:27b``) in configuration.
Different backends may require different identifiers (e.g., HuggingFace repo IDs).

This module resolves canonical names into backend-specific identifiers.
"""

from __future__ import annotations

from ai_psychiatrist.config import LLMBackend

# Canonical name -> backend-specific identifier (or None if unsupported)
MODEL_ALIASES: dict[str, dict[LLMBackend, str | None]] = {
    "gemma3:27b": {
        LLMBackend.OLLAMA: "gemma3:27b",
        LLMBackend.HUGGINGFACE: "google/gemma-3-27b-it",
    },
    # MedGemma is not available in the official Ollama library (community uploads exist).
    "medgemma:27b": {
        LLMBackend.OLLAMA: None,
        LLMBackend.HUGGINGFACE: "google/medgemma-27b-text-it",
    },
    "qwen3-embedding:8b": {
        LLMBackend.OLLAMA: "qwen3-embedding:8b",
        LLMBackend.HUGGINGFACE: "Qwen/Qwen3-Embedding-8B",
    },
}


def resolve_model_name(model: str, backend: LLMBackend) -> str:
    """Resolve a canonical model name to a backend-specific name.

    If ``model`` is not a canonical alias, it is returned unchanged. This enables
    advanced users to provide backend-native identifiers directly.
    """
    aliases = MODEL_ALIASES.get(model)
    if aliases is None:
        return model

    resolved = aliases.get(backend)
    if resolved is None:
        msg = f"Model '{model}' is not available for backend '{backend.value}'"
        raise ValueError(msg)
    return resolved
