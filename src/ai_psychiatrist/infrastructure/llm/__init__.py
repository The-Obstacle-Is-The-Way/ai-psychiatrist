"""LLM infrastructure layer.

This module provides abstractions for LLM interactions using the Strategy pattern.
It enables swapping between Ollama, OpenAI, or mock implementations without
changing business logic.

Paper Reference:
    - Section 2.2: Gemma 3 27B for chat, Qwen 3 8B for embeddings
    - Section 2.3.5: Ollama API integration
    - Appendix F: MedGemma 27B achieves 18% better MAE (0.505 vs 0.619)

Test Double Location Policy (BUG-001):
    MockLLMClient is a TEST-ONLY artifact and lives in tests/fixtures/mock_llm.py.
    It MUST NOT be exported from this production module.
    See: docs/developer/testing.md
"""

from ai_psychiatrist.infrastructure.llm.factory import create_llm_client
from ai_psychiatrist.infrastructure.llm.model_aliases import MODEL_ALIASES, resolve_model_name
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatClient,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingClient,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMClient,
)
from ai_psychiatrist.infrastructure.llm.responses import (
    SimpleChatClient,
    extract_json_from_response,
    extract_score_from_text,
    extract_xml_tags,
)

__all__ = [
    "MODEL_ALIASES",
    "ChatClient",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "EmbeddingClient",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "LLMClient",
    "OllamaClient",
    "SimpleChatClient",
    "create_llm_client",
    "extract_json_from_response",
    "extract_score_from_text",
    "extract_xml_tags",
    "resolve_model_name",
]
