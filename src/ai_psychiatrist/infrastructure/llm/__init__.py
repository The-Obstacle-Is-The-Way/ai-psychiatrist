"""LLM infrastructure layer.

This module provides abstractions for LLM interactions using the Strategy pattern.
It enables swapping between Ollama, OpenAI, or mock implementations without
changing business logic.

Paper Reference:
    - Section 2.2: Gemma 3 27B for chat, Qwen 3 8B for embeddings
    - Section 2.3.5: Ollama API integration
    - Appendix F: MedGemma 27B achieves 18% better MAE (0.505 vs 0.619)
"""

from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient
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
    "ChatClient",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "EmbeddingClient",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "LLMClient",
    "MockLLMClient",
    "OllamaClient",
    "SimpleChatClient",
    "extract_json_from_response",
    "extract_score_from_text",
    "extract_xml_tags",
]
