"""Abstract protocols for LLM interactions.

This module defines the core abstractions for LLM clients using Python Protocols
(structural subtyping). This enables the Strategy pattern - swapping implementations
without changing business logic.

Paper Reference:
    - Section 2.2: Gemma 3 27B for chat, Qwen 3 8B for embeddings
    - Section 2.3.5: Ollama API integration
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message in a chat conversation.

    Immutable value object representing a single message in the chat history.

    Attributes:
        role: Message role - "system", "user", or "assistant".
        content: The message text content.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if self.role not in ("system", "user", "assistant"):
            msg = f"Invalid role '{self.role}', must be 'system', 'user', or 'assistant'"
            raise ValueError(msg)
        if not self.content:
            msg = "Message content cannot be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Request for chat completion.

    Immutable value object containing all parameters for a chat completion request.

    Attributes:
        messages: Sequence of chat messages forming the conversation.
        model: Model identifier (e.g., "gemma3:27b").
        temperature: Sampling temperature (0.0 = deterministic, higher = more random).
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        timeout_seconds: Request timeout in seconds.
    """

    messages: Sequence[ChatMessage]
    model: str
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.8
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        """Validate request after initialization."""
        if not self.messages:
            msg = "Messages cannot be empty"
            raise ValueError(msg)
        if not self.model:
            msg = "Model cannot be empty"
            raise ValueError(msg)
        if not 0.0 <= self.temperature <= 2.0:
            msg = f"Temperature {self.temperature} must be between 0.0 and 2.0"
            raise ValueError(msg)
        if not 1 <= self.top_k <= 100:
            msg = f"top_k {self.top_k} must be between 1 and 100"
            raise ValueError(msg)
        if not 0.0 <= self.top_p <= 1.0:
            msg = f"top_p {self.top_p} must be between 0.0 and 1.0"
            raise ValueError(msg)
        if self.timeout_seconds < 1:
            msg = f"timeout_seconds {self.timeout_seconds} must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Response from chat completion.

    Immutable value object containing the LLM response and metadata.

    Attributes:
        content: Generated text content.
        model: Model that generated the response.
        done: Whether generation is complete.
        total_duration_ms: Total processing time in milliseconds.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
    """

    content: str
    model: str
    done: bool = True
    total_duration_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Request for text embedding.

    Immutable value object containing parameters for an embedding request.

    Attributes:
        text: Text to embed.
        model: Model identifier (e.g., "qwen3-embedding:8b").
        dimension: Optional dimension truncation for MRL support.
        timeout_seconds: Request timeout in seconds.
    """

    text: str
    model: str
    dimension: int | None = None
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        """Validate request after initialization."""
        if not self.text:
            msg = "Text cannot be empty"
            raise ValueError(msg)
        if not self.model:
            msg = "Model cannot be empty"
            raise ValueError(msg)
        if self.dimension is not None and self.dimension < 1:
            msg = f"dimension {self.dimension} must be >= 1"
            raise ValueError(msg)
        if self.timeout_seconds < 1:
            msg = f"timeout_seconds {self.timeout_seconds} must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Response from embedding request.

    Immutable value object containing the embedding vector and metadata.

    Attributes:
        embedding: L2-normalized embedding vector as tuple.
        model: Model that generated the embedding.
        dimension: Computed dimension from embedding length.
    """

    embedding: tuple[float, ...]
    model: str
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        """Set dimension from embedding length and validate."""
        if not self.embedding:
            msg = "Embedding cannot be empty"
            raise ValueError(msg)
        object.__setattr__(self, "dimension", len(self.embedding))


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat completion clients.

    Any class implementing this protocol can be used as a chat client.
    Uses structural subtyping - no explicit inheritance required.
    """

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute chat completion.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            Chat response with generated content.

        Raises:
            LLMError: If request fails.
            LLMTimeoutError: If request times out.
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients.

    Any class implementing this protocol can be used as an embedding client.
    Uses structural subtyping - no explicit inheritance required.
    """

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding for text.

        Args:
            request: Embedding request with text and parameters.

        Returns:
            Embedding response with L2-normalized vector.

        Raises:
            LLMError: If request fails.
            LLMTimeoutError: If request times out.
        """
        ...


@runtime_checkable
class LLMClient(ChatClient, EmbeddingClient, Protocol):
    """Combined protocol for full LLM client.

    Clients implementing both chat and embedding capabilities.
    This is the typical interface for production use.
    """

    pass
