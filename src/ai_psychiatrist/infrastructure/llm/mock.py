"""Mock LLM client for testing.

This module provides test doubles for LLM clients, enabling unit testing
without real LLM calls. Follows the principles of test isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class MockLLMClient:
    """Mock LLM client for testing.

    Allows specifying canned responses or response functions.
    Tracks all requests for assertion in tests.

    Example:
        >>> mock = MockLLMClient(chat_responses=["Hello!"])
        >>> response = await mock.simple_chat("Hi")
        >>> assert response == "Hello!"
        >>> assert mock.chat_call_count == 1
    """

    def __init__(
        self,
        chat_responses: list[str | ChatResponse] | None = None,
        embedding_responses: list[tuple[float, ...] | EmbeddingResponse] | None = None,
        chat_function: Callable[[ChatRequest], str] | None = None,
        embedding_function: Callable[[EmbeddingRequest], tuple[float, ...]] | None = None,
    ) -> None:
        """Initialize mock client.

        Args:
            chat_responses: List of responses to return in order.
            embedding_responses: List of embeddings to return in order.
            chat_function: Custom function to generate responses.
            embedding_function: Custom function to generate embeddings.
        """
        self._chat_responses = list(chat_responses or [])
        self._embedding_responses = list(embedding_responses or [])
        self._chat_function = chat_function
        self._embedding_function = embedding_function
        self._chat_call_count = 0
        self._embedding_call_count = 0
        self._chat_requests: list[ChatRequest] = []
        self._embedding_requests: list[EmbeddingRequest] = []

    @property
    def chat_call_count(self) -> int:
        """Number of chat calls made."""
        return self._chat_call_count

    @property
    def embedding_call_count(self) -> int:
        """Number of embedding calls made."""
        return self._embedding_call_count

    @property
    def chat_requests(self) -> list[ChatRequest]:
        """List of chat requests received."""
        return self._chat_requests.copy()

    @property
    def embedding_requests(self) -> list[EmbeddingRequest]:
        """List of embedding requests received."""
        return self._embedding_requests.copy()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Return mock chat response."""
        self._chat_requests.append(request)
        self._chat_call_count += 1

        if self._chat_function:
            content = self._chat_function(request)
        elif self._chat_responses:
            response = self._chat_responses.pop(0)
            if isinstance(response, ChatResponse):
                return response
            content = response
        else:
            content = '{"result": "mock response"}'

        return ChatResponse(
            content=content,
            model=request.model,
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return mock embedding response."""
        self._embedding_requests.append(request)
        self._embedding_call_count += 1

        if self._embedding_function:
            embedding = self._embedding_function(request)
        elif self._embedding_responses:
            response = self._embedding_responses.pop(0)
            if isinstance(response, EmbeddingResponse):
                return response
            embedding = response
        else:
            # Default: deterministic embedding based on dimension
            dim = request.dimension or 256
            embedding = tuple(0.1 * (i % 10) for i in range(dim))

        return EmbeddingResponse(
            embedding=embedding,
            model=request.model,
        )

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> str:
        """Simple chat interface matching OllamaClient.

        Args:
            user_prompt: User message content.
            system_prompt: Optional system message.
            model: Model to use (defaults to "mock").
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            Generated response content.
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        request = ChatRequest(
            messages=messages,
            model=model or "mock",
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = await self.chat(request)
        return response.content

    async def simple_embed(
        self,
        text: str,
        model: str | None = None,
        dimension: int | None = None,
    ) -> tuple[float, ...]:
        """Simple embed interface matching OllamaClient.

        Args:
            text: Text to embed.
            model: Model to use (defaults to "mock").
            dimension: Optional dimension truncation.

        Returns:
            Embedding vector.
        """
        request = EmbeddingRequest(
            text=text,
            model=model or "mock",
            dimension=dimension,
        )
        response = await self.embed(request)
        return response.embedding

    async def close(self) -> None:
        """No-op close for compatibility."""
        pass

    async def __aenter__(self) -> MockLLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        pass
