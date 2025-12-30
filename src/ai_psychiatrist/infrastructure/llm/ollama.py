"""Ollama LLM client implementation.

This module provides the production implementation for Ollama API integration.

Paper Reference:
    - Section 2.3.5: Ollama API integration
    - Section 2.2: Gemma 3 27B for chat, Qwen 3 8B for embeddings
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

import httpx

from ai_psychiatrist.config import get_model_name
from ai_psychiatrist.domain.exceptions import (
    LLMError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingBatchRequest,
    EmbeddingBatchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import ModelSettings, OllamaSettings

logger = get_logger(__name__)


def _l2_normalize(embedding: list[float]) -> tuple[float, ...]:
    """L2 normalize an embedding vector.

    Args:
        embedding: Raw embedding vector.

    Returns:
        L2-normalized embedding as tuple.
    """
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        return tuple(x / norm for x in embedding)
    return tuple(embedding)


class OllamaClient:
    """Ollama API client for chat and embeddings.

    Implements the ChatClient and EmbeddingClient protocols for
    production use with the Ollama API.

    Example:
        >>> from ai_psychiatrist.config import OllamaSettings
        >>> async with OllamaClient(OllamaSettings()) as client:
        ...     response = await client.simple_chat("Hello!")
        ...     print(response)
    """

    def __init__(
        self,
        ollama_settings: OllamaSettings,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize Ollama client.

        Args:
            ollama_settings: Ollama server configuration.
            model_settings: Optional model configuration for defaults.
        """
        self._base_url = ollama_settings.base_url
        self._chat_url = ollama_settings.chat_url
        self._embeddings_url = ollama_settings.embeddings_url
        self._default_timeout = ollama_settings.timeout_seconds
        self._model_settings = model_settings

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._default_timeout))

    async def __aenter__(self) -> OllamaClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def ping(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if server responds.

        Raises:
            LLMError: If ping fails.
        """
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error("Ollama ping failed", error=str(e))
            raise LLMError(f"Failed to ping Ollama: {e}") from e

    async def list_models(self) -> list[dict[str, Any]]:
        """List models available on the Ollama server.

        Returns:
            A list of model metadata dictionaries as returned by `/api/tags`.

        Raises:
            LLMError: If the request fails.
            LLMResponseParseError: If the response cannot be parsed.
        """
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error("Failed to list Ollama models", error=str(e))
            raise LLMError(f"Failed to list models: {e}") from e

        try:
            data = response.json()
            models = data.get("models", [])
            if not isinstance(models, list):
                raise TypeError("models must be a list")
        except (ValueError, TypeError) as e:
            logger.error("Failed to parse models response", raw_length=len(response.text))
            raise LLMResponseParseError(response.text, str(e)) from e

        # Filter to dictionaries (Ollama returns a list of objects with at least a `name` field)
        return [m for m in models if isinstance(m, dict)]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute chat completion via Ollama API.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            Chat response with generated content.

        Raises:
            LLMTimeoutError: If request times out.
            LLMError: If request fails.
        """
        payload = {
            "model": request.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        logger.debug(
            "Sending chat request",
            model=payload["model"],
            message_count=len(request.messages),
        )

        try:
            response = await self._client.post(
                self._chat_url,
                json=payload,
                timeout=request.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.error("Chat request timed out", timeout=request.timeout_seconds)
            raise LLMTimeoutError(request.timeout_seconds) from e
        except httpx.HTTPStatusError as e:
            logger.error(
                "Chat request failed",
                status_code=e.response.status_code,
                detail="response body redacted",
                response_length=len(e.response.text),
            )
            raise LLMError(f"HTTP {e.response.status_code}: response body redacted") from e
        except httpx.RequestError as e:
            logger.error("Chat request error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse chat response", raw_length=len(response.text))
            raise LLMResponseParseError(response.text, str(e)) from e

        logger.debug(
            "Chat response received",
            model=data.get("model"),
            content_length=len(content),
        )

        # Convert nanoseconds to milliseconds (Ollama returns timing in ns)
        total_duration_ns = data.get("total_duration")
        total_duration_ms = total_duration_ns // 1_000_000 if total_duration_ns else None

        return ChatResponse(
            content=content,
            model=data.get("model", request.model),
            done=data.get("done", True),
            total_duration_ms=total_duration_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding via Ollama API.

        Args:
            request: Embedding request with text and parameters.

        Returns:
            Embedding response with L2-normalized vector.

        Raises:
            LLMTimeoutError: If request times out.
            LLMError: If request fails.
        """
        payload = {
            "model": request.model,
            "prompt": request.text,
        }

        logger.debug(
            "Sending embedding request",
            model=payload["model"],
            text_length=len(request.text),
        )

        try:
            response = await self._client.post(
                self._embeddings_url,
                json=payload,
                timeout=request.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.error("Embedding request timed out", timeout=request.timeout_seconds)
            raise LLMTimeoutError(request.timeout_seconds) from e
        except httpx.HTTPStatusError as e:
            logger.error(
                "Embedding request failed",
                status_code=e.response.status_code,
                detail="response body redacted",
                response_length=len(e.response.text),
            )
            raise LLMError(f"HTTP {e.response.status_code}: response body redacted") from e
        except httpx.RequestError as e:
            logger.error("Embedding request error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e

        try:
            data = response.json()
            embedding = data["embedding"]
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse embedding response", raw_length=len(response.text))
            raise LLMResponseParseError(response.text, str(e)) from e

        # Truncate to requested dimension if specified (MRL support)
        if request.dimension is not None:
            embedding = embedding[: request.dimension]

        # L2 normalize
        normalized = _l2_normalize(embedding)

        logger.debug(
            "Embedding response received",
            dimension=len(normalized),
        )

        return EmbeddingResponse(
            embedding=normalized,
            model=data.get("model", request.model),
        )

    async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
        """Generate embeddings for multiple texts.

        Ollama does not support true batching for embeddings, so this is a correctness fallback
        that sequentially calls `embed(...)` with an overall timeout.
        """
        if not request.texts:
            return EmbeddingBatchResponse(embeddings=[], model=request.model)

        try:
            async with asyncio.timeout(request.timeout_seconds):
                embeddings: list[tuple[float, ...]] = []
                for text in request.texts:
                    resp = await self.embed(
                        EmbeddingRequest(
                            text=text,
                            model=request.model,
                            dimension=request.dimension,
                            timeout_seconds=request.timeout_seconds,
                        )
                    )
                    embeddings.append(resp.embedding)
        except TimeoutError as e:
            raise LLMTimeoutError(request.timeout_seconds) from e

        return EmbeddingBatchResponse(embeddings=embeddings, model=request.model)

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Simple chat completion with just user/system prompts.

        Args:
            user_prompt: User message content.
            system_prompt: Optional system message.
            model: Model to use (Paper Section 2.2: gemma3:27b).
            temperature: Sampling temperature (e.g., 0.0 for Judge agent).

        Returns:
            Generated response content.
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        # Fallback priority:
        # 1. explicit 'model' arg
        # 2. settings.qualitative_model (via get_model_name helper)
        default_model = get_model_name(self._model_settings, "qualitative")

        request = ChatRequest(
            messages=messages,
            model=model or default_model,
            temperature=temperature,
            timeout_seconds=self._default_timeout,
        )
        response = await self.chat(request)
        return response.content

    async def simple_embed(
        self,
        text: str,
        model: str | None = None,
        dimension: int | None = None,
    ) -> tuple[float, ...]:
        """Simple embedding generation.

        Args:
            text: Text to embed.
            model: Model to use (Paper Section 2.2: Qwen 3 8B Embedding).
            dimension: Optional dimension truncation (MRL support).

        Returns:
            L2-normalized embedding vector.
        """
        # Fallback priority:
        # 1. explicit 'model' arg
        # 2. settings.embedding_model (via get_model_name helper)
        default_model = get_model_name(self._model_settings, "embedding")

        request = EmbeddingRequest(
            text=text,
            model=model or default_model,
            dimension=dimension,
            timeout_seconds=self._default_timeout,
        )
        response = await self.embed(request)
        return response.embedding
