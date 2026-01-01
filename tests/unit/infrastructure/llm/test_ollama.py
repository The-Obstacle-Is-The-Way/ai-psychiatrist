"""Tests for OllamaClient.

Tests use respx to mock HTTP requests to the Ollama API.
"""

from __future__ import annotations

import math

import httpx
import pytest
import respx

from ai_psychiatrist.config import OllamaSettings
from ai_psychiatrist.domain.exceptions import (
    LLMError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient, _l2_normalize
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatClient,
    ChatMessage,
    ChatRequest,
    EmbeddingClient,
    EmbeddingRequest,
    LLMClient,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def ollama_settings() -> OllamaSettings:
    """Create test Ollama settings."""
    return OllamaSettings(host="localhost", port=11434, timeout_seconds=30)


@pytest.fixture
def ollama_client(ollama_settings: OllamaSettings) -> OllamaClient:
    """Create OllamaClient for testing."""
    return OllamaClient(ollama_settings)


class TestL2Normalize:
    """Tests for L2 normalization utility."""

    def test_normalizes_vector(self) -> None:
        """Should L2-normalize vector to unit length."""
        embedding = [3.0, 4.0]  # 3-4-5 right triangle
        result = _l2_normalize(embedding)

        assert result == pytest.approx((0.6, 0.8))

    def test_normalizes_to_unit_length(self) -> None:
        """Normalized vector should have length ~1."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        result = _l2_normalize(embedding)

        length = math.sqrt(sum(x * x for x in result))
        assert length == pytest.approx(1.0)

    def test_handles_zero_vector(self) -> None:
        """Should handle zero vector without division by zero."""
        embedding = [0.0, 0.0, 0.0]
        result = _l2_normalize(embedding)

        assert result == (0.0, 0.0, 0.0)

    def test_returns_tuple(self) -> None:
        """Should return tuple not list."""
        embedding = [1.0, 2.0]
        result = _l2_normalize(embedding)

        assert isinstance(result, tuple)


class TestOllamaClientPing:
    """Tests for ping functionality."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_ping_success(self, ollama_client: OllamaClient) -> None:
        """Should return True when server responds."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )

        result = await ollama_client.ping()

        assert result is True
        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_ping_failure_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMError when server unreachable."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(LLMError, match="Failed to ping"):
            await ollama_client.ping()

        await ollama_client.close()


class TestOllamaClientListModels:
    """Tests for model listing."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_client: OllamaClient) -> None:
        """Should return models list from /api/tags."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3-embedding:8b"},
                        {"name": "gemma3:27b"},
                    ]
                },
            )
        )

        models = await ollama_client.list_models()

        assert [m.get("name") for m in models] == ["qwen3-embedding:8b", "gemma3:27b"]
        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_invalid_payload_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMResponseParseError when payload is invalid."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": "not-a-list"})
        )

        with pytest.raises(LLMResponseParseError, match="models must be a list"):
            await ollama_client.list_models()

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_missing_key(self, ollama_client: OllamaClient) -> None:
        """Should return empty list when models key is missing."""
        respx.get("http://localhost:11434/api/tags").mock(return_value=httpx.Response(200, json={}))

        models = await ollama_client.list_models()

        assert models == []
        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_filters_non_dicts(self, ollama_client: OllamaClient) -> None:
        """Should filter out non-dict entries from models list."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "valid"},
                        "invalid-string",
                        None,
                        {"name": "also-valid"},
                    ]
                },
            )
        )

        models = await ollama_client.list_models()

        assert len(models) == 2
        assert [m.get("name") for m in models] == ["valid", "also-valid"]
        await ollama_client.close()


class TestOllamaClientChat:
    """Tests for chat completion."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_success(self, ollama_client: OllamaClient) -> None:
        """Should return ChatResponse on success."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gemma3:27b",
                    "message": {"role": "assistant", "content": "Hello!"},
                    "done": True,
                    "total_duration": 5000000000,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            )
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="gemma3:27b",
        )
        response = await ollama_client.chat(request)

        assert response.content == "Hello!"
        assert response.model == "gemma3:27b"
        assert response.done is True
        assert response.total_duration_ms == 5000  # 5 seconds in ms (converted from ns)
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_sends_correct_payload(self, ollama_client: OllamaClient) -> None:
        """Should send correct request payload."""
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "gemma3:27b",
                    "message": {"content": "response"},
                    "done": True,
                },
            )
        )

        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ],
            model="gemma3:27b",
            temperature=0.5,
        )
        await ollama_client.chat(request)

        assert route.called
        call = route.calls[0]
        payload = call.request.content.decode()
        assert '"model":"gemma3:27b"' in payload
        assert '"stream":false' in payload
        assert '"temperature":0.5' in payload

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_timeout_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMTimeoutError on timeout."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.TimeoutException("timeout")
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="test",
            timeout_seconds=5,
        )

        with pytest.raises(LLMTimeoutError, match="timed out after 5s"):
            await ollama_client.chat(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_http_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMError on HTTP error."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="test",
        )

        with pytest.raises(LLMError, match="HTTP 500"):
            await ollama_client.chat(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_request_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMError on request error."""
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.RequestError("Connection failed")
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="test",
        )

        with pytest.raises(LLMError, match="Request failed"):
            await ollama_client.chat(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_parse_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMResponseParseError on malformed response."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"invalid": "response"})
        )

        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="test",
        )

        with pytest.raises(LLMResponseParseError):
            await ollama_client.chat(request)

        await ollama_client.close()


class TestOllamaClientEmbed:
    """Tests for embedding generation."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_success(self, ollama_client: OllamaClient) -> None:
        """Should return EmbeddingResponse on success."""
        # Use simple vector for predictable normalization
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "qwen-embed",
                    "embedding": [3.0, 4.0],  # 3-4-5 triangle
                },
            )
        )

        request = EmbeddingRequest(text="test", model="qwen-embed")
        response = await ollama_client.embed(request)

        # Should be L2 normalized
        assert response.embedding == pytest.approx((0.6, 0.8))
        assert response.model == "qwen-embed"
        assert response.dimension == 2

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_with_dimension_truncation(self, ollama_client: OllamaClient) -> None:
        """Should truncate embedding to requested dimension (MRL)."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "qwen",
                    "embedding": [1.0, 2.0, 3.0, 4.0, 5.0],
                },
            )
        )

        request = EmbeddingRequest(text="test", model="qwen", dimension=3)
        response = await ollama_client.embed(request)

        # Should be truncated to first 3 elements, then normalized
        assert response.dimension == 3

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_timeout_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMTimeoutError on timeout."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            side_effect=httpx.TimeoutException("timeout")
        )

        request = EmbeddingRequest(text="test", model="test", timeout_seconds=5)

        with pytest.raises(LLMTimeoutError, match="timed out after 5s"):
            await ollama_client.embed(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_http_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMError on HTTP error."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(404, text="Model not found")
        )

        request = EmbeddingRequest(text="test", model="nonexistent")

        with pytest.raises(LLMError, match="HTTP 404"):
            await ollama_client.embed(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_request_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMError on request error."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            side_effect=httpx.RequestError("Connection refused")
        )

        request = EmbeddingRequest(text="test", model="test")

        with pytest.raises(LLMError, match="Request failed"):
            await ollama_client.embed(request)

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_embed_parse_error_raises(self, ollama_client: OllamaClient) -> None:
        """Should raise LLMResponseParseError on malformed response."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"no_embedding": True})
        )

        request = EmbeddingRequest(text="test", model="test")

        with pytest.raises(LLMResponseParseError):
            await ollama_client.embed(request)

        await ollama_client.close()


class TestOllamaClientSimpleMethods:
    """Tests for simple_chat and simple_embed convenience methods."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_chat_with_system(self, ollama_client: OllamaClient) -> None:
        """simple_chat should work with system prompt."""
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "response"}, "done": True},
            )
        )

        result = await ollama_client.simple_chat(
            user_prompt="Hello",
            system_prompt="You are a psychiatrist.",
            model="gemma3:27b",
        )

        assert result == "response"
        assert route.called
        timeout = route.calls[0].request.extensions.get("timeout")
        assert isinstance(timeout, dict)
        assert timeout.get("read") == 30

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_chat_without_system(self, ollama_client: OllamaClient) -> None:
        """simple_chat should work without system prompt."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "response"}, "done": True},
            )
        )

        result = await ollama_client.simple_chat("Hello")

        assert result == "response"

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_embed(self, ollama_client: OllamaClient) -> None:
        """simple_embed should return normalized embedding."""
        route = respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"embedding": [3.0, 4.0]},
            )
        )

        result = await ollama_client.simple_embed("test text")

        assert result == pytest.approx((0.6, 0.8))
        assert route.called
        timeout = route.calls[0].request.extensions.get("timeout")
        assert isinstance(timeout, dict)
        assert timeout.get("read") == 30

        await ollama_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_simple_embed_with_dimension(self, ollama_client: OllamaClient) -> None:
        """simple_embed should pass dimension for MRL truncation."""
        respx.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"embedding": [1.0, 2.0, 3.0, 4.0, 5.0]},
            )
        )

        result = await ollama_client.simple_embed("test", dimension=2)

        # Should truncate to 2 dimensions then normalize
        assert len(result) == 2

        await ollama_client.close()


class TestOllamaClientContextManager:
    """Tests for async context manager."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_context_manager(self, ollama_settings: OllamaSettings) -> None:
        """Should work as async context manager."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "test"}, "done": True},
            )
        )

        async with OllamaClient(ollama_settings) as client:
            result = await client.simple_chat("Hello")
            assert result == "test"


class TestOllamaClientProtocolCompliance:
    """Tests verifying OllamaClient implements protocols."""

    def test_implements_chat_client(self, ollama_client: OllamaClient) -> None:
        """Should implement ChatClient protocol."""
        assert isinstance(ollama_client, ChatClient)

    def test_implements_embedding_client(self, ollama_client: OllamaClient) -> None:
        """Should implement EmbeddingClient protocol."""
        assert isinstance(ollama_client, EmbeddingClient)

    def test_implements_llm_client(self, ollama_client: OllamaClient) -> None:
        """Should implement LLMClient protocol (combined)."""
        assert isinstance(ollama_client, LLMClient)
