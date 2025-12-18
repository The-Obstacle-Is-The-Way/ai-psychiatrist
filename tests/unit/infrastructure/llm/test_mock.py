"""Tests for MockLLMClient.

Tests verify the mock client behavior for unit testing without real LLM calls.
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class TestMockLLMClientChat:
    """Tests for MockLLMClient chat functionality."""

    @pytest.mark.asyncio
    async def test_canned_responses_in_order(self) -> None:
        """Should return canned responses in order."""
        mock = MockLLMClient(chat_responses=["first", "second", "third"])

        resp1 = await mock.simple_chat("prompt1")
        resp2 = await mock.simple_chat("prompt2")
        resp3 = await mock.simple_chat("prompt3")

        assert resp1 == "first"
        assert resp2 == "second"
        assert resp3 == "third"

    @pytest.mark.asyncio
    async def test_canned_response_objects(self) -> None:
        """Should return ChatResponse objects directly."""
        response = ChatResponse(
            content="custom response",
            model="custom-model",
            done=True,
            total_duration_ms=1000,
        )
        mock = MockLLMClient(chat_responses=[response])

        result = await mock.chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="test")],
                model="test-model",
            )
        )

        assert result.content == "custom response"
        assert result.model == "custom-model"
        assert result.total_duration_ms == 1000

    @pytest.mark.asyncio
    async def test_custom_function(self) -> None:
        """Should use custom function for responses."""

        def custom(req: ChatRequest) -> str:
            return f"Received: {req.messages[-1].content}"

        mock = MockLLMClient(chat_function=custom)
        resp = await mock.simple_chat("hello world")

        assert resp == "Received: hello world"

    @pytest.mark.asyncio
    async def test_custom_function_priority(self) -> None:
        """Function should take priority over canned responses."""

        def custom(_: ChatRequest) -> str:
            return "from function"

        mock = MockLLMClient(
            chat_responses=["from list"],
            chat_function=custom,
        )

        resp = await mock.simple_chat("test")
        assert resp == "from function"

    @pytest.mark.asyncio
    async def test_default_response_when_empty(self) -> None:
        """Should return default response when no responses configured."""
        mock = MockLLMClient()

        resp = await mock.simple_chat("test")

        assert "mock response" in resp

    @pytest.mark.asyncio
    async def test_tracks_call_count(self) -> None:
        """Should track number of chat calls."""
        mock = MockLLMClient(chat_responses=["a", "b", "c"])

        assert mock.chat_call_count == 0

        await mock.simple_chat("1")
        assert mock.chat_call_count == 1

        await mock.simple_chat("2")
        await mock.simple_chat("3")
        assert mock.chat_call_count == 3

    @pytest.mark.asyncio
    async def test_tracks_requests(self) -> None:
        """Should track all requests made."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat("test prompt", system_prompt="be helpful")

        assert len(mock.chat_requests) == 1
        req = mock.chat_requests[0]
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[0].content == "be helpful"
        assert req.messages[1].role == "user"
        assert req.messages[1].content == "test prompt"

    @pytest.mark.asyncio
    async def test_chat_request_model_propagated(self) -> None:
        """Response should use request model when returning string."""
        mock = MockLLMClient(chat_responses=["test"])

        result = await mock.chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content="test")],
                model="gemma3:27b",
            )
        )

        assert result.model == "gemma3:27b"
        assert result.done is True


class TestMockLLMClientEmbed:
    """Tests for MockLLMClient embedding functionality."""

    @pytest.mark.asyncio
    async def test_canned_embeddings_in_order(self) -> None:
        """Should return canned embeddings in order."""
        emb1 = (0.1, 0.2, 0.3)
        emb2 = (0.4, 0.5, 0.6)
        mock = MockLLMClient(embedding_responses=[emb1, emb2])

        resp1 = await mock.simple_embed("text1")
        resp2 = await mock.simple_embed("text2")

        assert resp1 == emb1
        assert resp2 == emb2

    @pytest.mark.asyncio
    async def test_canned_response_objects(self) -> None:
        """Should return EmbeddingResponse objects directly."""
        response = EmbeddingResponse(
            embedding=(0.1, 0.2, 0.3),
            model="custom-embed-model",
        )
        mock = MockLLMClient(embedding_responses=[response])

        result = await mock.embed(
            EmbeddingRequest(text="test", model="test-model")
        )

        assert result.embedding == (0.1, 0.2, 0.3)
        assert result.model == "custom-embed-model"

    @pytest.mark.asyncio
    async def test_custom_function(self) -> None:
        """Should use custom function for embeddings."""

        def custom(req: EmbeddingRequest) -> tuple[float, ...]:
            return tuple(float(i) for i in range(len(req.text)))

        mock = MockLLMClient(embedding_function=custom)
        resp = await mock.simple_embed("abc")

        assert resp == (0.0, 1.0, 2.0)

    @pytest.mark.asyncio
    async def test_default_embedding_respects_dimension(self) -> None:
        """Default embedding should respect dimension parameter."""
        mock = MockLLMClient()

        emb = await mock.simple_embed("text", dimension=128)

        assert len(emb) == 128

    @pytest.mark.asyncio
    async def test_default_embedding_dimension_256(self) -> None:
        """Default embedding should be 256 when no dimension specified."""
        mock = MockLLMClient()

        emb = await mock.simple_embed("text")

        assert len(emb) == 256

    @pytest.mark.asyncio
    async def test_tracks_call_count(self) -> None:
        """Should track number of embedding calls."""
        mock = MockLLMClient()

        assert mock.embedding_call_count == 0

        await mock.simple_embed("text1")
        assert mock.embedding_call_count == 1

        await mock.simple_embed("text2")
        await mock.simple_embed("text3")
        assert mock.embedding_call_count == 3

    @pytest.mark.asyncio
    async def test_tracks_requests(self) -> None:
        """Should track all embedding requests made."""
        mock = MockLLMClient()

        await mock.simple_embed("test text", model="qwen", dimension=512)

        assert len(mock.embedding_requests) == 1
        req = mock.embedding_requests[0]
        assert req.text == "test text"
        assert req.model == "qwen"
        assert req.dimension == 512


class TestMockLLMClientContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Should work as async context manager."""
        async with MockLLMClient(chat_responses=["test"]) as mock:
            resp = await mock.simple_chat("prompt")
            assert resp == "test"

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """Close should be no-op for compatibility."""
        mock = MockLLMClient()
        await mock.close()  # Should not raise


class TestMockLLMClientSimpleMethods:
    """Tests for simple_chat and simple_embed convenience methods."""

    @pytest.mark.asyncio
    async def test_simple_chat_with_system_prompt(self) -> None:
        """simple_chat should include system prompt when provided."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat(
            "user message",
            system_prompt="You are helpful.",
            model="test-model",
        )

        req = mock.chat_requests[0]
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[1].role == "user"
        assert req.model == "test-model"

    @pytest.mark.asyncio
    async def test_simple_chat_without_system_prompt(self) -> None:
        """simple_chat should work without system prompt."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat("user message")

        req = mock.chat_requests[0]
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_simple_chat_default_model(self) -> None:
        """simple_chat should use 'mock' as default model."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat("test")

        req = mock.chat_requests[0]
        assert req.model == "mock"

    @pytest.mark.asyncio
    async def test_simple_embed_default_model(self) -> None:
        """simple_embed should use 'mock' as default model."""
        mock = MockLLMClient()

        await mock.simple_embed("test")

        req = mock.embedding_requests[0]
        assert req.model == "mock"

    @pytest.mark.asyncio
    async def test_simple_chat_custom_params(self) -> None:
        """simple_chat should respect custom temperature/top_k/top_p."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat(
            "test",
            temperature=0.0,
            top_k=40,
            top_p=0.95,
        )

        req = mock.chat_requests[0]
        assert req.temperature == 0.0
        assert req.top_k == 40
        assert req.top_p == 0.95


class TestMockLLMClientProtocolCompliance:
    """Tests verifying MockLLMClient implements protocols correctly."""

    def test_implements_chat_client(self) -> None:
        """Should implement ChatClient protocol."""
        from ai_psychiatrist.infrastructure.llm.protocols import ChatClient

        mock = MockLLMClient()
        assert isinstance(mock, ChatClient)

    def test_implements_embedding_client(self) -> None:
        """Should implement EmbeddingClient protocol."""
        from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

        mock = MockLLMClient()
        assert isinstance(mock, EmbeddingClient)

    def test_implements_llm_client(self) -> None:
        """Should implement LLMClient protocol (combined)."""
        from ai_psychiatrist.infrastructure.llm.protocols import LLMClient

        mock = MockLLMClient()
        assert isinstance(mock, LLMClient)
