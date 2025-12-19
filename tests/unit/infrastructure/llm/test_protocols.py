"""Tests for LLM protocol dataclasses.

Tests verify value object behavior: immutability, validation, and equality.
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class TestChatMessage:
    """Tests for ChatMessage value object."""

    def test_valid_user_message(self) -> None:
        """Should create valid user message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_system_message(self) -> None:
        """Should create valid system message."""
        msg = ChatMessage(role="system", content="You are a psychiatrist.")
        assert msg.role == "system"
        assert msg.content == "You are a psychiatrist."

    def test_valid_assistant_message(self) -> None:
        """Should create valid assistant message."""
        msg = ChatMessage(role="assistant", content="How can I help?")
        assert msg.role == "assistant"
        assert msg.content == "How can I help?"

    def test_invalid_role_raises(self) -> None:
        """Should reject invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage(role="invalid", content="test")

    def test_empty_content_raises(self) -> None:
        """Should reject empty content."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ChatMessage(role="user", content="")

    def test_immutable(self) -> None:
        """Should be immutable (frozen)."""
        msg = ChatMessage(role="user", content="test")
        with pytest.raises(AttributeError):
            msg.content = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal messages should be equal."""
        msg1 = ChatMessage(role="user", content="test")
        msg2 = ChatMessage(role="user", content="test")
        assert msg1 == msg2

    def test_inequality(self) -> None:
        """Different messages should not be equal."""
        msg1 = ChatMessage(role="user", content="test1")
        msg2 = ChatMessage(role="user", content="test2")
        assert msg1 != msg2


class TestChatRequest:
    """Tests for ChatRequest value object."""

    def test_valid_request_minimal(self) -> None:
        """Should create request with minimal params."""
        messages = [ChatMessage(role="user", content="Hello")]
        req = ChatRequest(messages=messages, model="gemma3:27b")

        assert len(req.messages) == 1
        assert req.model == "gemma3:27b"
        assert req.temperature == 0.2
        assert req.top_k == 20
        assert req.top_p == 0.8
        assert req.timeout_seconds == 180

    def test_valid_request_custom_params(self) -> None:
        """Should create request with custom params."""
        messages = [
            ChatMessage(role="system", content="You are a psychiatrist."),
            ChatMessage(role="user", content="Hello"),
        ]
        req = ChatRequest(
            messages=messages,
            model="alibayram/medgemma:27b",
            temperature=0.0,
            top_k=40,
            top_p=0.95,
            timeout_seconds=300,
        )

        assert len(req.messages) == 2
        assert req.model == "alibayram/medgemma:27b"
        assert req.temperature == 0.0
        assert req.top_k == 40
        assert req.top_p == 0.95
        assert req.timeout_seconds == 300

    def test_empty_messages_raises(self) -> None:
        """Should reject empty messages."""
        with pytest.raises(ValueError, match="Messages cannot be empty"):
            ChatRequest(messages=[], model="gemma3:27b")

    def test_empty_model_raises(self) -> None:
        """Should reject empty model."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="Model cannot be empty"):
            ChatRequest(messages=messages, model="")

    def test_temperature_too_low_raises(self) -> None:
        """Should reject temperature below 0."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="Temperature"):
            ChatRequest(messages=messages, model="test", temperature=-0.1)

    def test_temperature_too_high_raises(self) -> None:
        """Should reject temperature above 2."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="Temperature"):
            ChatRequest(messages=messages, model="test", temperature=2.1)

    def test_top_k_too_low_raises(self) -> None:
        """Should reject top_k below 1."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="top_k"):
            ChatRequest(messages=messages, model="test", top_k=0)

    def test_top_k_too_high_raises(self) -> None:
        """Should reject top_k above 100."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="top_k"):
            ChatRequest(messages=messages, model="test", top_k=101)

    def test_top_p_too_low_raises(self) -> None:
        """Should reject top_p below 0."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="top_p"):
            ChatRequest(messages=messages, model="test", top_p=-0.1)

    def test_top_p_too_high_raises(self) -> None:
        """Should reject top_p above 1."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="top_p"):
            ChatRequest(messages=messages, model="test", top_p=1.1)

    def test_timeout_too_low_raises(self) -> None:
        """Should reject timeout below 1."""
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ValueError, match="timeout_seconds"):
            ChatRequest(messages=messages, model="test", timeout_seconds=0)

    def test_immutable(self) -> None:
        """Should be immutable (frozen)."""
        messages = [ChatMessage(role="user", content="Hello")]
        req = ChatRequest(messages=messages, model="test")
        with pytest.raises(AttributeError):
            req.model = "changed"  # type: ignore[misc]


class TestChatResponse:
    """Tests for ChatResponse value object."""

    def test_valid_response_minimal(self) -> None:
        """Should create response with minimal params."""
        resp = ChatResponse(content="Hello!", model="gemma3:27b")

        assert resp.content == "Hello!"
        assert resp.model == "gemma3:27b"
        assert resp.done is True
        assert resp.total_duration_ms is None
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None

    def test_valid_response_full(self) -> None:
        """Should create response with all params."""
        resp = ChatResponse(
            content="Assessment complete.",
            model="gemma3:27b",
            done=True,
            total_duration_ms=5000,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert resp.content == "Assessment complete."
        assert resp.model == "gemma3:27b"
        assert resp.done is True
        assert resp.total_duration_ms == 5000
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50

    def test_immutable(self) -> None:
        """Should be immutable (frozen)."""
        resp = ChatResponse(content="test", model="test")
        with pytest.raises(AttributeError):
            resp.content = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal responses should be equal."""
        resp1 = ChatResponse(content="test", model="m1")
        resp2 = ChatResponse(content="test", model="m1")
        assert resp1 == resp2


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest value object."""

    def test_valid_request_minimal(self) -> None:
        """Should create request with minimal params."""
        req = EmbeddingRequest(text="Hello world", model="qwen-embed")

        assert req.text == "Hello world"
        assert req.model == "qwen-embed"
        assert req.dimension is None
        assert req.timeout_seconds == 120

    def test_valid_request_with_dimension(self) -> None:
        """Should create request with MRL dimension truncation."""
        req = EmbeddingRequest(
            text="Hello world",
            model="qwen3-embedding:8b",
            dimension=4096,
            timeout_seconds=60,
        )

        assert req.dimension == 4096
        assert req.timeout_seconds == 60

    def test_empty_text_raises(self) -> None:
        """Should reject empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            EmbeddingRequest(text="", model="test")

    def test_empty_model_raises(self) -> None:
        """Should reject empty model."""
        with pytest.raises(ValueError, match="Model cannot be empty"):
            EmbeddingRequest(text="hello", model="")

    def test_invalid_dimension_raises(self) -> None:
        """Should reject dimension below 1."""
        with pytest.raises(ValueError, match="dimension"):
            EmbeddingRequest(text="hello", model="test", dimension=0)

    def test_invalid_timeout_raises(self) -> None:
        """Should reject timeout below 1."""
        with pytest.raises(ValueError, match="timeout_seconds"):
            EmbeddingRequest(text="hello", model="test", timeout_seconds=0)

    def test_immutable(self) -> None:
        """Should be immutable (frozen)."""
        req = EmbeddingRequest(text="test", model="test")
        with pytest.raises(AttributeError):
            req.text = "changed"  # type: ignore[misc]


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse value object."""

    def test_valid_response(self) -> None:
        """Should create response and compute dimension."""
        embedding = (0.1, 0.2, 0.3, 0.4)
        resp = EmbeddingResponse(embedding=embedding, model="qwen-embed")

        assert resp.embedding == embedding
        assert resp.model == "qwen-embed"
        assert resp.dimension == 4

    def test_dimension_computed(self) -> None:
        """Dimension should be computed from embedding length."""
        embedding = tuple(0.1 for _ in range(4096))
        resp = EmbeddingResponse(embedding=embedding, model="test")

        assert resp.dimension == 4096

    def test_empty_embedding_raises(self) -> None:
        """Should reject empty embedding."""
        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            EmbeddingResponse(embedding=(), model="test")

    def test_immutable(self) -> None:
        """Should be immutable (frozen)."""
        resp = EmbeddingResponse(embedding=(0.1, 0.2), model="test")
        with pytest.raises(AttributeError):
            resp.model = "changed"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal responses should be equal."""
        emb = (0.1, 0.2)
        resp1 = EmbeddingResponse(embedding=emb, model="m1")
        resp2 = EmbeddingResponse(embedding=emb, model="m1")
        assert resp1 == resp2
