# Spec 04: LLM Infrastructure

## Objective

Create a robust, testable abstraction for LLM interactions using the Strategy pattern. This enables swapping between Ollama, OpenAI, or mock implementations without changing business logic.

## Paper Reference

- **Section 2.2**: Gemma 3 27B for chat, Qwen 3 8B for embeddings
- **Section 2.3.5**: Ollama API integration
- **Appendix F**: MedGemma 27B achieves 18% better MAE (0.505 vs 0.619)

## Target Configuration (Paper-Optimal)

| Use | Spec Target | Paper Reference |
|-----|------------|-----------------|
| Qualitative/Judge/Meta chat | `gemma3:27b` | Section 2.2 (paper baseline) |
| Quantitative chat | MedGemma 27B (example Ollama tag: `alibayram/medgemma:27b`) | Appendix F (MAE 0.505; fewer predictions) |
| Embeddings | Qwen 3 8B Embedding (example Ollama tag: `dengcao/Qwen3-Embedding-8B:Q8_0`; quantization not specified in paper) | Section 2.2 |
| Quantitative fallback | `gemma3:27b` | Section 2.2 |

## As-Is Ollama Usage (Repo)

The current repo uses **three** Ollama endpoints, via two different client styles:

### HTTP (`requests`)

- `POST /api/generate` (streaming): used by `agents/qualitative_assessor_f.py`, `agents/qualitative_assessor_z.py`, `agents/quantitative_assessor_z.py`, `agents/interview_evaluator.py`
- `POST /api/chat` (non-stream): used by `agents/quantitative_assessor_f.py`, `agents/qualitive_evaluator.py`, and most cluster scripts/notebooks
- `POST /api/embeddings` (non-stream): used by `agents/quantitative_assessor_f.py` and `quantitative_assessment/embedding_batch_script.py`

### Python SDK (`ollama.Client`)

- used by `agents/meta_reviewer.py` (chat only)

### As-Is Defaults (Demo Pipeline)

- Host: `http://localhost:11434`
- Chat model: `llama3`
- Embedding model: `dengcao/Qwen3-Embedding-8B:Q4_K_M` (quantized)

### As-Is Defaults (Research/Cluster Scripts)

- Host is typically set via `OLLAMA_NODE = "arctrd..."`
- Models commonly used:
  - `gemma3:27b` / `gemma3-optimized:27b` (chat)
  - `dengcao/Qwen3-Embedding-8B:Q8_0` (embeddings)
  - `alibayram/medgemma:27b` (MedGemma variant)

## Deliverables

1. `src/ai_psychiatrist/infrastructure/llm/protocols.py` - Abstract interfaces
2. `src/ai_psychiatrist/infrastructure/llm/ollama.py` - Ollama implementation
3. `src/ai_psychiatrist/infrastructure/llm/responses.py` - Response parsing
4. `src/ai_psychiatrist/infrastructure/llm/mock.py` - Test doubles
5. `tests/unit/infrastructure/llm/` - Comprehensive tests

## Implementation

### 1. Protocols (protocols.py)

```python
"""Abstract protocols for LLM interactions."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Request for chat completion."""

    messages: Sequence[ChatMessage]
    model: str
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.8
    timeout_seconds: int = 180


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Response from chat completion."""

    content: str
    model: str
    done: bool = True
    total_duration_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Request for text embedding."""

    text: str
    model: str
    dimension: int | None = None
    timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Response from embedding request."""

    embedding: tuple[float, ...]
    model: str
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        """Set dimension from embedding length."""
        object.__setattr__(self, "dimension", len(self.embedding))


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat completion clients."""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute chat completion.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            Chat response with generated content.

        Raises:
            LLMError: If request fails.
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding for text.

        Args:
            request: Embedding request with text and parameters.

        Returns:
            Embedding response with vector.

        Raises:
            LLMError: If request fails.
        """
        ...


@runtime_checkable
class LLMClient(ChatClient, EmbeddingClient, Protocol):
    """Combined protocol for full LLM client."""

    pass
```

### 2. Ollama Implementation (ollama.py)

```python
"""Ollama LLM client implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import httpx

from ai_psychiatrist.domain.exceptions import (
    LLMError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import OllamaSettings

logger = get_logger(__name__)


def _l2_normalize(embedding: list[float]) -> tuple[float, ...]:
    """L2 normalize an embedding vector."""
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        return tuple(x / norm for x in embedding)
    return tuple(embedding)


class OllamaClient:
    """Ollama API client for chat and embeddings."""

    def __init__(
        self,
        ollama_settings: OllamaSettings,
    ) -> None:
        """Initialize Ollama client.

        Args:
            ollama_settings: Ollama server configuration.
        """
        self._base_url = ollama_settings.base_url
        self._chat_url = ollama_settings.chat_url
        self._embeddings_url = ollama_settings.embeddings_url
        self._default_timeout = ollama_settings.timeout_seconds

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._default_timeout))

    async def __aenter__(self) -> OllamaClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def ping(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if server responds, False otherwise.

        Raises:
            LLMError: If ping fails.
        """
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Ollama ping failed", error=str(e))
            raise LLMError(f"Failed to ping Ollama: {e}") from e

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
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
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
                detail=e.response.text,
            )
            raise LLMError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            logger.error("Chat request error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse chat response", raw=response.text[:500])
            raise LLMResponseParseError(response.text, str(e)) from e

        logger.debug(
            "Chat response received",
            model=data.get("model"),
            content_length=len(content),
        )

        return ChatResponse(
            content=content,
            model=data.get("model", request.model),
            done=data.get("done", True),
            total_duration_ms=data.get("total_duration"),
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
            )
            raise LLMError(f"HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            logger.error("Embedding request error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e

        try:
            data = response.json()
            embedding = data["embedding"]
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse embedding response")
            raise LLMResponseParseError(response.text, str(e)) from e

        # Truncate to requested dimension if specified
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

    # Convenience methods
    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str = "gemma3:27b",
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> str:
        """Simple chat completion with just user/system prompts.

        Args:
            user_prompt: User message content.
            system_prompt: Optional system message.
            model: Model to use.
            temperature: Sampling temperature (e.g., 0.0 for Judge agent).
            top_k: top-k sampling parameter.
            top_p: nucleus sampling parameter.

        Returns:
            Generated response content.
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        request = ChatRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        response = await self.chat(request)
        return response.content

    async def simple_embed(
        self,
        text: str,
        model: str = "dengcao/Qwen3-Embedding-8B:Q8_0",
        dimension: int | None = None,
    ) -> tuple[float, ...]:
        """Simple embedding generation.

        Args:
            text: Text to embed.
            model: Model to use.
            dimension: Optional dimension truncation.

        Returns:
            L2-normalized embedding vector.
        """
        request = EmbeddingRequest(
            text=text,
            model=model,
            dimension=dimension,
        )
        response = await self.embed(request)
        return response.embedding
```

### 3. Response Parsing Utilities (responses.py)

```python
"""Utilities for parsing LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


def extract_json_from_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM response.

    Handles common issues like markdown code blocks, smart quotes,
    and trailing commas.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        LLMResponseParseError: If no valid JSON found.
    """
    # Try extracting from <answer> tags first
    answer_match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw, flags=re.DOTALL)
    if answer_match:
        text = answer_match.group(1)
    else:
        text = raw

    # Strip markdown code blocks
    text = _strip_markdown_fences(text)

    # Normalize quotes and fix trailing commas
    text = _normalize_json_text(text)

    # Extract JSON object
    text = _extract_json_object(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed", error=str(e), text_preview=text[:200])
        raise LLMResponseParseError(raw, str(e)) from e


def extract_xml_tags(raw: str, tags: list[str]) -> dict[str, str]:
    """Extract content from XML-style tags.

    Args:
        raw: Raw text with XML tags.
        tags: List of tag names to extract.

    Returns:
        Dictionary mapping tag names to content.
    """
    result = {}
    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            result[tag] = match.group(1).strip()
        else:
            result[tag] = ""
    return result


def extract_score_from_text(text: str) -> int | None:
    """Extract numeric score from evaluation text.

    Args:
        text: Text containing score.

    Returns:
        Extracted score (1-5) or None if not found.
    """
    patterns = [
        r"[Ss]core[:\s]*(\d+)",
        r"(\d+)[/\s]*(?:out of\s*)?5",
        r"[Rr]ating[:\s]*(\d+)",
        r"^(\d+)\b",  # Number at start
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

    return None


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code block fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _normalize_json_text(text: str) -> str:
    """Normalize JSON text by fixing common issues."""
    # Replace smart quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    # Remove zero-width spaces
    text = text.replace("\u200b", "")

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def _extract_json_object(text: str) -> str:
    """Extract JSON object boundaries from text."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        raise LLMResponseParseError(text, "No JSON object found")

    return text[start : end + 1]


async def repair_json_with_llm(
    llm_client,  # OllamaClient
    broken_json: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    """Attempt to repair malformed JSON using LLM.

    Args:
        llm_client: LLM client for repair request.
        broken_json: Malformed JSON string.
        expected_keys: Expected keys in output.

    Returns:
        Repaired JSON as dictionary.

    Raises:
        LLMResponseParseError: If repair fails.
    """
    repair_prompt = f"""You will be given malformed JSON. Output ONLY a valid JSON object with these EXACT keys:
{', '.join(expected_keys)}

Each value must be an object: {{"evidence": <string>, "reason": <string>, "score": <int 0-3 or "N/A">}}.
If something is missing, fill with {{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}}.

Malformed JSON:
{broken_json}

Return only the fixed JSON. No prose, no markdown, no tags."""

    response = await llm_client.simple_chat(repair_prompt)
    return extract_json_from_response(response)
```

### 4. Mock Implementation (mock.py)

```python
"""Mock LLM client for testing."""

from __future__ import annotations

from typing import Any, Callable

from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)


class MockLLMClient:
    """Mock LLM client for testing.

    Allows specifying canned responses or response functions.
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
        return self._chat_requests

    @property
    def embedding_requests(self) -> list[EmbeddingRequest]:
        """List of embedding requests received."""
        return self._embedding_requests

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
            # Default: random-ish embedding
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
    ) -> str:
        """Simple chat interface matching OllamaClient."""
        from ai_psychiatrist.infrastructure.llm.protocols import ChatMessage

        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        request = ChatRequest(messages=messages, model=model or "mock")
        response = await self.chat(request)
        return response.content

    async def simple_embed(
        self,
        text: str,
        model: str | None = None,
        dimension: int | None = None,
    ) -> tuple[float, ...]:
        """Simple embed interface matching OllamaClient."""
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

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        pass
```

### 5. Tests

```python
"""Tests for LLM infrastructure."""

from __future__ import annotations

import pytest

from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
)
from ai_psychiatrist.infrastructure.llm.responses import (
    extract_json_from_response,
    extract_score_from_text,
    extract_xml_tags,
)


class TestExtractJson:
    """Tests for JSON extraction."""

    def test_clean_json(self) -> None:
        """Should parse clean JSON."""
        raw = '{"key": "value"}'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_json_in_answer_tags(self) -> None:
        """Should extract JSON from answer tags."""
        raw = 'Some text\n<answer>{"key": "value"}</answer>\nMore text'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_markdown_code_block(self) -> None:
        """Should strip markdown fences."""
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_smart_quotes(self) -> None:
        """Should handle smart quotes."""
        raw = '{"key": "value"}'  # curly quotes
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_trailing_comma(self) -> None:
        """Should handle trailing commas."""
        raw = '{"key": "value",}'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}


class TestExtractXmlTags:
    """Tests for XML tag extraction."""

    def test_single_tag(self) -> None:
        """Should extract single tag content."""
        raw = "<assessment>Test content</assessment>"
        result = extract_xml_tags(raw, ["assessment"])
        assert result == {"assessment": "Test content"}

    def test_multiple_tags(self) -> None:
        """Should extract multiple tags."""
        raw = "<a>First</a><b>Second</b>"
        result = extract_xml_tags(raw, ["a", "b"])
        assert result == {"a": "First", "b": "Second"}

    def test_missing_tag(self) -> None:
        """Should return empty string for missing tags."""
        raw = "<a>First</a>"
        result = extract_xml_tags(raw, ["a", "b"])
        assert result == {"a": "First", "b": ""}


class TestExtractScore:
    """Tests for score extraction."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Score: 4", 4),
            ("score: 3", 3),
            ("Rating: 5", 5),
            ("4/5", 4),
            ("3 out of 5", 3),
            ("The score is 4.", 4),
            ("No score here", None),
            ("Score: 6", None),  # Out of range
        ],
    )
    def test_score_patterns(self, text: str, expected: int | None) -> None:
        """Should extract scores from various formats."""
        assert extract_score_from_text(text) == expected


class TestMockLLMClient:
    """Tests for mock LLM client."""

    @pytest.mark.asyncio
    async def test_canned_responses(self) -> None:
        """Should return canned responses in order."""
        mock = MockLLMClient(chat_responses=["first", "second"])

        resp1 = await mock.simple_chat("prompt1")
        resp2 = await mock.simple_chat("prompt2")

        assert resp1 == "first"
        assert resp2 == "second"
        assert mock.chat_call_count == 2

    @pytest.mark.asyncio
    async def test_custom_function(self) -> None:
        """Should use custom function for responses."""

        def custom(req: ChatRequest) -> str:
            return f"Received: {req.messages[-1].content}"

        mock = MockLLMClient(chat_function=custom)
        resp = await mock.simple_chat("hello")

        assert resp == "Received: hello"

    @pytest.mark.asyncio
    async def test_tracks_requests(self) -> None:
        """Should track all requests made."""
        mock = MockLLMClient(chat_responses=["response"])

        await mock.simple_chat("test prompt", system_prompt="system")

        assert len(mock.chat_requests) == 1
        req = mock.chat_requests[0]
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[1].content == "test prompt"

    @pytest.mark.asyncio
    async def test_embedding_dimension(self) -> None:
        """Should respect dimension parameter."""
        mock = MockLLMClient()

        emb = await mock.simple_embed("text", dimension=128)

        assert len(emb) == 128
```

## Acceptance Criteria

- [ ] `OllamaClient` implements `ChatClient` and `EmbeddingClient` protocols
- [ ] All HTTP errors converted to domain exceptions
- [ ] Embeddings are L2-normalized
- [ ] JSON parsing handles markdown, smart quotes, trailing commas
- [ ] `MockLLMClient` enables unit testing without real LLM
- [ ] Comprehensive test coverage including error paths
- [ ] Async/await throughout for non-blocking I/O

## Dependencies

- **Spec 01**: Project structure
- **Spec 02**: Domain exceptions
- **Spec 03**: Configuration and logging

## Specs That Depend on This

- **Spec 06**: Qualitative Agent
- **Spec 07**: Judge Agent
- **Spec 08**: Embedding Service
- **Spec 09**: Quantitative Agent
- **Spec 10**: Meta-Review Agent
