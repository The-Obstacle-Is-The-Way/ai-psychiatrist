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
| Embeddings | Qwen 3 8B Embedding (example Ollama tag: `qwen3-embedding:8b`; quantization not specified in paper) | Section 2.2 |
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
- Embedding model: `qwen3-embedding:8b` (Ollama tag)

### As-Is Defaults (Research/Cluster Scripts)

- Host is typically set via `OLLAMA_NODE = "arctrd..."`
- Models commonly used:
  - `gemma3:27b` / `gemma3-optimized:27b` (chat)
  - `qwen3-embedding:8b` (embeddings)
  - `alibayram/medgemma:27b` (MedGemma variant)

## Deliverables

1. `src/ai_psychiatrist/infrastructure/llm/protocols.py` - Abstract interfaces
2. `src/ai_psychiatrist/infrastructure/llm/ollama.py` - Ollama implementation
3. `src/ai_psychiatrist/infrastructure/llm/responses.py` - Response parsing
4. `tests/fixtures/mock_llm.py` - Test doubles (TEST-ONLY, not in src/)
5. `tests/unit/infrastructure/llm/` - Comprehensive tests

## Test Double Location Policy

**MockLLMClient is a TEST-ONLY artifact and MUST NOT exist in `src/`.**

Location: `tests/fixtures/mock_llm.py`

Rationale:
- **Clean Architecture (Robert C. Martin)**: Test doubles are outer circle concerns. The Dependency Rule states that source code dependencies can only point inwards. Nothing in an inner circle can know anything about an outer circle.
- **ISO 27001 Control 8.31**: Development, testing and production environments should be separated to reduce risks of unauthorized access or changes to the production environment.
- **Safety**: For a medical AI system evaluating psychiatric assessments, mock responses contaminating production is a patient safety issue.

Import pattern for tests:
```python
# CORRECT: Import from tests/fixtures
from tests.fixtures.mock_llm import MockLLMClient

# WRONG: Never import from src (this file no longer exists)
# from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient
```

See: `docs/archive/bugs/BUG-001_MOCK_IN_PRODUCTION_PATH.md`

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
    """Request for chat completion."""

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
    """Response from embedding request."""

    embedding: tuple[float, ...]
    model: str
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        """Set dimension from embedding length."""
        if not self.embedding:
            msg = "Embedding cannot be empty"
            raise ValueError(msg)
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
        model: str = "qwen3-embedding:8b",
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
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
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
        r"score\s*[:\s]\s*(\d+)",  # Score: 4, score : 3, etc.
        r"score\s+of\s+(\d+)",  # score of 4
        r"rating\s*[:\s]\s*(\d+)",  # Rating: 5, rating: 3, etc.
        r"(\d+)\s*[/\s]\s*(?:out of\s*)?5",  # 4/5, 3 out of 5
        r"^(\d+)\b",  # Number at start
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
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
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

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
    value_template = (
        '{"evidence": <string>, "reason": <string>, "score": <int 0-3 or "N/A">}'
    )
    default_value = (
        '{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}'
    )
    repair_prompt = (
        "You will be given malformed JSON. Output ONLY a valid JSON object with these EXACT keys:\n"
        f"{', '.join(expected_keys)}\n\n"
        f"Each value must be an object: {value_template}.\n"
        f"If something is missing, fill with {default_value}.\n\n"
        "Malformed JSON:\n"
        f"{broken_json}\n\n"
        "Return only the fixed JSON. No prose, no markdown, no tags."
    )

    response = await llm_client.simple_chat(repair_prompt)
    return extract_json_from_response(response)
```

### 4. Mock Implementation (tests/fixtures/mock_llm.py)

**NOTE: This file lives in `tests/fixtures/`, NOT in `src/`. See Test Double Location Policy above.**

```python
"""Mock LLM client for testing.

IMPORTANT: This is a TEST-ONLY artifact. It MUST NOT be imported from
production code (src/). Per Clean Architecture, test doubles belong in
the outer test layer, not in production packages.
"""

from __future__ import annotations

import math
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


def _l2_normalize(embedding: tuple[float, ...]) -> tuple[float, ...]:
    """L2 normalize an embedding vector for test/prod parity."""
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        return tuple(x / norm for x in embedding)
    return embedding


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
            # Default: deterministic embedding based on dimension, L2-normalized
            dim = request.dimension or 256
            raw = tuple(0.1 * (i % 10) for i in range(dim))
            embedding = _l2_normalize(raw)

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
        """Simple chat interface matching OllamaClient."""
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

from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
)
from ai_psychiatrist.infrastructure.llm.responses import (
    extract_json_from_response,
    extract_score_from_text,
    extract_xml_tags,
    repair_json_with_llm,
)
# NOTE: MockLLMClient lives in tests/fixtures/ per BUG-001
from tests.fixtures.mock_llm import MockLLMClient


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

    def test_json_in_answer_tags_nested(self) -> None:
        """Should extract nested JSON from answer tags."""
        raw = '<answer>\n{"outer": {"inner": 1}}\n</answer>'
        result = extract_json_from_response(raw)
        assert result == {"outer": {"inner": 1}}

    def test_markdown_code_block(self) -> None:
        """Should strip markdown fences."""
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_smart_quotes(self) -> None:
        """Should handle smart quotes."""
        raw = "{\u201ckey\u201d: \u201cvalue\u201d}"
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


class TestRepairJsonWithLlm:
    """Tests for repair_json_with_llm."""

    @pytest.mark.asyncio
    async def test_repair_json_success(self) -> None:
        """Should return repaired JSON from LLM output."""
        mock = MockLLMClient(
            chat_responses=[
                '```json\\n{"a": {"evidence": "x", "reason": "y", "score": 1}}\\n```'
            ]
        )

        result = await repair_json_with_llm(
            mock,
            broken_json='{"a": {"evidence": "x", "reason": "y", "score": 1,}}',
            expected_keys=["a"],
        )

        assert result == {"a": {"evidence": "x", "reason": "y", "score": 1}}

    @pytest.mark.asyncio
    async def test_repair_json_invalid_raises(self) -> None:
        """Should raise when LLM output is not valid JSON."""
        mock = MockLLMClient(chat_responses=["not json"])

        with pytest.raises(LLMResponseParseError):
            await repair_json_with_llm(
                mock,
                broken_json='{"a": [}',
                expected_keys=["a"],
            )

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
