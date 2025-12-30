# Spec 37: Batch Query Embedding

| Field | Value |
|-------|-------|
| **Status** | READY |
| **Priority** | HIGH |
| **Addresses** | BUG-033 (timeouts), BUG-034 (participant regression), BUG-036 (no caching) |
| **Effort** | 1–2 days |
| **Impact** | 1 query-embedding op/participant; removes hard-coded 120s failure mode |

---

## Problem Statement

Run 4 is failing in `few_shot` mode with a **22% failure rate**:
- Output file: `data/outputs/both_paper_backfill-off_20251230_053108.json`
- `few_shot`: 32/41 succeeded, 9/41 failed
- All 9 failures: `LLM request timed out after 120s`

This is not a “Gemma scoring timeout” problem — it is a **query embedding timeout** problem:
- Run provenance shows `embedding_backend = huggingface` and `llm_backend = ollama`.
- The only 120-second default in the embedding path is `EmbeddingRequest.timeout_seconds = 120`.

---

## Root Cause (Code-Level, SSOT)

1. **Eight sequential query embeddings per participant (worst-case)**.
   - `EmbeddingService.build_reference_bundle()` loops all 8 `PHQ8Item`s and calls `embed_text()` for each item that has evidence.
   - File: `src/ai_psychiatrist/services/embedding.py:318`

2. **Hard-coded 120-second timeout is used for query embeddings**.
   - `EmbeddingService.embed_text()` constructs `EmbeddingRequest(...)` without passing `timeout_seconds`.
   - `EmbeddingRequest.timeout_seconds` default is **120 seconds**.
   - HuggingFace enforces this with `asyncio.wait_for(..., timeout=request.timeout_seconds)`.
   - Files:
     - `src/ai_psychiatrist/infrastructure/llm/protocols.py:104` (`EmbeddingRequest.timeout_seconds: int = 120`)
     - `src/ai_psychiatrist/services/embedding.py:121` (missing timeout override)
     - `src/ai_psychiatrist/infrastructure/llm/huggingface.py:121` (wait_for)

3. **Current configuration knob does not affect this path**.
   - `HF_DEFAULT_EMBED_TIMEOUT` only affects `HuggingFaceClient.simple_embed(...)`, but the service calls `embed(...)` directly.

---

## Solution Overview

Batch query embeddings per participant:
- Build all per-item evidence strings first.
- Make **one** batch embedding call (instead of up to 8 single calls).
- Apply a **configurable** query-embedding timeout (no more “stuck at 120s” default).

This is a performance and reliability change; it must not change retrieval semantics.

---

## Goals / Non-Goals

**Goals**
- Reduce query-embedding operations from “up to 8 per participant” → **1 per participant** for HuggingFace.
- Remove the unconfigurable 120s timeout default from the few-shot query embedding path.
- Keep outputs deterministic (no semantic caching, no similarity-based cache keys).
- Make behavior backend-safe (HuggingFace supports true batching; Ollama gets a correct fallback implementation).

**Non-Goals**
- Semantic caching (vector-similarity cache) — out of scope.
- Changing reference embeddings formats — out of scope.
- Changing retrieval algorithm, `top_k`, or scoring — out of scope.

---

## Implementation Plan (Exact Changes)

### Step 1 — Add batch request/response + protocol method

**File**: `src/ai_psychiatrist/infrastructure/llm/protocols.py`

**Insert after** `EmbeddingResponse` (currently around `src/ai_psychiatrist/infrastructure/llm/protocols.py:137`).

**Add dataclasses**

```python
@dataclass(frozen=True, slots=True)
class EmbeddingBatchRequest:
    """Request for embedding multiple texts in one operation."""

    texts: Sequence[str]
    model: str
    dimension: int | None = None
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not self.texts:
            return
        if not self.model:
            raise ValueError("Model cannot be empty")
        if self.dimension is not None and self.dimension < 1:
            raise ValueError(f"dimension {self.dimension} must be >= 1")
        if self.timeout_seconds < 1:
            raise ValueError(f"timeout_seconds {self.timeout_seconds} must be >= 1")
        # Fail fast on empty strings (service should not send them).
        if any(not t for t in self.texts):
            raise ValueError("texts cannot contain empty strings")


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResponse:
    """Response from a batch embedding request."""

    embeddings: list[tuple[float, ...]]
    model: str
```

**Update `EmbeddingClient` protocol** (add this method):

```python
@abstractmethod
async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    """Generate embeddings for multiple texts.

    Must return embeddings in the same order as request.texts.
    """
    ...
```

**Why this shape**
- Matches the existing request/response pattern (`EmbeddingRequest` / `EmbeddingResponse`).
- Avoids adding ad-hoc positional params that drift between backends.

---

### Step 2 — Implement `embed_batch()` for HuggingFaceClient

**File**: `src/ai_psychiatrist/infrastructure/llm/huggingface.py`

**Add imports**:
- `EmbeddingBatchRequest`, `EmbeddingBatchResponse` from `ai_psychiatrist.infrastructure.llm.protocols`

**Add method**: `HuggingFaceClient.embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse`

**Implementation requirements**
- Must call `SentenceTransformer.encode(...)` with list input.
- Must enforce timeout using `asyncio.wait_for(..., timeout=request.timeout_seconds)`.
- Must raise `LLMTimeoutError(request.timeout_seconds)` on timeout.
- Must truncate each embedding if `request.dimension` is not `None`.
- Must return `EmbeddingBatchResponse(embeddings=[...], model=model_id)`.

**Pseudo-code (copy/pasteable structure; final code should use existing helpers)**

```python
async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    model_id = resolve_model_name(request.model, LLMBackend.HUGGINGFACE)
    model = await self._get_embedding_model(model_id)

    async def _encode_async() -> list[tuple[float, ...]]:
        def _encode() -> list[tuple[float, ...]]:
            encoded = model.encode(
                list(request.texts),
                normalize_embeddings=True,
            )
            # sentence-transformers returns numpy array / list-like
            return [tuple(row.tolist()) for row in encoded]

        return await asyncio.to_thread(_encode)

    try:
        embeddings = await asyncio.wait_for(_encode_async(), timeout=request.timeout_seconds)
    except TimeoutError as e:
        raise LLMTimeoutError(request.timeout_seconds) from e

    if request.dimension is not None:
        embeddings = [emb[: request.dimension] for emb in embeddings]

    return EmbeddingBatchResponse(embeddings=embeddings, model=model_id)
```

**Best-practice knobs (optional, but recommended)**
- Use `SentenceTransformer.encode(..., batch_size=...)` if we later embed more than 8 texts.
- Source: SentenceTransformers docs (encode supports `batch_size`, `normalize_embeddings`): https://www.sbert.net/

---

### Step 3 — Implement `embed_batch()` for OllamaClient

**File**: `src/ai_psychiatrist/infrastructure/llm/ollama.py`

Ollama embeddings API does not accept batches, so this is a correctness fallback:
- Sequentially call `embed(EmbeddingRequest(...))` for each text.
- Enforce an **overall** timeout for the entire batch.

**Required behavior**
- Preserve request order.
- Use `asyncio.timeout(request.timeout_seconds)` (Python 3.11+) to cap total time.

**Pseudo-code**

```python
async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    if not request.texts:
        return EmbeddingBatchResponse(embeddings=[], model=request.model)

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

    return EmbeddingBatchResponse(embeddings=embeddings, model=request.model)
```

---

### Step 4 — Add config knobs (timeout + feature flag)

**File**: `src/ai_psychiatrist/config.py`

**Class**: `EmbeddingSettings` (currently starts around `src/ai_psychiatrist/config.py:222`)

Add:

```python
enable_batch_query_embedding: bool = Field(
    default=True,
    description="Use one batch query embedding call per participant (Spec 37).",
)

query_embed_timeout_seconds: int = Field(
    default=300,
    ge=30,
    le=3600,
    description="Timeout for query embedding (single or batch) in seconds (Spec 37).",
)
```

Environment variables:
- `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING=true|false`
- `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS=300`

---

### Step 5 — Use batch embeddings in `EmbeddingService.build_reference_bundle()`

**File**: `src/ai_psychiatrist/services/embedding.py`

**Also required**: update imports at top of file
- Before: `from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingRequest`
- After: `from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingBatchRequest, EmbeddingRequest`

**Add fields in `EmbeddingService.__init__`** (store settings):
- `self._enable_batch_query_embedding = settings.enable_batch_query_embedding`
- `self._query_embed_timeout_seconds = settings.query_embed_timeout_seconds`

**Update `embed_text()`** (remove hard-coded 120s default by passing timeout):

**Before** (`src/ai_psychiatrist/services/embedding.py:134`):
```python
response = await self._llm_client.embed(
    EmbeddingRequest(
        text=text,
        model=model,
        dimension=self._dimension,
    )
)
```

**After**:
```python
response = await self._llm_client.embed(
    EmbeddingRequest(
        text=text,
        model=model,
        dimension=self._dimension,
        timeout_seconds=self._query_embed_timeout_seconds,
    )
)
```

**Replace the per-item embedding loop in `build_reference_bundle()`** with the following (copy/pasteable “after”):

```python
async def build_reference_bundle(
    self,
    evidence_dict: dict[PHQ8Item, list[str]],
) -> ReferenceBundle:
    logger.info(
        "Building reference bundle",
        items_with_evidence=sum(1 for v in evidence_dict.values() if v),
    )

    item_references: dict[PHQ8Item, list[SimilarityMatch]] = {
        item: [] for item in PHQ8Item.all_items()
    }

    item_texts: list[tuple[PHQ8Item, str]] = []
    for item in PHQ8Item.all_items():
        evidence_quotes = evidence_dict.get(item, [])
        if not evidence_quotes:
            continue

        combined_text = "\n".join(evidence_quotes)
        if len(combined_text) < self._min_chars:
            continue

        item_texts.append((item, combined_text))

    if self._enable_batch_query_embedding and item_texts:
        model = get_model_name(self._model_settings, "embedding")
        batch_request = EmbeddingBatchRequest(
            texts=[text for _, text in item_texts],
            model=model,
            dimension=self._dimension,
            timeout_seconds=self._query_embed_timeout_seconds,
        )
        batch_response = await self._llm_client.embed_batch(batch_request)

        for (item, combined_text), query_emb in zip(
            item_texts,
            batch_response.embeddings,
            strict=True,
        ):
            matches = self._compute_similarities(query_emb, item=item)
            matches.sort(key=lambda x: x.similarity, reverse=True)

            if self._min_reference_similarity > 0.0:
                matches = [m for m in matches if m.similarity >= self._min_reference_similarity]

            top_matches = matches[: self._top_k]
            final_matches = await self._validate_matches(top_matches, item, combined_text)

            if self._enable_retrieval_audit:
                evidence_key = f"PHQ8_{item.value}"
                for rank, match in enumerate(final_matches, start=1):
                    logger.info(
                        "retrieved_reference",
                        item=item.value,
                        evidence_key=evidence_key,
                        rank=rank,
                        similarity=match.similarity,
                        participant_id=match.chunk.participant_id,
                        reference_score=match.reference_score,
                        chunk_preview=match.chunk.text[:160],
                        chunk_chars=len(match.chunk.text),
                    )

            item_references[item] = final_matches

            logger.debug(
                "Found references for item",
                item=item.value,
                match_count=len(final_matches),
                top_similarity=final_matches[0].similarity if final_matches else 0,
            )

        return ReferenceBundle(item_references=item_references)

    # Fallback: current sequential behavior (kept for safety / toggle).
    for item in PHQ8Item.all_items():
        evidence_quotes = evidence_dict.get(item, [])
        if not evidence_quotes:
            continue

        combined_text = "\n".join(evidence_quotes)
        if len(combined_text) < self._min_chars:
            continue

        query_emb = await self.embed_text(combined_text)
        if not query_emb:
            continue

        matches = self._compute_similarities(query_emb, item=item)
        matches.sort(key=lambda x: x.similarity, reverse=True)

        if self._min_reference_similarity > 0.0:
            matches = [m for m in matches if m.similarity >= self._min_reference_similarity]

        top_matches = matches[: self._top_k]
        final_matches = await self._validate_matches(top_matches, item, combined_text)

        if self._enable_retrieval_audit:
            evidence_key = f"PHQ8_{item.value}"
            for rank, match in enumerate(final_matches, start=1):
                logger.info(
                    "retrieved_reference",
                    item=item.value,
                    evidence_key=evidence_key,
                    rank=rank,
                    similarity=match.similarity,
                    participant_id=match.chunk.participant_id,
                    reference_score=match.reference_score,
                    chunk_preview=match.chunk.text[:160],
                    chunk_chars=len(match.chunk.text),
                )

        item_references[item] = final_matches

        logger.debug(
            "Found references for item",
            item=item.value,
            match_count=len(final_matches),
            top_similarity=final_matches[0].similarity if final_matches else 0,
        )

    return ReferenceBundle(item_references=item_references)
```

**Acceptance requirement**: HuggingFace backend makes exactly **one** `encode()` call per participant (for all items with evidence).

---

## Edge Cases

- **No items have evidence**: `embed_batch` must not be called; bundle returns all items with empty reference lists.
- **Evidence exists but `combined_text` < `min_evidence_chars`**: treat as no evidence (consistent with current behavior).
- **Backend is Ollama**: `embed_batch` fallback must preserve semantics; performance win may be smaller.
- **Timeout**: error must remain surfaced as `LLMTimeoutError(timeout_seconds)` (same class as today).

---

## Tests (Implementation-Ready, Deterministic)

### Unit: EmbeddingService uses batch once

**File**: `tests/unit/services/test_embedding.py`

Add a unit test that:
- Mocks `llm_client.embed_batch` as `AsyncMock` returning deterministic vectors.
- Ensures `llm_client.embed` is **not** called when `enable_batch_query_embedding=True`.
- Ensures `embed_batch` is called exactly once per `build_reference_bundle(...)` call.

Copy/pasteable test skeleton:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_psychiatrist.config import EmbeddingSettings
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingBatchResponse
from ai_psychiatrist.services.embedding import EmbeddingService
from ai_psychiatrist.services.reference_store import ReferenceStore


@pytest.mark.asyncio
async def test_build_reference_bundle_uses_embed_batch_once() -> None:
    llm_client = MagicMock()
    llm_client.embed = AsyncMock()
    llm_client.embed_batch = AsyncMock(
        return_value=EmbeddingBatchResponse(
            embeddings=[(1.0, 0.0), (0.0, 1.0)],
            model="mock",
        )
    )

    reference_store = MagicMock(spec=ReferenceStore)
    settings = EmbeddingSettings(
        dimension=2,
        enable_batch_query_embedding=True,
        query_embed_timeout_seconds=999,
    )

    service = EmbeddingService(llm_client, reference_store, settings)
    service._compute_similarities = MagicMock(return_value=[])
    service._validate_matches = AsyncMock(return_value=[])

    evidence = {
        PHQ8Item.SLEEP: ["I cannot sleep at night."],
        PHQ8Item.TIRED: ["I feel exhausted most days."],
    }

    _ = await service.build_reference_bundle(evidence)

    llm_client.embed_batch.assert_awaited_once()
    llm_client.embed.assert_not_awaited()
```

**Also required**: update `tests/fixtures/mock_llm.py`
- Add `embed_batch(...)` that loops over texts and calls `embed(...)` to generate a per-text embedding.
- This prevents unrelated tests from failing when they use `MockLLMClient` with `EmbeddingService`.

### Unit: HuggingFaceClient missing deps behavior

**File**: `tests/unit/infrastructure/llm/test_huggingface.py`

Add a test mirroring the existing “deps missing” tests:
- Monkeypatch `importlib.import_module` to raise.
- Assert `await client.embed_batch(...)` raises `MissingHuggingFaceDependenciesError`.

---

## Verification Criteria

1. Unit tests pass:
   - `uv run pytest tests/unit/services/test_embedding.py -v`
   - `uv run pytest tests/unit/infrastructure/llm/test_huggingface.py -v`
2. Reproduction run passes with zero few-shot failures:
   - `uv run python scripts/reproduce_results.py --mode few_shot --split paper`
3. Run 4 regression is gone:
   - `failed_subjects == 0` for `few_shot`
   - runtime returns toward ~95 minutes on the same machine (expect variance).

---

## Best-Practice Notes (2025)

- Batch encode is the standard way to reduce per-call overhead with SentenceTransformers (`encode(sentences=[...], batch_size=...)`): https://www.sbert.net/
- `asyncio.wait_for` is a standard timeout mechanism, but canceling CPU-bound `to_thread` work is not instantaneous, so reducing the number of embedding operations is a durable reliability win: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
