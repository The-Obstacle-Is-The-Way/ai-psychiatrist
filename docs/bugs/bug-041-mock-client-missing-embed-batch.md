# BUG-041: MockLLMClient Missing embed_batch Method

| Field | Value |
|-------|-------|
| **Status** | FIXED |
| **Severity** | HIGH (Blocks Spec 37) |
| **Affects** | Test suite |
| **Introduced** | N/A (missing implementation) |
| **Discovered** | 2025-12-30 |
| **Solution** | [Spec 37: Batch Query Embedding](../specs/37-batch-query-embedding.md) (Step: MockLLMClient update) |

## Summary

`MockLLMClient` in `tests/fixtures/mock_llm.py` does not implement the `embed_batch()` method that Spec 37 will add to the `EmbeddingClient` protocol. This will cause test failures once Spec 37 is implemented.

---

## Current State

**File**: `tests/fixtures/mock_llm.py`

```python
class MockLLMClient:
    # Has these methods:
    async def chat(self, request: ChatRequest) -> ChatResponse: ...
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...
    async def simple_chat(...) -> str: ...
    async def simple_embed(...) -> tuple[float, ...]: ...
    async def close(self) -> None: ...

    # MISSING:
    # async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse
```

---

## Impact

When Spec 37 is implemented:
1. `EmbeddingClient` protocol gains `embed_batch()` method
2. `EmbeddingService.build_reference_bundle()` calls `embed_batch()`
3. Tests using `MockLLMClient` with `enable_batch_query_embedding=True` will fail with:
   ```
   AttributeError: 'MockLLMClient' object has no attribute 'embed_batch'
   ```

**Affected tests** (at minimum):
- `TestEmbeddingService.test_build_reference_bundle`
- `TestEmbeddingService.test_build_reference_bundle_short_evidence`
- `TestEmbeddingService.test_build_reference_bundle_logs_audit_when_enabled`
- `TestEmbeddingService.test_reference_threshold_filters_low_similarity`
- `TestEmbeddingService.test_reference_budget_limits_included_matches`
- `TestEmbeddingService.test_defaults_preserve_existing_selection`

---

## Fix

### Add to MockLLMClient

```python
async def embed_batch(self, request: EmbeddingBatchRequest) -> EmbeddingBatchResponse:
    """Return mock batch embeddings (one per text in request)."""
    self._embed_batch_requests.append(request)
    self._embed_batch_call_count += 1

    embeddings: list[tuple[float, ...]] = []
    for text in request.texts:
        # Delegate to single-embed logic
        single_request = EmbeddingRequest(
            text=text,
            model=request.model,
            dimension=request.dimension,
            timeout_seconds=request.timeout_seconds,
        )
        response = await self.embed(single_request)
        embeddings.append(response.embedding)

    return EmbeddingBatchResponse(embeddings=embeddings, model=request.model)
```

### Add Tracking (Optional)

```python
def __init__(self, ...):
    # ... existing ...
    self._embed_batch_requests: list[EmbeddingBatchRequest] = []
    self._embed_batch_call_count = 0

@property
def embed_batch_call_count(self) -> int:
    return self._embed_batch_call_count

@property
def embed_batch_requests(self) -> list[EmbeddingBatchRequest]:
    return self._embed_batch_requests.copy()
```

### Update Imports

```python
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingBatchRequest,   # Add
    EmbeddingBatchResponse,  # Add
)
```

---

## Timing

**Must be done**: Before or immediately after Spec 37 protocols are added.

**Order of operations**:
1. Add `EmbeddingBatchRequest` / `EmbeddingBatchResponse` to protocols.py (Spec 37 Step 1)
2. Add `embed_batch()` to MockLLMClient (this bug)
3. Implement `embed_batch()` in HuggingFaceClient and OllamaClient (Spec 37 Step 2-3)
4. Update EmbeddingService (Spec 37 Step 5)

---

## Verification

- [ ] `MockLLMClient` has `embed_batch()` method
- [ ] Method signature matches protocol
- [ ] Existing tests pass
- [ ] New tests can mock batch behavior
