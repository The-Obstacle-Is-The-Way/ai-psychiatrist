# BUG-036: No Query Embedding Caching

| Field | Value |
|-------|-------|
| **Status** | SPEC'd |
| **Severity** | HIGH |
| **Affects** | few_shot mode performance |
| **Introduced** | Original design |
| **Discovered** | 2025-12-30 |
| **Solution** | [Spec 37: Batch Query Embedding](../specs/37-batch-query-embedding.md) |

> **Note**: Batch embedding (Spec 37) provides 8x reduction in embedding calls.
> LRU caching for repeated evidence text is a future enhancement tracked separately.

## Summary

Query embeddings (for evidence text) are computed fresh for every PHQ-8 item during quantitative assessment. There is no caching mechanism, leading to:
- Up to 8 separate embedding calls per transcript (one per PHQ-8 item **that has evidence**)
- Repeated work when evidence text is similar across items
- Runtime latency that contributes to BUG-033 timeouts (query embeddings use the `EmbeddingRequest.timeout_seconds = 120` default unless explicitly overridden)

## Current Architecture

### Reference Embeddings (PRECOMPUTED)
```
generate_embeddings.py → .npz files → ReferenceStore (lazy load once)
```

### Query Embeddings (RUNTIME - NO CACHING)
```
QuantitativeAssessmentAgent.assess()
  → EmbeddingService.build_reference_bundle()
      → for each PHQ-8 item:
            embed_text(evidence)  # NEW CALL EVERY TIME
```

## Code Path

```python
# embedding.py lines 318-398
async def build_reference_bundle(self, evidence_dict: dict[PHQ8Item, list[str]]) -> ReferenceBundle:
    for item in PHQ8Item.all_items():
        combined_text = "\n".join(evidence_quotes)
        query_emb = await self.embed_text(combined_text)  # NO CACHE
        # ...
```

## Impact

| Metric | Current | With Caching |
|--------|---------|--------------|
| Embedding calls per transcript | Up to 8 | 1 (batch) |
| Cache hits on repeated evidence | 0% | ~30-50% (estimated) |
| Cold start overhead | Every call | First call only |

## Proposed Solutions

### Option 1: Batch Embedding (Recommended)
**Effort**: 1-2 days | **Impact**: 8x reduction

```python
# Proposed change to build_reference_bundle()
all_texts = ["\n".join(evidence_dict.get(item, [])) for item in PHQ8Item.all_items()]
batch_request = EmbeddingBatchRequest(
    texts=all_texts,
    model=model,
    dimension=self._dimension,
    timeout_seconds=self._query_embed_timeout_seconds,
)
batch_response = await self._llm_client.embed_batch(batch_request)
all_embeddings = batch_response.embeddings
```

Requires:
- Add `embed_batch()` to `EmbeddingClient` protocol
- Implement batch encoding in `HuggingFaceClient`
- Update `build_reference_bundle()` to collect all texts first

### Option 2: LRU Cache
**Effort**: 1 day | **Impact**: Helps repeated evidence

```python
from functools import lru_cache
import hashlib

class EmbeddingService:
    _cache: dict[str, tuple[float, ...]] = {}

    async def embed_text(self, text: str) -> tuple[float, ...]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        embedding = await self._llm_client.embed(...)
        self._cache[key] = embedding
        return embedding
```

### Option 3: Redis Cache
**Effort**: 2-3 days | **Impact**: Cross-instance caching

For distributed systems:
- Store embeddings in Redis with TTL
- Use text hash as key
- Share cache across workers

### Option 4: Model Warm-up
**Effort**: Few hours | **Impact**: Eliminates cold start

```python
# server.py startup hook
@app.on_event("startup")
async def warm_up_embedding_model():
    dummy_text = "warm up the embedding model"
    await embedding_service.embed_text(dummy_text)
```

## 2025 Best Practices

Cross-checked sources (Dec 2025):
1. **Batch embeddings** are the primary lever when your backend supports vectorized encoding (`SentenceTransformer.encode(sentences=[...], batch_size=...)`): https://www.sbert.net/
2. **Timeouts**: `asyncio.wait_for(...)` is the standard mechanism, but canceling a `to_thread`-spawned CPU task is not instantaneous — so reducing call count (batching) and avoiding repeated work (caching) are the durable fixes: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
3. **Semantic caching** exists (e.g., GPTCache) but adds complexity/variance; deterministic caching by exact-text hash is usually the first step for reproducibility pipelines: https://github.com/zilliztech/GPTCache

## Implementation Priority

1. **Batch Embedding** (Option 1) - Immediate impact, reduces calls 8x
2. **Model Warm-up** (Option 4) - Quick win, eliminates cold start
3. **LRU Cache** (Option 2) - Medium-term, helps repeated evidence
4. **Redis Cache** (Option 3) - Long-term, for distributed deployment

## Verification

After implementation:
- [ ] Single batch embedding call per transcript
- [ ] No cold start delay (model pre-loaded)
- [ ] Cache hit rate > 0% for repeated evidence
- [ ] No 120s timeout failures

## Related

- BUG-033: Runtime query embedding timeouts
- BUG-034: Few-shot participant count regression
