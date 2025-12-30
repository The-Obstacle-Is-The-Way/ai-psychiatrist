# BUG-033: Runtime Query Embedding Timeouts

| Field | Value |
|-------|-------|
| **Status** | FIXED |
| **Severity** | CRITICAL |
| **Affects** | few_shot mode |
| **Introduced** | Unknown (design issue) |
| **Discovered** | 2025-12-30 |
| **Solution** | [Spec 37: Batch Query Embedding](../specs/37-batch-query-embedding.md) |

## Summary

HuggingFace query embeddings timeout after 120 seconds during few-shot assessment, causing participant failures. All 9 failed participants in latest run (77a2bdb8) were due to this timeout.

## Root Cause

The `EmbeddingService.embed_text()` function generates embeddings **at runtime** for evidence text during `build_reference_bundle()`. The timeout that is firing is **not** `HF_DEFAULT_EMBED_TIMEOUT` (HuggingFaceSettings); it is the hard-coded default on `EmbeddingRequest.timeout_seconds`.

Concretely:
- `EmbeddingService.embed_text()` constructs `EmbeddingRequest(...)` **without** setting `timeout_seconds` (so it uses the dataclass default).
- `EmbeddingRequest.timeout_seconds` defaults to **120 seconds**.
- `HuggingFaceClient.embed()` enforces the timeout via `asyncio.wait_for(..., timeout=request.timeout_seconds)`.

This creates an effectively “hard-coded 120s embed timeout” for query embeddings, which is too short for the HuggingFace/SentenceTransformers backend on slower machines and/or longer evidence strings.

## Evidence

### Failed Participants (Latest Run)
| PID | Error |
|-----|-------|
| 345 | `LLM request timed out after 120s` |
| 357 | `LLM request timed out after 120s` |
| 385 | `LLM request timed out after 120s` |
| 390 | `LLM request timed out after 120s` |
| 413 | `LLM request timed out after 120s` |
| 417 | `LLM request timed out after 120s` |
| 422 | `LLM request timed out after 120s` |
| 451 | `LLM request timed out after 120s` |
| 487 | `LLM request timed out after 120s` |

### Comparison
- **Dec 29 run** (5e62455): 41/41 few_shot success
- **Dec 30 run** (be35e35): 32/41 few_shot success (9 timeouts)

## Technical Details

### Code Path
```
QuantitativeAssessmentAgent.assess()
  → EmbeddingService.build_reference_bundle()
      → EmbeddingService.embed_text()
          → HuggingFaceClient.embed()
              → asyncio.wait_for(..., timeout=request.timeout_seconds)  # 120s default
```

### Key Files
- `src/ai_psychiatrist/services/embedding.py:121-151` - `EmbeddingService.embed_text()` (does not pass `timeout_seconds`)
- `src/ai_psychiatrist/infrastructure/llm/protocols.py:104-134` - `EmbeddingRequest.timeout_seconds: int = 120` (hard-coded default)
- `src/ai_psychiatrist/infrastructure/llm/huggingface.py:133-151` - `HuggingFaceClient.embed()` timeout enforcement (`asyncio.wait_for`)
- `src/ai_psychiatrist/config.py:59-70` - `HuggingFaceSettings.default_embed_timeout` (NOTE: **not** used by this failing code path)

### Configuration
```python
# NOTE: This does NOT affect the failing path.
# HF_DEFAULT_EMBED_TIMEOUT only applies to HuggingFaceClient.simple_embed(...),
# but EmbeddingService.embed_text() calls HuggingFaceClient.embed(...) directly
# and relies on EmbeddingRequest.timeout_seconds default (120s).
#
# In other words: setting HF_DEFAULT_EMBED_TIMEOUT will not fix this bug.
```

## Immediate Fix

There is currently **no env-only fix** because query embedding timeout is not wired to config; it is the `EmbeddingRequest` dataclass default (120s).

Short-term mitigation requires code change (covered by Spec 37):
- Add a configurable query embedding timeout and pass it into `EmbeddingRequest(timeout_seconds=...)`, or
- Stop making 8 sequential per-item calls by batching (Spec 37).

## Long-Term Solutions

### Option 1: Batch Embedding (Recommended)
Collect all evidence texts upfront and embed in one call:
- 8x fewer API calls (1 batch vs 8 individual)
- Effort: 1-2 days

### Option 2: Model Warm-up
Pre-load embedding model at server startup:
- Eliminates cold-start delay
- Effort: Few hours

### Option 3: LRU Cache
Cache query embeddings for repeated evidence text:
- Helps with repeated assessments
- Effort: 1 day

### Option 4: Parallel Embedding
Use `asyncio.TaskGroup` for concurrent embedding:
- Parallel items, single wait timeout
- Effort: 2-3 hours

## 2025 Best Practices

Cross-checked sources (Dec 2025):
- SentenceTransformers supports true batch embedding via `SentenceTransformer.encode(..., batch_size=..., normalize_embeddings=...)` which is the right primitive for Spec 37: https://www.sbert.net/
- Python timeouts are typically implemented with `asyncio.wait_for(...)` (what we do today), but cancellation does **not** stop CPU-bound `to_thread` work immediately — so reducing the number of embedding calls matters: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
- For retry/backoff patterns (when timeouts are transient), Tenacity is a common reference implementation: https://tenacity.readthedocs.io/

## Related

- BUG-027: Unified timeout configuration
- BUG-034: Few-shot participant count regression
- BUG-036: No query embedding caching
