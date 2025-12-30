# BUG-033: Runtime Query Embedding Timeouts

| Field | Value |
|-------|-------|
| **Status** | OPEN |
| **Severity** | CRITICAL |
| **Affects** | few_shot mode |
| **Introduced** | Unknown (design issue) |
| **Discovered** | 2025-12-30 |

## Summary

HuggingFace query embeddings timeout after 120 seconds during few-shot assessment, causing participant failures. All 9 failed participants in latest run (77a2bdb8) were due to this timeout.

## Root Cause

The `EmbeddingService.embed_text()` function generates embeddings **at runtime** for evidence text during `build_reference_bundle()`. HuggingFace backend has a 120-second default timeout that is too short for:
- Large combined evidence text
- First-call model loading overhead
- CPU-bound embedding computation

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
              → asyncio.wait_for(..., timeout=120)  # TIMEOUT HERE
```

### Key Files
- `src/ai_psychiatrist/services/embedding.py:121-151` - `embed_text()`
- `src/ai_psychiatrist/infrastructure/llm/huggingface.py:145-147` - timeout enforcement
- `src/ai_psychiatrist/config.py:63-66` - `default_embed_timeout=120`

### Configuration
```python
# In config.py
class HuggingFaceSettings(BaseSettings):
    default_embed_timeout: int = Field(
        default=120,  # TOO SHORT
        description="Default timeout for embedding requests",
    )
```

## Immediate Fix

Increase the embedding timeout in `.env`:
```bash
HF_DEFAULT_EMBED_TIMEOUT=300  # 5 minutes
# or
HF_DEFAULT_EMBED_TIMEOUT=600  # 10 minutes for slow machines
```

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

From web search:
- **Semantic caching** with similarity threshold (0.8+)
- **Matryoshka embeddings** for two-stage retrieval
- **Layered caching** architecture (in-memory + Redis + disk)
- **TTL expiration** for cache freshness

Sources:
- [Mastering Embedding Caching 2025](https://sparkco.ai/blog/mastering-embedding-caching-advanced-techniques-for-2025)
- [Semantic LLM Caching](https://www.marktechpost.com/2025/11/11/how-to-reduce-cost-and-latency-of-your-rag-application-using-semantic-llm-caching/)

## Related

- BUG-027: Unified timeout configuration
- BUG-034: Few-shot participant count regression
- BUG-036: No query embedding caching
