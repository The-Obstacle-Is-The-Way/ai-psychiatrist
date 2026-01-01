# Batch Query Embedding (Spec 37) — Fixing Query-Embedding Timeouts

**Audience**: Researchers running few-shot experiments
**Last Updated**: 2026-01-01

Spec 37 is a **performance + reliability** change to the few-shot retrieval pipeline:
- **Before**: up to **8 sequential query embeddings** per participant (one per PHQ-8 item with evidence)
- **After**: **1 batch query embedding** per participant (all item evidence embedded together)

This fixes the Run 4 regression where few-shot runs failed with `LLM request timed out after 120s` due to repeated query embedding calls.

SSOT implementation:
- `EmbeddingService.build_reference_bundle()` in `src/ai_psychiatrist/services/embedding.py`
- `EmbeddingBatchRequest` / `EmbeddingClient.embed_batch()` in `src/ai_psychiatrist/infrastructure/llm/protocols.py`
- `HuggingFaceClient.embed_batch()` in `src/ai_psychiatrist/infrastructure/llm/huggingface.py`
- `OllamaClient.embed_batch()` in `src/ai_psychiatrist/infrastructure/llm/ollama.py` (sequential fallback)

---

## Why This Exists (Root Cause)

Few-shot retrieval embeds the **query evidence** (not the reference corpus) to retrieve similar reference chunks.

Evidence is extracted per PHQ-8 item, so a participant can produce up to 8 evidence texts:
- `PHQ8_Sleep` evidence text
- `PHQ8_Tired` evidence text
- …

Historically, the system embedded these queries **one-by-one**. In the worst case:
- 8 embeddings per participant × 41 participants (paper-test) = 328 embedding calls

That increases:
- wall-clock runtime
- timeout exposure (a single timeout fails the participant)

Spec 37 reduces this to:
- 1 embedding operation per participant (HuggingFace true batching; Ollama fallback still correct)

---

## Configuration

### Batch Embedding Toggle

```bash
EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING=true
```

- **Code default**: `true`
- Disable only for debugging parity with older runs:

```bash
EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING=false
```

### Query Embedding Timeout

```bash
EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS=300
```

- **Code default**: `300`
- This timeout applies to:
  - `embed_text()` (single)
  - `embed_batch()` (batch)

---

## How It Works (At a High Level)

1. Build per-item evidence texts from extracted evidence quotes.
2. If batch embedding is enabled:
   - call `EmbeddingClient.embed_batch(...)` once with all evidence texts
3. For each embedded item evidence:
   - retrieve candidate references (`_compute_similarities`)
   - apply guardrails (Spec 33), optional item-tag filtering (Spec 34), optional chunk-score source (Spec 35), optional CRAG validation (Spec 36)
4. Format the unified `<Reference Examples>` block (Spec 31 + Spec 33 XML fix).

---

## Verification (What To Check)

### 1) Reproduce results output should not include 120s embedding timeouts

Run few-shot on a small limit:

```bash
uv run python scripts/reproduce_results.py --split paper-test --few-shot-only --limit 3
```

If you still see `LLM request timed out after 120s`, you are likely:
- running an older code revision, or
- using an entry point that bypasses `EmbeddingService.query_embed_timeout_seconds`

### 2) Confirm the effective settings snapshot

`scripts/reproduce_results.py` prints the effective embedding settings at startup (including batch + timeout). Treat that output as the first sanity check.

---

## Relationship to Other Retrieval Fixes

Spec 37 is **not** a retrieval-quality fix. It does not change similarity, filtering, or scoring. It exists to:
- reduce repeated embedding overhead
- reduce the chance of query embedding timeouts

See also:
- Retrieval debugging: `docs/embeddings/debugging-retrieval-quality.md`
- Feature index: `docs/pipeline-internals/features.md`
