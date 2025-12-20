# BUG-009: Silent Embedding Dimension Mismatch

**Severity**: HIGH (P1)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-20
**Spec Reference**: `docs/specs/08_EMBEDDING_SERVICE.md`, `docs/specs/09_QUANTITATIVE_AGENT.md`

---

## Executive Summary

Embedding dimension mismatches are silently ignored during similarity search. When the reference embeddings do not match the configured dimension, the system **drops all mismatched vectors** without warning and returns no reference examples. This silently degrades few-shot performance and can make the quantitative agent behave like zero-shot without obvious errors.

---

## Evidence

- Embedding dimension mismatch is skipped without logging or raising an error. (`src/ai_psychiatrist/services/embedding.py:160-161`)
- A domain exception `EmbeddingDimensionMismatchError` exists but is never used. (`src/ai_psychiatrist/domain/exceptions.py:202-218`)
- Reference embeddings are truncated to `EmbeddingSettings.dimension`, but if the stored vectors are shorter (e.g., older 1024-dim embeddings), they remain shorter and will all be skipped by the comparison logic.

---

## Impact

- Few-shot retrieval can silently return **zero matches** even when embeddings exist.
- QuantitativeAgent continues without references, lowering accuracy and diverging from paper results.
- Debugging is difficult because no error or warning indicates the mismatch.

---

## Scope & Disposition

- **Code Path**: Current implementation (`src/ai_psychiatrist/services/...`).
- **Fix Category**: Data integrity and error signaling.
- **Recommended Action**: Fix now; fail loudly or explicitly disable few-shot when dimensions mismatch.

---

## Recommended Fix

- Validate embedding dimensionality at load time and **raise `EmbeddingDimensionMismatchError`** if mismatched.
- Alternatively, log a clear error and disable few-shot with an explicit warning.
- Add tests for mismatched reference dimensions to ensure a failure is surfaced.

---

## Files Involved

- `src/ai_psychiatrist/services/embedding.py`
- `src/ai_psychiatrist/services/reference_store.py`
- `src/ai_psychiatrist/domain/exceptions.py`

---

## Resolution

Fixed dimension mismatch handling in `ReferenceStore._load_embeddings()`:

1. **Explicit validation at load time**: Embeddings shorter than configured dimension are
   logged with a warning and skipped.

2. **Fail-fast on total mismatch**: If ALL embeddings have insufficient dimension,
   raises `EmbeddingDimensionMismatchError` with expected and actual dimensions.

3. **Logging for partial mismatch**: When some embeddings are skipped, logs an error
   with the count of skipped vs total chunks.

4. **Warning in EmbeddingService**: Added logging when query/reference dimensions mismatch
   during similarity computation.

Tests added:
- `TestEmbeddingDimensionMismatch.test_all_embeddings_mismatched_raises_error`
- `TestEmbeddingDimensionMismatch.test_partial_mismatch_skips_bad_embeddings`
- `TestEmbeddingDimensionMismatch.test_embedding_truncation`

---

## Verification

```bash
pytest tests/unit/services/test_embedding.py -v --no-cov -k "dimension"
# 3 passed
```
