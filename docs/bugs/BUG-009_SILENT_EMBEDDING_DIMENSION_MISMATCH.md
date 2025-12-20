# BUG-009: Silent Embedding Dimension Mismatch

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
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

## Recommended Fix

- Validate embedding dimensionality at load time and **raise `EmbeddingDimensionMismatchError`** if mismatched.
- Alternatively, log a clear error and disable few-shot with an explicit warning.
- Add tests for mismatched reference dimensions to ensure a failure is surfaced.

---

## Files Involved

- `src/ai_psychiatrist/services/embedding.py`
- `src/ai_psychiatrist/services/reference_store.py`
- `src/ai_psychiatrist/domain/exceptions.py`
