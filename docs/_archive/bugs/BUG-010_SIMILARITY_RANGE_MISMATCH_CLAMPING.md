# BUG-010: Cosine Similarity Range Mismatch (Clamping)

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-20
**Spec Reference**: `docs/specs/08_EMBEDDING_SERVICE.md`, `docs/specs/02_CORE_DOMAIN.md`

---

## Executive Summary

Cosine similarity values are **clamped to [0, 1]** before creating `SimilarityMatch`. This conflicts with standard cosine similarity range **[-1, 1]** as referenced in the spec and literature, and can distort ranking when negative similarities exist. The clamp is a workaround for the domain constraint but changes the meaning of similarity.

---

## Evidence

- Similarity is clamped to [0, 1] before constructing `SimilarityMatch`. (`src/ai_psychiatrist/services/embedding.py:166-167`)
- `SimilarityMatch` value object explicitly enforces `0.0 <= similarity <= 1.0` in `__post_init__`, raising ValueError for negative values. (`src/ai_psychiatrist/domain/value_objects.py`)
- Spec 08 references raw cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) which yields values in [-1, 1]. (`docs/specs/08_EMBEDDING_SERVICE.md`)
- Domain enforces similarity in [0, 1]. (`docs/specs/02_CORE_DOMAIN.md:253-255`)

---

## Impact

- Negative similarities are mapped to 0, collapsing distinct values and altering ranking.
- This deviates from the paper’s cosine similarity semantics and may affect reference selection quality.

---

## Scope & Disposition

- **Code Path**: Current implementation + domain constraints (`src/ai_psychiatrist/...`).
- **Fix Category**: Domain/model alignment.
- **Recommended Action**: Fix now; choose one semantic and update domain + spec + implementation together.

---

## Recommended Fix

Choose one consistent approach and update both domain + implementation:

Option A: Allow [-1, 1] in `SimilarityMatch` and remove clamping.
Option B: Convert cosine similarity to [0, 1] using `(1 + cos) / 2` and update spec and tests accordingly.

---

## Files Involved

- `src/ai_psychiatrist/services/embedding.py`
- `src/ai_psychiatrist/domain/value_objects.py`
- `docs/specs/08_EMBEDDING_SERVICE.md`
- `docs/specs/02_CORE_DOMAIN.md`

---

## Resolution

Chose **Option B**: Transform cosine similarity to [0, 1] using `(1 + cos) / 2`.

Changes:
1. **EmbeddingService._compute_similarities()**: Replaced clamping with proper transformation:
   ```python
   raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])
   sim = (1.0 + raw_cos) / 2.0
   ```

2. **SimilarityMatch docstring**: Updated to document the transformation semantics:
   - 0 = opposite vectors (raw cos = -1)
   - 0.5 = orthogonal vectors (raw cos = 0)
   - 1.0 = identical vectors (raw cos = 1)

Tests added:
- `TestSimilarityTransformation.test_similarity_transformation_range`
- `TestSimilarityTransformation.test_similarity_transformation_values`

This approach:
- Preserves the [0, 1] domain constraint (no breaking change)
- Eliminates semantic distortion from clamping negative values
- Provides meaningful similarity values for ranking

---

## Verification

```bash
pytest tests/unit/services/test_embedding.py -v --no-cov -k "transformation"
# 2 passed
```
