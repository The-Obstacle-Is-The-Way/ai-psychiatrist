# Spec 055: Embedding NaN Detection and Validation

**Status**: Ready to Implement
**Priority**: High
**Complexity**: Low
**Related**: PIPELINE-BRITTLENESS.md

---

## Problem Statement

NaN (Not a Number) values in embedding vectors propagate silently through the pipeline:

1. **Source**: Embedding backend returns NaN (rare but possible with malformed input)
2. **Propagation**: L2 normalization with NaN → NaN persists
3. **Corruption**: Cosine similarity with NaN → NaN similarity scores
4. **Result**: Reference ranking becomes meaningless

This is a **silent corruption** that produces unpredictable results without any error.

---

## Current Behavior

```python
# reference_store.py - L2 normalization
def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb  # Returns zero vector unchanged
    return emb / norm
```

If `emb` contains NaN:
- `np.linalg.norm(emb)` returns NaN
- `emb / NaN` returns array of NaN
- No error raised

```python
# embedding.py - similarity computation
similarities = reference_matrix @ query_vector  # NaN propagates
```

---

## Proposed Solution

Add NaN detection at all embedding generation and loading points.

---

## Implementation

### Core Validation Function

```python
# New: src/ai_psychiatrist/infrastructure/validation.py

import numpy as np
from ai_psychiatrist.domain.exceptions import EmbeddingValidationError


def validate_embedding(
    embedding: np.ndarray,
    context: str = "embedding",
    *,
    check_nan: bool = True,
    check_inf: bool = True,
    check_zero: bool = True,
) -> np.ndarray:
    """Validate embedding vector for common corruption patterns.

    Args:
        embedding: Vector to validate
        context: Description for error messages (e.g., "query embedding for participant 300")
        check_nan: Raise if NaN detected
        check_inf: Raise if Inf detected
        check_zero: Raise if all-zero vector detected

    Returns:
        The validated embedding (unchanged)

    Raises:
        EmbeddingValidationError: If validation fails
    """
    if check_nan and np.isnan(embedding).any():
        nan_count = np.isnan(embedding).sum()
        nan_positions = np.where(np.isnan(embedding))[0][:5]  # First 5 positions
        raise EmbeddingValidationError(
            f"NaN detected in {context}: {nan_count} NaN values at positions {nan_positions.tolist()}"
        )

    if check_inf and np.isinf(embedding).any():
        inf_count = np.isinf(embedding).sum()
        raise EmbeddingValidationError(
            f"Inf detected in {context}: {inf_count} Inf values"
        )

    if check_zero and np.allclose(embedding, 0):
        raise EmbeddingValidationError(
            f"All-zero vector in {context}: L2 norm is 0, cosine similarity undefined"
        )

    return embedding


def validate_embedding_matrix(
    matrix: np.ndarray,
    context: str = "embedding matrix",
) -> np.ndarray:
    """Validate entire embedding matrix.

    Args:
        matrix: 2D array of shape (n_samples, n_dims)
        context: Description for error messages

    Returns:
        The validated matrix (unchanged)

    Raises:
        EmbeddingValidationError: If validation fails
    """
    if matrix.ndim != 2:
        raise EmbeddingValidationError(
            f"Expected 2D matrix in {context}, got shape {matrix.shape}"
        )

    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        nan_rows = np.where(nan_mask.any(axis=1))[0]
        raise EmbeddingValidationError(
            f"NaN detected in {context}: {len(nan_rows)} rows contain NaN "
            f"(first few: {nan_rows[:5].tolist()})"
        )

    inf_mask = np.isinf(matrix)
    if inf_mask.any():
        inf_rows = np.where(inf_mask.any(axis=1))[0]
        raise EmbeddingValidationError(
            f"Inf detected in {context}: {len(inf_rows)} rows contain Inf"
        )

    zero_rows = np.where(~matrix.any(axis=1))[0]
    if len(zero_rows) > 0:
        raise EmbeddingValidationError(
            f"All-zero rows in {context}: {len(zero_rows)} rows "
            f"(first few: {zero_rows[:5].tolist()})"
        )

    return matrix
```

### New Exception Type

```python
# domain/exceptions.py - add new exception

class EmbeddingValidationError(DomainException):
    """Raised when embedding vector validation fails (NaN, Inf, zero)."""

    def __init__(self, message: str):
        super().__init__(message)
```

### Integration Points

#### 1. Query Embedding Generation

```python
# embedding.py - EmbeddingService._embed_text()

async def _embed_text(self, text: str) -> np.ndarray:
    """Generate embedding for text."""
    embedding = await self._client.embed(text, model=self._model)
    vector = np.array(embedding, dtype=np.float32)

    # NEW: Validate before returning
    from ai_psychiatrist.infrastructure.validation import validate_embedding
    validate_embedding(
        vector,
        context=f"query embedding for text '{text[:50]}...'",
    )

    return vector
```

#### 2. Reference Store Loading

```python
# reference_store.py - ReferenceStore._load_embeddings()

def _load_embeddings(self) -> None:
    """Load and validate reference embeddings."""
    # ... existing loading code ...

    # After building the matrix:
    from ai_psychiatrist.infrastructure.validation import validate_embedding_matrix

    # NEW: Validate full matrix after loading
    validate_embedding_matrix(
        self._reference_matrix,
        context=f"reference embeddings from {self._embeddings_path}",
    )

    logger.info(
        "reference_embeddings_validated",
        shape=self._reference_matrix.shape,
        dtype=str(self._reference_matrix.dtype),
    )
```

#### 3. Similarity Computation

```python
# embedding.py - compute similarity

def compute_similarities(
    query: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities with validation."""
    from ai_psychiatrist.infrastructure.validation import validate_embedding

    validate_embedding(query, context="query vector in similarity computation")

    similarities = reference_matrix @ query

    # Validate output (should never have NaN if inputs are clean)
    if np.isnan(similarities).any():
        raise EmbeddingValidationError(
            "NaN in similarity output despite valid inputs - numerical instability"
        )

    return similarities
```

---

## Testing

```python
# tests/unit/infrastructure/test_validation.py

import numpy as np
import pytest
from ai_psychiatrist.infrastructure.validation import (
    validate_embedding,
    validate_embedding_matrix,
)
from ai_psychiatrist.domain.exceptions import EmbeddingValidationError


class TestValidateEmbedding:
    def test_valid_embedding_passes(self):
        emb = np.array([0.1, 0.2, 0.3, 0.4])
        result = validate_embedding(emb, "test")
        assert np.array_equal(result, emb)

    def test_nan_raises(self):
        emb = np.array([0.1, np.nan, 0.3])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding(emb, "test embedding")
        assert "NaN" in str(exc.value)
        assert "test embedding" in str(exc.value)

    def test_inf_raises(self):
        emb = np.array([0.1, np.inf, 0.3])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding(emb, "test")
        assert "Inf" in str(exc.value)

    def test_negative_inf_raises(self):
        emb = np.array([0.1, -np.inf, 0.3])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding(emb, "test")
        assert "Inf" in str(exc.value)

    def test_zero_vector_raises(self):
        emb = np.array([0.0, 0.0, 0.0])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding(emb, "test")
        assert "zero" in str(exc.value).lower()

    def test_near_zero_passes(self):
        emb = np.array([1e-10, 1e-10, 1e-10])
        result = validate_embedding(emb, "test")  # Should not raise
        assert result is not None

    def test_checks_can_be_disabled(self):
        emb = np.array([np.nan, np.inf, 0.0])
        result = validate_embedding(
            emb, "test",
            check_nan=False,
            check_inf=False,
            check_zero=False,
        )
        assert np.isnan(result[0])


class TestValidateEmbeddingMatrix:
    def test_valid_matrix_passes(self):
        matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = validate_embedding_matrix(matrix, "test matrix")
        assert np.array_equal(result, matrix)

    def test_nan_row_identified(self):
        matrix = np.array([
            [0.1, 0.2],
            [np.nan, 0.4],  # Row 1 has NaN
            [0.5, 0.6],
        ])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding_matrix(matrix, "test")
        assert "1" in str(exc.value)  # Row index mentioned

    def test_multiple_nan_rows_reported(self):
        matrix = np.array([
            [np.nan, 0.2],  # Row 0
            [0.3, 0.4],
            [0.5, np.nan],  # Row 2
        ])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding_matrix(matrix, "test")
        assert "2 rows" in str(exc.value)

    def test_zero_row_detected(self):
        matrix = np.array([
            [0.1, 0.2],
            [0.0, 0.0],  # Zero row
            [0.5, 0.6],
        ])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding_matrix(matrix, "test")
        assert "zero" in str(exc.value).lower()

    def test_1d_array_rejected(self):
        vector = np.array([0.1, 0.2, 0.3])
        with pytest.raises(EmbeddingValidationError) as exc:
            validate_embedding_matrix(vector, "test")
        assert "2D" in str(exc.value)
```

---

## Performance Considerations

| Operation | Overhead | Notes |
|-----------|----------|-------|
| `np.isnan(embedding).any()` | ~1μs for 4096-dim | Negligible |
| `np.isnan(matrix).any()` | ~1ms for 10K×4096 | Done once at load time |
| Per-query validation | ~2μs | Negligible per participant |

**Conclusion**: Overhead is negligible. Always validate.

---

## Failure Modes After Implementation

| Scenario | Before | After |
|----------|--------|-------|
| NaN in query embedding | Silent corruption | `EmbeddingValidationError` raised |
| NaN in reference matrix | Silent corruption | Fails at load time |
| Zero vector after normalization | Undefined similarity | `EmbeddingValidationError` raised |
| Inf from numerical overflow | Silent corruption | `EmbeddingValidationError` raised |

---

## Rollout Plan

1. **Phase 1**: Implement validation functions and exception
2. **Phase 2**: Add validation at query embedding generation
3. **Phase 3**: Add validation at reference matrix loading
4. **Phase 4**: Add validation at similarity computation output

All phases can be deployed together - this is a pure strictness improvement.

---

## Success Criteria

1. All NaN/Inf/zero embeddings detected at point of origin
2. Clear error messages with context (which participant, which text)
3. Test coverage for all edge cases
4. <1ms additional latency per participant

---

## Future Enhancements

1. **Automatic retry**: If embedding fails validation, retry with cleaned input
2. **Metric tracking**: Log validation failure rates for monitoring
3. **Partial matrix handling**: Skip invalid rows instead of failing entire load (opt-in)
