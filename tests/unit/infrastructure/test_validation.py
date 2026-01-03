"""Unit tests for embedding validation helpers (Spec 055)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_psychiatrist.domain.exceptions import EmbeddingValidationError
from ai_psychiatrist.infrastructure.validation import (
    validate_embedding,
    validate_embedding_matrix,
)


class TestValidateEmbedding:
    def test_valid_embedding_passes(self) -> None:
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = validate_embedding(emb, context="test")
        assert np.array_equal(result, emb)

    def test_nan_raises(self) -> None:
        emb = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="NaN"):
            validate_embedding(emb, context="test embedding")

    def test_inf_raises(self) -> None:
        emb = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="Inf"):
            validate_embedding(emb, context="test embedding")

    def test_all_zero_raises(self) -> None:
        emb = np.zeros(4, dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="All-zero"):
            validate_embedding(emb, context="test embedding")


class TestValidateEmbeddingMatrix:
    def test_valid_matrix_passes(self) -> None:
        mat = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = validate_embedding_matrix(mat, context="matrix")
        assert np.array_equal(result, mat)

    def test_nan_rows_raise(self) -> None:
        mat = np.array([[0.1, np.nan], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="NaN"):
            validate_embedding_matrix(mat, context="matrix")

    def test_inf_rows_raise(self) -> None:
        mat = np.array([[0.1, np.inf], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="Inf"):
            validate_embedding_matrix(mat, context="matrix")

    def test_zero_rows_raise(self) -> None:
        mat = np.array([[0.0, 0.0], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(EmbeddingValidationError, match="All-zero"):
            validate_embedding_matrix(mat, context="matrix")
