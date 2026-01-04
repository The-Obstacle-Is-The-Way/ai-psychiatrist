"""Validation helpers for detecting silent numeric corruption.

These checks are intentionally strict because NaN/Inf/zero embeddings silently
corrupt similarity computations and downstream retrieval (Spec 055).
"""

from __future__ import annotations

import numpy as np

from ai_psychiatrist.domain.exceptions import EmbeddingValidationError


def validate_embedding(
    embedding: np.ndarray,
    *,
    context: str,
    check_nan: bool = True,
    check_inf: bool = True,
    check_zero: bool = True,
) -> np.ndarray:
    """Validate a single embedding vector.

    Args:
        embedding: Vector to validate (expected 1D).
        context: Privacy-safe description for error messages.
        check_nan: Raise if NaN detected.
        check_inf: Raise if Inf detected.
        check_zero: Raise if all-zero vector detected.

    Returns:
        The input embedding (unchanged).

    Raises:
        EmbeddingValidationError: If validation fails.
    """
    if embedding.ndim != 1:
        raise EmbeddingValidationError(
            f"Expected 1D embedding in {context}, got shape {embedding.shape}"
        )

    if check_nan and np.isnan(embedding).any():
        nan_positions = np.where(np.isnan(embedding))[0][:5]
        raise EmbeddingValidationError(
            f"NaN detected in {context}: {int(np.isnan(embedding).sum())} NaN value(s) "
            f"(first positions: {nan_positions.tolist()})"
        )

    if check_inf and np.isinf(embedding).any():
        raise EmbeddingValidationError(
            f"Inf detected in {context}: {int(np.isinf(embedding).sum())} Inf value(s)"
        )

    if check_zero and np.allclose(embedding, 0.0):
        raise EmbeddingValidationError(f"All-zero vector in {context}")

    return embedding


def validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Validate a 2D embedding matrix.

    Args:
        matrix: Matrix to validate (shape: n_rows x n_dims).
        context: Privacy-safe description for error messages.

    Returns:
        The input matrix (unchanged).

    Raises:
        EmbeddingValidationError: If validation fails.
    """
    if matrix.ndim != 2:
        raise EmbeddingValidationError(f"Expected 2D matrix in {context}, got shape {matrix.shape}")

    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        nan_rows = np.where(nan_mask.any(axis=1))[0][:5]
        raise EmbeddingValidationError(
            f"NaN detected in {context}: row(s) {nan_rows.tolist()} contain NaN"
        )

    inf_mask = np.isinf(matrix)
    if inf_mask.any():
        inf_rows = np.where(inf_mask.any(axis=1))[0][:5]
        raise EmbeddingValidationError(
            f"Inf detected in {context}: row(s) {inf_rows.tolist()} contain Inf"
        )

    row_norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.where(np.isclose(row_norms, 0.0))[0][:5]
    if zero_rows.size > 0:
        raise EmbeddingValidationError(f"All-zero row(s) in {context}: {zero_rows.tolist()}")

    return matrix
