"""Unit tests for consistency-based confidence metrics (Spec 050)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_psychiatrist.confidence.consistency import compute_consistency_metrics

pytestmark = pytest.mark.unit


def test_compute_consistency_metrics_majority_mode() -> None:
    metrics = compute_consistency_metrics((2, 2, 1, 2, 1))

    assert metrics.modal_score == 2
    assert metrics.modal_count == 3
    assert metrics.modal_confidence == pytest.approx(0.6)
    assert metrics.na_rate == pytest.approx(0.0)

    expected_std = float(np.std(np.array([2, 2, 1, 2, 1], dtype=float)))
    assert metrics.score_std == pytest.approx(expected_std)


def test_compute_consistency_metrics_all_na() -> None:
    metrics = compute_consistency_metrics((None, None, None))
    assert metrics.modal_score is None
    assert metrics.modal_count == 3
    assert metrics.modal_confidence == pytest.approx(1.0)
    assert metrics.na_rate == pytest.approx(1.0)
    assert metrics.score_std == pytest.approx(0.0)


def test_compute_consistency_metrics_tie_prefers_none() -> None:
    # Tie on count 2 vs 2; prefer None (conservative).
    metrics = compute_consistency_metrics((None, 1, 1, None))
    assert metrics.modal_score is None
    assert metrics.modal_count == 2
    assert metrics.modal_confidence == pytest.approx(0.5)


def test_compute_consistency_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="samples cannot be empty"):
        compute_consistency_metrics(())
