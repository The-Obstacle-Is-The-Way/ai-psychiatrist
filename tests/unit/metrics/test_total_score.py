"""Unit tests for total PHQ-8 score metrics (Spec 061)."""

from __future__ import annotations

import math

import pytest

from ai_psychiatrist.metrics.total_score import compute_total_score_metrics

pytestmark = pytest.mark.unit


def test_compute_total_score_metrics_handles_abstentions() -> None:
    predicted = [10, None, 8]
    actual = [12, 10, 9]

    metrics = compute_total_score_metrics(predicted=predicted, actual=actual)

    assert metrics.n_total == 3
    assert metrics.n_predicted == 2
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.mae == pytest.approx(1.5)
    assert metrics.rmse == pytest.approx(math.sqrt(2.5))
    assert metrics.severity_tier_accuracy == pytest.approx(1.0)
    assert metrics.pearson_r == pytest.approx(1.0)


def test_compute_total_score_metrics_returns_none_when_insufficient_data() -> None:
    metrics = compute_total_score_metrics(predicted=[None], actual=[10])
    assert metrics.n_predicted == 0
    assert metrics.coverage == 0.0
    assert metrics.mae is None
    assert metrics.rmse is None
    assert metrics.pearson_r is None
    assert metrics.severity_tier_accuracy is None


def test_compute_total_score_metrics_returns_none_for_constant_predictions() -> None:
    metrics = compute_total_score_metrics(predicted=[10, 10, 10], actual=[0, 12, 24])
    assert metrics.n_predicted == 3
    assert metrics.coverage == 1.0
    assert metrics.pearson_r is None


def test_compute_total_score_metrics_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="predicted and actual must have the same length"):
        compute_total_score_metrics(predicted=[10, None], actual=[10])
