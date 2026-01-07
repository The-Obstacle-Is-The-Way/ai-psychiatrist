"""Unit tests for binary depression classification metrics (Spec 062)."""

from __future__ import annotations

import pytest

from ai_psychiatrist.metrics.binary_classification import (
    BinaryLabel,
    compute_binary_classification_metrics,
)

pytestmark = pytest.mark.unit


def test_compute_binary_classification_metrics_handles_abstentions() -> None:
    predicted: list[BinaryLabel | None] = ["depressed", None, "not_depressed", "depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed", "not_depressed", "not_depressed"]
    scores = [12, 9, 3, 15]

    metrics = compute_binary_classification_metrics(
        predicted=predicted,
        actual=actual,
        scores=scores,
    )

    assert metrics.n_total == 4
    assert metrics.n_predicted == 3
    assert metrics.coverage == pytest.approx(0.75)
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))
    assert metrics.confusion_matrix is not None
    assert metrics.confusion_matrix["true_positive"] == 1
    assert metrics.confusion_matrix["true_negative"] == 1
    assert metrics.confusion_matrix["false_positive"] == 1
    assert metrics.confusion_matrix["false_negative"] == 0
    assert metrics.auroc is not None


def test_compute_binary_classification_metrics_returns_none_when_no_predictions() -> None:
    predicted: list[BinaryLabel | None] = [None, None]
    actual: list[BinaryLabel] = ["depressed", "not_depressed"]
    scores: list[int | None] = [None, None]

    metrics = compute_binary_classification_metrics(
        predicted=predicted, actual=actual, scores=scores
    )

    assert metrics.coverage == 0.0
    assert metrics.accuracy is None
    assert metrics.confusion_matrix is None
    assert metrics.auroc is None
