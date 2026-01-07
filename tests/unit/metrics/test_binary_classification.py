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


def test_compute_binary_classification_metrics_raises_on_length_mismatch() -> None:
    predicted: list[BinaryLabel | None] = ["depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed"]

    with pytest.raises(ValueError, match="predicted and actual must have the same length"):
        compute_binary_classification_metrics(predicted=predicted, actual=actual)


def test_compute_binary_classification_metrics_raises_on_scores_length_mismatch() -> None:
    predicted: list[BinaryLabel | None] = ["depressed", "not_depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed"]
    scores = [10]  # Wrong length

    with pytest.raises(ValueError, match="scores and actual must have the same length"):
        compute_binary_classification_metrics(predicted=predicted, actual=actual, scores=scores)


def test_compute_binary_classification_metrics_auroc_none_when_single_class() -> None:
    """AUROC is undefined when all samples are the same class."""
    predicted: list[BinaryLabel | None] = ["depressed", "depressed"]
    actual: list[BinaryLabel] = ["depressed", "depressed"]
    scores = [15, 18]

    metrics = compute_binary_classification_metrics(
        predicted=predicted, actual=actual, scores=scores
    )

    # AUROC requires both positive and negative samples
    assert metrics.auroc is None


def test_compute_binary_classification_metrics_precision_none_when_no_positive_predictions() -> (
    None
):
    """Precision is undefined when TP + FP = 0."""
    predicted: list[BinaryLabel | None] = ["not_depressed", "not_depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed"]

    metrics = compute_binary_classification_metrics(predicted=predicted, actual=actual)

    assert metrics.precision is None
    assert metrics.recall is not None  # TP + FN > 0


def test_compute_binary_classification_metrics_recall_none_when_no_actual_positives() -> None:
    """Recall is undefined when TP + FN = 0."""
    predicted: list[BinaryLabel | None] = ["depressed", "not_depressed"]
    actual: list[BinaryLabel] = ["not_depressed", "not_depressed"]

    metrics = compute_binary_classification_metrics(predicted=predicted, actual=actual)

    assert metrics.recall is None
    assert metrics.precision is not None  # TP + FP > 0


def test_compute_binary_classification_metrics_f1_none_when_precision_or_recall_none() -> None:
    """F1 is undefined when precision or recall is undefined."""
    # All predictions negative, no actual positives → recall undefined
    predicted: list[BinaryLabel | None] = ["not_depressed", "not_depressed"]
    actual: list[BinaryLabel] = ["not_depressed", "not_depressed"]

    metrics = compute_binary_classification_metrics(predicted=predicted, actual=actual)

    assert metrics.f1 is None


def test_compute_binary_classification_metrics_skips_none_scores() -> None:
    """Scores that are None should be skipped in AUROC computation."""
    predicted: list[BinaryLabel | None] = ["depressed", "not_depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed"]
    scores: list[int | None] = [None, None]  # Both scores None

    metrics = compute_binary_classification_metrics(
        predicted=predicted, actual=actual, scores=scores
    )

    # AUROC should be None when all scores are None
    assert metrics.auroc is None
    # But accuracy etc. should still be computed
    assert metrics.accuracy == 1.0


def test_compute_binary_classification_metrics_partial_none_scores() -> None:
    """When some scores are None, AUROC uses only valid scores."""
    predicted: list[BinaryLabel | None] = ["depressed", "not_depressed", "depressed"]
    actual: list[BinaryLabel] = ["depressed", "not_depressed", "not_depressed"]
    scores: list[int | None] = [15, None, 12]  # Middle score is None

    metrics = compute_binary_classification_metrics(
        predicted=predicted, actual=actual, scores=scores
    )

    # AUROC computed from valid scores only
    assert metrics.auroc is not None
    assert metrics.accuracy == pytest.approx(2 / 3)
