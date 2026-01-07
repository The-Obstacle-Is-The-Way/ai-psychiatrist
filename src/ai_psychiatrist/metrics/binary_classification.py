"""Binary depression classification metrics (Spec 062)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

BinaryLabel = Literal["depressed", "not_depressed"]

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """Coverage-aware metrics for binary depression classification."""

    n_total: int
    n_predicted: int
    coverage: float

    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    auroc: float | None

    confusion_matrix: dict[str, int] | None


_LABEL_TO_INT: dict[BinaryLabel, int] = {"depressed": 1, "not_depressed": 0}


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return None

    diffs = pos[:, None] - neg[None, :]
    auc = (np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / (pos.size * neg.size)
    return None if math.isnan(float(auc)) else float(auc)


def compute_binary_classification_metrics(
    *,
    predicted: Sequence[BinaryLabel | None],
    actual: Sequence[BinaryLabel],
    scores: Sequence[int | None] | None = None,
) -> BinaryClassificationMetrics:
    """Compute coverage-aware binary classification metrics.

    All thresholded metrics are computed on the subset where `predicted` is not None.
    AUROC is computed on the same subset (and requires `scores`).
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if scores is not None and len(scores) != len(actual):
        raise ValueError("scores and actual must have the same length")

    n_total = len(actual)
    included = [i for i, p in enumerate(predicted) if p is not None]
    n_predicted = len(included)
    coverage = (n_predicted / n_total) if n_total else 0.0

    if n_predicted == 0:
        return BinaryClassificationMetrics(
            n_total=n_total,
            n_predicted=0,
            coverage=0.0,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            auroc=None,
            confusion_matrix=None,
        )

    pred_int = np.asarray(
        [_LABEL_TO_INT[cast("BinaryLabel", predicted[i])] for i in included],
        dtype=int,
    )
    act_int = np.asarray([_LABEL_TO_INT[actual[i]] for i in included], dtype=int)

    tp = int(np.sum((pred_int == 1) & (act_int == 1)))
    tn = int(np.sum((pred_int == 0) & (act_int == 0)))
    fp = int(np.sum((pred_int == 1) & (act_int == 0)))
    fn = int(np.sum((pred_int == 0) & (act_int == 1)))

    accuracy = (tp + tn) / n_predicted

    precision: float | None = None
    if tp + fp > 0:
        precision = tp / (tp + fp)

    recall: float | None = None
    if tp + fn > 0:
        recall = tp / (tp + fn)

    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    auroc: float | None = None
    if scores is not None:
        score_values: list[float] = []
        auc_labels: list[int] = []
        for i in included:
            s = scores[i]
            if s is None:
                continue
            score_values.append(float(s))
            auc_labels.append(_LABEL_TO_INT[actual[i]])
        if score_values:
            auroc = _compute_auroc(
                scores=np.asarray(score_values, dtype=float),
                labels=np.asarray(auc_labels, dtype=int),
            )

    return BinaryClassificationMetrics(
        n_total=n_total,
        n_predicted=n_predicted,
        coverage=coverage,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        confusion_matrix={
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    )
