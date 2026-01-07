"""Total PHQ-8 score metrics (Spec 061)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ai_psychiatrist.domain.enums import SeverityLevel

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class TotalScoreMetrics:
    """Coverage-aware regression metrics for total PHQ-8 score prediction."""

    n_total: int
    n_predicted: int
    coverage: float

    mae: float | None
    rmse: float | None
    pearson_r: float | None
    severity_tier_accuracy: float | None


def compute_total_score_metrics(
    *,
    predicted: Sequence[int | None],
    actual: Sequence[int],
) -> TotalScoreMetrics:
    """Compute coverage-aware metrics for total PHQ-8 score prediction.

    Metrics are computed on the subset where `predicted` is not None.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")

    n_total = len(actual)
    pairs = [(p, a) for p, a in zip(predicted, actual, strict=True) if p is not None]
    n_predicted = len(pairs)

    coverage = (n_predicted / n_total) if n_total else 0.0

    if n_predicted == 0:
        return TotalScoreMetrics(
            n_total=n_total,
            n_predicted=0,
            coverage=0.0,
            mae=None,
            rmse=None,
            pearson_r=None,
            severity_tier_accuracy=None,
        )

    pred_arr = np.asarray([p for p, _ in pairs], dtype=float)
    actual_arr = np.asarray([a for _, a in pairs], dtype=float)
    diffs = pred_arr - actual_arr

    mae = float(np.mean(np.abs(diffs)))
    rmse = math.sqrt(float(np.mean(diffs**2)))

    pearson_r: float | None = None
    if n_predicted >= 2 and float(np.std(pred_arr)) > 0.0 and float(np.std(actual_arr)) > 0.0:
        r = float(np.corrcoef(pred_arr, actual_arr)[0, 1])
        pearson_r = None if math.isnan(r) else r

    tier_correct = 0
    for pred, act in pairs:
        if SeverityLevel.from_total_score(int(pred)) == SeverityLevel.from_total_score(int(act)):
            tier_correct += 1
    severity_tier_accuracy = tier_correct / n_predicted

    return TotalScoreMetrics(
        n_total=n_total,
        n_predicted=n_predicted,
        coverage=coverage,
        mae=mae,
        rmse=rmse,
        pearson_r=pearson_r,
        severity_tier_accuracy=severity_tier_accuracy,
    )
