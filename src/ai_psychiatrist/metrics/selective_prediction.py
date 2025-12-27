"""Selective prediction metrics for PHQ-8 assessment.

Implements Spec 25: Risk-Coverage curves, AURC, AUGRC, and related metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ItemPrediction:
    """Prediction for a single item instance."""

    participant_id: int
    item_index: int
    pred: int | None
    gt: int
    confidence: float


@dataclass(frozen=True, slots=True)
class RiskCoverageCurve:
    """Risk-coverage curve data."""

    coverage: list[float]
    selective_risk: list[float]
    generalized_risk: list[float]
    threshold: list[float]
    cmax: float


def compute_cmax(items: Sequence[ItemPrediction]) -> float:
    """Compute maximum achievable coverage (Cmax)."""
    if not items:
        return 0.0
    k = sum(1 for x in items if x.pred is not None)
    return k / len(items)


def _compute_loss(item: ItemPrediction, loss_type: Literal["abs", "abs_norm"]) -> float:
    """Compute loss for a single item."""
    if item.pred is None:
        raise ValueError("Cannot compute loss for abstained item")
    abs_err = abs(item.pred - item.gt)
    if loss_type == "abs_norm":
        return abs_err / 3.0
    return float(abs_err)


def compute_risk_coverage_curve(
    items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]
) -> RiskCoverageCurve:
    """Compute the risk-coverage curve.

    Args:
        items: Sequence of item predictions.
        loss: Loss function to use ("abs" or "abs_norm").

    Returns:
        RiskCoverageCurve object.
    """
    n = len(items)
    if n == 0:
        return RiskCoverageCurve([], [], [], [], 0.0)

    # Filter to predicted items S
    # We store tuples of (loss, confidence)
    predictions = []
    for item in items:
        if item.pred is not None:
            l_val = _compute_loss(item, loss)
            predictions.append((l_val, item.confidence))

    k = len(predictions)
    cmax = k / n

    if k == 0:
        return RiskCoverageCurve([], [], [], [], 0.0)

    # Sort by confidence descending
    # Use stable sort to match potential tie-breaking if ever needed,
    # though plateau logic makes local order irrelevant.
    # We just need grouped by confidence.
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Identify unique confidence thresholds (working points)
    # We want to process "batches" of items with the same confidence together.
    thresholds: list[float] = []
    coverages: list[float] = []
    selective_risks: list[float] = []
    generalized_risks: list[float] = []

    current_sum_loss = 0.0
    current_k = 0

    i = 0
    while i < k:
        # Current batch confidence
        conf = predictions[i][1]

        # Process all items with this confidence (plateau)
        while i < k and predictions[i][1] == conf:
            current_sum_loss += predictions[i][0]
            current_k += 1
            i += 1

        # Record working point
        cov = current_k / n
        sel_risk = current_sum_loss / current_k
        gen_risk = current_sum_loss / n

        thresholds.append(conf)
        coverages.append(cov)
        selective_risks.append(sel_risk)
        generalized_risks.append(gen_risk)

    return RiskCoverageCurve(
        coverage=coverages,
        selective_risk=selective_risks,
        generalized_risk=generalized_risks,
        threshold=thresholds,
        cmax=cmax,
    )


def compute_aurc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC)."""
    curve = compute_risk_coverage_curve(items, loss=loss)
    return _integrate_curve(curve.coverage, curve.selective_risk, curve.cmax, mode="aurc")


def compute_augrc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Compute Area Under the Generalized Risk-Coverage Curve (AUGRC)."""
    curve = compute_risk_coverage_curve(items, loss=loss)
    return _integrate_curve(curve.coverage, curve.generalized_risk, curve.cmax, mode="augrc")


def _integrate_curve(
    coverages: list[float], risks: list[float], _cmax: float, mode: Literal["aurc", "augrc"]
) -> float:
    """Integrate curve using trapezoidal rule on [0, Cmax]."""
    if not coverages:
        return 0.0

    # Augment points for [0, coverage_1] segment
    # For AURC: right-continuous at 0 => risk(0) = risk(coverage_1)
    # For AUGRC: risk(0) = 0

    xs = [0.0, *coverages]
    ys = [risks[0], *risks] if mode == "aurc" else [0.0, *risks]

    return float(np.trapezoid(ys, xs))


def compute_risk_at_coverage(
    items: Sequence[ItemPrediction], *, target_coverage: float, loss: Literal["abs", "abs_norm"]
) -> float | None:
    """Compute risk at a specific target coverage (MAE@coverage)."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    curve = compute_risk_coverage_curve(items, loss=loss)

    # Find smallest working point j such that coverage_j >= target_coverage
    for cov, risk in zip(curve.coverage, curve.selective_risk, strict=True):
        if cov >= target_coverage:
            return risk

    # If no such working point exists (target_c > Cmax)
    return None


def _interpolate_at_c(coverages: list[float], risks: list[float], target_c: float) -> float:
    """Linearly interpolate risk at target_c given sorted coverages."""
    # Since coverages are sorted, we can use np.interp, but we need to handle the step behavior vs linear.  # noqa: E501
    # Spec 6.6 says "linearly interpolate the corresponding risk value".
    # We know coverages start from coverage_1 > 0.
    # We need to consider the 0 point based on the metric definition.
    # BUT, this helper is usually called with the augmented arrays from _integrate_truncated.
    # Let's let the caller handle augmentation.
    return float(np.interp(target_c, coverages, risks))


def compute_aurc_at_coverage(
    items: Sequence[ItemPrediction], *, max_coverage: float, loss: Literal["abs", "abs_norm"]
) -> float:
    """Compute truncated AURC up to a maximum coverage."""
    if not 0.0 < max_coverage <= 1.0:
        raise ValueError(f"max_coverage must be in (0, 1], got {max_coverage}")
    curve = compute_risk_coverage_curve(items, loss=loss)
    return _integrate_truncated(curve.coverage, curve.selective_risk, max_coverage, mode="aurc")


def compute_augrc_at_coverage(
    items: Sequence[ItemPrediction], *, max_coverage: float, loss: Literal["abs", "abs_norm"]
) -> float:
    """Compute truncated AUGRC up to a maximum coverage."""
    if not 0.0 < max_coverage <= 1.0:
        raise ValueError(f"max_coverage must be in (0, 1], got {max_coverage}")
    curve = compute_risk_coverage_curve(items, loss=loss)
    return _integrate_truncated(curve.coverage, curve.generalized_risk, max_coverage, mode="augrc")


def _integrate_truncated(
    coverages: list[float],
    risks: list[float],
    max_c: float,
    mode: Literal["aurc", "augrc"],
) -> float:
    """Integrate curve up to min(max_c, Cmax)."""
    if not coverages:
        return 0.0

    cmax = coverages[-1]
    c_prime = min(max_c, cmax)

    # Augment points exactly like full integration
    xs = [0.0, *coverages]
    ys = [risks[0], *risks] if mode == "aurc" else [0.0, *risks]

    # If C' is exactly 0 (shouldn't happen given cmax check but safety first)
    if c_prime <= 0:
        return 0.0

    # Find where C' falls in xs
    # xs is sorted increasing [0, c1, c2, ... cmax]

    # We want to integrate up to C'.
    # Find indices where x <= C'
    mask = np.array(xs) <= c_prime
    xs_trunc = list(np.array(xs)[mask])
    ys_trunc = list(np.array(ys)[mask])

    # If C' is not exactly the last point, append interpolated point
    if xs_trunc[-1] < c_prime:
        # Interpolate between last kept point and next point
        last_idx = len(xs_trunc) - 1
        # next point in original arrays
        next_x = xs[last_idx + 1]
        next_y = ys[last_idx + 1]

        prev_x = xs[last_idx]
        prev_y = ys[last_idx]

        # Linear interpolation
        interpolated_risk = prev_y + (c_prime - prev_x) * (next_y - prev_y) / (next_x - prev_x)

        xs_trunc.append(c_prime)
        ys_trunc.append(interpolated_risk)

    return float(np.trapezoid(ys_trunc, xs_trunc))
