"""Bootstrap inference utilities for participant-level analysis.

Implements participant-cluster bootstrap for CIs and paired comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ai_psychiatrist.metrics.selective_prediction import ItemPrediction

T = TypeVar("T", int, float)


@dataclass(frozen=True, slots=True)
class BootstrapResult(Generic[T]):
    """Result of a bootstrap analysis."""

    point_estimate: T | None
    ci95: tuple[float, float]
    std_error: float
    drop_rate: float
    n_resamples: int


@dataclass(frozen=True, slots=True)
class BootstrapDeltaResult:
    """Result of a paired bootstrap delta analysis."""

    point_estimate_left: float | None
    point_estimate_right: float | None
    delta_point_estimate: float | None
    ci95: tuple[float, float]
    std_error: float
    drop_rate: float
    n_resamples: int


def bootstrap_by_participant(
    items: Sequence[ItemPrediction],
    *,
    metric_fn: Callable[[Sequence[ItemPrediction]], T | None],
    n_resamples: int,
    seed: int,
) -> BootstrapResult[T]:
    """Compute participant-cluster bootstrap CI for a metric.

    Args:
        items: List of all item predictions.
        metric_fn: Function computing metric from items (returns None if invalid/unachievable).
        n_resamples: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        BootstrapResult with point estimate and CI.
    """
    if not items:
        # Return empty/safe defaults
        return BootstrapResult(None, (0.0, 0.0), 0.0, 0.0, n_resamples)

    # Group items by participant
    items_by_pid = defaultdict(list)
    for item in items:
        items_by_pid[item.participant_id].append(item)

    participant_ids = sorted(items_by_pid.keys())
    n_participants = len(participant_ids)

    # Point estimate on original data
    point_est = metric_fn(items)

    # Bootstrap
    rng = np.random.RandomState(seed)
    replicates: list[float] = []

    for _ in range(n_resamples):
        # Sample participant IDs with replacement
        # We assume n_participants > 0 because items is not empty
        sampled_indices = rng.randint(0, n_participants, size=n_participants)
        sampled_pids = [participant_ids[i] for i in sampled_indices]

        # Collect all items from sampled participants
        resample_items: list[ItemPrediction] = []
        for pid in sampled_pids:
            resample_items.extend(items_by_pid[pid])

        val = metric_fn(resample_items)
        if val is not None:
            replicates.append(float(val))

    # Compute stats
    valid_count = len(replicates)
    drop_rate = 1.0 - (valid_count / n_resamples) if n_resamples > 0 else 0.0

    if valid_count < 2:
        # Not enough data for CI
        # If point_est is valid, return it as tight CI? Or (NaN, NaN)?
        # Spec says: "single participant does not crash (degenerates to point estimate)"
        # So let's return (point, point) if we have one.
        fallback = float(point_est) if point_est is not None else 0.0
        return BootstrapResult(point_est, (fallback, fallback), 0.0, drop_rate, n_resamples)

    # Percentile CI (2.5, 97.5)
    ci_low = float(np.percentile(replicates, 2.5))
    ci_high = float(np.percentile(replicates, 97.5))
    std_err = float(np.std(replicates, ddof=1))

    return BootstrapResult(
        point_estimate=point_est,
        ci95=(ci_low, ci_high),
        std_error=std_err,
        drop_rate=drop_rate,
        n_resamples=n_resamples,
    )


def paired_bootstrap_delta_by_participant(
    items_left: Sequence[ItemPrediction],
    items_right: Sequence[ItemPrediction],
    *,
    metric_fn: Callable[[Sequence[ItemPrediction]], float | None],
    n_resamples: int,
    seed: int,
) -> BootstrapDeltaResult:
    """Compute paired bootstrap CI for Δ (right - left).

    Requires that items_left and items_right cover the same set of participants.
    (Or at least we will intersect them or assume alignment - Spec says intersect/validate
    before calling).
    We group by participant ID and require matching IDs in resamples.

    Args:
        items_left: Items for system A.
        items_right: Items for system B.
        metric_fn: Function to compute metric.
        n_resamples: Number of iterations.
        seed: Random seed.
    """
    # Group items by PID
    left_by_pid = defaultdict(list)
    for item in items_left:
        left_by_pid[item.participant_id].append(item)

    right_by_pid = defaultdict(list)
    for item in items_right:
        right_by_pid[item.participant_id].append(item)

    # Get common participants (Intersection)
    # Spec says evaluation script handles strict vs intersection.
    # Here we just operate on the intersection of keys present in inputs.
    pids_left = set(left_by_pid.keys())
    pids_right = set(right_by_pid.keys())
    common_pids = sorted(pids_left.intersection(pids_right))

    if not common_pids:
        return BootstrapDeltaResult(None, None, None, (0.0, 0.0), 0.0, 1.0, n_resamples)

    # Prepare "aligned" lists for point estimate on common set
    # Note: caller might have passed non-overlapping PIDs, but this function implies paired analysis
    # on the OVERLAP.
    items_left_common = [i for pid in common_pids for i in left_by_pid[pid]]
    items_right_common = [i for pid in common_pids for i in right_by_pid[pid]]

    est_left = metric_fn(items_left_common)
    est_right = metric_fn(items_right_common)

    delta_est: float | None = None
    if est_left is not None and est_right is not None:
        delta_est = est_right - est_left

    # Bootstrap
    rng = np.random.RandomState(seed)
    deltas: list[float] = []

    n_participants = len(common_pids)

    for _ in range(n_resamples):
        sampled_indices = rng.randint(0, n_participants, size=n_participants)
        sampled_pids = [common_pids[i] for i in sampled_indices]

        # Construct resampled sets
        res_left: list[ItemPrediction] = []
        res_right: list[ItemPrediction] = []

        for pid in sampled_pids:
            res_left.extend(left_by_pid[pid])
            res_right.extend(right_by_pid[pid])

        val_left = metric_fn(res_left)
        val_right = metric_fn(res_right)

        if val_left is not None and val_right is not None:
            deltas.append(val_right - val_left)

    valid_count = len(deltas)
    drop_rate = 1.0 - (valid_count / n_resamples) if n_resamples > 0 else 0.0

    if valid_count < 2:
        fallback = float(delta_est) if delta_est is not None else 0.0
        return BootstrapDeltaResult(
            est_left, est_right, delta_est, (fallback, fallback), 0.0, drop_rate, n_resamples
        )

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    std_err = float(np.std(deltas, ddof=1))

    return BootstrapDeltaResult(
        point_estimate_left=est_left,
        point_estimate_right=est_right,
        delta_point_estimate=delta_est,
        ci95=(ci_low, ci_high),
        std_error=std_err,
        drop_rate=drop_rate,
        n_resamples=n_resamples,
    )
