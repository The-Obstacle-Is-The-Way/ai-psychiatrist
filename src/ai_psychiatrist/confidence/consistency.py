"""Consistency-based confidence metrics (Spec 050).

This module provides deterministic, per-item agreement metrics across multiple
stochastic scoring samples.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ConsistencyMetrics:
    """Agreement statistics across N samples for a single item."""

    modal_score: int | None
    modal_count: int
    modal_confidence: float
    score_std: float
    na_rate: float


def compute_consistency_metrics(samples: Sequence[int | None]) -> ConsistencyMetrics:
    """Compute agreement metrics across multiple samples.

    Args:
        samples: A sequence of item scores (0-3) or None (N/A) across samples.

    Returns:
        ConsistencyMetrics with modal score and summary statistics.
    """
    if not samples:
        raise ValueError("samples cannot be empty")

    sample_tuple = tuple(samples)
    counts = Counter(sample_tuple)
    max_count = max(counts.values())
    modal_candidates = [score for score, count in counts.items() if count == max_count]

    # Deterministic tie-break: prefer abstention (None) when tied; else smallest score.
    if None in modal_candidates:
        modal_score: int | None = None
    else:
        modal_score = min(s for s in modal_candidates if s is not None)

    modal_count = int(counts[modal_score])
    modal_confidence = modal_count / len(sample_tuple)
    na_rate = float(counts.get(None, 0) / len(sample_tuple))

    numeric = [s for s in sample_tuple if s is not None]
    score_std = 0.0
    if len(numeric) >= 2:
        score_std = float(np.std(np.array(numeric, dtype=float)))

    return ConsistencyMetrics(
        modal_score=modal_score,
        modal_count=modal_count,
        modal_confidence=float(modal_confidence),
        score_std=score_std,
        na_rate=na_rate,
    )
