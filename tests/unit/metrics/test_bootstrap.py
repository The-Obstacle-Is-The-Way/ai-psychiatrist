"""Unit tests for bootstrap inference utilities (Spec 25)."""

from collections.abc import Sequence

import numpy as np
import pytest

from ai_psychiatrist.metrics.bootstrap import (
    bootstrap_by_participant,
    paired_bootstrap_delta_by_participant,
)
from ai_psychiatrist.metrics.selective_prediction import ItemPrediction, compute_cmax

pytestmark = [pytest.mark.unit]


@pytest.fixture
def items_participant_1() -> list[ItemPrediction]:
    """Participant 1: 2 items, cov=1.0, err=0.0."""
    return [
        ItemPrediction(1, 0, 1, 1, 1.0),
        ItemPrediction(1, 1, 1, 1, 1.0),
    ]


@pytest.fixture
def items_participant_2() -> list[ItemPrediction]:
    """Participant 2: 2 items, cov=0.5, err=1.0 (on predicted)."""
    return [
        ItemPrediction(2, 0, 2, 1, 1.0),  # Error
        ItemPrediction(2, 1, None, 1, 0.0),  # Abstain
    ]


class TestBootstrap:
    def test_bootstrap_single_participant_deterministic(
        self, items_participant_1: list[ItemPrediction]
    ) -> None:
        """With P=1, all bootstrap resamples should be identical."""
        result = bootstrap_by_participant(
            items_participant_1, metric_fn=compute_cmax, n_resamples=100, seed=42
        )
        assert result.point_estimate == 1.0
        # CI should be point estimate (collapsed)
        assert result.ci95 == (1.0, 1.0)
        # Standard error should be 0
        assert result.std_error == 0.0

    def test_bootstrap_cmax_distribution(
        self,
        items_participant_1: list[ItemPrediction],
        items_participant_2: list[ItemPrediction],
    ) -> None:
        """P=2. P1 has Cmax=1.0, P2 has Cmax=0.5.

        Resamples will be (P1, P1) -> Cmax=1.0
                         (P1, P2) -> Cmax=0.75
                         (P2, P2) -> Cmax=0.5

        Point estimate on original: 0.75
        """
        items = items_participant_1 + items_participant_2
        result = bootstrap_by_participant(items, metric_fn=compute_cmax, n_resamples=1000, seed=42)
        assert result.point_estimate == 0.75
        # CI should contain 0.5 and 1.0
        assert 0.5 <= result.ci95[0] < 0.75
        assert 0.75 < result.ci95[1] <= 1.0

    def test_bootstrap_handles_none_returns(
        self,
        items_participant_1: list[ItemPrediction],
        items_participant_2: list[ItemPrediction],
    ) -> None:
        """If metric returns None (e.g. MAE@Coverage not reached), drop it."""

        def flaky_metric(items: Sequence[ItemPrediction]) -> float | None:
            # Return None if we picked Participant 2 (which has abstention) twice?
            # Or just simulate randomness.
            # Here: if total items < 4 (meaning we picked P2 only?), return None.
            # Wait, bootstrap resamples P items.
            # If we sample (P2, P2), len=4 (2 items each).
            # Let's say: if cmax < 0.8 return None.
            c = compute_cmax(items)
            return c if c >= 0.8 else None

        # P1 (cmax=1.0), P2 (cmax=0.5).
        # (P1, P1) -> 1.0 -> Keep
        # (P1, P2) -> 0.75 -> Drop
        # (P2, P2) -> 0.5 -> Drop

        items = items_participant_1 + items_participant_2
        result = bootstrap_by_participant(items, metric_fn=flaky_metric, n_resamples=1000, seed=42)

        # Point estimate on original (0.75) returns None -> flaky_metric(P1+P2) = None
        # So point_estimate should be None.
        assert result.point_estimate is None

        # But we should have a valid CI from the (P1, P1) cases.
        # (P1, P1) happens ~25% of time.
        assert result.ci95 == (1.0, 1.0)
        assert result.drop_rate > 0.5  # Expect ~75% drop rate

    def test_paired_bootstrap_delta(self) -> None:
        """Test paired bootstrap delta.

        System A: P1 has loss=0, P2 has loss=0 -> Mean=0.0
        System B: P1 has loss=1, P2 has loss=0 -> Mean=0.5
        Delta on original: 0.5 - 0.0 = 0.5

        Resamples:
        - (P1, P1): A=0, B=1 -> D=1
        - (P2, P2): A=0, B=0 -> D=0
        - (P1, P2): A=0, B=0.5 -> D=0.5
        """
        items_a = [
            ItemPrediction(1, 0, 0, 0, 1.0),  # loss 0
            ItemPrediction(2, 0, 0, 0, 1.0),  # loss 0
        ]

        items_b = [
            ItemPrediction(1, 0, 1, 0, 1.0),  # loss 1
            ItemPrediction(2, 0, 0, 0, 1.0),  # loss 0
        ]

        def mean_loss(items: Sequence[ItemPrediction]) -> float:
            errs = [abs(x.pred - x.gt) for x in items if x.pred is not None]
            return float(np.mean(errs)) if errs else 0.0

        result = paired_bootstrap_delta_by_participant(
            items_a, items_b, metric_fn=mean_loss, n_resamples=1000, seed=42
        )

        # Original: A=0, B=0.5 -> Delta = 0.5
        assert result.point_estimate_left == 0.0
        assert result.point_estimate_right == 0.5
        assert result.delta_point_estimate == 0.5

        # CI should range 0 to 1 (covers possible delta outcomes)
        assert result.ci95[0] >= 0.0
        assert result.ci95[1] <= 1.0
