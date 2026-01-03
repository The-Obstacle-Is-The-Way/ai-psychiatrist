"""Unit tests for selective prediction metrics (Spec 25).

Implements the "Canonical numeric test vector" requirement from Spec 25 Section 9.1.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest

from ai_psychiatrist.metrics.selective_prediction import (
    ItemPrediction,
    compute_augrc,
    compute_augrc_at_coverage,
    compute_augrc_optimal,
    compute_aurc,
    compute_aurc_achievable,
    compute_aurc_at_coverage,
    compute_aurc_optimal,
    compute_eaurc,
    compute_risk_at_coverage,
    compute_risk_coverage_curve,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def canonical_vectors_abs_loss() -> Sequence[ItemPrediction]:
    """Canonical numeric test vector from Spec 25 Section 9.1.

    Given N=4 item instances with (pred, gt, confidence):
    1. (2, 2, 2)    -> abs_err=0
    2. (3, 1, 2)    -> abs_err=2
    3. (1, 1, 1)    -> abs_err=0
    4. (None, 0, 0) -> abstain (excluded from S, but included in N)
    """
    return [
        ItemPrediction(participant_id=1, item_index=0, pred=2, gt=2, confidence=2.0),
        ItemPrediction(participant_id=1, item_index=1, pred=3, gt=1, confidence=2.0),
        ItemPrediction(participant_id=1, item_index=2, pred=1, gt=1, confidence=1.0),
        ItemPrediction(participant_id=1, item_index=3, pred=None, gt=0, confidence=0.0),
    ]


class TestSelectivePredictionCanonical:
    """Strict numeric validation against Spec 25 canonical vectors."""

    def test_risk_coverage_curve_structure(
        self, canonical_vectors_abs_loss: Sequence[ItemPrediction]
    ) -> None:
        """Verify RC curve working points (ties treated as plateaus).

        Expected:
        - threshold = [2.0, 1.0]
        - coverage = [2/4, 3/4] = [0.5, 0.75]
        """
        curve = compute_risk_coverage_curve(canonical_vectors_abs_loss, loss="abs")

        np.testing.assert_allclose(curve.threshold, [2.0, 1.0], err_msg="Thresholds mismatch")
        np.testing.assert_allclose(curve.coverage, [0.5, 0.75], err_msg="Coverages mismatch")
        assert curve.cmax == 0.75

    def test_metrics_abs_loss(self, canonical_vectors_abs_loss: Sequence[ItemPrediction]) -> None:
        """Verify metrics for loss="abs".

        Expected:
        - selective_risk = [1.0, 2/3]
        - generalized_risk = [0.5, 0.5]
        - AURC_full = 17/24  (~0.708333)
        - AUGRC_full = 1/4   (0.25)
        """
        curve = compute_risk_coverage_curve(canonical_vectors_abs_loss, loss="abs")

        # Risk values at working points
        np.testing.assert_allclose(
            curve.selective_risk, [1.0, 2 / 3], err_msg="Selective risk mismatch"
        )
        np.testing.assert_allclose(
            curve.generalized_risk, [0.5, 0.5], err_msg="Generalized risk mismatch"
        )

        # Areas
        aurc = compute_aurc(canonical_vectors_abs_loss, loss="abs")
        augrc = compute_augrc(canonical_vectors_abs_loss, loss="abs")

        assert math.isclose(aurc, 17 / 24), f"AURC mismatch: got {aurc}, expected {17 / 24}"
        assert math.isclose(augrc, 1 / 4), f"AUGRC mismatch: got {augrc}, expected {0.25}"

    def test_metrics_at_coverage_abs_loss(
        self, canonical_vectors_abs_loss: Sequence[ItemPrediction]
    ) -> None:
        """Verify metrics at specific coverage target (0.6) for loss="abs".

        Expected:
        - risk_at_coverage(target_c=0.6) returns 2/3 with achieved coverage 0.75
        - AURC@0.6 = 89/150 (~0.593333)
        - AUGRC@0.6 = 7/40  (0.175)
        """
        # Risk at coverage
        risk_at_06 = compute_risk_at_coverage(
            canonical_vectors_abs_loss, target_coverage=0.6, loss="abs"
        )
        assert risk_at_06 is not None
        assert math.isclose(risk_at_06, 2 / 3), (
            f"Risk@0.6 mismatch: got {risk_at_06}, expected {2 / 3}"
        )

        # Truncated Areas
        aurc_at_06 = compute_aurc_at_coverage(
            canonical_vectors_abs_loss, max_coverage=0.6, loss="abs"
        )
        augrc_at_06 = compute_augrc_at_coverage(
            canonical_vectors_abs_loss, max_coverage=0.6, loss="abs"
        )

        assert math.isclose(aurc_at_06, 89 / 150), (
            f"AURC@0.6 mismatch: got {aurc_at_06}, expected {89 / 150}"
        )
        assert math.isclose(augrc_at_06, 7 / 40), (
            f"AUGRC@0.6 mismatch: got {augrc_at_06}, expected {7 / 40}"
        )

    def test_metrics_abs_norm_loss(
        self, canonical_vectors_abs_loss: Sequence[ItemPrediction]
    ) -> None:
        """Verify metrics for loss="abs_norm" (exactly scaled by 1/3).

        Expected:
        - selective_risk = [1/3, 2/9]
        - generalized_risk = [1/6, 1/6]
        - AURC_full = 17/72
        - AUGRC_full = 1/12
        """
        curve = compute_risk_coverage_curve(canonical_vectors_abs_loss, loss="abs_norm")

        np.testing.assert_allclose(
            curve.selective_risk, [1 / 3, 2 / 9], err_msg="Sel risk (norm) mismatch"
        )
        np.testing.assert_allclose(
            curve.generalized_risk, [1 / 6, 1 / 6], err_msg="Gen risk (norm) mismatch"
        )

        aurc = compute_aurc(canonical_vectors_abs_loss, loss="abs_norm")
        augrc = compute_augrc(canonical_vectors_abs_loss, loss="abs_norm")

        assert math.isclose(aurc, 17 / 72), f"AURC (norm) mismatch: got {aurc}, expected {17 / 72}"
        assert math.isclose(augrc, 1 / 12), f"AUGRC (norm) mismatch: got {augrc}, expected {1 / 12}"

    def test_metrics_at_coverage_abs_norm_loss(
        self, canonical_vectors_abs_loss: Sequence[ItemPrediction]
    ) -> None:
        """Verify metrics at coverage 0.6 for loss="abs_norm".

        Expected:
        - risk_at_coverage(target_c=0.6) returns 2/9 with achieved coverage 0.75
        - AURC@0.6 = 89/450
        - AUGRC@0.6 = 7/120
        """
        risk_at_06 = compute_risk_at_coverage(
            canonical_vectors_abs_loss, target_coverage=0.6, loss="abs_norm"
        )
        assert risk_at_06 is not None
        assert math.isclose(risk_at_06, 2 / 9), f"Risk@0.6 (norm) mismatch: got {risk_at_06}"

        aurc_at_06 = compute_aurc_at_coverage(
            canonical_vectors_abs_loss, max_coverage=0.6, loss="abs_norm"
        )
        augrc_at_06 = compute_augrc_at_coverage(
            canonical_vectors_abs_loss, max_coverage=0.6, loss="abs_norm"
        )

        assert math.isclose(aurc_at_06, 89 / 450), f"AURC@0.6 (norm) mismatch: got {aurc_at_06}"
        assert math.isclose(augrc_at_06, 7 / 120), f"AUGRC@0.6 (norm) mismatch: got {augrc_at_06}"


class TestSelectivePredictionEdgeCases:
    """Test cases for edge conditions defined in Spec 25."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions -> AURC=0 and AUGRC=0."""
        items = [
            ItemPrediction(1, 0, 2, 2, 1.0),
            ItemPrediction(1, 1, 1, 1, 0.9),
        ]
        assert compute_aurc(items, loss="abs") == 0.0
        assert compute_augrc(items, loss="abs") == 0.0

    def test_all_abstain(self) -> None:
        """All abstain (K=0) -> Cmax=0, AURC=0, AUGRC=0, empty curve."""
        items = [
            ItemPrediction(1, 0, None, 2, 0.0),
            ItemPrediction(1, 1, None, 1, 0.0),
        ]
        curve = compute_risk_coverage_curve(items, loss="abs")
        assert curve.cmax == 0.0
        assert len(curve.coverage) == 0
        assert compute_aurc(items, loss="abs") == 0.0
        assert compute_augrc(items, loss="abs") == 0.0

    def test_single_prediction_with_abstention(self) -> None:
        """Single prediction + abstentions.

        Verifies right-continuous convention at 0.
        Item 1: pred=2, gt=2 (err=0), conf=1.0.
        Item 2: abstain.
        Total N=2. Cmax=0.5.
        Curve: [(cov=0.5, risk=0)]
        AURC = coverage * risk = 0.5 * 0 = 0.
        """
        items = [
            ItemPrediction(1, 0, 2, 2, 1.0),
            ItemPrediction(1, 1, None, 1, 0.0),
        ]
        # To make it non-zero risk, let's make the prediction wrong
        items[0] = ItemPrediction(1, 0, 3, 2, 1.0)  # err=1

        curve = compute_risk_coverage_curve(items, loss="abs")
        assert curve.cmax == 0.5
        assert len(curve.coverage) == 1
        assert curve.selective_risk[0] == 1.0

        # AURC = integral from 0 to 0.5 of risk 1.0 => 0.5
        aurc = compute_aurc(items, loss="abs")
        assert math.isclose(aurc, 0.5)

    def test_risk_at_coverage_target_too_high(self) -> None:
        """If target_c > Cmax, return None (or handle gracefully, here we expect None or similar).

        Spec says: If no such working point exists (target_c > Cmax), return None.
        But the function signature in implementation might return optional float.
        """
        # If we have abstentions...
        items_ab = [ItemPrediction(1, 0, None, 2, 1.0)]  # Cmax=0.0

        assert compute_risk_at_coverage(items_ab, target_coverage=0.1, loss="abs") is None

    def test_exact_working_point_truncation(self) -> None:
        """When truncation coverage C equals an exact working point coverage.

        Items:
        1. (err=0, conf=2) -> cov=0.5, risk=0
        2. (err=1, conf=1) -> cov=1.0, risk=0.5

        AURC@0.5 should be exactly 0 (risk 0 up to 0.5).
        AURC@1.0 should be area under steps.
        """
        items = [
            ItemPrediction(1, 0, 2, 2, 2.0),
            ItemPrediction(1, 1, 3, 2, 1.0),
        ]
        # Selective Risk Curve:
        # [0, 0.5) -> risk=0 (from right continuous assumption of first point? No, from 0 to cov1 is risk1)
        # Spec 6.3: selective_risk(0) = selective_risk_1
        # Point 1: cov=0.5, risk=0.
        # Point 2: cov=1.0, risk=0.5.

        # Integral [0, 0.5] of 0 = 0.
        aurc_05 = compute_aurc_at_coverage(items, max_coverage=0.5, loss="abs")
        assert aurc_05 == 0.0

    def test_mae_at_coverage_snaps_to_next_plateau(self) -> None:
        """Spec 9.1: MAE@coverage at target between plateaus snaps to next plateau.

        Items (3 total, 2 predicted):
        - (pred=1, gt=1, conf=2) -> err=0 at cov=0.33
        - (pred=2, gt=1, conf=1) -> err=1 at cov=0.67
        - (pred=None, gt=0, conf=0) -> abstain

        Target 0.5 is between 0.33 and 0.67, should snap to 0.67.
        """
        items = [
            ItemPrediction(1, 0, 1, 1, 2.0),  # err=0
            ItemPrediction(1, 1, 2, 1, 1.0),  # err=1
            ItemPrediction(1, 2, None, 0, 0.0),  # abstain
        ]

        # Target 0.5: should snap to coverage 0.67 with risk = (0+1)/2 = 0.5
        risk = compute_risk_at_coverage(items, target_coverage=0.5, loss="abs")
        assert risk is not None
        assert math.isclose(risk, 0.5), f"Expected 0.5, got {risk}"

    def test_augrc_consistency_at_working_points(self) -> None:
        """Spec 9.1: verify generalized_risk_j == coverage_j * selective_risk_j."""
        items = [
            ItemPrediction(1, 0, 2, 2, 2.0),  # err=0
            ItemPrediction(1, 1, 3, 1, 2.0),  # err=2
            ItemPrediction(1, 2, 1, 1, 1.0),  # err=0
            ItemPrediction(1, 3, None, 0, 0.0),  # abstain
        ]

        curve = compute_risk_coverage_curve(items, loss="abs")

        for i, (cov, sel_risk, gen_risk) in enumerate(
            zip(curve.coverage, curve.selective_risk, curve.generalized_risk, strict=True)
        ):
            expected_gen_risk = cov * sel_risk
            assert math.isclose(gen_risk, expected_gen_risk), (
                f"Working point {i}: gen_risk={gen_risk} != cov*sel_risk={expected_gen_risk}"
            )

    def test_all_same_confidence_single_plateau(self) -> None:
        """Spec 9.1: all predicted items have identical confidence -> single working point.

        All 3 predicted items have conf=1.0, so there's exactly one threshold.
        Cmax = 3/4 = 0.75 (one abstention).
        """
        items = [
            ItemPrediction(1, 0, 1, 1, 1.0),  # err=0
            ItemPrediction(1, 1, 2, 1, 1.0),  # err=1
            ItemPrediction(1, 2, 0, 1, 1.0),  # err=1
            ItemPrediction(1, 3, None, 0, 0.0),  # abstain
        ]

        curve = compute_risk_coverage_curve(items, loss="abs")

        # Single working point
        assert len(curve.coverage) == 1, f"Expected 1 working point, got {len(curve.coverage)}"
        assert curve.coverage[0] == curve.cmax
        assert curve.cmax == 0.75
        # Risk = (0+1+1)/3 = 2/3
        assert math.isclose(curve.selective_risk[0], 2 / 3)

    def test_aurc_optimal_perfect(self) -> None:
        """Perfect ranking -> optimal = actual, excess = 0."""
        # Perfect order: correct (0 err) first, wrong (1 err) last
        items = [
            ItemPrediction(1, 0, 0, 0, 0.9),  # err=0
            ItemPrediction(1, 1, 1, 0, 0.1),  # err=1
        ]

        aurc = compute_aurc(items, loss="abs")
        aurc_opt = compute_aurc_optimal(items, loss="abs")
        eaurc = compute_eaurc(items, loss="abs")

        assert math.isclose(aurc, aurc_opt)
        assert math.isclose(eaurc, 0.0)

    def test_aurc_optimal_all_wrong(self) -> None:
        """All wrong -> optimal is still just the area under risk=1 line?

        If all are wrong (err=1), any ranking gives risk curve = 1.0 everywhere.
        So optimal should be equal to actual, which is 1.0 * 1.0 / 2 ??
        Wait. Risk=1 everywhere.
        Coverage [0, 1]. Risk [1, 1].
        Area = 1.0 (if risk constant 1).

        Let's check.
        Items: 2 items, both wrong.
        """
        items = [
            ItemPrediction(1, 0, 1, 0, 0.9),  # err=1
            ItemPrediction(1, 1, 1, 0, 0.1),  # err=1
        ]

        # Curve:
        # 1. conf 0.9: cov=0.5, risk=1
        # 2. conf 0.1: cov=1.0, risk=1
        # Area: Trapz([1, 1], [0, 0.5]) + Trapz([1, 1], [0.5, 1]) ??
        # No, _integrate_curve uses [0, c1, c2]. ys=[risk1, risk1, risk2].
        # ys=[1, 1, 1], xs=[0, 0.5, 1.0]. Area = 1.0.

        aurc = compute_aurc(items, loss="abs")
        assert math.isclose(aurc, 1.0)

        aurc_opt = compute_aurc_optimal(items, loss="abs")
        # Optimal ranking sorts by loss. Both loss=1. Order doesn't matter.
        assert math.isclose(aurc_opt, 1.0)

        eaurc = compute_eaurc(items, loss="abs")
        assert math.isclose(eaurc, 0.0)

    def test_eaurc_positive(self) -> None:
        """Suboptimal ranking -> positive excess AURC."""
        # Bad order: wrong first, correct last
        items = [
            ItemPrediction(1, 0, 1, 0, 0.9),  # err=1, high conf
            ItemPrediction(1, 1, 0, 0, 0.1),  # err=0, low conf
        ]

        # Actual Curve:
        # 1. cov=0.5, risk=1
        # 2. cov=1.0, risk=0.5
        # xs=[0, 0.5, 1.0], ys=[1, 1, 0.5]
        # Area = 0.5*1 + 0.5*(1+0.5)/2 = 0.5 + 0.375 = 0.875
        aurc = compute_aurc(items, loss="abs")
        assert math.isclose(aurc, 0.875)

        # Optimal (Oracle) Curve:
        # Correct first (err=0), Wrong last (err=1)
        # 1. cov=0.5, risk=0
        # 2. cov=1.0, risk=0.5
        # xs=[0, 0.5, 1.0], ys=[0, 0, 0.5]
        # Area = 0 + 0.5*(0+0.5)/2 = 0.125
        aurc_opt = compute_aurc_optimal(items, loss="abs")
        assert math.isclose(aurc_opt, 0.125)

        eaurc = compute_eaurc(items, loss="abs")
        assert math.isclose(eaurc, 0.75)  # 0.875 - 0.125

    def test_aurc_achievable_convex_hull(self) -> None:
        """Test achievable AURC uses convex hull points."""
        # Construct a jagged curve.
        # 1. Good (err=0)
        # 2. Bad (err=1)
        # 3. Good (err=0)
        # Order: 1 (0.9), 2 (0.5), 3 (0.1)
        items = [
            ItemPrediction(1, 0, 0, 0, 0.9),  # err=0
            ItemPrediction(1, 1, 1, 0, 0.5),  # err=1
            ItemPrediction(1, 2, 0, 0, 0.1),  # err=0
        ]

        # Points:
        # 1. cov=0.33, risk=0
        # 2. cov=0.66, risk=0.5 ( (0+1)/2 )
        # 3. cov=1.00, risk=0.33 ( (0+1+0)/3 )

        # Curve points: (0.33, 0), (0.66, 0.5), (1.0, 0.33)
        # Convex hull (lower) should skip point 2 because point 3 is lower risk at higher coverage?
        # Wait, risk at 1.0 is 0.33 which is < 0.5.
        # Yes, standard hull logic.

        # Check regular AURC first
        # xs=[0, 0.33, 0.66, 1.0], ys=[0, 0, 0.5, 0.33]
        # A1 (0-0.33): 0
        # A2 (0.33-0.66): 0.33 * (0+0.5)/2 = 0.33 * 0.25 = 0.0825
        # A3 (0.66-1.0): 0.33 * (0.5+0.33)/2 = 0.33 * 0.415 = 0.13695
        # Total ~ 0.22

        aurc = compute_aurc(items, loss="abs")

        # Achievable:
        # Hull points should be (0.33, 0) and (1.0, 0.33).
        # xs=[0, 0.33, 1.0], ys=[0, 0, 0.33]

        aurc_ach = compute_aurc_achievable(items, loss="abs")
        assert aurc_ach < aurc

    def test_augrc_optimal(self) -> None:
        """Test AUGRC optimal is computed correctly."""
        # Suboptimal case from before
        items = [
            ItemPrediction(1, 0, 1, 0, 0.9),  # err=1
            ItemPrediction(1, 1, 0, 0, 0.1),  # err=0
        ]
        # Optimal AUGRC: Correct first.
        # 1. cov=0.5, risk=0. GenRisk = 0.5*0 = 0.
        # 2. cov=1.0, risk=0.5. GenRisk = 1.0*0.5 = 0.5.
        # xs=[0, 0.5, 1.0], ys=[0, 0, 0.5]
        # Area = 0 + 0.5*(0+0.5)/2 = 0.125

        augrc_opt = compute_augrc_optimal(items, loss="abs")
        assert math.isclose(augrc_opt, 0.125)
