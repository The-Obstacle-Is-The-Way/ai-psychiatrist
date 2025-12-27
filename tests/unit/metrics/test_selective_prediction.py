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
    compute_aurc,
    compute_aurc_at_coverage,
    compute_risk_at_coverage,
    compute_risk_coverage_curve,
)


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

    def test_risk_coverage_curve_structure(self, canonical_vectors_abs_loss):
        """Verify RC curve working points (ties treated as plateaus).

        Expected:
        - threshold = [2.0, 1.0]
        - coverage = [2/4, 3/4] = [0.5, 0.75]
        """
        curve = compute_risk_coverage_curve(canonical_vectors_abs_loss, loss="abs")

        np.testing.assert_allclose(curve.threshold, [2.0, 1.0], err_msg="Thresholds mismatch")
        np.testing.assert_allclose(curve.coverage, [0.5, 0.75], err_msg="Coverages mismatch")
        assert curve.cmax == 0.75

    def test_metrics_abs_loss(self, canonical_vectors_abs_loss):
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

    def test_metrics_at_coverage_abs_loss(self, canonical_vectors_abs_loss):
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

    def test_metrics_abs_norm_loss(self, canonical_vectors_abs_loss):
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

    def test_metrics_at_coverage_abs_norm_loss(self, canonical_vectors_abs_loss):
        """Verify metrics at coverage 0.6 for loss="abs_norm".

        Expected:
        - risk_at_coverage(target_c=0.6) returns 2/9 with achieved coverage 0.75
        - AURC@0.6 = 89/450
        - AUGRC@0.6 = 7/120
        """
        risk_at_06 = compute_risk_at_coverage(
            canonical_vectors_abs_loss, target_coverage=0.6, loss="abs_norm"
        )
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

    def test_perfect_predictions(self):
        """Perfect predictions -> AURC=0 and AUGRC=0."""
        items = [
            ItemPrediction(1, 0, 2, 2, 1.0),
            ItemPrediction(1, 1, 1, 1, 0.9),
        ]
        assert compute_aurc(items, loss="abs") == 0.0
        assert compute_augrc(items, loss="abs") == 0.0

    def test_all_abstain(self):
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

    def test_single_prediction_with_abstention(self):
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

    def test_risk_at_coverage_target_too_high(self):
        """If target_c > Cmax, return None (or handle gracefully, here we expect None or similar).

        Spec says: If no such working point exists (target_c > Cmax), return None.
        But the function signature in implementation might return optional float.
        """
        items = [ItemPrediction(1, 0, 2, 2, 1.0)]  # Cmax=1.0
        # If we have abstentions...
        items_ab = [ItemPrediction(1, 0, None, 2, 1.0)]  # Cmax=0.0

        assert compute_risk_at_coverage(items_ab, target_coverage=0.1, loss="abs") is None

    def test_exact_working_point_truncation(self):
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
