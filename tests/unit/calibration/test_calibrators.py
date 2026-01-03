"""Unit tests for confidence calibration utilities (Spec 048/049)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_psychiatrist.calibration.calibrators import (
    TemperatureScalingCalibrator,
    compute_binary_nll,
)

pytestmark = pytest.mark.unit


class TestTemperatureScalingCalibrator:
    def test_overconfident_predictions_increase_temperature(self) -> None:
        # Overconfident: high p but low accuracy
        p = np.array([0.9, 0.9, 0.9, 0.9], dtype=float)
        y = np.array([1, 0, 0, 0], dtype=int)

        before = compute_binary_nll(p, y)
        calibrator = TemperatureScalingCalibrator.fit(p, y)
        after = compute_binary_nll(calibrator.apply(p), y)

        assert calibrator.temperature > 1.0
        assert after < before

    def test_apply_preserves_ordering(self) -> None:
        p = np.array([0.2, 0.5, 0.8], dtype=float)
        y = np.array([0, 1, 1], dtype=int)
        calibrator = TemperatureScalingCalibrator.fit(p, y)
        p_cal = calibrator.apply(p)

        assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))
        assert list(np.argsort(p)) == list(np.argsort(p_cal))
