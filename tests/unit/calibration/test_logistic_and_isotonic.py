"""Unit tests for supervised confidence calibrators (Spec 049)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_psychiatrist.calibration.calibrators import (
    IsotonicCalibrator,
    LogisticCalibrator,
    StandardScalerParams,
)

pytestmark = pytest.mark.unit


class TestLogisticCalibrator:
    def test_predict_proba_shape_and_ordering(self) -> None:
        calibrator = LogisticCalibrator(
            coefficients=np.array([1.0, 1.0], dtype=float),
            intercept=0.0,
            scaler=StandardScalerParams(
                mean=np.array([0.0, 0.0], dtype=float),
                std=np.array([1.0, 1.0], dtype=float),
            ),
        )

        x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
        p = calibrator.predict_proba(x)

        assert p.shape == (2,)
        assert p[0] == pytest.approx(0.5)
        assert p[1] > p[0]


class TestIsotonicCalibrator:
    def test_predict_proba_monotonic(self) -> None:
        calibrator = IsotonicCalibrator(
            x_thresholds=np.array([0.0, 0.5, 1.0], dtype=float),
            y_thresholds=np.array([0.2, 0.5, 0.9], dtype=float),
        )

        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        p = calibrator.predict_proba(x)

        assert p.shape == (5,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert list(p) == sorted(p)
