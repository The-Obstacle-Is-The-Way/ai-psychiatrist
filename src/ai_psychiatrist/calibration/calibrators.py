"""Post-hoc confidence calibrators.

These utilities implement lightweight calibration methods used by the selective
prediction pipeline. They are designed to be:
- Deterministic (no stochastic training)
- Small and testable
- Serializable via JSON artifacts (handled in scripts)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np

_EPS = 1e-6


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return cast("np.ndarray", np.clip(p, _EPS, 1.0 - _EPS))


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probs(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return cast("np.ndarray", 1.0 / (1.0 + np.exp(-x)))


def apply_temperature_scaling(p: np.ndarray, *, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities.

    This is the probability-space equivalent of classic temperature scaling on
    logits: p' = sigmoid(logit(p) / T).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return _sigmoid(_logit(p) / temperature)


def compute_binary_nll(p: np.ndarray, y: np.ndarray) -> float:
    """Compute mean negative log-likelihood for binary labels."""
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")
    p = _clip_probs(p.astype(float))
    y = y.astype(float)
    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(nll))


def compute_ece(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected calibration error (ECE) for binary outcomes."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")

    p = np.clip(p.astype(float), 0.0, 1.0)
    y = y.astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in pairwise(bins):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        p_bin = p[mask]
        y_bin = y[mask]
        acc = float(np.mean(y_bin))
        conf = float(np.mean(p_bin))
        ece += float(np.mean(mask)) * abs(acc - conf)

    return float(ece)


@dataclass(frozen=True, slots=True)
class TemperatureScalingCalibrator:
    """Single-parameter temperature scaling calibrator."""

    temperature: float

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray) -> TemperatureScalingCalibrator:
        """Fit temperature scaling by minimizing binary NLL."""
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)

        if p.ndim != 1 or y.ndim != 1:
            raise ValueError("p and y must be 1D arrays")
        if p.shape != y.shape:
            raise ValueError("p and y must have the same length")
        if p.size == 0:
            raise ValueError("Cannot fit calibrator on empty data")

        # Log-spaced search over a bounded range to avoid scipy dependency.
        min_t = 0.05
        max_t = 10.0

        low = min_t
        high = max_t
        best_t = 1.0
        best_nll = compute_binary_nll(apply_temperature_scaling(p, temperature=best_t), y)

        for _ in range(3):
            grid = np.exp(np.linspace(np.log(low), np.log(high), 200))
            nlls = np.array(
                [compute_binary_nll(apply_temperature_scaling(p, temperature=t), y) for t in grid]
            )
            idx = int(np.argmin(nlls))
            cand_t = float(grid[idx])
            cand_nll = float(nlls[idx])
            if cand_nll < best_nll:
                best_nll = cand_nll
                best_t = cand_t

            # Narrow search window around best_t
            low = max(min_t, best_t / 1.5)
            high = min(max_t, best_t * 1.5)

        return cls(temperature=best_t)

    def apply(self, p: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        return apply_temperature_scaling(np.asarray(p, dtype=float), temperature=self.temperature)


@dataclass(frozen=True, slots=True)
class StandardScalerParams:
    """Frozen parameters for standardization: (x - mean) / std."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected last dimension {self.mean.shape[0]}, got {x.shape[-1]} for features"
            )
        return cast("np.ndarray", (x - self.mean) / self.std)


@dataclass(frozen=True, slots=True)
class LogisticCalibrator:
    """Logistic regression calibrator with frozen parameters."""

    coefficients: np.ndarray
    intercept: float
    scaler: StandardScalerParams

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        logits = self.intercept + x_scaled @ self.coefficients
        return cast("np.ndarray", _sigmoid(np.asarray(logits, dtype=float)).reshape(-1))


@dataclass(frozen=True, slots=True)
class LinearCalibrator:
    """Linear regression calibrator with frozen parameters.

    Used for continuous targets such as `1 - normalized_abs_error`.
    """

    coefficients: np.ndarray
    intercept: float
    scaler: StandardScalerParams

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        y = self.intercept + x_scaled @ self.coefficients
        y = np.clip(y, 0.0, 1.0)
        return cast("np.ndarray", np.asarray(y, dtype=float).reshape(-1))


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Isotonic regression calibrator (piecewise-linear interpolation)."""

    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.interp(x, self.x_thresholds, self.y_thresholds)
        y = np.clip(y, 0.0, 1.0)
        return cast("np.ndarray", y)
