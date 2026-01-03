"""Feature extraction for supervised confidence calibration (Spec 049)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibratorFeatureExtractor:
    """Extract numeric feature vectors from `item_signals`."""

    feature_names: tuple[str, ...]

    def __init__(self, feature_names: list[str]) -> None:
        object.__setattr__(self, "feature_names", tuple(feature_names))

    def extract(self, item_signals: dict[str, Any]) -> np.ndarray:
        """Extract features as a 1D float array.

        Missing values are filled with conservative defaults:
        - `*similarity*` -> 0.0
        - `*confidence*` -> 3.0 (middle of 1-5 scale)
        - otherwise -> 0.0
        """
        values: list[float] = []
        for name in self.feature_names:
            raw = item_signals.get(name)
            if raw is None:
                if "similarity" in name:
                    values.append(0.0)
                elif "confidence" in name:
                    values.append(3.0)
                else:
                    values.append(0.0)
                continue

            if isinstance(raw, bool):
                raise TypeError(f"Feature '{name}' must be numeric or null, got bool")
            if isinstance(raw, (int, float)):
                values.append(float(raw))
                continue

            raise TypeError(f"Feature '{name}' must be numeric or null, got {type(raw).__name__}")

        return np.asarray(values, dtype=float)
