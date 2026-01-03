"""Unit tests for Spec 048 verbalized confidence calibration script."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scripts.calibrate_verbalized_confidence import extract_training_pairs

from ai_psychiatrist.calibration.calibrators import TemperatureScalingCalibrator, compute_binary_nll
from ai_psychiatrist.domain.enums import PHQ8Item

pytestmark = pytest.mark.unit


def _make_item_keys() -> list[str]:
    return [item.value for item in PHQ8Item.all_items()]


def _make_experiment(*, item_signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item_keys = _make_item_keys()
    pred_items: dict[str, int | None] = dict.fromkeys(item_keys, None)
    gt_items: dict[str, int] = dict.fromkeys(item_keys, 0)

    # Create four predicted items with varying confidence/correctness.
    pred_items[item_keys[0]] = 1
    gt_items[item_keys[0]] = 1  # correct

    pred_items[item_keys[1]] = 2
    gt_items[item_keys[1]] = 0  # incorrect

    pred_items[item_keys[2]] = 1
    gt_items[item_keys[2]] = 1  # correct

    pred_items[item_keys[3]] = 0
    gt_items[item_keys[3]] = 0  # correct

    return {
        "provenance": {"mode": "few_shot"},
        "results": {
            "mode": "few_shot",
            "results": [
                {
                    "participant_id": 101,
                    "success": True,
                    "predicted_items": pred_items,
                    "ground_truth_items": gt_items,
                    "item_signals": item_signals,
                }
            ],
        },
    }


def _signals_with_verbalized_confidence(values: dict[str, int | None]) -> dict[str, dict[str, Any]]:
    signals = {}
    for key in _make_item_keys():
        signals[key] = {"verbalized_confidence": values.get(key)}
    return signals


class TestCalibrateVerbalizedConfidence:
    def test_extract_training_pairs(self) -> None:
        item_keys = _make_item_keys()
        exp = _make_experiment(
            item_signals=_signals_with_verbalized_confidence(
                {
                    item_keys[0]: 5,
                    item_keys[1]: 5,
                    item_keys[2]: 3,
                    item_keys[3]: 2,
                }
            )
        )
        p_raw, y = extract_training_pairs(exp)

        assert p_raw.shape == (4,)
        assert y.shape == (4,)
        assert np.all((p_raw >= 0.0) & (p_raw <= 1.0))
        assert set(y.tolist()) <= {0, 1}

    def test_temperature_scaling_improves_nll(self) -> None:
        item_keys = _make_item_keys()
        exp = _make_experiment(
            item_signals=_signals_with_verbalized_confidence(
                {
                    item_keys[0]: 5,
                    item_keys[1]: 5,
                    item_keys[2]: 3,
                    item_keys[3]: 2,
                }
            )
        )
        p_raw, y = extract_training_pairs(exp)
        before = compute_binary_nll(p_raw, y)
        calibrator = TemperatureScalingCalibrator.fit(p_raw, y)
        after = compute_binary_nll(calibrator.apply(p_raw), y)

        assert after <= before
