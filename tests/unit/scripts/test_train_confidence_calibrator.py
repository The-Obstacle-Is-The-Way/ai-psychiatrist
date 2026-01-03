"""Unit tests for Spec 049 confidence calibrator training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scripts.train_confidence_calibrator import _build_dataset, _fit_logistic_model

from ai_psychiatrist.domain.enums import PHQ8Item

pytestmark = pytest.mark.unit


def _make_item_keys() -> list[str]:
    return [item.value for item in PHQ8Item.all_items()]


def _make_experiment(*, signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item_keys = _make_item_keys()

    # Predict every item to get non-trivial sample size.
    pred_items: dict[str, int | None] = {}
    gt_items: dict[str, int] = {}
    for idx, key in enumerate(item_keys):
        pred = idx % 4
        pred_items[key] = pred
        gt_items[key] = pred if idx % 2 == 0 else (idx + 1) % 4

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
                    "item_signals": signals,
                }
            ],
        },
    }


def _signals_for_features() -> dict[str, dict[str, Any]]:
    signals = {}
    for idx, key in enumerate(_make_item_keys()):
        signals[key] = {
            "llm_evidence_count": idx % 4,
            "retrieval_similarity_mean": 0.25 * (idx % 4),
            "verbalized_confidence": 1 + (idx % 5),
        }
    return signals


class TestTrainConfidenceCalibrator:
    def test_build_dataset_and_fit_logistic(self) -> None:
        exp = _make_experiment(signals=_signals_for_features())
        x, y = _build_dataset(
            exp,
            features=("llm_evidence_count", "retrieval_similarity_mean", "verbalized_confidence"),
            target="correctness",
        )

        assert x.shape == (8, 3)
        assert y.shape == (8,)

        calibrator = _fit_logistic_model(x, y)
        p = calibrator.predict_proba(x)

        assert p.shape == (8,)
        assert np.all((p >= 0.0) & (p <= 1.0))
