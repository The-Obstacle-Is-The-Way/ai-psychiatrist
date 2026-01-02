"""Unit tests for selective prediction confidence variants (Spec 046)."""

from __future__ import annotations

from typing import Any

import pytest
from scripts.evaluate_selective_prediction import parse_items

from ai_psychiatrist.domain.enums import PHQ8Item

pytestmark = pytest.mark.unit


def _make_item_keys() -> list[str]:
    return [item.value for item in PHQ8Item.all_items()]


def _make_experiment(*, signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item_keys = _make_item_keys()
    pred_items: dict[str, int | None] = dict.fromkeys(item_keys, None)
    gt_items: dict[str, int] = dict.fromkeys(item_keys, 0)

    # Ensure at least one predicted item exists to avoid degenerate metrics paths.
    pred_items[item_keys[0]] = 2
    gt_items[item_keys[0]] = 1

    return {
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
        }
    }


def _signals_without_retrieval() -> dict[str, dict[str, Any]]:
    return {k: {"llm_evidence_count": 1, "keyword_evidence_count": 0} for k in _make_item_keys()}


def _signals_with_retrieval_mean(mean: float | None) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "llm_evidence_count": 3,
            "keyword_evidence_count": 0,
            "retrieval_reference_count": 2,
            "retrieval_similarity_mean": mean,
            "retrieval_similarity_max": mean,
        }
        for k in _make_item_keys()
    }


class TestConfidenceVariants:
    """Tests for Spec 046 confidence variants in parse_items()."""

    def test_retrieval_confidence_errors_on_missing_signals(self) -> None:
        experiment = _make_experiment(signals=_signals_without_retrieval())
        with pytest.raises(ValueError, match="retrieval_similarity_mean"):
            parse_items(experiment, "retrieval_similarity_mean")

    def test_hybrid_confidence_deterministic_and_bounded(self) -> None:
        experiment = _make_experiment(signals=_signals_with_retrieval_mean(0.4))
        items, included, failed = parse_items(experiment, "hybrid_evidence_similarity")

        assert included == {101}
        assert failed == set()

        first = items[0]
        assert 0.0 <= first.confidence <= 1.0
        # llm_evidence_count=3 -> e=1.0, s=0.4 => 0.5*1.0 + 0.5*0.4 = 0.7
        assert first.confidence == pytest.approx(0.7)

    def test_retrieval_similarity_none_yields_zero_confidence(self) -> None:
        experiment = _make_experiment(signals=_signals_with_retrieval_mean(None))
        items, _, _ = parse_items(experiment, "retrieval_similarity_mean")
        assert items[0].confidence == 0.0
