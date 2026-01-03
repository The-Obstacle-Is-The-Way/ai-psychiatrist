"""Unit tests for selective prediction confidence variants (Spec 046)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scripts.evaluate_selective_prediction import SupervisedCalibration, parse_items

from ai_psychiatrist.calibration.calibrators import LogisticCalibrator, StandardScalerParams
from ai_psychiatrist.calibration.feature_extraction import CalibratorFeatureExtractor
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
    return {k: {"llm_evidence_count": 1} for k in _make_item_keys()}


def _signals_with_retrieval_mean(mean: float | None) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "llm_evidence_count": 3,
            "retrieval_reference_count": 2,
            "retrieval_similarity_mean": mean,
            "retrieval_similarity_max": mean,
        }
        for k in _make_item_keys()
    }


def _signals_with_verbalized_confidence(
    *, verbalized_confidence: int | None, retrieval_similarity_mean: float
) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "llm_evidence_count": 3,
            "retrieval_reference_count": 2,
            "retrieval_similarity_mean": retrieval_similarity_mean,
            "retrieval_similarity_max": retrieval_similarity_mean,
            "verbalized_confidence": verbalized_confidence,
        }
        for k in _make_item_keys()
    }


def _signals_with_token_signals(
    *,
    token_msp: float,
    token_pe: float,
    token_energy: float,
    retrieval_similarity_mean: float = 0.4,
) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "llm_evidence_count": 2,
            "retrieval_similarity_mean": retrieval_similarity_mean,
            "retrieval_similarity_max": retrieval_similarity_mean,
            "token_msp": token_msp,
            "token_pe": token_pe,
            "token_energy": token_energy,
        }
        for k in _make_item_keys()
    }


def _signals_with_consistency_signals(
    *,
    consistency_modal_confidence: float,
    consistency_score_std: float,
    retrieval_similarity_mean: float = 0.4,
    llm_evidence_count: int = 3,
) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "llm_evidence_count": llm_evidence_count,
            "retrieval_similarity_mean": retrieval_similarity_mean,
            "retrieval_similarity_max": retrieval_similarity_mean,
            "consistency_modal_confidence": consistency_modal_confidence,
            "consistency_score_std": consistency_score_std,
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

    def test_verbalized_confidence_normalized(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_verbalized_confidence(
                verbalized_confidence=5, retrieval_similarity_mean=0.4
            )
        )
        items, _, _ = parse_items(experiment, "verbalized")
        assert items[0].confidence == 1.0

    def test_hybrid_verbalized_confidence_deterministic_and_bounded(self) -> None:
        # llm_evidence_count=3 -> e=1.0, s=0.4, v=3 -> 0.5
        # conf = 0.4*v + 0.3*e + 0.3*s = 0.62
        experiment = _make_experiment(
            signals=_signals_with_verbalized_confidence(
                verbalized_confidence=3, retrieval_similarity_mean=0.4
            )
        )
        items, _, _ = parse_items(experiment, "hybrid_verbalized")
        assert 0.0 <= items[0].confidence <= 1.0
        assert items[0].confidence == pytest.approx(0.62)

    def test_verbalized_confidence_none_yields_neutral(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_verbalized_confidence(
                verbalized_confidence=None, retrieval_similarity_mean=0.4
            )
        )
        items, _, _ = parse_items(experiment, "verbalized")
        assert items[0].confidence == pytest.approx(0.5)

    def test_calibrated_confidence_requires_calibration(self) -> None:
        experiment = _make_experiment(signals=_signals_with_retrieval_mean(0.4))
        with pytest.raises(ValueError, match="calibrated"):
            parse_items(experiment, "calibrated")

    def test_calibrated_confidence_uses_supervised_calibrator(self) -> None:
        item_keys = _make_item_keys()
        experiment = _make_experiment(
            signals={
                k: {
                    "llm_evidence_count": 2,
                    "retrieval_similarity_mean": 0.4,
                    "retrieval_similarity_max": 0.4,
                    "verbalized_confidence": 4,
                }
                for k in item_keys
            }
        )

        calibration = SupervisedCalibration(
            method="logistic",
            target="correctness",
            features=("llm_evidence_count",),
            extractor=CalibratorFeatureExtractor(["llm_evidence_count"]),
            calibrator=LogisticCalibrator(
                coefficients=np.array([0.0], dtype=float),
                intercept=0.0,
                scaler=StandardScalerParams(
                    mean=np.array([0.0], dtype=float),
                    std=np.array([1.0], dtype=float),
                ),
            ),
        )

        items, _, _ = parse_items(experiment, "calibrated", calibration=calibration)
        assert items[0].confidence == pytest.approx(0.5)

    def test_token_msp_confidence_uses_signal(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_token_signals(token_msp=0.8, token_pe=0.7, token_energy=-0.1)
        )
        items, _, _ = parse_items(experiment, "token_msp")
        assert items[0].confidence == pytest.approx(0.8)

    def test_token_pe_confidence_inverts_entropy(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_token_signals(token_msp=0.8, token_pe=1.0, token_energy=-0.1)
        )
        items, _, _ = parse_items(experiment, "token_pe")
        assert items[0].confidence == pytest.approx(1.0 / 2.0)

    def test_token_energy_confidence_exponentiates_logsumexp(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_token_signals(
                token_msp=0.8, token_pe=0.7, token_energy=-0.2876820724517809
            )
        )
        items, _, _ = parse_items(experiment, "token_energy")
        assert items[0].confidence == pytest.approx(0.75)

    def test_secondary_confidence_variant_supported(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_token_signals(
                token_msp=0.4, token_pe=0.7, token_energy=-0.1, retrieval_similarity_mean=0.6
            )
        )
        items, _, _ = parse_items(
            experiment, "secondary:token_msp+retrieval_similarity_mean:average"
        )
        assert items[0].confidence == pytest.approx(0.5)

    def test_consistency_confidence_uses_modal_confidence(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_consistency_signals(
                consistency_modal_confidence=0.8, consistency_score_std=0.4
            )
        )
        items, _, _ = parse_items(experiment, "consistency")
        assert items[0].confidence == pytest.approx(0.8)

    def test_consistency_inverse_std_confidence(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_consistency_signals(
                consistency_modal_confidence=0.8, consistency_score_std=0.5
            )
        )
        items, _, _ = parse_items(experiment, "consistency_inverse_std")
        assert items[0].confidence == pytest.approx(1.0 / 1.5)

    def test_hybrid_consistency_confidence(self) -> None:
        experiment = _make_experiment(
            signals=_signals_with_consistency_signals(
                consistency_modal_confidence=0.8,
                consistency_score_std=0.5,
                retrieval_similarity_mean=0.4,
                llm_evidence_count=3,
            )
        )
        # c=0.8, e=1.0, s=0.4 => 0.4*c + 0.3*e + 0.3*s = 0.74
        items, _, _ = parse_items(experiment, "hybrid_consistency")
        assert items[0].confidence == pytest.approx(0.74)
