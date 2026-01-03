"""Unit tests for calibration feature extraction (Spec 049)."""

from __future__ import annotations

import numpy as np
import pytest

from ai_psychiatrist.calibration.feature_extraction import CalibratorFeatureExtractor

pytestmark = pytest.mark.unit


class TestCalibratorFeatureExtractor:
    def test_extract_applies_defaults(self) -> None:
        extractor = CalibratorFeatureExtractor(
            ["llm_evidence_count", "retrieval_similarity_mean", "verbalized_confidence"]
        )
        x = extractor.extract({})
        assert x.shape == (3,)
        assert np.allclose(x, np.array([0.0, 0.0, 3.0]))

    def test_extract_preserves_numeric_values(self) -> None:
        extractor = CalibratorFeatureExtractor(
            ["llm_evidence_count", "retrieval_similarity_mean", "verbalized_confidence"]
        )
        x = extractor.extract(
            {"llm_evidence_count": 2, "retrieval_similarity_mean": 0.75, "verbalized_confidence": 4}
        )
        assert np.allclose(x, np.array([2.0, 0.75, 4.0]))

    def test_extract_rejects_bool(self) -> None:
        extractor = CalibratorFeatureExtractor(["llm_evidence_count"])
        with pytest.raises(TypeError, match="must be numeric or null, got bool"):
            extractor.extract({"llm_evidence_count": True})
