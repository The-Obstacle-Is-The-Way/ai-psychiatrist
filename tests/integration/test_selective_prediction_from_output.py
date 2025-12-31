"""Integration test for selective prediction evaluation script (Spec 25 Section 9.2)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_psychiatrist.domain.enums import PHQ8Item

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_output_json(tmp_path: Path) -> Path:
    """Create a mock JSON output file."""
    item_keys = [item.value for item in PHQ8Item.all_items()]

    # One successful participant with all 8 items and mixed predicted/abstained.
    # We craft the first 3 items to match an easy-to-verify RC curve:
    # - confidences (2, 2, 1), losses (0, 2, 0), and 5 abstentions.
    # This yields (N=8, K=3):
    # - Cmax = 3/8
    # - AURC_full (abs) = 17/48
    # - AUGRC_full (abs) = 1/16
    gt_items = dict.fromkeys(item_keys, 0)
    pred_items = dict.fromkeys(item_keys, None)
    signals = {k: {"llm_evidence_count": 0, "keyword_evidence_count": 0} for k in item_keys}

    # Map the first three PHQ-8 items to the canonical structure.
    k0, k1, k2 = item_keys[0], item_keys[1], item_keys[2]
    gt_items[k0] = 2
    gt_items[k1] = 1
    gt_items[k2] = 1
    pred_items[k0] = 2  # err=0
    pred_items[k1] = 3  # err=2
    pred_items[k2] = 1  # err=0
    signals[k0]["llm_evidence_count"] = 2
    signals[k1]["llm_evidence_count"] = 2
    signals[k2]["llm_evidence_count"] = 1

    data = {
        "run_metadata": {"run_id": "test_run", "git_commit": "abc"},
        "experiments": [
            {
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
                        },
                        {
                            "participant_id": 102,
                            "success": False,
                            "error": "mock failure",
                        },
                    ],
                },
            }
        ],
    }
    path = tmp_path / "output.json"
    with path.open("w") as f:
        json.dump(data, f)
    return path


def test_evaluate_cli_runs(mock_output_json: Path) -> None:
    """Evaluate a saved run artifact and validate computed metrics."""

    cmd = [
        sys.executable,
        "scripts/evaluate_selective_prediction.py",
        "--input",
        str(mock_output_json),
        "--loss",
        "abs",
        "--confidence",
        "llm",
        "--coverage-grid",
        "0.25,0.50",
        "--bootstrap-resamples",
        "0",
        "--output",
        str(mock_output_json.parent / "metrics.json"),
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"Stderr: {result.stderr}"

    # Check output
    metrics_path = mock_output_json.parent / "metrics.json"
    assert metrics_path.exists()

    with metrics_path.open() as f:
        m = json.load(f)

    assert m["loss"]["name"] == "abs"
    assert m["population"]["participants_total"] == 2
    assert m["population"]["participants_included"] == 1
    assert m["population"]["participants_failed"] == 1
    assert m["population"]["items_total"] == 8

    assert "llm" in m["confidence_variants"]
    llm = m["confidence_variants"]["llm"]

    assert llm["cmax"] == pytest.approx(3 / 8)
    assert llm["aurc_full"] == pytest.approx(17 / 48)
    assert llm["augrc_full"] == pytest.approx(1 / 16)

    assert llm["aurc_at_c"]["used"] == pytest.approx(3 / 8)
    assert llm["aurc_at_c"]["value"] == pytest.approx(17 / 48)
    assert llm["augrc_at_c"]["used"] == pytest.approx(3 / 8)
    assert llm["augrc_at_c"]["value"] == pytest.approx(1 / 16)

    # MAE@coverage: target 0.25 is achievable (coverage_1=2/8), 0.50 is not.
    assert llm["mae_at_coverage"]["0.25"]["achieved"] == pytest.approx(2 / 8)
    assert llm["mae_at_coverage"]["0.25"]["value"] == pytest.approx(1.0)
    assert llm["mae_at_coverage"]["0.50"] is None
