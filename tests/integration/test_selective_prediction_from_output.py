"""Integration test for selective prediction evaluation script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Import main function for testing via subprocess or direct import?
# Direct import is easier for checking internals if we refactor, but script is scripts/
# We will use run_shell_command to test CLI invocation or just subprocess.
# Actually, verifying the logic via direct import of helper functions or just mocking.
# Spec says "Fixture must include... at least 1 participant... item_signals".


@pytest.fixture
def mock_output_json(tmp_path: Path) -> Path:
    """Create a mock JSON output file."""
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
                            "predicted_items": {"PHQ8_NoInterest": 2, "PHQ8_Depressed": 3},
                            "ground_truth_items": {"PHQ8_NoInterest": 2, "PHQ8_Depressed": 1},
                            "item_signals": {
                                "PHQ8_NoInterest": {
                                    "llm_evidence_count": 5,
                                    "keyword_evidence_count": 0,
                                },
                                "PHQ8_Depressed": {
                                    "llm_evidence_count": 1,
                                    "keyword_evidence_count": 2,
                                },
                            },
                        },
                        {
                            "participant_id": 102,
                            "success": True,
                            "predicted_items": {"PHQ8_NoInterest": None},
                            "ground_truth_items": {"PHQ8_NoInterest": 0},
                            "item_signals": {"PHQ8_NoInterest": {"llm_evidence_count": 0}},
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
    """Test that the evaluation CLI runs without error."""

    cmd = [
        sys.executable,
        "scripts/evaluate_selective_prediction.py",
        "--input",
        str(mock_output_json),
        "--bootstrap-resamples",
        "10",
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

    # Validation
    # 2 participants included.
    # P1: 2 items (conf 5, conf 1/3). P2: 1 item (conf 0).
    # Total items? P1 has 2 items listed. P2 has 1 item listed.
    # Wait, the script iterates keys in PHQ8Item.all_items().
    # Our mock only provided a subset of keys.
    # The script should handle missing keys by skipping.

    assert m["population"]["participants_included"] == 2
    # Check if confidence variants present
    assert "llm" in m["confidence_variants"]
    assert "total_evidence" in m["confidence_variants"]
