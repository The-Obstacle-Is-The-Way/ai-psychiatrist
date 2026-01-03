"""Unit tests for experiment tracking utilities."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from ai_psychiatrist.services.experiment_tracking import (
    ExperimentProvenance,
    RunMetadata,
    compute_file_checksum,
    generate_output_filename,
    get_git_info,
    update_experiment_registry,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:Data directory does not exist.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Few-shot enabled but embeddings not found.*:UserWarning"),
]


def test_generate_output_filename() -> None:
    ts = datetime(2025, 12, 25, 14, 30, 22)
    name = generate_output_filename(mode="few_shot", split="paper-test", timestamp=ts)
    assert name == "few_shot_paper-test_20251225_143022.json"


def test_compute_file_checksum(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"test content")

    checksum = compute_file_checksum(path)
    assert checksum is not None
    assert len(checksum) == 16

    missing = compute_file_checksum(tmp_path / "missing.bin")
    assert missing is None


def test_get_git_info_returns_tuple() -> None:
    commit, dirty = get_git_info()
    assert isinstance(commit, str)
    assert isinstance(dirty, bool)


def test_update_experiment_registry_writes_yaml(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    output_file = tmp_path / "out.json"

    run_metadata = RunMetadata(
        run_id="a1b2c3d4",
        timestamp="2025-12-25T14:30:00Z",
        git_commit="abc1234",
        git_dirty=False,
        python_version="3.13.0",
        platform="darwin-arm64",
        ollama_base_url="http://localhost:11434",
    )

    experiments: list[dict[str, object]] = [
        {
            "provenance": ExperimentProvenance(
                mode="zero_shot",
                split="dev",
                quantitative_model="gemma3:27b",
                embedding_model="qwen3-embedding:8b",
                llm_backend="ollama",
                embedding_backend="huggingface",
                embeddings_path=None,
                embeddings_checksum=None,
                embeddings_meta_checksum=None,
                participants_requested=10,
                participants_evaluated=5,
            ).to_dict(),
            "results": {
                "prediction_coverage": 0.5,
                "item_mae_weighted": 0.123,
            },
        }
    ]

    update_experiment_registry(
        run_metadata=run_metadata,
        experiments=experiments,
        output_file=output_file,
        registry_path=registry_path,
    )

    assert registry_path.exists()
    content = registry_path.read_text()
    assert "runs:" in content
    assert "a1b2c3d4:" in content
