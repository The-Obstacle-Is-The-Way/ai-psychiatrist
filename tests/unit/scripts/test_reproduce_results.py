"""Unit tests for scripts/reproduce_results.py helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.reproduce_results import print_run_configuration

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingSettings,
    ModelSettings,
    OllamaSettings,
    Settings,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_print_run_configuration_displays_embedding_settings(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()
    embeddings_dir = base_dir / "embeddings"
    embeddings_dir.mkdir()

    embeddings_path = embeddings_dir / "embeddings.npz"
    tags_path = embeddings_path.with_suffix(".tags.json")
    tags_path.write_text("{}", encoding="utf-8")

    settings = Settings(
        enable_few_shot=False,
        data=DataSettings(
            base_dir=base_dir,
            transcripts_dir=transcripts_dir,
            embeddings_path=embeddings_path,
            train_csv=base_dir / "train.csv",
            dev_csv=base_dir / "dev.csv",
        ),
        model=ModelSettings(
            quantitative_model="gemma3:27b-it-qat",
            embedding_model="qwen3-embedding:8b",
        ),
        ollama=OllamaSettings(host="127.0.0.1", port=11434),
        embedding=EmbeddingSettings(
            enable_item_tag_filter=True,
            enable_retrieval_audit=True,
            min_reference_similarity=0.5,
            max_reference_chars_per_item=2000,
        ),
    )

    print_run_configuration(settings=settings, split="paper-test")

    output = capsys.readouterr().out
    assert f"Tags Sidecar: {tags_path} (FOUND)" in output
    assert "Item Tag Filter: True" in output
    assert "Retrieval Audit: True" in output
    assert "Min Reference Similarity: 0.5" in output
    assert "Max Reference Chars Per Item: 2000" in output
