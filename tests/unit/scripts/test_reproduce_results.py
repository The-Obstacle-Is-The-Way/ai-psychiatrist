"""Unit tests for scripts/reproduce_results.py helpers."""

from __future__ import annotations

import argparse
import importlib.util
from typing import TYPE_CHECKING, Any, cast

import pytest
from scripts.reproduce_results import init_embedding_service, print_run_configuration

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingBackendSettings,
    EmbeddingSettings,
    ModelSettings,
    OllamaSettings,
    Settings,
)
from ai_psychiatrist.infrastructure.llm.huggingface import MissingHuggingFaceDependenciesError

if TYPE_CHECKING:
    from pathlib import Path

    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

pytestmark = pytest.mark.unit


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


def test_init_embedding_service_fails_fast_when_hf_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()

    def fake_find_spec(module: str, *_args: Any, **_kwargs: Any) -> object | None:
        if module == "torch":
            return None
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    args = argparse.Namespace(zero_shot_only=False, split="paper-test")

    with pytest.raises(MissingHuggingFaceDependenciesError, match="make dev-hf"):
        init_embedding_service(
            args=args,
            data_settings=DataSettings(base_dir=base_dir, transcripts_dir=transcripts_dir),
            embedding_backend_settings=EmbeddingBackendSettings(
                backend=EmbeddingBackend.HUGGINGFACE
            ),
            embedding_settings=EmbeddingSettings(
                embeddings_file="huggingface_qwen3_8b_paper_train_participant_only"
            ),
            model_settings=ModelSettings(),
            embedding_client=cast("EmbeddingClient", object()),
            chat_client=cast("OllamaClient", object()),
        )
