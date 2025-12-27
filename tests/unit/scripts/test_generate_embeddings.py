"""Unit tests for scripts/generate_embeddings.py helpers."""

import hashlib
from pathlib import Path

from scripts.generate_embeddings import (
    calculate_split_hash,
    calculate_split_ids_hash,
    get_output_filename,
    slugify_model,
)

from ai_psychiatrist.config import DataSettings


def _make_data_settings(tmp_path: Path) -> DataSettings:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()
    return DataSettings(
        base_dir=base_dir,
        transcripts_dir=transcripts_dir,
        embeddings_path=base_dir / "embeddings.npz",
        train_csv=base_dir / "train.csv",
        dev_csv=base_dir / "dev.csv",
    )


def test_slugify_model_normalizes_common_patterns() -> None:
    assert slugify_model("Qwen/Qwen3-Embedding-8B") == "qwen3_8b"
    assert slugify_model("qwen3-embedding:8b") == "qwen3_8b"
    assert slugify_model("qwen3-embedding:8b-it-q8_0") == "qwen3_8b_it_q8_0"


def test_get_output_filename_is_deterministic() -> None:
    assert (
        get_output_filename(backend="huggingface", model="qwen3-embedding:8b", split="paper-train")
        == "huggingface_qwen3_8b_paper_train"
    )


def test_calculate_split_hash_paper_train(tmp_path: Path) -> None:
    data_settings = _make_data_settings(tmp_path)
    paper_splits_dir = data_settings.base_dir / "paper_splits"
    paper_splits_dir.mkdir()
    csv_path = paper_splits_dir / "paper_split_train.csv"
    csv_bytes = b"Participant_ID,Gender\n2,F\n1,M\n"
    csv_path.write_bytes(csv_bytes)

    expected = hashlib.sha256(csv_bytes).hexdigest()[:12]
    assert calculate_split_hash(data_settings, "paper-train") == expected


def test_calculate_split_ids_hash_paper_train(tmp_path: Path) -> None:
    data_settings = _make_data_settings(tmp_path)
    paper_splits_dir = data_settings.base_dir / "paper_splits"
    paper_splits_dir.mkdir()
    csv_path = paper_splits_dir / "paper_split_train.csv"
    csv_path.write_text("Participant_ID,Gender\n2,F\n1,M\n", encoding="utf-8")

    expected = hashlib.sha256(b"1,2").hexdigest()[:12]
    assert calculate_split_ids_hash(data_settings, "paper-train") == expected


def test_calculate_split_ids_hash_missing_participant_id_column(tmp_path: Path) -> None:
    data_settings = _make_data_settings(tmp_path)
    paper_splits_dir = data_settings.base_dir / "paper_splits"
    paper_splits_dir.mkdir()
    csv_path = paper_splits_dir / "paper_split_train.csv"
    csv_path.write_text("NotParticipant_ID\n1\n2\n", encoding="utf-8")

    assert calculate_split_ids_hash(data_settings, "paper-train") == "error"


def test_calculate_split_ids_hash_missing_file(tmp_path: Path) -> None:
    data_settings = _make_data_settings(tmp_path)
    assert calculate_split_ids_hash(data_settings, "paper-train") == "missing"


def test_calculate_split_ids_hash_unknown_split(tmp_path: Path) -> None:
    data_settings = _make_data_settings(tmp_path)
    assert calculate_split_ids_hash(data_settings, "not-a-split") == "unknown"
