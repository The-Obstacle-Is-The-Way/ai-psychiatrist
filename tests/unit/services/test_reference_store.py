"""Tests for ReferenceStore."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from numpy.lib.npyio import NpzFile

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingBackendSettings,
    EmbeddingSettings,
    ModelSettings,
)
from ai_psychiatrist.domain.exceptions import (
    EmbeddingArtifactMismatchError,
)
from ai_psychiatrist.services.reference_store import ReferenceStore


@pytest.fixture
def data_settings(tmp_path: Path) -> DataSettings:
    """Create data settings with temp paths."""
    base = tmp_path / "data"
    base.mkdir()
    return DataSettings(
        base_dir=base,
        transcripts_dir=base / "transcripts",
        embeddings_path=base / "embeddings.npz",
    )


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    """Create embedding settings."""
    return EmbeddingSettings(
        dimension=4096,
        chunk_size=8,
    )


@pytest.fixture
def embedding_backend_settings() -> EmbeddingBackendSettings:
    """Create embedding backend settings."""
    return EmbeddingBackendSettings(backend=EmbeddingBackend.HUGGINGFACE)


class TestReferenceStoreMetadata:
    """Tests for metadata validation in ReferenceStore."""

    def test_validation_success(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should succeed when metadata matches config."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        # Setup files
        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "dimension": 4096,
                    "chunk_size": 8,
                }
            )
        )

        npz_path = data_settings.embeddings_path
        npz_path.touch()
        json_path = data_settings.embeddings_path.with_suffix(".json")
        json_path.write_text("{}")

        # Mock numpy load
        mock_npz = MagicMock(spec_set=NpzFile)
        mock_npz.__getitem__.return_value = []
        mock_npz.__contains__.return_value = True

        with patch("numpy.load", return_value=mock_npz):
            store._load_embeddings()

    def test_invalid_metadata_type_skips_validation(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should not crash if .meta.json is valid JSON but not a dict."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        # Setup files
        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(["not-a-dict"]), encoding="utf-8")

        npz_path = data_settings.embeddings_path
        npz_path.touch()
        json_path = data_settings.embeddings_path.with_suffix(".json")
        json_path.write_text("{}", encoding="utf-8")

        # Mock numpy load
        mock_npz = MagicMock(spec_set=NpzFile)
        mock_npz.__getitem__.return_value = []
        mock_npz.__contains__.return_value = True

        with patch("numpy.load", return_value=mock_npz):
            store._load_embeddings()

    def test_validation_fail_backend(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when backend mismatches."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "ollama",  # Mismatch (expect huggingface)
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="backend mismatch"),
        ):
            store._load_embeddings()

    def test_validation_fail_dimension(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when dimension mismatches."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "dimension": 1024,  # Mismatch (expect 4096)
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="dimension mismatch"),
        ):
            store._load_embeddings()

    def test_validation_fail_chunk_size(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when chunk_size mismatches."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "chunk_size": 16,  # Mismatch (expect 8)
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="chunk_size mismatch"),
        ):
            store._load_embeddings()

    def test_validation_fail_chunk_step(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when chunk_step mismatches."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "chunk_step": 3,  # Mismatch (expect default 2)
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="chunk_step mismatch"),
        ):
            store._load_embeddings()

    def test_validation_fail_model(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when resolved model mismatches."""
        model_settings = ModelSettings(embedding_model="qwen3-embedding:8b")
        store = ReferenceStore(
            data_settings,
            embedding_settings,
            embedding_backend_settings,
            model_settings,
        )

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "model": "some/other-model",  # mismatch vs resolved Qwen/Qwen3-Embedding-8B
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="model mismatch"),
        ):
            store._load_embeddings()

    def test_validation_semantic_success(
        self,
        tmp_path: Path,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should succeed when split_ids_hash matches even if split_csv_hash mismatches."""
        base = tmp_path / "data"
        base.mkdir()
        transcripts_dir = base / "transcripts"
        transcripts_dir.mkdir()

        train_csv = base / "train.csv"
        # Create CSV with IDs 1, 2
        train_csv.write_text("Participant_ID,Gender\n1,M\n2,F\n", encoding="utf-8")

        # Calculate correct IDs hash for "1,2"
        correct_ids_hash = hashlib.sha256(b"1,2").hexdigest()[:12]

        data_settings = DataSettings(
            base_dir=base,
            transcripts_dir=transcripts_dir,
            embeddings_path=base / "embeddings.npz",
            train_csv=train_csv,
            dev_csv=base / "dev.csv",
        )

        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "split": "avec-train",
                    "split_csv_hash": "deadbeef",  # Mismatched CSV hash
                    "split_ids_hash": correct_ids_hash,  # Matching IDs hash
                }
            ),
            encoding="utf-8",
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}", encoding="utf-8")

        mock_npz = MagicMock(spec_set=NpzFile)
        mock_npz.__getitem__.return_value = []
        mock_npz.__contains__.return_value = True

        with patch("numpy.load", return_value=mock_npz):
            # Should NOT raise
            store._load_embeddings()

    def test_validation_fail_ids_mismatch(
        self,
        tmp_path: Path,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when split_ids_hash mismatches."""
        base = tmp_path / "data"
        base.mkdir()
        transcripts_dir = base / "transcripts"
        transcripts_dir.mkdir()

        train_csv = base / "train.csv"
        train_csv.write_text("Participant_ID,Gender\n1,M\n2,F\n", encoding="utf-8")

        data_settings = DataSettings(
            base_dir=base,
            transcripts_dir=transcripts_dir,
            embeddings_path=base / "embeddings.npz",
            train_csv=train_csv,
            dev_csv=base / "dev.csv",
        )

        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "split": "avec-train",
                    "split_ids_hash": "badhash123",  # Mismatched IDs hash
                }
            ),
            encoding="utf-8",
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}", encoding="utf-8")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="split_ids_hash mismatch"),
        ):
            store._load_embeddings()

    def test_validation_legacy_fallback_success(
        self,
        tmp_path: Path,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Legacy artifact: split_csv_hash mismatch but derived IDs match -> Success."""
        base = tmp_path / "data"
        base.mkdir()
        transcripts_dir = base / "transcripts"
        transcripts_dir.mkdir()

        train_csv = base / "train.csv"
        train_csv.write_text("Participant_ID,Gender\n1,M\n", encoding="utf-8")

        data_settings = DataSettings(
            base_dir=base,
            transcripts_dir=transcripts_dir,
            embeddings_path=base / "embeddings.npz",
            train_csv=train_csv,
            dev_csv=base / "dev.csv",
        )

        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        # Meta has mismatched CSV hash and NO split_ids_hash
        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "split": "avec-train",
                    "split_csv_hash": "deadbeef",
                }
            ),
            encoding="utf-8",
        )

        data_settings.embeddings_path.touch()
        # Sidecar has ID "1" which matches the CSV
        data_settings.embeddings_path.with_suffix(".json").write_text(
            '{"1": ["text"]}', encoding="utf-8"
        )

        mock_npz = MagicMock(spec_set=NpzFile)
        mock_npz.__contains__.return_value = True
        mock_npz.__getitem__.return_value = []

        with patch("numpy.load", return_value=mock_npz):
            store._load_embeddings()

    def test_validation_legacy_fallback_fail(
        self,
        tmp_path: Path,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Legacy artifact: split_csv_hash mismatch AND derived IDs mismatch -> Fail."""
        base = tmp_path / "data"
        base.mkdir()
        transcripts_dir = base / "transcripts"
        transcripts_dir.mkdir()

        train_csv = base / "train.csv"
        # CSV has ID 1
        train_csv.write_text("Participant_ID,Gender\n1,M\n", encoding="utf-8")

        data_settings = DataSettings(
            base_dir=base,
            transcripts_dir=transcripts_dir,
            embeddings_path=base / "embeddings.npz",
            train_csv=train_csv,
            dev_csv=base / "dev.csv",
        )

        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "split": "avec-train",
                    "split_csv_hash": "deadbeef",
                }
            ),
            encoding="utf-8",
        )

        data_settings.embeddings_path.touch()
        # Sidecar has ID "2" which DOES NOT match CSV ID "1"
        data_settings.embeddings_path.with_suffix(".json").write_text(
            '{"2": ["text"]}', encoding="utf-8"
        )

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="split_ids mismatch"),
        ):
            store._load_embeddings()

    def test_validation_fail_min_evidence_chars(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should fail when min_evidence_chars mismatches."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        meta_path = data_settings.embeddings_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "huggingface",
                    "min_evidence_chars": 9,  # mismatch vs default 8
                }
            )
        )

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="min_evidence_chars mismatch"),
        ):
            store._load_embeddings()

    def test_legacy_file_skip_validation(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
        embedding_backend_settings: EmbeddingBackendSettings,
    ) -> None:
        """Should skip validation if .meta.json is missing."""
        store = ReferenceStore(data_settings, embedding_settings, embedding_backend_settings)

        data_settings.embeddings_path.touch()
        data_settings.embeddings_path.with_suffix(".json").write_text("{}")

        mock_npz = MagicMock(spec_set=NpzFile)
        mock_npz.__getitem__.return_value = []
        mock_npz.__contains__.return_value = True

        with patch("numpy.load", return_value=mock_npz):
            store._load_embeddings()
