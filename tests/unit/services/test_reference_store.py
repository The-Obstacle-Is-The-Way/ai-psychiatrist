"""Tests for ReferenceStore."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingBackendSettings,
    EmbeddingSettings,
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
        mock_npz = MagicMock()
        mock_npz.__getitem__.return_value = []
        mock_npz.close = MagicMock()
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

        mock_npz = MagicMock()
        mock_npz.close = MagicMock()

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

        mock_npz = MagicMock()
        mock_npz.close = MagicMock()

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

        mock_npz = MagicMock()
        mock_npz.close = MagicMock()

        with (
            patch("numpy.load", return_value=mock_npz),
            pytest.raises(EmbeddingArtifactMismatchError, match="chunk_size mismatch"),
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

        mock_npz = MagicMock()
        mock_npz.__getitem__.return_value = []
        mock_npz.close = MagicMock()
        mock_npz.__contains__.return_value = True

        with patch("numpy.load", return_value=mock_npz):
            store._load_embeddings()
