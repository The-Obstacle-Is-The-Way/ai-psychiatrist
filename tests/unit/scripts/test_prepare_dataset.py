"""Tests for dataset preparation script."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


def _load_prepare_dataset_module() -> object:
    """Load scripts/prepare_dataset.py as a module for testing."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load prepare_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDatasetPreparation:
    """Tests for dataset preparation."""

    def test_transcript_path_format(self) -> None:
        """Transcript paths should follow expected format."""
        # Expected path structure
        expected_dir = Path("data/transcripts/300_P")
        expected_file = expected_dir / "300_TRANSCRIPT.csv"

        assert expected_dir.name == "300_P"
        assert expected_file.name == "300_TRANSCRIPT.csv"

    def test_extract_transcripts_idempotent_with_audio(self, tmp_path: Path) -> None:
        """Extraction is idempotent and can add audio later."""
        module = _load_prepare_dataset_module()

        downloads_dir = tmp_path / "downloads"
        participants_dir = downloads_dir / "participants"
        participants_dir.mkdir(parents=True)

        zip_path = participants_dir / "300_P.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "300_TRANSCRIPT.csv",
                "start_time\tstop_time\tspeaker\tvalue\n0\t1\tParticipant\thello\n",
            )
            zf.writestr("300_AUDIO.wav", b"RIFF0000")
            zf.writestr("300_CLNF_hog.txt", "ignore me")

        output_dir = tmp_path / "data"

        stats = module.extract_transcripts(downloads_dir, output_dir, include_audio=False)
        assert stats["extracted"] == 1
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert (output_dir / "transcripts/300_P/300_TRANSCRIPT.csv").exists()
        assert not (output_dir / "audio/300_AUDIO.wav").exists()

        stats = module.extract_transcripts(downloads_dir, output_dir, include_audio=True)
        assert stats["extracted"] == 1
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert (output_dir / "audio/300_AUDIO.wav").exists()

        stats = module.extract_transcripts(downloads_dir, output_dir, include_audio=True)
        assert stats["extracted"] == 0
        assert stats["skipped"] == 1

    def test_copy_split_csvs_includes_full_test(self, tmp_path: Path) -> None:
        """Copying split CSVs includes full_test_split.csv when present."""
        module = _load_prepare_dataset_module()

        downloads_dir = tmp_path / "downloads"
        downloads_dir.mkdir(parents=True)
        output_dir = tmp_path / "data"
        output_dir.mkdir(parents=True)

        for name in [
            "train_split_Depression_AVEC2017.csv",
            "dev_split_Depression_AVEC2017.csv",
            "test_split_Depression_AVEC2017.csv",
            "full_test_split.csv",
        ]:
            (downloads_dir / name).write_text("Participant_ID\n300\n")

        copied = module.copy_split_csvs(downloads_dir, output_dir)
        assert copied == 4
        assert (output_dir / "full_test_split.csv").exists()

    def test_validate_dataset_handles_column_variants(self, tmp_path: Path) -> None:
        """Validation accepts Participant_ID column variants."""
        module = _load_prepare_dataset_module()

        if not module.HAS_PANDAS:
            pytest.skip("pandas not available")

        output_dir = tmp_path / "data"
        transcript_dir = output_dir / "transcripts/300_P"
        transcript_dir.mkdir(parents=True)
        (transcript_dir / "300_TRANSCRIPT.csv").write_text(
            "start_time\tstop_time\tspeaker\tvalue\n0\t1\tParticipant\thello\n"
        )

        (output_dir / "train_split_Depression_AVEC2017.csv").write_text("participant_ID\n300\n")

        results = module.validate_dataset(output_dir)
        assert results["valid"] is True
        assert results["train_count"] == 1
        assert results["missing_transcripts"] == []
