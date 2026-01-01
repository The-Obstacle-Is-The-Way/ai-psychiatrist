"""Unit tests for DAIC-WOZ transcript preprocessing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from scripts.preprocess_daic_woz_transcripts import (
    apply_variant_filter,
    clean_transcript,
    is_in_interruption_window,
    is_sync_marker,
    main,
    normalize_speaker,
    remove_preamble,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class TestSpeakerNormalization:
    """Tests for speaker normalization."""

    def test_ellie_lowercase_normalized(self) -> None:
        assert normalize_speaker("ellie") == "Ellie"

    def test_ellie_uppercase_normalized(self) -> None:
        assert normalize_speaker("ELLIE") == "Ellie"

    def test_participant_lowercase_normalized(self) -> None:
        assert normalize_speaker("participant") == "Participant"

    def test_participant_mixed_case_normalized(self) -> None:
        assert normalize_speaker("Participant") == "Participant"

    def test_unknown_speaker_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown speaker"):
            normalize_speaker("Unknown")


class TestSyncMarkerDetection:
    """Tests for sync marker detection."""

    @pytest.mark.parametrize(
        "marker",
        [
            "<sync>",
            "<synch>",
            "[sync]",
            "[synch]",
            "[syncing]",
            "[synching]",
            "<SYNC>",
            "[SYNC]",
            "  <sync>  ",
            "[syncing]  ",
        ],
    )
    def test_sync_markers_detected(self, marker: str) -> None:
        assert is_sync_marker(marker) is True

    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "<laughter>",
            "[laughing]",
            "sync",
            "synchronize",
        ],
    )
    def test_non_sync_not_detected(self, text: str) -> None:
        assert is_sync_marker(text) is False


class TestInterruptionWindowDetection:
    """Tests for interruption window overlap detection."""

    def test_373_inside_window(self) -> None:
        """373 window overlaps [395, 428]."""
        assert is_in_interruption_window(373, 400.0, 410.0) is True

    def test_373_before_window(self) -> None:
        assert is_in_interruption_window(373, 100.0, 200.0) is False

    def test_373_after_window(self) -> None:
        assert is_in_interruption_window(373, 500.0, 600.0) is False

    def test_373_overlaps_start(self) -> None:
        assert is_in_interruption_window(373, 390.0, 400.0) is True

    def test_373_overlaps_end(self) -> None:
        assert is_in_interruption_window(373, 425.0, 435.0) is True

    def test_444_inside_window(self) -> None:
        """444 window overlaps [286, 387]."""
        assert is_in_interruption_window(444, 300.0, 350.0) is True

    def test_other_participant_no_window(self) -> None:
        assert is_in_interruption_window(300, 400.0, 410.0) is False


class TestPreambleRemoval:
    """Tests for pre-interview preamble removal."""

    def test_removes_rows_before_first_ellie(self) -> None:
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0, 2.0, 3.0],
                "stop_time": [0.5, 1.5, 2.5, 3.5],
                "speaker": ["Participant", "Participant", "Ellie", "Participant"],
                "value": ["noise", "setup", "hi im ellie", "hello"],
            }
        )
        result = remove_preamble(df)
        assert len(result) == 2
        assert result.iloc[0]["speaker"] == "Ellie"

    def test_no_ellie_known_session_ok(self) -> None:
        """Sessions 451, 458, 480 have no Ellie — should not fail."""
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "speaker": ["Participant", "Participant"],
                "value": ["hello", "world"],
            }
        )
        result = remove_preamble(df, participant_id=451)
        assert len(result) == 2


class TestVariantFilters:
    """Tests for variant-specific filtering."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "stop_time": [0.5, 1.5, 2.5, 3.5, 4.5],
                "speaker": ["Ellie", "Participant", "Ellie", "Participant", "Participant"],
                "value": ["question1", "answer1", "question2", "answer2", "answer3"],
            }
        )

    def test_participant_only(self, sample_df: pd.DataFrame) -> None:
        result = apply_variant_filter(sample_df, "participant_only")
        assert len(result) == 3
        assert all(result["speaker"] == "Participant")

    def test_both_speakers_clean(self, sample_df: pd.DataFrame) -> None:
        result = apply_variant_filter(sample_df, "both_speakers_clean")
        assert len(result) == 5  # All rows kept

    def test_participant_qa_includes_one_prompt_per_block(self) -> None:
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "stop_time": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                "speaker": [
                    "Ellie",
                    "Ellie",
                    "Participant",
                    "Participant",
                    "Ellie",
                    "Participant",
                ],
                "value": ["q1a", "q1b", "a1", "a2", "q2", "a3"],
            }
        )
        result = apply_variant_filter(df, "participant_qa")
        assert list(result["value"]) == ["q1b", "a1", "a2", "q2", "a3"]


class TestFullCleaningPipeline:
    """Integration tests for full cleaning pipeline."""

    def test_sync_markers_removed(self) -> None:
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0, 2.0],
                "stop_time": [0.5, 1.5, 2.5],
                "speaker": ["Ellie", "Participant", "Participant"],
                "value": ["hello", "[sync]", "actual content"],
            }
        )
        result = clean_transcript(df, participant_id=300, variant="both_speakers_clean")
        assert len(result) == 2
        assert "[sync]" not in result["value"].values

    def test_preserves_nonverbal_annotations(self) -> None:
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "speaker": ["Ellie", "Participant"],
                "value": ["hello", "um <laughter> yeah"],
            }
        )
        result = clean_transcript(df, participant_id=300, variant="participant_only")
        assert "<laughter>" in result.iloc[0]["value"]

    def test_preserves_original_case(self) -> None:
        df = pd.DataFrame(
            {
                "start_time": [0.0, 1.0],
                "stop_time": [0.5, 1.5],
                "speaker": ["Ellie", "Participant"],
                "value": ["hello", "I'm REALLY tired"],
            }
        )
        result = clean_transcript(df, participant_id=300, variant="participant_only")
        assert "REALLY" in result.iloc[0]["value"]


class TestCLISafety:
    """Tests for CLI safety constraints and outputs."""

    def test_refuses_same_input_output_dir(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "transcripts"
        input_dir.mkdir()

        exit_code = main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(input_dir),
                "--variant",
                "participant_only",
                "--dry-run",
            ]
        )
        assert exit_code == 1

    def test_atomic_write_on_success(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "transcripts"
        output_dir = tmp_path / "out"
        transcript_dir = input_dir / "300_P"
        transcript_dir.mkdir(parents=True)
        (transcript_dir / "300_TRANSCRIPT.csv").write_text(
            "start_time\tstop_time\tspeaker\tvalue\n0\t1\tEllie\thello\n1\t2\tParticipant\tSECRET_TEXT_SHOULD_NOT_APPEAR\n",
            encoding="utf-8",
        )

        exit_code = main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--variant",
                "participant_only",
                "--overwrite",
            ]
        )
        assert exit_code == 0
        assert (output_dir / "300_P" / "300_TRANSCRIPT.csv").exists()
        assert (output_dir / "preprocess_manifest.json").exists()
        assert not output_dir.with_name(output_dir.name + ".tmp").exists()

    def test_manifest_written_no_transcript_text(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "transcripts"
        output_dir = tmp_path / "out"
        transcript_dir = input_dir / "300_P"
        transcript_dir.mkdir(parents=True)
        secret = "SECRET_TEXT_SHOULD_NOT_APPEAR"
        (transcript_dir / "300_TRANSCRIPT.csv").write_text(
            f"start_time\tstop_time\tspeaker\tvalue\n0\t1\tEllie\thello\n1\t2\tParticipant\t{secret}\n",
            encoding="utf-8",
        )

        exit_code = main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--variant",
                "both_speakers_clean",
                "--overwrite",
            ]
        )
        assert exit_code == 0
        manifest_path = output_dir / "preprocess_manifest.json"
        assert manifest_path.exists()

        manifest_raw = manifest_path.read_text(encoding="utf-8")
        assert secret not in manifest_raw

        payload = json.loads(manifest_raw)
        assert payload["variant"] == "both_speakers_clean"
