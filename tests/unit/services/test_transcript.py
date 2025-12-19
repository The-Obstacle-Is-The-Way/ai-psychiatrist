"""Tests for transcript service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.exceptions import EmptyTranscriptError, TranscriptError
from ai_psychiatrist.services.transcript import TranscriptService


class MockDataSettings:
    """Mock data settings for testing."""

    def __init__(self, transcripts_dir: Path | None = None) -> None:
        self.transcripts_dir = transcripts_dir or Path("/tmp/nonexistent")
        self.train_csv = Path("/tmp/nonexistent.csv")
        self.dev_csv = Path("/tmp/nonexistent.csv")


class TestTranscriptService:
    """Tests for transcript service."""

    def test_load_from_text_valid(self) -> None:
        """Should create transcript from raw text."""
        service = TranscriptService(data_settings=MockDataSettings())
        transcript = service.load_transcript_from_text(123, "Hello world")

        assert transcript.participant_id == 123
        assert transcript.text == "Hello world"
        assert isinstance(transcript, Transcript)

    def test_load_from_text_multiline(self) -> None:
        """Should handle multiline transcript text."""
        service = TranscriptService(data_settings=MockDataSettings())
        text = "Ellie: How are you?\nParticipant: I'm fine."
        transcript = service.load_transcript_from_text(456, text)

        assert transcript.participant_id == 456
        assert transcript.line_count == 2
        assert "Ellie:" in transcript.text
        assert "Participant:" in transcript.text

    def test_reject_empty_text(self) -> None:
        """Should reject empty transcript text."""
        service = TranscriptService(data_settings=MockDataSettings())

        with pytest.raises(EmptyTranscriptError):
            service.load_transcript_from_text(123, "")

    def test_reject_whitespace_only_text(self) -> None:
        """Should reject whitespace-only transcript text."""
        service = TranscriptService(data_settings=MockDataSettings())

        with pytest.raises(EmptyTranscriptError):
            service.load_transcript_from_text(123, "   \n\t  ")

    def test_load_transcript_not_found(self, tmp_path: Path) -> None:
        """Should raise TranscriptError when transcript file not found."""
        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)

        with pytest.raises(TranscriptError, match="not found"):
            service.load_transcript(999)

    def test_load_transcript_from_csv(self, tmp_path: Path) -> None:
        """Should load transcript from DAIC-WOZ CSV format."""
        # Create directory structure
        participant_dir = tmp_path / "300_P"
        participant_dir.mkdir()

        # Create mock transcript CSV
        transcript_path = participant_dir / "300_TRANSCRIPT.csv"
        df = pd.DataFrame({
            "speaker": ["Ellie", "Participant", "Ellie"],
            "value": ["How are you?", "I'm doing okay.", "That's good."],
        })
        df.to_csv(transcript_path, sep="\t", index=False)

        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)
        transcript = service.load_transcript(300)

        assert transcript.participant_id == 300
        assert "Ellie: How are you?" in transcript.text
        assert "Participant: I'm doing okay." in transcript.text
        assert transcript.line_count == 3

    def test_load_transcript_handles_missing_columns(self, tmp_path: Path) -> None:
        """Should handle CSV with missing speaker/value entries."""
        participant_dir = tmp_path / "301_P"
        participant_dir.mkdir()

        transcript_path = participant_dir / "301_TRANSCRIPT.csv"
        df = pd.DataFrame({
            "speaker": ["Ellie", None, "Participant"],
            "value": ["Hello", "invalid", None],
        })
        df.to_csv(transcript_path, sep="\t", index=False)

        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)
        transcript = service.load_transcript(301)

        # Should only have 1 valid line (Ellie: Hello)
        assert transcript.line_count == 1
        assert "Ellie: Hello" in transcript.text

    def test_list_available_participants(self, tmp_path: Path) -> None:
        """Should list all participant IDs from directory structure."""
        # Create participant directories
        (tmp_path / "300_P").mkdir()
        (tmp_path / "301_P").mkdir()
        (tmp_path / "302_P").mkdir()
        (tmp_path / "invalid_dir").mkdir()  # Should be ignored

        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)
        participants = service.list_available_participants()

        assert participants == [300, 301, 302]

    def test_list_participants_empty_directory(self, tmp_path: Path) -> None:
        """Should return empty list for empty directory."""
        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)
        participants = service.list_available_participants()

        assert participants == []

    def test_list_participants_nonexistent_directory(self) -> None:
        """Should return empty list for nonexistent directory."""
        settings = MockDataSettings(transcripts_dir=Path("/nonexistent"))
        service = TranscriptService(data_settings=settings)
        participants = service.list_available_participants()

        assert participants == []

    def test_get_transcript_path(self, tmp_path: Path) -> None:
        """Should construct correct path for participant."""
        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)

        path = service._get_transcript_path(300)
        expected = tmp_path / "300_P" / "300_TRANSCRIPT.csv"

        assert path == expected

    def test_parse_empty_csv(self, tmp_path: Path) -> None:
        """Should raise EmptyTranscriptError for empty CSV."""
        participant_dir = tmp_path / "302_P"
        participant_dir.mkdir()

        transcript_path = participant_dir / "302_TRANSCRIPT.csv"
        df = pd.DataFrame({"speaker": [], "value": []})
        df.to_csv(transcript_path, sep="\t", index=False)

        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)

        with pytest.raises(EmptyTranscriptError):
            service.load_transcript(302)

    def test_parse_malformed_csv(self, tmp_path: Path) -> None:
        """Should raise TranscriptError for malformed CSV."""
        participant_dir = tmp_path / "303_P"
        participant_dir.mkdir()

        transcript_path = participant_dir / "303_TRANSCRIPT.csv"
        # Write invalid content
        transcript_path.write_text("not,a,valid,csv\nno\ttabs\there")

        settings = MockDataSettings(transcripts_dir=tmp_path)
        service = TranscriptService(data_settings=settings)

        with pytest.raises(TranscriptError, match="Failed to parse"):
            service.load_transcript(303)
