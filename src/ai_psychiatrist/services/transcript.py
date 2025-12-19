"""Transcript loading and management service.

Handles DAIC-WOZ dataset format for interview transcript loading
and provides clean interfaces for agents.

Paper Reference:
- Section 2.1: DAIC-WOZ dataset structure
- Section 2.4.2: Transcript chunking (N_chunk=8, step=2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.exceptions import EmptyTranscriptError, TranscriptError
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ai_psychiatrist.config import DataSettings

logger = get_logger(__name__)


class TranscriptService:
    """Service for loading and managing interview transcripts.

    Loads DAIC-WOZ format transcripts (tab-separated CSV with speaker/value columns)
    and provides methods for transcript access and management.
    """

    def __init__(self, data_settings: DataSettings) -> None:
        """Initialize transcript service.

        Args:
            data_settings: Data path configuration.
        """
        self._transcripts_dir = data_settings.transcripts_dir

    def load_transcript(self, participant_id: int) -> Transcript:
        """Load transcript for a specific participant.

        Args:
            participant_id: DAIC-WOZ participant ID.

        Returns:
            Loaded transcript entity.

        Raises:
            TranscriptError: If transcript cannot be loaded.
            EmptyTranscriptError: If transcript is empty.
        """
        transcript_path = self._get_transcript_path(participant_id)

        logger.info(
            "Loading transcript",
            participant_id=participant_id,
            path=str(transcript_path),
        )

        if not transcript_path.exists():
            raise TranscriptError(f"Transcript not found: {transcript_path}")

        try:
            text = self._parse_daic_woz_transcript(transcript_path)
        except Exception as e:
            logger.error(
                "Failed to parse transcript",
                participant_id=participant_id,
                error=str(e),
            )
            raise TranscriptError(f"Failed to parse transcript: {e}") from e

        if not text.strip():
            raise EmptyTranscriptError(f"Empty transcript for participant {participant_id}")

        logger.info(
            "Transcript loaded",
            participant_id=participant_id,
            word_count=len(text.split()),
            line_count=len(text.splitlines()),
        )

        return Transcript(participant_id=participant_id, text=text)

    def load_transcript_from_text(self, participant_id: int, text: str) -> Transcript:
        """Create transcript entity from raw text.

        Args:
            participant_id: Participant identifier.
            text: Raw transcript text.

        Returns:
            Transcript entity.

        Raises:
            EmptyTranscriptError: If text is empty.
        """
        if not text.strip():
            raise EmptyTranscriptError("Transcript text cannot be empty")

        return Transcript(participant_id=participant_id, text=text)

    def list_available_participants(self) -> list[int]:
        """List all participant IDs with available transcripts.

        Returns:
            Sorted list of participant IDs.
        """
        if not self._transcripts_dir.exists():
            logger.warning(
                "Transcripts directory not found",
                path=str(self._transcripts_dir),
            )
            return []

        participant_ids = []
        for item in self._transcripts_dir.iterdir():
            if item.is_dir() and item.name.endswith("_P"):
                try:
                    pid = int(item.name.replace("_P", ""))
                    participant_ids.append(pid)
                except ValueError:
                    continue

        return sorted(participant_ids)

    def _get_transcript_path(self, participant_id: int) -> Path:
        """Get path to transcript file for participant."""
        return (
            self._transcripts_dir
            / f"{participant_id}_P"
            / f"{participant_id}_TRANSCRIPT.csv"
        )

    def _parse_daic_woz_transcript(self, path: Path) -> str:
        """Parse DAIC-WOZ transcript CSV format.

        Format: tab-separated with columns including 'speaker' and 'value'.

        Args:
            path: Path to the transcript CSV file.

        Returns:
            Formatted transcript text with 'speaker: value' per line.
        """
        df = pd.read_csv(path, sep="\t")

        # Filter and format dialogue
        df = df.dropna(subset=["speaker", "value"])
        df["dialogue"] = df["speaker"] + ": " + df["value"]

        return "\n".join(df["dialogue"].tolist())
