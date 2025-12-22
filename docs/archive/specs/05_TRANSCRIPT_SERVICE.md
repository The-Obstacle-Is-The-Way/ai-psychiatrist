# Spec 05: Transcript Service

## Objective

Create a robust transcript loading and chunking service that handles DAIC-WOZ dataset format and provides clean interfaces for agents.

## Paper Reference

- **Section 2.1**: DAIC-WOZ dataset structure
- **Section 2.4.2**: Transcript chunking (N_chunk=8, step=2)

## As-Is Transcript Loading (Repo)

There are two transcript ingestion paths in the current repo:

### 1) Demo Pipeline (FastAPI)

- `agents/interview_simulator.py` loads a **single fixed transcript text file**.
- Path is controlled by `TRANSCRIPT_PATH` (env var) or defaults to `agents/transcript.txt`.
- `server.py` always uses this loader; the API does not accept transcript text today.

### 2) Research / DAIC-WOZ Scripts + Notebooks

- Scripts and notebooks read DAIC-WOZ transcripts from:
  `.../{participant_id}_P/{participant_id}_TRANSCRIPT.csv` (tab-separated)
- They typically join to a single string as either:
  - `"speaker : value"` (note spaces) or
  - `"speaker: value"` (no spaces)
- The chunking implementation used for embeddings is in:
  - `quantitative_assessment/embedding_batch_script.py:create_sliding_chunks(...)`
  - `quantitative_assessment/embedding_quantitative_analysis.ipynb` (same logic)

## Deliverables

1. `src/ai_psychiatrist/services/transcript.py` - Transcript loading service
2. `src/ai_psychiatrist/services/chunking.py` - Transcript chunking utilities
3. `tests/unit/services/test_transcript.py` - Comprehensive tests

## Implementation

### 1. Transcript Service (transcript.py)

```python
"""Transcript loading and management service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.exceptions import EmptyTranscriptError, TranscriptError
from ai_psychiatrist.domain.value_objects import TranscriptChunk
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import DataSettings

logger = get_logger(__name__)


class TranscriptService:
    """Service for loading and managing interview transcripts."""

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

        logger.info("Loading transcript", participant_id=participant_id, path=str(transcript_path))

        if not transcript_path.exists():
            raise TranscriptError(f"Transcript not found: {transcript_path}")

        try:
            text = self._parse_daic_woz_transcript(transcript_path)
        except Exception as e:
            logger.error("Failed to parse transcript", participant_id=participant_id, error=str(e))
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
            logger.warning("Transcripts directory not found", path=str(self._transcripts_dir))
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
        """
        df = pd.read_csv(path, sep="\t")

        # Filter and format dialogue
        df = df.dropna(subset=["speaker", "value"])
        df["dialogue"] = df["speaker"] + ": " + df["value"]

        return "\n".join(df["dialogue"].tolist())


class TranscriptChunker:
    """Utility for chunking transcripts for embedding."""

    def __init__(self, chunk_size: int = 8, step_size: int = 2) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Number of lines per chunk.
            step_size: Sliding window step.
        """
        if chunk_size < 2:
            raise ValueError("Chunk size must be at least 2")
        if step_size < 1:
            raise ValueError("Step size must be at least 1")
        if step_size > chunk_size:
            raise ValueError("Step size cannot exceed chunk size")

        self._chunk_size = chunk_size
        self._step_size = step_size

    def chunk_transcript(self, transcript: Transcript) -> list[TranscriptChunk]:
        """Split transcript into overlapping chunks.

        Args:
            transcript: Transcript to chunk.

        Returns:
            List of transcript chunks.
        """
        lines = transcript.text.strip().splitlines()
        chunks = []

        for i in range(0, len(lines), self._step_size):
            chunk_lines = lines[i : i + self._chunk_size]
            if not chunk_lines:
                break

            chunk_text = "\n".join(chunk_lines)
            if chunk_text.strip():
                chunks.append(
                    TranscriptChunk(
                        text=chunk_text,
                        participant_id=transcript.participant_id,
                        line_start=i,
                        line_end=i + len(chunk_lines) - 1,
                    )
                )

        logger.debug(
            "Chunked transcript",
            participant_id=transcript.participant_id,
            total_lines=len(lines),
            chunk_count=len(chunks),
            chunk_size=self._chunk_size,
            step_size=self._step_size,
        )

        return chunks

    def chunk_text(self, text: str, participant_id: int) -> list[TranscriptChunk]:
        """Chunk raw text directly.

        Args:
            text: Raw transcript text.
            participant_id: Participant identifier.

        Returns:
            List of transcript chunks.
        """
        transcript = Transcript(participant_id=participant_id, text=text)
        return self.chunk_transcript(transcript)
```

### 2. Ground Truth Service

```python
"""Ground truth data loading service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import DataSettings

logger = get_logger(__name__)


class GroundTruthService:
    """Service for loading PHQ-8 ground truth scores."""

    # Column mapping from CSV to PHQ8Item
    COLUMN_MAPPING = {
        "PHQ8_NoInterest": PHQ8Item.NO_INTEREST,
        "PHQ8_Depressed": PHQ8Item.DEPRESSED,
        "PHQ8_Sleep": PHQ8Item.SLEEP,
        "PHQ8_Tired": PHQ8Item.TIRED,
        "PHQ8_Appetite": PHQ8Item.APPETITE,
        "PHQ8_Failure": PHQ8Item.FAILURE,
        "PHQ8_Concentrating": PHQ8Item.CONCENTRATING,
        "PHQ8_Moving": PHQ8Item.MOVING,
    }

    def __init__(self, data_settings: DataSettings) -> None:
        """Initialize ground truth service.

        Args:
            data_settings: Data path configuration.
        """
        self._train_csv = data_settings.train_csv
        self._dev_csv = data_settings.dev_csv
        self._df: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        """Load and combine train/dev ground truth data."""
        if self._df is not None:
            return self._df

        dfs = []
        for path in [self._train_csv, self._dev_csv]:
            if path.exists():
                df = pd.read_csv(path)
                df["Participant_ID"] = df["Participant_ID"].astype(int)
                dfs.append(df)
                logger.debug("Loaded ground truth", path=str(path), count=len(df))
            else:
                logger.warning("Ground truth file not found", path=str(path))

        if not dfs:
            logger.error("No ground truth data loaded")
            return pd.DataFrame()

        self._df = pd.concat(dfs, ignore_index=True)
        self._df = self._df.sort_values("Participant_ID").reset_index(drop=True)

        logger.info("Ground truth loaded", total_participants=len(self._df))
        return self._df

    def get_scores(self, participant_id: int) -> dict[PHQ8Item, int | None]:
        """Get PHQ-8 scores for a participant.

        Args:
            participant_id: Participant ID.

        Returns:
            Dictionary mapping PHQ8Item to score (0-3) or None if unavailable.
        """
        df = self._load_data()
        row = df[df["Participant_ID"] == participant_id]

        if row.empty:
            logger.warning("No ground truth for participant", participant_id=participant_id)
            return {item: None for item in PHQ8Item}

        scores = {}
        for col, item in self.COLUMN_MAPPING.items():
            if col in row.columns:
                val = row[col].iloc[0]
                try:
                    scores[item] = int(val)
                except (ValueError, TypeError):
                    scores[item] = None
            else:
                scores[item] = None

        return scores

    def get_total_score(self, participant_id: int) -> int | None:
        """Get total PHQ-8 score for a participant.

        Args:
            participant_id: Participant ID.

        Returns:
            Total score (0-24) or None if unavailable.
        """
        df = self._load_data()
        row = df[df["Participant_ID"] == participant_id]

        if row.empty:
            return None

        if "PHQ8_Score" in row.columns:
            try:
                return int(row["PHQ8_Score"].iloc[0])
            except (ValueError, TypeError):
                pass

        # Calculate from items
        scores = self.get_scores(participant_id)
        if all(s is not None for s in scores.values()):
            return sum(s for s in scores.values() if s is not None)

        return None

    def list_participants(self) -> list[int]:
        """List all participant IDs with ground truth data."""
        df = self._load_data()
        return df["Participant_ID"].tolist()
```

### 3. Tests

```python
"""Tests for transcript service."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.exceptions import EmptyTranscriptError, TranscriptError
from ai_psychiatrist.services.transcript import TranscriptChunker, TranscriptService


class TestTranscriptChunker:
    """Tests for transcript chunking."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript with 10 lines."""
        text = "\n".join(f"Line {i}" for i in range(10))
        return Transcript(participant_id=123, text=text)

    def test_chunk_basic(self, sample_transcript: Transcript) -> None:
        """Should create correct number of chunks."""
        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        chunks = chunker.chunk_transcript(sample_transcript)

        # With 10 lines, step 2, we get chunks starting at 0, 2, 4, 6, 8
        assert len(chunks) == 5
        assert all(c.participant_id == 123 for c in chunks)

    def test_chunk_content(self, sample_transcript: Transcript) -> None:
        """Chunks should contain correct lines."""
        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        chunks = chunker.chunk_transcript(sample_transcript)

        # First chunk: lines 0-3
        assert "Line 0" in chunks[0].text
        assert "Line 3" in chunks[0].text
        assert chunks[0].line_start == 0
        assert chunks[0].line_end == 3

        # Second chunk: lines 2-5
        assert "Line 2" in chunks[1].text
        assert "Line 5" in chunks[1].text
        assert chunks[1].line_start == 2

    def test_chunk_overlap(self, sample_transcript: Transcript) -> None:
        """Chunks should overlap correctly."""
        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        chunks = chunker.chunk_transcript(sample_transcript)

        # Check overlap between first two chunks
        first_lines = set(chunks[0].text.splitlines())
        second_lines = set(chunks[1].text.splitlines())
        overlap = first_lines & second_lines

        # With chunk_size=4 and step=2, should have 2 overlapping lines
        assert len(overlap) == 2

    def test_invalid_chunk_size(self) -> None:
        """Should reject invalid chunk size."""
        with pytest.raises(ValueError, match="at least 2"):
            TranscriptChunker(chunk_size=1, step_size=1)

    def test_invalid_step_size(self) -> None:
        """Should reject step size > chunk size."""
        with pytest.raises(ValueError, match="cannot exceed"):
            TranscriptChunker(chunk_size=4, step_size=5)


class TestTranscriptService:
    """Tests for transcript service."""

    def test_load_from_text(self) -> None:
        """Should create transcript from raw text."""
        service = TranscriptService(data_settings=MockDataSettings())
        transcript = service.load_transcript_from_text(123, "Hello world")

        assert transcript.participant_id == 123
        assert transcript.text == "Hello world"

    def test_reject_empty_text(self) -> None:
        """Should reject empty transcript text."""
        service = TranscriptService(data_settings=MockDataSettings())

        with pytest.raises(EmptyTranscriptError):
            service.load_transcript_from_text(123, "   ")


# Helper for tests
class MockDataSettings:
    """Mock data settings for testing."""

    def __init__(self) -> None:
        self.transcripts_dir = Path("/tmp/nonexistent")
        self.train_csv = Path("/tmp/nonexistent.csv")
        self.dev_csv = Path("/tmp/nonexistent.csv")
```

## Acceptance Criteria

- [ ] Loads DAIC-WOZ transcript CSV format correctly
- [ ] Handles missing transcripts with clear errors
- [ ] Chunks use configurable size and step (paper: 8, 2)
- [ ] Chunk overlap is correct
- [ ] Ground truth scores loaded from CSV
- [ ] All participant IDs are integers
- [ ] Comprehensive error handling

## Dependencies

- **Spec 01**: Project structure
- **Spec 02**: Domain entities (Transcript, TranscriptChunk)
- **Spec 03**: Configuration and logging

## Specs That Depend on This

- **Spec 08**: Embedding Service
- **Spec 09**: Quantitative Agent
- **Spec 11**: Full Pipeline
