"""Transcript chunking utilities.

Provides sliding window chunking for transcripts, used for embedding-based
retrieval in few-shot prompting.

Paper Reference:
- Section 2.4.2: Transcript chunking (N_chunk=8, step=2)
- Appendix D: Optimal hyperparameters (chunk_size=8, step_size=2)
"""

from __future__ import annotations

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.value_objects import TranscriptChunk
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


class TranscriptChunker:
    """Utility for chunking transcripts for embedding.

    Uses a sliding window approach to create overlapping chunks
    from interview transcripts. This enables semantic similarity
    search for few-shot example retrieval.
    """

    def __init__(self, chunk_size: int = 8, step_size: int = 2) -> None:
        """Initialize chunker.

        Args:
            chunk_size: Number of lines per chunk (Paper: 8).
            step_size: Sliding window step (Paper: 2).

        Raises:
            ValueError: If chunk_size < 2 or step_size < 1 or step_size > chunk_size.
        """
        if chunk_size < 2:
            raise ValueError("Chunk size must be at least 2")
        if step_size < 1:
            raise ValueError("Step size must be at least 1")
        if step_size > chunk_size:
            raise ValueError("Step size cannot exceed chunk size")

        self._chunk_size = chunk_size
        self._step_size = step_size

    @property
    def chunk_size(self) -> int:
        """Get the chunk size."""
        return self._chunk_size

    @property
    def step_size(self) -> int:
        """Get the step size."""
        return self._step_size

    def chunk_transcript(self, transcript: Transcript) -> list[TranscriptChunk]:
        """Split transcript into overlapping chunks.

        Args:
            transcript: Transcript to chunk.

        Returns:
            List of transcript chunks.
        """
        lines = transcript.text.strip().splitlines()

        # Remove empty lines at the end
        while lines and lines[-1] == "":
            lines.pop()

        chunks = []

        # If fewer lines than chunk_size, return the whole thing as one chunk
        if len(lines) <= self._chunk_size:
            if lines:
                chunk_text = "\n".join(lines)
                if chunk_text.strip():
                    chunks.append(
                        TranscriptChunk(
                            text=chunk_text,
                            participant_id=transcript.participant_id,
                            line_start=0,
                            line_end=len(lines) - 1,
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

        # Create sliding windows
        for i in range(0, len(lines) - self._chunk_size + 1, self._step_size):
            chunk_lines = lines[i : i + self._chunk_size]
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

        # If the last chunk doesn't include the final lines, add one more chunk
        # Note: last_chunk_start % step_size != 0 guarantees this position was never
        # visited in the sliding window loop, so no duplicate check is needed.
        last_chunk_start = len(lines) - self._chunk_size
        if last_chunk_start > 0 and (last_chunk_start % self._step_size) != 0:
            final_chunk_lines = lines[last_chunk_start:]
            final_chunk_text = "\n".join(final_chunk_lines)
            chunks.append(
                TranscriptChunk(
                    text=final_chunk_text,
                    participant_id=transcript.participant_id,
                    line_start=last_chunk_start,
                    line_end=len(lines) - 1,
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
