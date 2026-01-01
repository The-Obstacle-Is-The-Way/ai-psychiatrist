"""Tests for transcript chunking service."""

from __future__ import annotations

import pytest

from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.value_objects import TranscriptChunk
from ai_psychiatrist.services.chunking import TranscriptChunker

pytestmark = pytest.mark.unit


class TestTranscriptChunker:
    """Tests for transcript chunking."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript with 10 lines."""
        text = "\n".join(f"Line {i}" for i in range(10))
        return Transcript(participant_id=123, text=text)

    @pytest.fixture
    def paper_chunker(self) -> TranscriptChunker:
        """Create chunker with paper-optimal parameters (8, 2)."""
        return TranscriptChunker(chunk_size=8, step_size=2)

    def test_init_default_values(self) -> None:
        """Should use paper-optimal defaults (8, 2)."""
        chunker = TranscriptChunker()

        assert chunker.chunk_size == 8
        assert chunker.step_size == 2

    def test_init_custom_values(self) -> None:
        """Should accept custom chunk and step sizes."""
        chunker = TranscriptChunker(chunk_size=4, step_size=1)

        assert chunker.chunk_size == 4
        assert chunker.step_size == 1

    def test_chunk_basic(self, sample_transcript: Transcript) -> None:
        """Should create correct number of chunks."""
        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        chunks = chunker.chunk_transcript(sample_transcript)

        # With 10 lines, step 2, chunk 4: positions 0, 2, 4, 6, then last chunk at 6
        # So we get chunks starting at 0, 2, 4, 6 (4 chunks)
        assert len(chunks) == 4
        assert all(c.participant_id == 123 for c in chunks)
        assert all(isinstance(c, TranscriptChunk) for c in chunks)

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
        assert chunks[1].line_end == 5

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
        assert "Line 2" in overlap
        assert "Line 3" in overlap

    def test_chunk_paper_params(self) -> None:
        """Should work with paper parameters (chunk=8, step=2)."""
        # Create a 20-line transcript
        text = "\n".join(f"Line {i}" for i in range(20))
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        # With 20 lines, chunk 8, step 2:
        # Positions 0, 2, 4, 6, 8, 10, 12 (7 positions before 20-8=12 limit)
        assert len(chunks) == 7

        # Each chunk should have 8 lines
        assert all(len(c.text.splitlines()) == 8 for c in chunks)

    def test_chunk_short_transcript(self) -> None:
        """Should handle transcript shorter than chunk size."""
        text = "Line 0\nLine 1\nLine 2"
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        # Should return single chunk with all content
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].line_start == 0
        assert chunks[0].line_end == 2

    def test_chunk_exactly_chunk_size(self) -> None:
        """Should handle transcript exactly equal to chunk size."""
        text = "\n".join(f"Line {i}" for i in range(8))
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        assert len(chunks) == 1
        assert len(chunks[0].text.splitlines()) == 8

    def test_chunk_text_directly(self) -> None:
        """Should chunk raw text directly without creating Transcript first."""
        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        text = "\n".join(f"Line {i}" for i in range(10))
        chunks = chunker.chunk_text(text, participant_id=999)

        assert len(chunks) == 4
        assert all(c.participant_id == 999 for c in chunks)

    def test_invalid_chunk_size_too_small(self) -> None:
        """Should reject chunk size < 2."""
        with pytest.raises(ValueError, match="at least 2"):
            TranscriptChunker(chunk_size=1, step_size=1)

    def test_invalid_chunk_size_zero(self) -> None:
        """Should reject zero chunk size."""
        with pytest.raises(ValueError, match="at least 2"):
            TranscriptChunker(chunk_size=0, step_size=1)

    def test_invalid_step_size_zero(self) -> None:
        """Should reject zero step size."""
        with pytest.raises(ValueError, match="at least 1"):
            TranscriptChunker(chunk_size=4, step_size=0)

    def test_invalid_step_exceeds_chunk(self) -> None:
        """Should reject step size > chunk size."""
        with pytest.raises(ValueError, match="cannot exceed"):
            TranscriptChunker(chunk_size=4, step_size=5)

    def test_step_equals_chunk(self) -> None:
        """Should accept step size equal to chunk size (no overlap)."""
        chunker = TranscriptChunker(chunk_size=4, step_size=4)
        text = "\n".join(f"Line {i}" for i in range(8))
        transcript = Transcript(participant_id=100, text=text)
        chunks = chunker.chunk_transcript(transcript)

        # With no overlap, should get 2 chunks
        assert len(chunks) == 2

        # No overlap between chunks
        first_lines = set(chunks[0].text.splitlines())
        second_lines = set(chunks[1].text.splitlines())
        assert len(first_lines & second_lines) == 0

    def test_chunk_preserves_whitespace(self) -> None:
        """Should preserve internal whitespace in lines."""
        text = "Speaker: Hello   there\nParticipant: Hi    back"
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        assert "Hello   there" in chunks[0].text
        assert "Hi    back" in chunks[0].text

    def test_chunk_handles_trailing_newlines(self) -> None:
        """Should handle transcripts with trailing newlines."""
        text = "Line 0\nLine 1\nLine 2\n\n\n"
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        # Should trim trailing empty lines
        assert len(chunks) == 1
        assert chunks[0].text == "Line 0\nLine 1\nLine 2"

    def test_chunk_line_indices_correct(self) -> None:
        """Chunk line start/end indices should be correct."""
        text = "\n".join(f"Line {i}" for i in range(12))
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=4, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        # Verify each chunk's line indices
        assert chunks[0].line_start == 0
        assert chunks[0].line_end == 3
        assert chunks[1].line_start == 2
        assert chunks[1].line_end == 5
        assert chunks[2].line_start == 4
        assert chunks[2].line_end == 7

    def test_chunk_word_count(self) -> None:
        """Chunks should have correct word counts."""
        text = "One two three\nFour five six\nSeven eight nine"
        transcript = Transcript(participant_id=100, text=text)

        chunker = TranscriptChunker(chunk_size=8, step_size=2)
        chunks = chunker.chunk_transcript(transcript)

        assert chunks[0].word_count == 9  # 3 words per line * 3 lines
