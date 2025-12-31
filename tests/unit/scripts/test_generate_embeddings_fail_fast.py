"""Unit tests for scripts/generate_embeddings.py fail-fast behavior (Spec 40)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Import path is set up in conftest
from scripts.generate_embeddings import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    EmbeddingGenerationError,
    process_participant,
)


class TestProcessParticipantStrictMode:
    """Tests for default strict mode (allow_partial=False)."""

    @pytest.mark.asyncio
    async def test_transcript_load_failure_raises(self) -> None:
        """Transcript load failure must raise in strict mode."""
        mock_client = AsyncMock()
        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.side_effect = FileNotFoundError("missing")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=50,
                allow_partial=False,  # STRICT MODE
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index is None
        assert "participant 100" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_transcript_raises(self) -> None:
        """Empty transcript (no chunks) must raise in strict mode."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(text="")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=10,
                allow_partial=False,
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index is None
        assert "empty transcript" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embedding_failure_raises(self) -> None:
        """Chunk embedding failure must raise in strict mode."""
        mock_client = AsyncMock()
        mock_client.embed.side_effect = RuntimeError("API timeout")

        mock_transcript_service = MagicMock()
        mock_transcript = MagicMock()
        # Create enough text for at least one chunk
        mock_transcript.text = (
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\n" * 10
        )
        mock_transcript_service.load_transcript.return_value = mock_transcript

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=10,
                allow_partial=False,  # STRICT MODE
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index == 0  # First chunk
        assert "chunk 0" in str(exc_info.value)


class TestProcessParticipantPartialMode:
    """Tests for explicit partial mode (allow_partial=True)."""

    @pytest.mark.asyncio
    async def test_transcript_load_failure_returns_empty(self) -> None:
        """Transcript load failure returns empty in partial mode."""
        mock_client = AsyncMock()
        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.side_effect = FileNotFoundError("missing")

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=2,
            min_chars=50,
            allow_partial=True,  # PARTIAL MODE
        )

        assert results == []
        assert tags == []
        assert skipped_chunks == 0

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_empty(self) -> None:
        """Empty transcript skips participant in partial mode."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(text="")

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=2,
            min_chars=10,
            allow_partial=True,
        )

        assert results == []
        assert tags == []
        assert skipped_chunks == 0

    @pytest.mark.asyncio
    async def test_embedding_failure_skips_chunk(self) -> None:
        """Chunk embedding failure skips chunk in partial mode."""
        mock_client = AsyncMock()
        # First call fails, second succeeds
        mock_client.embed.side_effect = [
            RuntimeError("API timeout"),
            MagicMock(embedding=[1.0] * 3),
        ]

        mock_transcript_service = MagicMock()
        mock_transcript = MagicMock()
        # Create enough text for 2 chunks (8 lines each, no overlap)
        mock_transcript.text = ("Line content here.\n" * 8) * 2
        mock_transcript_service.load_transcript.return_value = mock_transcript

        results, _tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=8,  # No overlap for predictable chunks
            min_chars=10,
            allow_partial=True,  # PARTIAL MODE
        )

        # Should have 1 result (second chunk), first was skipped
        assert len(results) == 1
        assert skipped_chunks == 1

    @pytest.mark.asyncio
    async def test_tagger_failure_raises_even_in_partial(self) -> None:
        """Tagging failures must remain fail-fast even when tagger is enabled."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(
            text=("Line content here.\n" * 8),
        )

        tagger = MagicMock()
        tagger.tag_chunk.side_effect = RuntimeError("tagger broke")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=8,
                min_chars=10,
                tagger=tagger,
                allow_partial=True,  # Tagger should STILL fail-fast
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index == 0

    @pytest.mark.asyncio
    async def test_all_chunks_fail_returns_empty(self) -> None:
        """All chunks failing returns empty in partial mode (not crash)."""
        mock_client = AsyncMock()
        # All calls fail
        mock_client.embed.side_effect = RuntimeError("API timeout")

        mock_transcript_service = MagicMock()
        mock_transcript = MagicMock()
        # Create enough text for 2 chunks
        mock_transcript.text = ("Line content here.\n" * 8) * 2
        mock_transcript_service.load_transcript.return_value = mock_transcript

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=8,
            min_chars=10,
            allow_partial=True,
        )

        # All chunks failed, should return empty
        assert results == []
        assert tags == []
        assert skipped_chunks == 2  # Both chunks were skipped


class TestExitCodes:
    """Tests for correct exit codes."""

    def test_exit_codes_defined(self) -> None:
        """Exit codes must be correctly defined."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_PARTIAL == 2


class TestEmbeddingGenerationError:
    """Tests for the custom exception."""

    def test_exception_stores_participant_id(self) -> None:
        """Exception must store participant_id."""
        error = EmbeddingGenerationError("test error", participant_id=123)
        assert error.participant_id == 123
        assert error.chunk_index is None
        assert "test error" in str(error)

    def test_exception_stores_chunk_index(self) -> None:
        """Exception must store chunk_index when provided."""
        error = EmbeddingGenerationError("test error", participant_id=123, chunk_index=5)
        assert error.participant_id == 123
        assert error.chunk_index == 5

    def test_exception_chaining_preserves_cause(self) -> None:
        """Exception chaining must preserve original cause."""
        original = ValueError("original error")
        try:
            try:
                raise original
            except ValueError as e:
                raise EmbeddingGenerationError("wrapped", participant_id=100) from e
        except EmbeddingGenerationError as wrapped:
            assert wrapped.__cause__ is original
