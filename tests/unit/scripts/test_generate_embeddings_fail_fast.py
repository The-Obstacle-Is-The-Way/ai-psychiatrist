"""Unit tests for scripts/generate_embeddings.py fail-fast behavior (Spec 40)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import path is set up in conftest
from scripts import generate_embeddings as generate_embeddings_module
from scripts.generate_embeddings import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    EmbeddingGenerationError,
    GenerationConfig,
    GenerationResult,
    process_participant,
    save_embeddings,
)

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackendSettings,
    EmbeddingSettings,
    ModelSettings,
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


def _make_generation_config(tmp_path: Path, *, write_item_tags: bool) -> GenerationConfig:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir()

    return GenerationConfig(
        data_settings=DataSettings(base_dir=base_dir, transcripts_dir=transcripts_dir),
        embedding_settings=EmbeddingSettings(),
        backend_settings=EmbeddingBackendSettings(),
        model_settings=ModelSettings(),
        chunk_size=8,
        step_size=2,
        dimension=3,
        min_chars=10,
        model="qwen3-embedding:8b",
        resolved_model="qwen3-embedding:8b",
        output_path=tmp_path / "out.npz",
        split="paper-train",
        dry_run=False,
        write_item_tags=write_item_tags,
        tagger_type="keyword",
        keywords_path=tmp_path / "keywords.yaml",
        allow_partial=False,
    )


class TestSaveEmbeddingsAtomicWrites:
    def test_save_embeddings_leaves_no_temp_files(self, tmp_path: Path) -> None:
        config = _make_generation_config(tmp_path, write_item_tags=True)
        result = GenerationResult(
            embeddings={1: [("chunk", [0.1, 0.2, 0.3])]},
            tags={1: [["PHQ8_Sleep"]]},
            total_chunks=1,
        )

        save_embeddings(result, config)

        assert config.output_path.exists()
        assert config.output_path.with_suffix(".json").exists()
        assert config.output_path.with_suffix(".meta.json").exists()
        assert config.output_path.with_suffix(".tags.json").exists()

        assert not config.output_path.with_suffix(".tmp.npz").exists()
        assert not config.output_path.with_suffix(".tmp.json").exists()
        assert not config.output_path.with_suffix(".tmp.meta.json").exists()
        assert not config.output_path.with_suffix(".tmp.tags.json").exists()

    def test_save_embeddings_cleans_temp_files_on_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _make_generation_config(tmp_path, write_item_tags=True)
        result = GenerationResult(
            embeddings={1: [("chunk", [0.1, 0.2, 0.3])]},
            tags={1: [["PHQ8_Sleep"]]},
            total_chunks=1,
        )

        original_dump = json.dump
        call_count = 0

        def flaky_dump(obj: Any, fp: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            original_dump(obj, fp, *args, **kwargs)

        monkeypatch.setattr(
            generate_embeddings_module,
            "json",
            SimpleNamespace(dump=flaky_dump),
        )
        with pytest.raises(RuntimeError, match="boom"):
            save_embeddings(result, config)

        assert not config.output_path.exists()
        assert not config.output_path.with_suffix(".json").exists()
        assert not config.output_path.with_suffix(".meta.json").exists()
        assert not config.output_path.with_suffix(".tags.json").exists()

        assert not config.output_path.with_suffix(".tmp.npz").exists()
        assert not config.output_path.with_suffix(".tmp.json").exists()
        assert not config.output_path.with_suffix(".tmp.meta.json").exists()
        assert not config.output_path.with_suffix(".tmp.tags.json").exists()
