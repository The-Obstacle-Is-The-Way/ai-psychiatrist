"""Tests for embedding service and reference store."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.value_objects import SimilarityMatch, TranscriptChunk
from ai_psychiatrist.services.embedding import EmbeddingService, ReferenceBundle
from ai_psychiatrist.services.reference_store import PHQ8_COLUMN_MAP, ReferenceStore
from tests.fixtures.mock_llm import MockLLMClient


class TestReferenceBundle:
    """Tests for ReferenceBundle."""

    def test_format_empty_bundle(self) -> None:
        """Should format empty bundle with 'no evidence' for all items."""
        bundle = ReferenceBundle(item_references={})
        formatted = bundle.format_for_prompt()

        # Should have sections for all 8 PHQ-8 items
        for item in PHQ8Item.all_items():
            assert f"[{item.value}]" in formatted
            assert "No valid evidence found" in formatted

    def test_format_with_matches(self) -> None:
        """Should format bundle with matches correctly."""
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="I can't enjoy anything anymore", participant_id=123),
            similarity=0.95,
            reference_score=2,
        )

        bundle = ReferenceBundle(item_references={PHQ8Item.NO_INTEREST: [match]})
        formatted = bundle.format_for_prompt()

        assert "[NoInterest]" in formatted
        assert "(Score: 2)" in formatted
        assert "I can't enjoy anything anymore" in formatted
        assert "<Reference Examples>" in formatted
        assert "</Reference Examples>" in formatted

    def test_format_with_none_score(self) -> None:
        """Should handle None reference score."""
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="Some text", participant_id=123),
            similarity=0.8,
            reference_score=None,
        )

        bundle = ReferenceBundle(item_references={PHQ8Item.SLEEP: [match]})
        formatted = bundle.format_for_prompt()

        assert "(Score: N/A)" in formatted

    def test_format_multiple_matches(self) -> None:
        """Should format multiple matches for same item."""
        matches = [
            SimilarityMatch(
                chunk=TranscriptChunk(text="First reference", participant_id=100),
                similarity=0.9,
                reference_score=3,
            ),
            SimilarityMatch(
                chunk=TranscriptChunk(text="Second reference", participant_id=101),
                similarity=0.85,
                reference_score=2,
            ),
        ]

        bundle = ReferenceBundle(item_references={PHQ8Item.TIRED: matches})
        formatted = bundle.format_for_prompt()

        assert "First reference" in formatted
        assert "Second reference" in formatted
        assert "(Score: 3)" in formatted
        assert "(Score: 2)" in formatted


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_llm_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def mock_reference_store(self) -> MagicMock:
        """Create mock reference store."""
        store = MagicMock(spec=ReferenceStore)
        # Return normalized embeddings (sum of squares = 1)
        store.get_all_embeddings.return_value = {
            100: [
                ("Sample reference text", [0.1] * 256),
                ("Another reference", [0.2] * 256),
            ],
            101: [
                ("Different participant", [0.3] * 256),
            ],
        }
        store.get_score.return_value = 2
        return store

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock embedding settings."""
        settings = MagicMock()
        settings.dimension = 256
        settings.top_k_references = 2
        settings.min_evidence_chars = 8
        return settings

    @pytest.mark.asyncio
    async def test_embed_text(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should generate embedding for text."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        embedding = await service.embed_text("Test text to embed for testing")

        assert len(embedding) == 256
        assert mock_llm_client.embedding_call_count == 1

    @pytest.mark.asyncio
    async def test_embed_short_text_returns_empty(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should return empty tuple for text below minimum chars."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        embedding = await service.embed_text("short")

        assert embedding == ()
        assert mock_llm_client.embedding_call_count == 0

    @pytest.mark.asyncio
    async def test_embed_chunk(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should embed a transcript chunk."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        chunk = TranscriptChunk(
            text="This is a test transcript chunk",
            participant_id=123,
        )

        embedded = await service.embed_chunk(chunk)

        assert embedded.chunk == chunk
        assert len(embedded.embedding) == 256

    @pytest.mark.asyncio
    async def test_find_similar_chunks(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should find similar chunks from reference store."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        query_emb = tuple([0.1] * 256)
        matches = await service.find_similar_chunks(query_emb)

        assert len(matches) <= 2  # top_k = 2
        assert all(0 <= m.similarity <= 1 for m in matches)
        # Should be sorted by similarity descending
        if len(matches) > 1:
            assert matches[0].similarity >= matches[1].similarity

    @pytest.mark.asyncio
    async def test_find_similar_chunks_empty_query(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should return empty list for empty query embedding."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        matches = await service.find_similar_chunks(())

        assert matches == []

    @pytest.mark.asyncio
    async def test_find_similar_chunks_no_references(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should handle empty reference store gracefully."""
        mock_reference_store.get_all_embeddings.return_value = {}

        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        query_emb = tuple([0.1] * 256)
        matches = await service.find_similar_chunks(query_emb)

        assert matches == []

    @pytest.mark.asyncio
    async def test_build_reference_bundle(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should build reference bundle for evidence."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        evidence = {
            PHQ8Item.NO_INTEREST: ["I can't enjoy anything anymore"],
            PHQ8Item.SLEEP: ["I wake up at 4am every night"],
            PHQ8Item.APPETITE: [],  # No evidence
        }

        bundle = await service.build_reference_bundle(evidence)

        # Items with evidence should have references
        assert len(bundle.item_references.get(PHQ8Item.NO_INTEREST, [])) > 0
        assert len(bundle.item_references.get(PHQ8Item.SLEEP, [])) > 0
        # Items without evidence should be empty
        assert len(bundle.item_references.get(PHQ8Item.APPETITE, [])) == 0

    @pytest.mark.asyncio
    async def test_build_reference_bundle_short_evidence(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should skip items with evidence too short."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        evidence = {
            PHQ8Item.NO_INTEREST: ["short"],  # Below min_evidence_chars
        }

        bundle = await service.build_reference_bundle(evidence)

        assert len(bundle.item_references.get(PHQ8Item.NO_INTEREST, [])) == 0


class TestReferenceStore:
    """Tests for ReferenceStore."""

    def test_l2_normalize_unit_vector(self) -> None:
        """Unit vector should stay the same after normalization."""
        unit = [1.0, 0.0, 0.0]
        normalized = ReferenceStore._l2_normalize(unit)
        np.testing.assert_array_almost_equal(normalized, [1.0, 0.0, 0.0])

    def test_l2_normalize_general_vector(self) -> None:
        """General vector should be normalized to unit length."""
        vec = [3.0, 4.0, 0.0]
        normalized = ReferenceStore._l2_normalize(vec)
        expected = [0.6, 0.8, 0.0]  # 3/5, 4/5, 0
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_l2_normalize_zero_vector(self) -> None:
        """Zero vector should remain zero (avoid division by zero)."""
        zero = [0.0, 0.0, 0.0]
        normalized = ReferenceStore._l2_normalize(zero)
        np.testing.assert_array_almost_equal(normalized, [0.0, 0.0, 0.0])

    def test_l2_normalize_result_has_unit_length(self) -> None:
        """Normalized vector should have unit length."""
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = ReferenceStore._l2_normalize(vec)
        length = np.linalg.norm(normalized)
        np.testing.assert_almost_equal(length, 1.0)

    def test_phq8_column_map_complete(self) -> None:
        """PHQ8_COLUMN_MAP should have entries for all PHQ-8 items."""
        for item in PHQ8Item.all_items():
            assert item in PHQ8_COLUMN_MAP
            assert PHQ8_COLUMN_MAP[item].startswith("PHQ8_")

    def test_get_score_missing_participant(self) -> None:
        """Should return None for missing participant."""
        mock_data = MagicMock()
        mock_data.embeddings_path.exists.return_value = False
        mock_data.train_csv.exists.return_value = False
        mock_data.dev_csv.exists.return_value = False

        mock_embed = MagicMock()
        mock_embed.dimension = 256

        store = ReferenceStore(mock_data, mock_embed)

        score = store.get_score(99999, PHQ8Item.NO_INTEREST)
        assert score is None

    def test_list_participants_empty(self) -> None:
        """Should return empty list when no embeddings."""
        mock_data = MagicMock()
        mock_data.embeddings_path.exists.return_value = False

        mock_embed = MagicMock()
        mock_embed.dimension = 256

        store = ReferenceStore(mock_data, mock_embed)

        participants = store.list_participants()
        assert participants == []

    def test_is_loaded_initially_false(self) -> None:
        """Should be False before loading."""
        mock_data = MagicMock()
        mock_embed = MagicMock()
        mock_embed.dimension = 256

        store = ReferenceStore(mock_data, mock_embed)

        assert store.is_loaded is False


class TestSimilarityMatchValidation:
    """Tests for SimilarityMatch validation."""

    def test_valid_similarity(self) -> None:
        """Should accept valid similarity values."""
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="test", participant_id=1),
            similarity=0.5,
        )
        assert match.similarity == 0.5

    def test_boundary_similarities(self) -> None:
        """Should accept boundary values 0 and 1."""
        match_zero = SimilarityMatch(
            chunk=TranscriptChunk(text="test", participant_id=1),
            similarity=0.0,
        )
        match_one = SimilarityMatch(
            chunk=TranscriptChunk(text="test", participant_id=1),
            similarity=1.0,
        )
        assert match_zero.similarity == 0.0
        assert match_one.similarity == 1.0

    def test_invalid_similarity_raises(self) -> None:
        """Should raise ValueError for invalid similarity."""
        with pytest.raises(ValueError, match="Similarity must be 0-1"):
            SimilarityMatch(
                chunk=TranscriptChunk(text="test", participant_id=1),
                similarity=1.5,
            )

        with pytest.raises(ValueError, match="Similarity must be 0-1"):
            SimilarityMatch(
                chunk=TranscriptChunk(text="test", participant_id=1),
                similarity=-0.1,
            )
