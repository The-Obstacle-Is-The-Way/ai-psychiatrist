"""Tests for embedding service and reference store."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ai_psychiatrist.config import DataSettings
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import EmbeddingDimensionMismatchError
from ai_psychiatrist.domain.value_objects import SimilarityMatch, TranscriptChunk
from ai_psychiatrist.services.embedding import EmbeddingService, ReferenceBundle
from ai_psychiatrist.services.reference_store import PHQ8_COLUMN_MAP, ReferenceStore
from tests.fixtures.mock_llm import MockLLMClient

if TYPE_CHECKING:
    from pathlib import Path


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
    async def test_find_similar_chunks_dimension_mismatch_raises(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Should raise when query embedding dimension mismatches config."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        # Wrong dimensionality (expected 256 per mock_settings)
        with pytest.raises(EmbeddingDimensionMismatchError):
            await service.find_similar_chunks((0.1, 0.2))

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

    @pytest.fixture
    def mock_data_settings(self, tmp_path: Path) -> DataSettings:
        """Create data settings with temporary paths."""
        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        return DataSettings(
            base_dir=tmp_path,
            transcripts_dir=transcripts_dir,
            embeddings_path=tmp_path / "embeddings.pkl",
            train_csv=tmp_path / "train.csv",
            dev_csv=tmp_path / "dev.csv",
        )

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

    def test_load_embeddings_success(self, mock_data_settings: DataSettings) -> None:
        """Should successfully load and normalize embeddings from file."""
        # Create dummy embeddings file
        raw_data = {
            "100": [("chunk1", [3.0, 4.0])],  # Norm = 5.0 -> [0.6, 0.8]
            "101": [("chunk2", [1.0, 0.0])],
        }
        with mock_data_settings.embeddings_path.open("wb") as f:
            pickle.dump(raw_data, f)

        mock_embed = MagicMock()
        mock_embed.dimension = 2  # Match dummy data dimension

        store = ReferenceStore(mock_data_settings, mock_embed)
        embeddings = store.get_all_embeddings()

        assert len(embeddings) == 2
        assert 100 in embeddings
        assert 101 in embeddings

        # Check normalization
        text, vector = embeddings[100][0]
        assert text == "chunk1"
        np.testing.assert_array_almost_equal(vector, [0.6, 0.8])

        assert store.is_loaded is True
        assert store.participant_count == 2

    def test_load_scores_success(self, mock_data_settings: DataSettings) -> None:
        """Should successfully load scores from CSVs."""
        # Create dummy train CSV
        train_df = pd.DataFrame(
            {
                "Participant_ID": [100],
                "PHQ8_NoInterest": [1],
                "PHQ8_Depressed": [2],
            }
        )
        train_df.to_csv(mock_data_settings.train_csv, index=False)

        # Create dummy dev CSV
        dev_df = pd.DataFrame(
            {
                "Participant_ID": [101],
                "PHQ8_NoInterest": [3],
                "PHQ8_Depressed": [0],
            }
        )
        dev_df.to_csv(mock_data_settings.dev_csv, index=False)

        mock_embed = MagicMock()
        store = ReferenceStore(mock_data_settings, mock_embed)

        # Check scores from train split
        assert store.get_score(100, PHQ8Item.NO_INTEREST) == 1
        assert store.get_score(100, PHQ8Item.DEPRESSED) == 2

        # Check scores from dev split
        assert store.get_score(101, PHQ8Item.NO_INTEREST) == 3
        assert store.get_score(101, PHQ8Item.DEPRESSED) == 0

        # Check missing score
        assert store.get_score(999, PHQ8Item.NO_INTEREST) is None

    def test_get_participant_embeddings(self, mock_data_settings: DataSettings) -> None:
        """Should return embeddings for specific participant."""
        raw_data = {
            "100": [("text1", [1.0, 0.0])],
            "101": [("text2", [0.0, 1.0])],
        }
        with mock_data_settings.embeddings_path.open("wb") as f:
            pickle.dump(raw_data, f)

        mock_embed = MagicMock()
        mock_embed.dimension = 2

        store = ReferenceStore(mock_data_settings, mock_embed)

        p100 = store.get_participant_embeddings(100)
        assert len(p100) == 1
        assert p100[0][0] == "text1"

        p999 = store.get_participant_embeddings(999)
        assert len(p999) == 0

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


class TestEmbeddingDimensionMismatch:
    """Tests for BUG-009: Dimension mismatch handling."""

    def test_all_embeddings_mismatched_raises_error(self, tmp_path: Path) -> None:
        """Should raise EmbeddingDimensionMismatchError when ALL embeddings are too short."""
        # Create embeddings with dimension 2 when we expect 256
        raw_data = {
            "100": [("chunk1", [1.0, 0.0])],  # Only 2 dims
            "101": [("chunk2", [0.0, 1.0])],  # Only 2 dims
        }

        # Create temp files
        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = tmp_path / "embeddings.pkl"
        with embeddings_path.open("wb") as f:
            pickle.dump(raw_data, f)

        data_settings = DataSettings(
            base_dir=tmp_path,
            transcripts_dir=transcripts_dir,
            embeddings_path=embeddings_path,
            train_csv=tmp_path / "train.csv",
            dev_csv=tmp_path / "dev.csv",
        )

        mock_embed = MagicMock()
        mock_embed.dimension = 256  # Expect 256, but embeddings only have 2

        store = ReferenceStore(data_settings, mock_embed)

        with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
            store.get_all_embeddings()

        assert excinfo.value.expected == 256
        assert excinfo.value.actual == 2

    def test_partial_mismatch_skips_bad_embeddings(self, tmp_path: Path) -> None:
        """Should skip embeddings that are too short and keep valid ones."""
        # Mix of valid and invalid embeddings
        raw_data = {
            "100": [("chunk1", [1.0, 0.0])],  # Only 2 dims - will be skipped
            "101": [("chunk2", [0.5] * 10)],  # 10 dims - valid for dim=4
        }

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = tmp_path / "embeddings.pkl"
        with embeddings_path.open("wb") as f:
            pickle.dump(raw_data, f)

        data_settings = DataSettings(
            base_dir=tmp_path,
            transcripts_dir=transcripts_dir,
            embeddings_path=embeddings_path,
            train_csv=tmp_path / "train.csv",
            dev_csv=tmp_path / "dev.csv",
        )

        mock_embed = MagicMock()
        mock_embed.dimension = 4  # Need at least 4, so 2-dim is skipped, 10-dim is valid

        store = ReferenceStore(data_settings, mock_embed)
        embeddings = store.get_all_embeddings()

        # Only participant 101 should remain (100 was skipped)
        assert 101 in embeddings
        assert 100 not in embeddings  # Participant 100 had no valid embeddings
        assert len(embeddings[101]) == 1

    def test_embedding_truncation(self, tmp_path: Path) -> None:
        """Should truncate embeddings to configured dimension."""
        # Embedding with 10 dimensions
        raw_data = {
            "100": [("chunk1", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])],
        }

        transcripts_dir = tmp_path / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = tmp_path / "embeddings.pkl"
        with embeddings_path.open("wb") as f:
            pickle.dump(raw_data, f)

        data_settings = DataSettings(
            base_dir=tmp_path,
            transcripts_dir=transcripts_dir,
            embeddings_path=embeddings_path,
            train_csv=tmp_path / "train.csv",
            dev_csv=tmp_path / "dev.csv",
        )

        mock_embed = MagicMock()
        mock_embed.dimension = 4  # Truncate to 4

        store = ReferenceStore(data_settings, mock_embed)
        embeddings = store.get_all_embeddings()

        # Check truncation happened
        chunk_text, emb = embeddings[100][0]
        assert chunk_text == "chunk1"
        assert len(emb) == 4

        # Check L2 normalization (should be unit length)
        length = np.linalg.norm(emb)
        np.testing.assert_almost_equal(length, 1.0)


class TestSimilarityTransformation:
    """Tests for BUG-010: Cosine similarity transformation."""

    @pytest.fixture
    def mock_llm_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock embedding settings."""
        settings = MagicMock()
        settings.dimension = 256
        settings.top_k_references = 2
        settings.min_evidence_chars = 8
        return settings

    @pytest.fixture
    def mock_reference_store(self) -> MagicMock:
        """Create mock reference store with known embeddings."""
        store = MagicMock(spec=ReferenceStore)
        # Create embeddings that will produce known cosine similarities
        # Unit vector [1, 0, 0, ...] - identical to query
        identical = [1.0] + [0.0] * 255
        # Orthogonal vector [0, 1, 0, ...] - cosine = 0
        orthogonal = [0.0, 1.0] + [0.0] * 254
        # Opposite vector [-1, 0, 0, ...] - cosine = -1
        opposite = [-1.0] + [0.0] * 255

        store.get_all_embeddings.return_value = {
            100: [("identical match", identical)],
            101: [("orthogonal match", orthogonal)],
            102: [("opposite match", opposite)],
        }
        store.get_score.return_value = 2
        return store

    @pytest.mark.asyncio
    async def test_similarity_transformation_range(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Similarity values should be transformed to [0, 1] range."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        # Query with unit vector [1, 0, 0, ...]
        query_emb = tuple([1.0] + [0.0] * 255)
        matches = await service.find_similar_chunks(query_emb, top_k=10)

        # All similarities should be in [0, 1]
        assert all(0.0 <= m.similarity <= 1.0 for m in matches)

    @pytest.mark.asyncio
    async def test_similarity_transformation_values(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Transformed similarity should follow (1 + cos) / 2 formula."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        # Query with unit vector [1, 0, 0, ...]
        query_emb = tuple([1.0] + [0.0] * 255)
        matches = await service.find_similar_chunks(query_emb, top_k=10)

        # Create a map for easier lookup
        similarity_map = {m.chunk.participant_id: m.similarity for m in matches}

        # Identical (cos=1): (1+1)/2 = 1.0
        np.testing.assert_almost_equal(similarity_map[100], 1.0, decimal=5)

        # Orthogonal (cos=0): (1+0)/2 = 0.5
        np.testing.assert_almost_equal(similarity_map[101], 0.5, decimal=5)

        # Opposite (cos=-1): (1-1)/2 = 0.0
        np.testing.assert_almost_equal(similarity_map[102], 0.0, decimal=5)
