# Spec 08: Embedding Service

## Objective

Implement the embedding-based similarity search service for few-shot reference retrieval, as described in Section 2.4.2 of the paper.

## Paper Reference

- **Section 2.4.2**: Few-shot prompting workflow
- **Appendix D**: Optimal hyperparameters (chunk_size=8, N_example=2, dim=4096)
- **Appendix E**: Retrieval statistics and t-SNE visualization

## Target Configuration (Paper-Optimal)

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| Embedding model family | Qwen 3 8B Embedding (example Ollama tag: `qwen3-embedding:8b`; quantization not specified in paper) | Section 2.2 |
| Dimension | 4096 | Appendix D |
| Chunk size | 8 lines | Appendix D |
| Step size | 2 lines | Appendix D |
| top_k references | 2 per item | Appendix D |

## As-Is Implementation (Repo)

The current repo’s few-shot retrieval is implemented inside the quantitative agent (not as a standalone service):

- File: `agents/quantitative_assessor_f.py`
- Reference store: a pickle of `{participant_id: [(raw_text, embedding_vec), ...]}` loaded from
  `pickle_path="agents/chunk_8_step_2_participant_embedded_transcripts.pkl"` (note: this file is not checked into the repo)
- Embedding endpoint: `POST /api/embeddings`
- Similarity: cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`)
- Default demo `top_k`: **3** (paper optimal: **2**)

### As-Is Reference Formatting (Verbatim)

The reference bundle inserted into the scoring prompt is formatted with a pseudo-tag that is **not valid XML** (opening and “closing” markers are both `<Reference Examples>`):

```python
# agents/quantitative_assessor_f.py
if len(text) < min_chars:
    return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>", []

...
lines.append(f"({item_key} Score: {val})\n{h['raw_text']}")

...
block = "<Reference Examples>\n\n" + "\n\n".join(lines) + "\n\n<Reference Examples>"
return block, sims
```

## Key Technical Details

### Dimension Truncation (Repo Behavior)

Both the demo quantitative agent and the research embedding scripts support **dimension truncation** by slicing the embedding vector to `dim` and then L2-normalizing:

- `agents/quantitative_assessor_f.py:ollama_embed(..., dim=...)`
- `quantitative_assessment/embedding_batch_script.py:get_embedding(..., dim=...)`

The paper evaluated multiple dimensions (64/256/1024/4096) and selected 4096 as optimal (Appendix D).

### Embedding Processing Pipeline
1. Generate raw embedding from LLM
2. Truncate to target dimension (4096 optimal per paper)
3. L2 normalize for cosine similarity calculations

## Deliverables

1. `src/ai_psychiatrist/services/embedding.py` - Embedding generation and search
2. `src/ai_psychiatrist/services/reference_store.py` - Pre-computed embeddings store
3. `tests/unit/services/test_embedding.py` - Comprehensive tests

## Implementation

### 1. Embedding Service (embedding.py)

```python
"""Embedding generation and similarity search service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.value_objects import EmbeddedChunk, SimilarityMatch, TranscriptChunk
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import EmbeddingSettings
    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
    from ai_psychiatrist.services.reference_store import ReferenceStore

logger = get_logger(__name__)


@dataclass
class ReferenceBundle:
    """Bundle of reference examples for all PHQ-8 items."""

    item_references: dict[PHQ8Item, list[SimilarityMatch]]

    def format_for_prompt(self) -> str:
        """Format references as prompt text.

        Returns:
            Formatted reference string for LLM prompt.
        """
        blocks = []
        for item in PHQ8Item.all_items():
            matches = self.item_references.get(item, [])
            if not matches:
                block = f"[{item.value}]\n<Reference Examples>\nNo valid evidence found\n</Reference Examples>"
            else:
                lines = []
                for match in matches:
                    score_text = (
                        f"(Score: {match.reference_score})"
                        if match.reference_score is not None
                        else "(Score: N/A)"
                    )
                    lines.append(f"{score_text}\n{match.chunk.text}")
                block = (
                    f"[{item.value}]\n<Reference Examples>\n\n"
                    + "\n\n".join(lines)
                    + "\n\n</Reference Examples>"
                )
            blocks.append(block)
        return "\n\n".join(blocks)


class EmbeddingService:
    """Service for generating embeddings and finding similar chunks.

    Implements the embedding-based few-shot prompting approach from Section 2.4.2.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        reference_store: ReferenceStore,
        settings: EmbeddingSettings,
    ) -> None:
        """Initialize embedding service.

        Args:
            llm_client: LLM client for generating embeddings.
            reference_store: Pre-computed reference embeddings.
            settings: Embedding configuration.
        """
        self._llm_client = llm_client
        self._reference_store = reference_store
        self._dimension = settings.dimension
        self._top_k = settings.top_k_references
        self._min_chars = settings.min_evidence_chars

    async def embed_text(self, text: str) -> tuple[float, ...]:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            L2-normalized embedding vector.
        """
        if len(text) < self._min_chars:
            logger.debug("Text too short for embedding", length=len(text))
            return ()

        embedding = await self._llm_client.simple_embed(
            text=text,
            dimension=self._dimension,
        )

        logger.debug(
            "Generated embedding",
            text_length=len(text),
            dimension=len(embedding),
        )

        return embedding

    async def embed_chunk(self, chunk: TranscriptChunk) -> EmbeddedChunk:
        """Generate embedding for a transcript chunk.

        Args:
            chunk: Transcript chunk to embed.

        Returns:
            Embedded chunk with vector.
        """
        embedding = await self.embed_text(chunk.text)
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    async def find_similar_chunks(
        self,
        query_embedding: tuple[float, ...],
        top_k: int | None = None,
    ) -> list[SimilarityMatch]:
        """Find most similar chunks from reference store.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results (defaults to configured value).

        Returns:
            List of similarity matches sorted by similarity (descending).
        """
        if not query_embedding:
            return []

        k = top_k or self._top_k

        # Get all reference embeddings
        all_refs = self._reference_store.get_all_embeddings()

        if not all_refs:
            logger.warning("No reference embeddings available")
            return []

        # Compute similarities
        query_array = np.array([query_embedding])
        similarities = []

        for participant_id, chunks in all_refs.items():
            for chunk, embedding in chunks:
                if len(embedding) != len(query_embedding):
                    logger.warning(
                        "Dimension mismatch between query and reference",
                        query_dim=len(query_embedding),
                        ref_dim=len(embedding),
                        participant_id=participant_id,
                    )
                    continue

                ref_array = np.array([embedding])
                raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])
                # Transform cosine similarity from [-1, 1] to [0, 1] to match SimilarityMatch validation.
                sim = (1.0 + raw_cos) / 2.0

                similarities.append(
                    SimilarityMatch(
                        chunk=TranscriptChunk(
                            text=chunk,
                            participant_id=participant_id,
                        ),
                        similarity=sim,
                        reference_score=self._reference_store.get_score(
                            participant_id, PHQ8Item.NO_INTEREST  # placeholder
                        ),
                    )
                )

        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:k]

    async def build_reference_bundle(
        self,
        evidence_dict: dict[PHQ8Item, list[str]],
    ) -> ReferenceBundle:
        """Build reference bundle for all PHQ-8 items.

        Args:
            evidence_dict: Dictionary of PHQ8Item -> list of evidence quotes.

        Returns:
            ReferenceBundle with similar references for each item.
        """
        logger.info(
            "Building reference bundle",
            items_with_evidence=sum(1 for v in evidence_dict.values() if v),
        )

        item_references: dict[PHQ8Item, list[SimilarityMatch]] = {}

        for item in PHQ8Item.all_items():
            evidence_quotes = evidence_dict.get(item, [])

            if not evidence_quotes:
                item_references[item] = []
                continue

            # Concatenate evidence for embedding
            combined_text = "\n".join(evidence_quotes)

            if len(combined_text) < self._min_chars:
                item_references[item] = []
                continue

            # Get embedding and find similar
            query_emb = await self.embed_text(combined_text)
            matches = await self._find_similar_for_item(query_emb, item)

            item_references[item] = matches

            logger.debug(
                "Found references for item",
                item=item.value,
                match_count=len(matches),
                top_similarity=matches[0].similarity if matches else 0,
            )

        return ReferenceBundle(item_references=item_references)

    async def _find_similar_for_item(
        self,
        query_embedding: tuple[float, ...],
        item: PHQ8Item,
    ) -> list[SimilarityMatch]:
        """Find similar chunks with item-specific scores.

        Args:
            query_embedding: Query embedding.
            item: PHQ-8 item for score lookup.

        Returns:
            List of matches with item-specific reference scores.
        """
        if not query_embedding:
            return []

        all_refs = self._reference_store.get_all_embeddings()
        matches = []

        query_array = np.array([query_embedding])

        for participant_id, chunks in all_refs.items():
            for chunk_text, embedding in chunks:
                if len(embedding) != len(query_embedding):
                    logger.warning(
                        "Dimension mismatch between query and reference",
                        query_dim=len(query_embedding),
                        ref_dim=len(embedding),
                        participant_id=participant_id,
                    )
                    continue

                ref_array = np.array([embedding])
                raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])
                # Transform cosine similarity from [-1, 1] to [0, 1] to match SimilarityMatch validation.
                sim = (1.0 + raw_cos) / 2.0

                # Get item-specific score for this participant
                score = self._reference_store.get_score(participant_id, item)

                matches.append(
                    SimilarityMatch(
                        chunk=TranscriptChunk(
                            text=chunk_text,
                            participant_id=participant_id,
                        ),
                        similarity=sim,
                        reference_score=score,
                    )
                )

        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[: self._top_k]
```

### 2. Reference Store (reference_store.py)

```python
"""Pre-computed reference embeddings store."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import EmbeddingDimensionMismatchError
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import DataSettings, EmbeddingSettings

logger = get_logger(__name__)


# Mapping from PHQ8Item to CSV column names
PHQ8_COLUMN_MAP = {
    PHQ8Item.NO_INTEREST: "PHQ8_NoInterest",
    PHQ8Item.DEPRESSED: "PHQ8_Depressed",
    PHQ8Item.SLEEP: "PHQ8_Sleep",
    PHQ8Item.TIRED: "PHQ8_Tired",
    PHQ8Item.APPETITE: "PHQ8_Appetite",
    PHQ8Item.FAILURE: "PHQ8_Failure",
    PHQ8Item.CONCENTRATING: "PHQ8_Concentrating",
    PHQ8Item.MOVING: "PHQ8_Moving",
}


class ReferenceStore:
    """Store for pre-computed reference embeddings and scores.

    Loads and manages the knowledge base of embedded transcript chunks
    with their associated PHQ-8 scores.
    """

    def __init__(
        self,
        data_settings: DataSettings,
        embedding_settings: EmbeddingSettings,
    ) -> None:
        """Initialize reference store.

        Args:
            data_settings: Data path configuration.
            embedding_settings: Embedding configuration.
        """
        self._embeddings_path = data_settings.embeddings_path
        self._train_csv = data_settings.train_csv
        self._dev_csv = data_settings.dev_csv
        self._dimension = embedding_settings.dimension

        # Lazy-loaded data
        self._embeddings: dict[int, list[tuple[str, list[float]]]] | None = None
        self._scores_df: pd.DataFrame | None = None

    def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
        """Load pre-computed embeddings from pickle file."""
        if self._embeddings is not None:
            return self._embeddings

        if not self._embeddings_path.exists():
            logger.warning(
                "Embeddings file not found",
                path=str(self._embeddings_path),
            )
            self._embeddings = {}
            return self._embeddings

        logger.info("Loading reference embeddings", path=str(self._embeddings_path))

        with open(self._embeddings_path, "rb") as f:
            raw_data = pickle.load(f)

        # Normalize embeddings and convert participant IDs to int
        normalized: dict[int, list[tuple[str, list[float]]]] = {}
        skipped_chunks = 0
        total_chunks = 0

        for pid, pairs in raw_data.items():
            pid_int = int(pid)
            norm_pairs: list[tuple[str, list[float]]] = []
            for text, embedding in pairs:
                total_chunks += 1
                # Validate dimension - embeddings must be at least as long as configured
                if len(embedding) < self._dimension:
                    skipped_chunks += 1
                    logger.warning(
                        "Skipping embedding with insufficient dimension",
                        participant_id=pid_int,
                        expected=self._dimension,
                        actual=len(embedding),
                    )
                    continue
                # Truncate to configured dimension
                emb = embedding[: self._dimension]
                # L2 normalize
                emb = self._l2_normalize(emb)
                norm_pairs.append((text, emb))
            if norm_pairs:
                normalized[pid_int] = norm_pairs

        # Fail loudly if ALL embeddings are mismatched (BUG-009 fix)
        if total_chunks > 0 and skipped_chunks == total_chunks:
            raise EmbeddingDimensionMismatchError(
                expected=self._dimension,
                actual=len(next(iter(raw_data.values()))[0][1]) if raw_data else 0,
            )

        if skipped_chunks > 0:
            logger.error(
                "Some reference embeddings had dimension mismatch",
                skipped=skipped_chunks,
                total=total_chunks,
                expected_dimension=self._dimension,
            )

        self._embeddings = normalized

        total_chunks = sum(len(v) for v in normalized.values())
        logger.info(
            "Embeddings loaded",
            participants=len(normalized),
            total_chunks=total_chunks,
            dimension=self._dimension,
        )

        return self._embeddings

    def _load_scores(self) -> pd.DataFrame:
        """Load ground truth PHQ-8 scores."""
        if self._scores_df is not None:
            return self._scores_df

        dfs = []
        for path in [self._train_csv, self._dev_csv]:
            if path.exists():
                df = pd.read_csv(path)
                df["Participant_ID"] = df["Participant_ID"].astype(int)
                dfs.append(df)

        if not dfs:
            logger.warning("No ground truth files found")
            self._scores_df = pd.DataFrame()
            return self._scores_df

        self._scores_df = pd.concat(dfs, ignore_index=True)
        self._scores_df = self._scores_df.sort_values("Participant_ID")

        logger.info("Scores loaded", participants=len(self._scores_df))
        return self._scores_df

    @staticmethod
    def _l2_normalize(embedding: list[float]) -> list[float]:
        """L2 normalize an embedding vector."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def get_all_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
        """Get all reference embeddings.

        Returns:
            Dictionary mapping participant_id -> list of (text, embedding) pairs.
        """
        return self._load_embeddings()

    def get_participant_embeddings(
        self, participant_id: int
    ) -> list[tuple[str, list[float]]]:
        """Get embeddings for a specific participant.

        Args:
            participant_id: Participant ID.

        Returns:
            List of (text, embedding) pairs.
        """
        embeddings = self._load_embeddings()
        return embeddings.get(participant_id, [])

    def get_score(self, participant_id: int, item: PHQ8Item) -> int | None:
        """Get PHQ-8 item score for a participant.

        Args:
            participant_id: Participant ID.
            item: PHQ-8 item.

        Returns:
            Score (0-3) or None if unavailable.
        """
        df = self._load_scores()
        row = df[df["Participant_ID"] == participant_id]

        if row.empty:
            return None

        col_name = PHQ8_COLUMN_MAP.get(item)
        if col_name is None or col_name not in row.columns:
            return None

        try:
            return int(row[col_name].iloc[0])
        except (ValueError, TypeError):
            return None

    def list_participants(self) -> list[int]:
        """List all participant IDs with embeddings.

        Returns:
            Sorted list of participant IDs.
        """
        embeddings = self._load_embeddings()
        return sorted(embeddings.keys())

    @property
    def is_loaded(self) -> bool:
        """Check if embeddings are loaded."""
        return self._embeddings is not None

    @property
    def participant_count(self) -> int:
        """Get number of participants with embeddings."""
        return len(self._load_embeddings())
```

### 3. Tests (test_embedding.py)

```python
"""Tests for embedding service."""

from __future__ import annotations

import pytest
import numpy as np

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.services.embedding import EmbeddingService, ReferenceBundle


class TestReferenceBundle:
    """Tests for ReferenceBundle."""

    def test_format_empty_bundle(self) -> None:
        """Should format empty bundle correctly."""
        bundle = ReferenceBundle(item_references={})
        formatted = bundle.format_for_prompt()

        # Should have sections for all items
        for item in PHQ8Item.all_items():
            assert f"[{item.value}]" in formatted
            assert "No valid evidence found" in formatted

    def test_format_with_matches(self) -> None:
        """Should format bundle with matches."""
        from ai_psychiatrist.domain.value_objects import SimilarityMatch, TranscriptChunk

        match = SimilarityMatch(
            chunk=TranscriptChunk(text="Test evidence", participant_id=123),
            similarity=0.95,
            reference_score=2,
        )

        bundle = ReferenceBundle(
            item_references={PHQ8Item.NO_INTEREST: [match]}
        )
        formatted = bundle.format_for_prompt()

        assert "[NoInterest]" in formatted
        assert "(Score: 2)" in formatted
        assert "Test evidence" in formatted


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        from tests.fixtures.mock_llm import MockLLMClient

        def embed_func(req):
            # Return deterministic embedding based on text length
            dim = req.dimension or 256
            return tuple(float(i % 10) / 10 for i in range(dim))

        return MockLLMClient(embedding_function=embed_func)

    @pytest.fixture
    def mock_reference_store(self):
        """Create mock reference store."""
        from unittest.mock import MagicMock

        store = MagicMock()
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
    def mock_settings(self):
        """Create mock embedding settings."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.dimension = 256
        settings.top_k_references = 2
        settings.min_evidence_chars = 8
        return settings

    @pytest.mark.asyncio
    async def test_embed_text(
        self,
        mock_llm_client,
        mock_reference_store,
        mock_settings,
    ) -> None:
        """Should generate embedding for text."""
        service = EmbeddingService(
            mock_llm_client,
            mock_reference_store,
            mock_settings,
        )

        embedding = await service.embed_text("Test text to embed")

        assert len(embedding) == 256
        assert mock_llm_client.embedding_call_count == 1

    @pytest.mark.asyncio
    async def test_embed_short_text_returns_empty(
        self,
        mock_llm_client,
        mock_reference_store,
        mock_settings,
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
    async def test_find_similar_chunks(
        self,
        mock_llm_client,
        mock_reference_store,
        mock_settings,
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
    async def test_build_reference_bundle(
        self,
        mock_llm_client,
        mock_reference_store,
        mock_settings,
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


class TestReferenceStore:
    """Tests for ReferenceStore."""

    def test_l2_normalize(self) -> None:
        """Should correctly L2 normalize vectors."""
        from ai_psychiatrist.services.reference_store import ReferenceStore

        # Unit vector should stay the same
        unit = [1.0, 0.0, 0.0]
        normalized = ReferenceStore._l2_normalize(unit)
        np.testing.assert_array_almost_equal(normalized, [1.0, 0.0, 0.0])

        # General vector
        vec = [3.0, 4.0, 0.0]
        normalized = ReferenceStore._l2_normalize(vec)
        expected = [0.6, 0.8, 0.0]  # 3/5, 4/5, 0
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_zero_vector_unchanged(self) -> None:
        """Zero vector should remain zero (avoid division by zero)."""
        from ai_psychiatrist.services.reference_store import ReferenceStore

        zero = [0.0, 0.0, 0.0]
        normalized = ReferenceStore._l2_normalize(zero)
        np.testing.assert_array_almost_equal(normalized, [0.0, 0.0, 0.0])
```

## Acceptance Criteria

- [ ] Generates embeddings via Ollama API
- [ ] Embeddings are L2-normalized
- [ ] Dimension truncation works (paper optimal: 4096)
- [ ] Cosine similarity search finds top-k matches
- [ ] Reference bundle formats correctly for prompts
- [ ] Loads pre-computed embeddings from pickle
- [ ] Handles missing embeddings gracefully
- [ ] Comprehensive test coverage

## Dependencies

- **Spec 03**: Configuration (EmbeddingSettings, DataSettings)
- **Spec 04**: LLM infrastructure (embedding generation)
- **Spec 05**: Transcript service (TranscriptChunk)

## Specs That Depend on This

- **Spec 09**: Quantitative Agent (uses reference bundle)
