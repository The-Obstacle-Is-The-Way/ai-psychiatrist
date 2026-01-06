"""Embedding generation and similarity search service.

Implements the embedding-based few-shot prompting approach from
Paper Section 2.4.2. Generates embeddings via the configured LLM backend and performs
cosine similarity search against pre-computed reference embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ai_psychiatrist.config import get_model_name
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingValidationError,
)
from ai_psychiatrist.domain.value_objects import EmbeddedChunk, SimilarityMatch, TranscriptChunk
from ai_psychiatrist.infrastructure.hashing import stable_text_hash
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingBatchRequest, EmbeddingRequest
from ai_psychiatrist.infrastructure.logging import get_logger
from ai_psychiatrist.infrastructure.validation import validate_embedding
from ai_psychiatrist.services.reference_validation import (
    NoOpReferenceValidator,
    ReferenceValidationRequest,
    ReferenceValidator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ai_psychiatrist.config import EmbeddingSettings, ModelSettings
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient
    from ai_psychiatrist.services.reference_store import ReferenceStore

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalSimilarityStats:
    """Aggregated retrieval similarity statistics for one PHQ-8 item.

    These signals are used as confidence ranking inputs for selective prediction
    evaluation (Spec 046).
    """

    retrieval_reference_count: int
    retrieval_similarity_mean: float | None
    retrieval_similarity_max: float | None


def compute_retrieval_similarity_stats(
    matches: Sequence[SimilarityMatch],
) -> RetrievalSimilarityStats:
    """Compute retrieval similarity stats from final matches used in the prompt.

    Notes:
    - We only include matches with a non-null `reference_score`, matching
      `ReferenceBundle.format_for_prompt()` behavior.
    - If no valid references exist, mean/max are `None` (not 0.0) to preserve
      observability in run artifacts.
    """
    usable = [m.similarity for m in matches if m.reference_score is not None]
    if not usable:
        return RetrievalSimilarityStats(
            retrieval_reference_count=0,
            retrieval_similarity_mean=None,
            retrieval_similarity_max=None,
        )

    mean = sum(usable) / len(usable)
    return RetrievalSimilarityStats(
        retrieval_reference_count=len(usable),
        retrieval_similarity_mean=mean,
        retrieval_similarity_max=max(usable),
    )


@dataclass
class ReferenceBundle:
    """Bundle of reference examples for all PHQ-8 items.

    Used to provide few-shot examples in quantitative assessment prompts.
    """

    item_references: dict[PHQ8Item, list[SimilarityMatch]]

    def format_for_prompt(self) -> str:
        """Format references as prompt text (baseline prompt format).

        Paper notebook behavior (cell 49f51ff5) + Spec 33 XML fix:
        - Single unified <Reference Examples> block.
        - Each reference entry is labeled like: (PHQ8_Sleep Score: 2)
        - Items with no matches are omitted (no empty per-item blocks).
        - Proper XML closing tag: </Reference Examples> (Spec 33 fix).
        """
        entries: list[str] = []

        for item in PHQ8Item.all_items():
            evidence_key = f"PHQ8_{item.value}"
            for match in self.item_references.get(item, []):
                # Notebook behavior: only include references with available ground truth.
                if match.reference_score is None:
                    continue
                entries.append(
                    f"({evidence_key} Score: {match.reference_score})\n{match.chunk.text}"
                )

        if entries:
            return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n</Reference Examples>"

        return "<Reference Examples>\nNo valid evidence found\n</Reference Examples>"


class EmbeddingService:
    """Service for generating embeddings and finding similar chunks.

    Implements the embedding-based few-shot prompting approach from Section 2.4.2.
    Uses the configured LLM backend for embedding generation and cosine similarity for retrieval.
    """

    def __init__(
        self,
        llm_client: EmbeddingClient,
        reference_store: ReferenceStore,
        settings: EmbeddingSettings,
        model_settings: ModelSettings | None = None,
        reference_validator: ReferenceValidator | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            llm_client: LLM client for generating embeddings.
            reference_store: Pre-computed reference embeddings.
            settings: Embedding configuration.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
            reference_validator: Optional validator for retrieved references (Spec 36).
        """
        self._llm_client = llm_client
        self._reference_store = reference_store
        self._dimension = settings.dimension
        self._top_k = settings.top_k_references
        self._min_chars = settings.min_evidence_chars
        self._enable_batch_query_embedding = settings.enable_batch_query_embedding
        self._query_embed_timeout_seconds = settings.query_embed_timeout_seconds
        self._model_settings = model_settings
        self._enable_retrieval_audit = settings.enable_retrieval_audit

        # Validation (Spec 36)
        self._reference_validator = reference_validator or NoOpReferenceValidator()
        self._validation_max_refs_per_item = settings.validation_max_refs_per_item

        # Retrieval quality guardrails (Spec 33, post-hoc improvements not in paper)
        # - min_reference_similarity: Floor for filtering low-similarity (0.0 = disabled)
        # - max_reference_chars_per_item: Character budget per item (0 = unlimited)
        self._min_reference_similarity = settings.min_reference_similarity
        self._max_reference_chars_per_item = settings.max_reference_chars_per_item
        self._enable_item_tag_filter = settings.enable_item_tag_filter

        self._reference_score_source = settings.reference_score_source
        if self._reference_score_source == "chunk" and not self._reference_store.has_chunk_scores():
            raise ValueError(
                "reference_score_source='chunk' requires <embeddings>.chunk_scores.json"
            )

    async def embed_text(self, text: str) -> tuple[float, ...]:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            L2-normalized embedding vector, or empty tuple if text too short.
        """
        if len(text) < self._min_chars:
            logger.debug("Text too short for embedding", length=len(text))
            return ()

        # Use model settings if provided (Paper Section 2.2: Qwen 3 8B Embedding)
        model = get_model_name(self._model_settings, "embedding")
        response = await self._llm_client.embed(
            EmbeddingRequest(
                text=text,
                model=model,
                dimension=self._dimension,
                timeout_seconds=self._query_embed_timeout_seconds,
            )
        )
        embedding = response.embedding

        logger.debug(
            "Generated embedding",
            text_length=len(text),
            dimension=len(embedding),
        )

        validate_embedding(
            np.array(embedding, dtype=np.float32),
            context="query embedding (no text logged)",
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

    def _compute_similarities(
        self,
        query_embedding: tuple[float, ...],
        item: PHQ8Item | None = None,
    ) -> list[SimilarityMatch]:
        """Compute similarities between query and all reference embeddings.

        Args:
            query_embedding: Query embedding vector.
            item: Optional PHQ-8 item for score lookup. If None, uses NO_INTEREST.

        Returns:
            List of all similarity matches (unsorted).
        """
        # BUG-004 Fix: Use vectorized matrix operations
        matrix, metadata = self._reference_store.get_vectorized_data()

        if matrix.shape[0] == 0:
            logger.warning("No reference embeddings available")
            return []

        if len(query_embedding) != self._dimension:
            raise EmbeddingDimensionMismatchError(
                expected=self._dimension,
                actual=len(query_embedding),
            )

        # Compute Cosine Similarity: sims = Matrix . Query
        # Assumes matrix rows are already L2 normalized (ReferenceStore ensures this).
        query_vec = np.array(query_embedding, dtype=np.float32)
        validate_embedding(query_vec, context="query embedding (pre-similarity)")

        # Dot product: (N, D) @ (D,) -> (N,)
        similarities = matrix @ query_vec
        if not np.isfinite(similarities).all():
            raise EmbeddingValidationError("Non-finite similarity scores (NaN/Inf) detected")

        # Transform cosine similarity from [-1, 1] to [0, 1] range
        similarities = (1.0 + similarities) / 2.0
        # Clip for float precision safety
        similarities = np.clip(similarities, 0.0, 1.0)

        matches: list[SimilarityMatch] = []
        lookup_item = item or PHQ8Item.NO_INTEREST
        target_tag = (
            f"PHQ8_{item.value}" if (self._enable_item_tag_filter and item is not None) else None
        )

        # Iterate over results
        # While this is still a loop, it's just unpacking/filtering, not computing dot products.
        for idx, sim in enumerate(similarities):
            meta = metadata[idx]

            # Tag filtering
            if target_tag and target_tag not in meta["tags"]:
                continue

            pid = meta["participant_id"]
            chunk_idx = meta["chunk_index"]

            # Get item-specific score
            if self._reference_score_source == "chunk":
                score = self._reference_store.get_chunk_score(pid, chunk_idx, lookup_item)
            else:
                score = self._reference_store.get_score(pid, lookup_item)

            matches.append(
                SimilarityMatch(
                    chunk=TranscriptChunk(
                        text=meta["text"],
                        participant_id=pid,
                    ),
                    similarity=float(sim),
                    reference_score=score,
                )
            )

        return matches

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
        matches = self._compute_similarities(query_embedding)

        # Sort by similarity (descending) and take top k
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:k]

    async def _validate_matches(
        self,
        matches: list[SimilarityMatch],
        item: PHQ8Item,
        evidence_text: str,
    ) -> list[SimilarityMatch]:
        """Apply budget constraints and CRAG-style validation to matches."""
        # 1. Apply budget constraints (character limit)
        if self._max_reference_chars_per_item > 0:
            budgeted: list[SimilarityMatch] = []
            used = 0
            for m in matches:
                cost = len(m.chunk.text)
                if used + cost > self._max_reference_chars_per_item:
                    break
                budgeted.append(m)
                used += cost
            matches = budgeted

        # 2. Apply CRAG-style validation
        validated_matches: list[SimilarityMatch] = []
        for match in matches:
            if len(validated_matches) >= self._validation_max_refs_per_item:
                break

            request = ReferenceValidationRequest(
                item=item,
                evidence_text=evidence_text,
                reference_text=match.chunk.text,
                reference_score=match.reference_score,
            )

            decision = await self._reference_validator.validate(request)

            if decision == "accept":
                validated_matches.append(match)

        return validated_matches

    async def _retrieve_item_references(
        self,
        *,
        item: PHQ8Item,
        evidence_text: str,
        query_emb: tuple[float, ...],
    ) -> list[SimilarityMatch]:
        matches = self._compute_similarities(query_emb, item=item)
        matches.sort(key=lambda x: x.similarity, reverse=True)

        if self._min_reference_similarity > 0.0:
            matches = [m for m in matches if m.similarity >= self._min_reference_similarity]

        top_matches = matches[: self._top_k]
        final_matches = await self._validate_matches(top_matches, item, evidence_text)

        if self._enable_retrieval_audit:
            evidence_key = f"PHQ8_{item.value}"
            for rank, match in enumerate(final_matches, start=1):
                logger.info(
                    "retrieved_reference",
                    item=item.value,
                    evidence_key=evidence_key,
                    rank=rank,
                    similarity=match.similarity,
                    participant_id=match.chunk.participant_id,
                    reference_score=match.reference_score,
                    chunk_hash=stable_text_hash(match.chunk.text),
                    chunk_chars=len(match.chunk.text),
                )

        logger.debug(
            "Found references for item",
            item=item.value,
            match_count=len(final_matches),
            top_similarity=final_matches[0].similarity if final_matches else 0,
        )

        return final_matches

    async def build_reference_bundle(
        self,
        evidence_dict: dict[PHQ8Item, list[str]],
    ) -> ReferenceBundle:
        """Build reference bundle for all PHQ-8 items.

        For each PHQ-8 item with evidence, generates an embedding from the
        combined evidence text and retrieves the top-k most similar reference
        chunks with their associated scores.

        Args:
            evidence_dict: Dictionary of PHQ8Item -> list of evidence quotes.

        Returns:
            ReferenceBundle with similar references for each item.
        """
        logger.info(
            "Building reference bundle",
            items_with_evidence=sum(1 for v in evidence_dict.values() if v),
        )

        item_references: dict[PHQ8Item, list[SimilarityMatch]] = {
            item: [] for item in PHQ8Item.all_items()
        }

        item_texts = [
            (item, "\n".join(evidence_dict.get(item, []))) for item in PHQ8Item.all_items()
        ]
        item_texts = [(item, text) for item, text in item_texts if len(text) >= self._min_chars]

        if not item_texts:
            return ReferenceBundle(item_references=item_references)

        model = get_model_name(self._model_settings, "embedding")
        if self._enable_batch_query_embedding:
            batch_request = EmbeddingBatchRequest(
                texts=[text for _, text in item_texts],
                model=model,
                dimension=self._dimension,
                timeout_seconds=self._query_embed_timeout_seconds,
            )
            embeddings = (await self._llm_client.embed_batch(batch_request)).embeddings

            for idx, emb in enumerate(embeddings):
                validate_embedding(
                    np.array(emb, dtype=np.float32),
                    context=f"batch query embedding {idx} (no text logged)",
                )
        else:
            embeddings = [await self.embed_text(text) for _, text in item_texts]

        for (item, evidence_text), query_emb in zip(item_texts, embeddings, strict=True):
            item_references[item] = await self._retrieve_item_references(
                item=item,
                evidence_text=evidence_text,
                query_emb=query_emb,
            )

        return ReferenceBundle(item_references=item_references)
