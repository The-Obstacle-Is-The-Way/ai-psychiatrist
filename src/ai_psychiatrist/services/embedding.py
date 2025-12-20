"""Embedding generation and similarity search service.

Implements the embedding-based few-shot prompting approach from
Paper Section 2.4.2. Generates embeddings via Ollama and performs
cosine similarity search against pre-computed reference embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.value_objects import EmbeddedChunk, SimilarityMatch, TranscriptChunk
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import EmbeddingSettings, ModelSettings
    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
    from ai_psychiatrist.services.reference_store import ReferenceStore

logger = get_logger(__name__)


@dataclass
class ReferenceBundle:
    """Bundle of reference examples for all PHQ-8 items.

    Used to provide few-shot examples in quantitative assessment prompts.
    """

    item_references: dict[PHQ8Item, list[SimilarityMatch]]

    def format_for_prompt(self) -> str:
        """Format references as prompt text.

        Returns:
            Formatted reference string for LLM prompt.
        """
        blocks: list[str] = []
        for item in PHQ8Item.all_items():
            matches = self.item_references.get(item, [])
            if not matches:
                block = (
                    f"[{item.value}]\n"
                    "<Reference Examples>\n"
                    "No valid evidence found\n"
                    "</Reference Examples>"
                )
            else:
                lines: list[str] = []
                for match in matches:
                    score_text = (
                        f"(Score: {match.reference_score})"
                        if match.reference_score is not None
                        else "(Score: N/A)"
                    )
                    lines.append(f"{score_text}\n{match.chunk.text}")
                block = (
                    f"[{item.value}]\n"
                    "<Reference Examples>\n\n" + "\n\n".join(lines) + "\n\n</Reference Examples>"
                )
            blocks.append(block)
        return "\n\n".join(blocks)


class EmbeddingService:
    """Service for generating embeddings and finding similar chunks.

    Implements the embedding-based few-shot prompting approach from Section 2.4.2.
    Uses Ollama for embedding generation and cosine similarity for retrieval.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        reference_store: ReferenceStore,
        settings: EmbeddingSettings,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            llm_client: LLM client for generating embeddings.
            reference_store: Pre-computed reference embeddings.
            settings: Embedding configuration.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm_client = llm_client
        self._reference_store = reference_store
        self._dimension = settings.dimension
        self._top_k = settings.top_k_references
        self._min_chars = settings.min_evidence_chars
        self._model_settings = model_settings

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
        model = self._model_settings.embedding_model if self._model_settings else None

        embedding = await self._llm_client.simple_embed(
            text=text,
            model=model,
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
        all_refs = self._reference_store.get_all_embeddings()

        if not all_refs:
            logger.warning("No reference embeddings available")
            return []

        query_array = np.array([query_embedding])
        matches: list[SimilarityMatch] = []
        lookup_item = item or PHQ8Item.NO_INTEREST

        for participant_id, chunks in all_refs.items():
            for chunk_text, embedding in chunks:
                if len(embedding) != len(query_embedding):
                    # Log mismatch - this shouldn't happen if ReferenceStore loads correctly
                    logger.warning(
                        "Dimension mismatch between query and reference",
                        query_dim=len(query_embedding),
                        ref_dim=len(embedding),
                        participant_id=participant_id,
                    )
                    continue

                ref_array = np.array([embedding])
                raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])

                # Transform cosine similarity from [-1, 1] to [0, 1] (BUG-010 fix)
                # This gives: -1 (opposite) -> 0, 0 (orthogonal) -> 0.5, 1 (identical) -> 1
                sim = (1.0 + raw_cos) / 2.0

                # Get item-specific score
                score = self._reference_store.get_score(participant_id, lookup_item)

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

            # Get embedding and find similar with item-specific scores
            query_emb = await self.embed_text(combined_text)

            if not query_emb:
                item_references[item] = []
                continue

            # Find similar chunks with this item's scores
            matches = self._compute_similarities(query_emb, item=item)
            matches.sort(key=lambda x: x.similarity, reverse=True)
            top_matches = matches[: self._top_k]

            item_references[item] = top_matches

            logger.debug(
                "Found references for item",
                item=item.value,
                match_count=len(top_matches),
                top_similarity=top_matches[0].similarity if top_matches else 0,
            )

        return ReferenceBundle(item_references=item_references)
