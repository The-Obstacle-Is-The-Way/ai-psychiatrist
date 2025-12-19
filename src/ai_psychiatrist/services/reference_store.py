"""Pre-computed reference embeddings store.

Implements storage for pre-computed transcript embeddings used in
few-shot prompting (Paper Section 2.4.2). Provides lazy loading,
L2 normalization, and ground truth score lookup.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import DataSettings, EmbeddingSettings

logger = get_logger(__name__)


# Mapping from PHQ8Item to CSV column names in DAIC-WOZ ground truth
PHQ8_COLUMN_MAP: dict[PHQ8Item, str] = {
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
    with their associated PHQ-8 scores. Embeddings are truncated to
    configured dimension and L2-normalized for cosine similarity.
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
        """Load pre-computed embeddings from pickle file.

        Returns:
            Dictionary mapping participant_id -> list of (text, embedding) pairs.
        """
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

        with self._embeddings_path.open("rb") as f:
            raw_data = pickle.load(f)

        # Normalize embeddings and convert participant IDs to int
        normalized: dict[int, list[tuple[str, list[float]]]] = {}

        for pid, pairs in raw_data.items():
            pid_int = int(pid)
            norm_pairs: list[tuple[str, list[float]]] = []
            for text, embedding in pairs:
                # Truncate to configured dimension
                emb = list(embedding[: self._dimension])
                # L2 normalize
                emb = self._l2_normalize(emb)
                norm_pairs.append((text, emb))
            normalized[pid_int] = norm_pairs

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
        """Load ground truth PHQ-8 scores from CSV files.

        Returns:
            DataFrame with participant scores.
        """
        if self._scores_df is not None:
            return self._scores_df

        dfs: list[pd.DataFrame] = []
        for path in [self._train_csv, self._dev_csv]:
            if path.exists():
                df = pd.read_csv(path)
                df["Participant_ID"] = df["Participant_ID"].astype(int)
                dfs.append(df)
                logger.debug("Loaded scores file", path=str(path), rows=len(df))

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
        """L2 normalize an embedding vector.

        Args:
            embedding: Raw embedding vector.

        Returns:
            L2-normalized embedding (unit length).
        """
        arr = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        result: list[float] = arr.tolist()
        return result

    def get_all_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
        """Get all reference embeddings.

        Returns:
            Dictionary mapping participant_id -> list of (text, embedding) pairs.
        """
        return self._load_embeddings()

    def get_participant_embeddings(self, participant_id: int) -> list[tuple[str, list[float]]]:
        """Get embeddings for a specific participant.

        Args:
            participant_id: Participant ID.

        Returns:
            List of (text, embedding) pairs for the participant.
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
        if df.empty:
            return None

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
