"""Pre-computed reference embeddings store.

Implements storage for pre-computed transcript embeddings used in
few-shot prompting (Paper Section 2.4.2). Provides lazy loading,
L2 normalization, and ground truth score lookup.

Storage Format (NPZ + JSON sidecar):
    - {embeddings_path}.npz: Embeddings as numpy arrays (key: "emb_{pid}")
    - {embeddings_path}.json: Text chunks (key: str(pid) -> list[str])

This format replaces pickle for security (no arbitrary code execution).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ai_psychiatrist.config import resolve_reference_embeddings_path
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import (
    EmbeddingArtifactMismatchError,
    EmbeddingDimensionMismatchError,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ai_psychiatrist.config import (
        DataSettings,
        EmbeddingBackendSettings,
        EmbeddingSettings,
    )

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
        embedding_backend_settings: EmbeddingBackendSettings | None = None,
    ) -> None:
        """Initialize reference store.

        Args:
            data_settings: Data path configuration.
            embedding_settings: Embedding configuration.
            embedding_backend_settings: Embedding backend configuration (optional).
        """
        self._embeddings_path = resolve_reference_embeddings_path(data_settings, embedding_settings)
        self._train_csv = data_settings.train_csv
        self._dev_csv = data_settings.dev_csv
        self._dimension = embedding_settings.dimension
        self._embedding_settings = embedding_settings

        if embedding_backend_settings is None:
            from ai_psychiatrist.config import get_settings  # noqa: PLC0415

            embedding_backend_settings = get_settings().embedding_backend
        self._embedding_backend = embedding_backend_settings

        # Lazy-loaded data
        self._embeddings: dict[int, list[tuple[str, list[float]]]] | None = None
        self._scores_df: pd.DataFrame | None = None

    def _get_texts_path(self) -> Path:
        """Get path to the JSON sidecar file containing text chunks."""
        return self._embeddings_path.with_suffix(".json")

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate embedding artifact matches current config."""
        errors: list[str] = []

        # Backend check
        stored_backend = metadata.get("backend")
        current_backend = self._embedding_backend.backend.value
        if stored_backend is None:
            logger.debug("Metadata missing 'backend' field, skipping backend validation")
        elif stored_backend != current_backend:
            errors.append(
                f"backend mismatch: artifact='{stored_backend}', config='{current_backend}'"
            )

        # Dimension check
        stored_dim = metadata.get("dimension")
        current_dim = self._embedding_settings.dimension
        if stored_dim is None:
            logger.debug("Metadata missing 'dimension' field, skipping dimension validation")
        elif stored_dim != current_dim:
            errors.append(f"dimension mismatch: artifact={stored_dim}, config={current_dim}")

        # Chunk params check
        stored_chunk = metadata.get("chunk_size")
        current_chunk = self._embedding_settings.chunk_size
        if stored_chunk is None:
            logger.debug("Metadata missing 'chunk_size' field, skipping chunk_size validation")
        elif stored_chunk != current_chunk:
            errors.append(f"chunk_size mismatch: artifact={stored_chunk}, config={current_chunk}")

        stored_step = metadata.get("chunk_step")
        current_step = self._embedding_settings.chunk_step
        if stored_step is None:
            logger.debug("Metadata missing 'chunk_step' field, skipping chunk_step validation")
        elif stored_step != current_step:
            errors.append(f"chunk_step mismatch: artifact={stored_step}, config={current_step}")

        if errors:
            raise EmbeddingArtifactMismatchError(
                "Embedding artifact validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nRegenerate embeddings or update config to match."
            )

    def _load_embeddings(  # noqa: PLR0912, PLR0915
        self,
    ) -> dict[int, list[tuple[str, list[float]]]]:
        """Load pre-computed embeddings from NPZ + JSON sidecar files.

        Format:
            - NPZ file: Contains arrays keyed by "emb_{pid}" for each participant
            - JSON file: Contains {"pid": ["text1", "text2", ...], ...}
            - .meta.json: Provenance metadata (optional)

        Returns:
            Dictionary mapping participant_id -> list of (text, embedding) pairs.
        """
        if self._embeddings is not None:
            return self._embeddings

        npz_path = self._embeddings_path
        json_path = self._get_texts_path()

        if not npz_path.exists():
            logger.warning(
                "Embeddings NPZ file not found",
                path=str(npz_path),
            )
            self._embeddings = {}
            return self._embeddings

        if not json_path.exists():
            logger.warning(
                "Embeddings JSON sidecar not found",
                path=str(json_path),
            )
            self._embeddings = {}
            return self._embeddings

        # Load and validate metadata (if present)
        meta_path = self._embeddings_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self._validate_metadata(metadata)
            except Exception as e:
                # If validation fails explicitly, re-raise.
                if isinstance(e, EmbeddingArtifactMismatchError):
                    raise
                logger.warning("Failed to load/validate metadata", error=str(e))
        elif self._embedding_backend.backend.value != "ollama":
            logger.warning(
                "Embeddings metadata not found; skipping artifact validation",
                meta_path=str(meta_path),
            )

        logger.info(
            "Loading reference embeddings",
            npz_path=str(npz_path),
            json_path=str(json_path),
        )

        # Load text chunks from JSON (safe, no code execution)
        with json_path.open("r", encoding="utf-8") as f:
            texts_data: dict[str, Any] = json.load(f)

        # Load embeddings from NPZ (safe, numpy arrays only)
        npz_data = np.load(npz_path, allow_pickle=False)

        # Normalize embeddings and combine with texts
        normalized: dict[int, list[tuple[str, list[float]]]] = {}
        skipped_chunks = 0
        total_chunks = 0
        actual_dim_sample = 0

        for pid_str, texts in texts_data.items():
            pid_int = int(pid_str)
            emb_key = f"emb_{pid_int}"

            if emb_key not in npz_data:
                logger.warning(
                    "No embeddings found for participant",
                    participant_id=pid_int,
                )
                continue

            embeddings = npz_data[emb_key]

            if len(texts) != len(embeddings):
                logger.warning(
                    "Text/embedding count mismatch",
                    participant_id=pid_int,
                    texts=len(texts),
                    embeddings=len(embeddings),
                )
                continue

            norm_pairs: list[tuple[str, list[float]]] = []
            for text, embedding in zip(texts, embeddings, strict=True):
                total_chunks += 1
                emb_len = len(embedding)

                # Track actual dimension for error reporting
                if actual_dim_sample == 0:
                    actual_dim_sample = emb_len

                # Validate dimension
                if emb_len < self._dimension:
                    skipped_chunks += 1
                    logger.warning(
                        "Skipping embedding with insufficient dimension",
                        participant_id=pid_int,
                        expected=self._dimension,
                        actual=emb_len,
                    )
                    continue

                # Truncate to configured dimension
                emb = list(embedding[: self._dimension])
                # L2 normalize
                emb = self._l2_normalize(emb)
                norm_pairs.append((text, emb))

            if norm_pairs:
                normalized[pid_int] = norm_pairs

        npz_data.close()

        # Fail loudly if ALL embeddings are mismatched (BUG-009 fix)
        if total_chunks > 0 and skipped_chunks == total_chunks:
            raise EmbeddingDimensionMismatchError(
                expected=self._dimension,
                actual=actual_dim_sample,
            )

        if skipped_chunks > 0:
            logger.error(
                "Some reference embeddings had dimension mismatch",
                skipped=skipped_chunks,
                total=total_chunks,
                expected_dimension=self._dimension,
            )

        self._embeddings = normalized

        loaded_chunks = sum(len(v) for v in normalized.values())
        logger.info(
            "Embeddings loaded",
            participants=len(normalized),
            total_chunks=loaded_chunks,
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
