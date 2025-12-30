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

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ai_psychiatrist.config import LLMBackend, resolve_reference_embeddings_path
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.domain.exceptions import (
    EmbeddingArtifactMismatchError,
    EmbeddingDimensionMismatchError,
)
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.logging import get_logger
from ai_psychiatrist.services.chunk_scoring import PHQ8_ITEM_KEY_SET, chunk_scoring_prompt_hash

if TYPE_CHECKING:
    from pathlib import Path

    from ai_psychiatrist.config import (
        DataSettings,
        EmbeddingBackendSettings,
        EmbeddingSettings,
        ModelSettings,
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


@dataclass
class EmbeddingPaths:
    """Paths to embedding artifacts."""

    npz: Path
    json: Path
    meta: Path


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
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize reference store.

        Args:
            data_settings: Data path configuration.
            embedding_settings: Embedding configuration.
            embedding_backend_settings: Embedding backend configuration (optional).
            model_settings: Model configuration (optional).
        """
        self._data_settings = data_settings
        self._embeddings_path = resolve_reference_embeddings_path(data_settings, embedding_settings)
        self._train_csv = data_settings.train_csv
        self._dev_csv = data_settings.dev_csv
        self._dimension = embedding_settings.dimension
        self._embedding_settings = embedding_settings

        if embedding_backend_settings is None:
            from ai_psychiatrist.config import get_settings  # noqa: PLC0415

            embedding_backend_settings = get_settings().embedding_config
        self._embedding_backend = embedding_backend_settings

        if model_settings is None:
            from ai_psychiatrist.config import get_settings  # noqa: PLC0415

            model_settings = get_settings().model
        self._model_settings = model_settings

        # Lazy-loaded data
        self._embeddings: dict[int, list[tuple[str, list[float]]]] | None = None
        self._tags: dict[int, list[list[str]]] | None = None
        self._chunk_scores: dict[int, list[dict[str, int | None]]] | None = None
        self._scores_df: pd.DataFrame | None = None

    def _get_texts_path(self) -> Path:
        """Get path to the JSON sidecar file containing text chunks."""
        return self._embeddings_path.with_suffix(".json")

    def _get_tags_path(self) -> Path:
        """Get path to the JSON sidecar file containing item tags."""
        return self._embeddings_path.with_suffix(".tags.json")

    def _get_chunk_scores_path(self) -> Path:
        """Get path to the JSON sidecar file containing chunk scores."""
        return self._embeddings_path.with_suffix(".chunk_scores.json")

    def _get_chunk_scores_meta_path(self) -> Path:
        """Get path to the JSON metadata sidecar for chunk scores."""
        return self._embeddings_path.with_suffix(".chunk_scores.meta.json")

    def _calculate_split_hash(self, split: str) -> str | None:
        """Calculate hash of the split CSV referenced by an embeddings artifact (if available)."""
        if split == "avec-train":
            csv_path = self._data_settings.train_csv
        elif split == "paper-train":
            csv_path = self._data_settings.base_dir / "paper_splits" / "paper_split_train.csv"
        else:
            return None

        if not csv_path.exists():
            return None

        return hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]

    def _calculate_split_ids_hash(self, split: str) -> str | None:
        """Calculate hash of the sorted participant IDs in the split (semantic provenance)."""
        if split == "avec-train":
            csv_path = self._data_settings.train_csv
        elif split == "paper-train":
            csv_path = self._data_settings.base_dir / "paper_splits" / "paper_split_train.csv"
        else:
            return None

        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            if "Participant_ID" not in df.columns:
                return None
            ids = sorted(df["Participant_ID"].astype(int).tolist())
            ids_str = ",".join(map(str, ids))
            return hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:12]
        except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None

    def _derive_artifact_ids_hash(self) -> str | None:
        """Derive IDs hash from the JSON sidecar (legacy fallback)."""
        json_path = self._get_texts_path()
        if not json_path.exists():
            return None
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # JSON keys are participant IDs (str)
                ids = sorted(int(k) for k in data)
                ids_str = ",".join(map(str, ids))
                return hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:12]
        except (TypeError, ValueError, OSError):
            return None

    def _validate_backend(self, metadata: dict[str, Any]) -> list[str]:
        """Validate backend configuration."""
        errors: list[str] = []
        stored_backend = metadata.get("backend")
        current_backend = self._embedding_backend.backend.value
        if stored_backend is None:
            logger.debug("Metadata missing 'backend' field, skipping backend validation")
        elif stored_backend != current_backend:
            errors.append(
                f"backend mismatch: artifact='{stored_backend}', config='{current_backend}'"
            )
        return errors

    def _validate_model(self, metadata: dict[str, Any]) -> list[str]:
        """Validate model configuration."""
        errors: list[str] = []
        stored_model = metadata.get("model")
        if stored_model is None:
            logger.debug("Metadata missing 'model' field, skipping model validation")
        else:
            current_model = self._model_settings.embedding_model
            resolved_model = resolve_model_name(
                current_model, LLMBackend(self._embedding_backend.backend.value)
            )
            if stored_model != resolved_model:
                errors.append(
                    f"model mismatch: artifact='{stored_model}', config='{resolved_model}'"
                )
        return errors

    def _validate_dimension(self, metadata: dict[str, Any]) -> list[str]:
        """Validate dimension configuration."""
        errors: list[str] = []
        stored_dim = metadata.get("dimension")
        current_dim = self._embedding_settings.dimension
        if stored_dim is None:
            logger.debug("Metadata missing 'dimension' field, skipping dimension validation")
        elif stored_dim != current_dim:
            errors.append(f"dimension mismatch: artifact={stored_dim}, config={current_dim}")
        return errors

    def _validate_chunk_params(self, metadata: dict[str, Any]) -> list[str]:
        """Validate chunking parameters."""
        errors: list[str] = []
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

        stored_min_chars = metadata.get("min_evidence_chars")
        current_min_chars = self._embedding_settings.min_evidence_chars
        if stored_min_chars is None:
            logger.debug(
                "Metadata missing 'min_evidence_chars' field, "
                "skipping min_evidence_chars validation"
            )
        elif stored_min_chars != current_min_chars:
            errors.append(
                "min_evidence_chars mismatch: "
                f"artifact={stored_min_chars}, config={current_min_chars}"
            )
        return errors

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate embedding artifact matches current config."""
        errors: list[str] = []

        errors.extend(self._validate_backend(metadata))
        errors.extend(self._validate_model(metadata))
        errors.extend(self._validate_dimension(metadata))
        errors.extend(self._validate_chunk_params(metadata))

        # Split integrity check (Semantic > strict)
        stored_split = metadata.get("split")
        if stored_split is None:
            logger.debug("Metadata missing 'split', skipping split validation")
        else:
            self._validate_split_integrity(metadata, str(stored_split), errors)

        if errors:
            raise EmbeddingArtifactMismatchError(
                "Embedding artifact validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nRegenerate embeddings or update config to match."
            )

    def _validate_split_integrity(
        self,
        metadata: dict[str, Any],
        split: str,
        errors: list[str],
    ) -> None:
        """Validate split integrity using IDs hash (preferred) or CSV hash (legacy)."""
        stored_ids_hash = metadata.get("split_ids_hash")
        stored_csv_hash = metadata.get("split_csv_hash")

        # 1. Check if current split is accessible
        current_ids_hash = self._calculate_split_ids_hash(split)
        if current_ids_hash is None:
            logger.debug("Split CSV not available/readable; skipping split validation", split=split)
            return

        # 2. Path 1: Semantic validation (Preferred - New Artifacts)
        if stored_ids_hash:
            if stored_ids_hash != current_ids_hash:
                errors.append(
                    f"split_ids_hash mismatch: artifact='{stored_ids_hash}', "
                    f"current='{current_ids_hash}' (split='{split}')"
                )
            elif stored_csv_hash:
                # Audit check (warn only)
                current_csv_hash = self._calculate_split_hash(split)
                if stored_csv_hash != current_csv_hash:
                    logger.warning(
                        "split_csv_hash mismatch (safe, IDs match)",
                        artifact=stored_csv_hash,
                        current=current_csv_hash,
                        split=split,
                    )
            return

        # 3. Path 2: Legacy validation (No split_ids_hash)
        if isinstance(stored_csv_hash, str) and stored_csv_hash in {"missing", "unknown"}:
            logger.debug("Legacy artifact with unknown hash; skipping")
            return

        # Try strict CSV hash first
        current_csv_hash = self._calculate_split_hash(split)
        if stored_csv_hash and stored_csv_hash == current_csv_hash:
            return  # Exact match, all good

        # If strict CSV hash fails, we MUST verify IDs semantically (Fallback)
        # We need to derive artifact IDs from the sidecar
        derived_ids_hash = self._derive_artifact_ids_hash()

        if derived_ids_hash and derived_ids_hash == current_ids_hash:
            logger.warning(
                "split_csv_hash mismatch (safe, derived IDs match)",
                artifact=stored_csv_hash,
                current=current_csv_hash,
                split=split,
            )
            return

        # Real failure
        msg = "split_csv_hash mismatch"
        if derived_ids_hash:
            msg += " AND split_ids mismatch (derived)"

        errors.append(
            f"{msg}: artifact='{stored_csv_hash}', current='{current_csv_hash}' (split='{split}')"
        )

    def _resolve_embedding_paths(self) -> EmbeddingPaths | None:
        """Resolve and validate existence of embedding artifact paths."""
        npz_path = self._embeddings_path
        json_path = self._get_texts_path()

        if not npz_path.exists():
            logger.warning("Embeddings NPZ file not found", path=str(npz_path))
            return None

        if not json_path.exists():
            logger.warning("Embeddings JSON sidecar not found", path=str(json_path))
            return None

        return EmbeddingPaths(
            npz=npz_path,
            json=json_path,
            meta=self._embeddings_path.with_suffix(".meta.json"),
        )

    def _load_texts_json(self, path: Path) -> dict[str, Any]:
        """Load text chunks from JSON sidecar."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict in {path}, got {type(data)}")
            return data

    def _combine_and_normalize(
        self,
        texts_data: dict[str, Any],
        npz_data: Any,  # numpy.lib.npyio.NpzFile
    ) -> dict[int, list[tuple[str, list[float]]]]:
        """Combine texts with embeddings and normalize."""
        normalized: dict[int, list[tuple[str, list[float]]]] = {}
        skipped_chunks = 0
        total_chunks = 0
        actual_dim_sample = 0
        require_alignment = (
            self._embedding_settings.enable_item_tag_filter
            or self._embedding_settings.reference_score_source == "chunk"
        )

        for pid_str, texts in texts_data.items():
            pid_int = int(pid_str)
            emb_key = f"emb_{pid_int}"

            if emb_key not in npz_data:
                if require_alignment:
                    raise EmbeddingArtifactMismatchError(
                        f"Missing embeddings for participant {pid_int} "
                        "(required for aligned sidecars)"
                    )
                logger.warning("No embeddings found for participant", participant_id=pid_int)
                continue

            embeddings = npz_data[emb_key]

            if len(texts) != len(embeddings):
                if require_alignment:
                    raise EmbeddingArtifactMismatchError(
                        f"Text/embedding count mismatch for participant {pid_int}: "
                        f"texts={len(texts)}, embeddings={len(embeddings)}"
                    )
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
                    if require_alignment:
                        raise EmbeddingDimensionMismatchError(
                            expected=self._dimension,
                            actual=emb_len,
                        )
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

        return normalized

    @staticmethod
    def _validate_tags_top_level(
        tags_data: Any,
        texts_data: dict[str, Any],
        tags_path: Path,
    ) -> dict[str, Any]:
        if not isinstance(tags_data, dict):
            raise EmbeddingArtifactMismatchError(
                f"Invalid tags schema in {tags_path.name}: expected object at top level"
            )

        # Validate participant ids match exactly
        texts_pids = set(texts_data.keys())
        tags_pids = set(tags_data.keys())
        missing_pids = texts_pids - tags_pids
        if missing_pids:
            missing_preview = ", ".join(sorted(missing_pids)[:5])
            raise EmbeddingArtifactMismatchError(
                f"Missing tags for participants in {tags_path.name}: {missing_preview}"
            )

        extra_pids = tags_pids - texts_pids
        if extra_pids:
            extra_preview = ", ".join(sorted(extra_pids)[:5])
            raise EmbeddingArtifactMismatchError(
                f"Extra participants present in {tags_path.name}: {extra_preview}"
            )

        return tags_data

    @staticmethod
    def _validate_chunk_tags(
        pid: int,
        chunk_idx: int,
        raw_chunk_tags: Any,
        tags_path: Path,
        valid_tags: set[str],
    ) -> list[str]:
        if not isinstance(raw_chunk_tags, list):
            raise EmbeddingArtifactMismatchError(
                f"Invalid tags schema for participant {pid} chunk {chunk_idx} "
                f"in {tags_path.name}: expected list"
            )

        chunk_tags: list[str] = []
        for raw_tag in raw_chunk_tags:
            if not isinstance(raw_tag, str):
                raise EmbeddingArtifactMismatchError(
                    f"Invalid tag type for participant {pid} chunk {chunk_idx} "
                    f"in {tags_path.name}: expected string"
                )
            if raw_tag not in valid_tags:
                raise EmbeddingArtifactMismatchError(
                    f"Unknown tag '{raw_tag}' for participant {pid} chunk {chunk_idx} "
                    f"in {tags_path.name}"
                )
            chunk_tags.append(raw_tag)

        return chunk_tags

    @staticmethod
    def _validate_participant_tags(
        pid: int,
        raw_p_tags: Any,
        expected_len: int,
        tags_path: Path,
        valid_tags: set[str],
    ) -> list[list[str]]:
        if not isinstance(raw_p_tags, list):
            raise EmbeddingArtifactMismatchError(
                f"Invalid tags schema for participant {pid} in {tags_path.name}: expected list"
            )

        if len(raw_p_tags) != expected_len:
            raise EmbeddingArtifactMismatchError(
                f"Tag count mismatch for participant {pid}: "
                f"texts={expected_len}, tags={len(raw_p_tags)}"
            )

        participant_tags: list[list[str]] = []
        for chunk_idx, raw_chunk_tags in enumerate(raw_p_tags):
            participant_tags.append(
                ReferenceStore._validate_chunk_tags(
                    pid=pid,
                    chunk_idx=chunk_idx,
                    raw_chunk_tags=raw_chunk_tags,
                    tags_path=tags_path,
                    valid_tags=valid_tags,
                )
            )

        return participant_tags

    def _load_tags(self, texts_data: dict[str, Any]) -> None:
        """Load and validate item tags sidecar.

        Called only when enable_item_tag_filter=True.
        Raises on any error (no silent fallbacks).
        """
        tags_path = self._get_tags_path()
        if not tags_path.exists():
            raise EmbeddingArtifactMismatchError(
                f"Tag filtering enabled but tags file missing: {tags_path}. "
                "Either disable tag filtering (EMBEDDING_ENABLE_ITEM_TAG_FILTER=false) "
                "or regenerate embeddings with tags."
            )

        with tags_path.open("r", encoding="utf-8") as f:
            tags_data = json.load(f)

        tags_data = self._validate_tags_top_level(tags_data, texts_data, tags_path)

        # Validate tags integrity against texts
        valid_tags = {f"PHQ8_{item.value}" for item in PHQ8Item}
        validated_tags: dict[int, list[list[str]]] = {}
        for pid_str, texts in texts_data.items():
            pid = int(pid_str)
            validated_tags[pid] = self._validate_participant_tags(
                pid=pid,
                raw_p_tags=tags_data[pid_str],
                expected_len=len(texts),
                tags_path=tags_path,
                valid_tags=valid_tags,
            )

        self._tags = validated_tags
        logger.info("Item tags loaded", participants=len(self._tags))

    def _raise_or_warn_chunk_scores_provenance(self, message: str) -> None:
        if self._embedding_settings.allow_chunk_scores_prompt_hash_mismatch:
            logger.warning(message)
            return
        raise EmbeddingArtifactMismatchError(message)

    def _validate_chunk_scores_prompt_hash(self) -> None:
        meta_path = self._get_chunk_scores_meta_path()
        if not meta_path.exists():
            self._raise_or_warn_chunk_scores_provenance(
                f"Chunk scores metadata missing: {meta_path.name}"
            )
            return

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise EmbeddingArtifactMismatchError(
                f"Invalid chunk scores metadata in {meta_path.name}: expected object"
            )

        stored_hash = meta.get("prompt_hash")
        expected_hash = chunk_scoring_prompt_hash()
        if stored_hash != expected_hash:
            self._raise_or_warn_chunk_scores_provenance(
                "Chunk scores prompt hash mismatch "
                f"(artifact={stored_hash!r}, expected={expected_hash!r})"
            )

    def _validate_chunk_scores_participant_ids(
        self,
        raw_data: dict[str, Any],
        *,
        scores_path: Path,
    ) -> None:
        if self._embeddings is None:
            raise RuntimeError("Embeddings must be loaded before validating chunk scores")

        expected_pids = {str(pid) for pid in self._embeddings}
        data_pids = set(raw_data)

        missing_pids = expected_pids - data_pids
        if missing_pids:
            missing_preview = ", ".join(sorted(missing_pids)[:5])
            raise EmbeddingArtifactMismatchError(
                f"Missing chunk scores for participants in {scores_path.name}: {missing_preview}"
            )

        extra_pids = data_pids - expected_pids
        if extra_pids:
            extra_preview = ", ".join(sorted(extra_pids)[:5])
            raise EmbeddingArtifactMismatchError(
                f"Extra participants present in {scores_path.name}: {extra_preview}"
            )

    @staticmethod
    def _validate_chunk_score_map(
        *,
        pid: int,
        chunk_idx: int,
        raw_map: Any,
        valid_keys: set[str],
    ) -> dict[str, int | None]:
        if not isinstance(raw_map, dict):
            raise EmbeddingArtifactMismatchError(
                f"Invalid score map for {pid} chunk {chunk_idx}: expected object"
            )

        key_set = set(raw_map)
        if key_set != valid_keys:
            missing = sorted(valid_keys - key_set)
            extra = sorted(key_set - valid_keys)
            raise EmbeddingArtifactMismatchError(
                f"Invalid chunk score keys for {pid} chunk {chunk_idx}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

        validated_map: dict[str, int | None] = {}
        for key in valid_keys:
            val = raw_map[key]
            if val is None:
                validated_map[key] = None
                continue

            # Reject bool explicitly (bool is a subclass of int).
            if isinstance(val, bool) or not isinstance(val, int) or not (0 <= val <= 3):
                raise EmbeddingArtifactMismatchError(
                    f"Invalid value for {pid} chunk {chunk_idx} key {key}: {val!r}"
                )
            validated_map[key] = val

        return validated_map

    def _validate_chunk_scores_data(
        self,
        raw_data: dict[str, Any],
        valid_keys: set[str],
    ) -> dict[int, list[dict[str, int | None]]]:
        """Validate raw chunk scores against embeddings structure."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings must be loaded before validating chunk scores")

        scores_path = self._get_chunk_scores_path()
        self._validate_chunk_scores_participant_ids(raw_data, scores_path=scores_path)

        validated_scores: dict[int, list[dict[str, int | None]]] = {}

        for pid_str, chunks_data in raw_data.items():
            try:
                pid = int(pid_str)
            except ValueError as e:
                raise EmbeddingArtifactMismatchError(
                    f"Invalid participant id in {scores_path.name}: {pid_str!r}"
                ) from e
            if pid not in self._embeddings:
                raise EmbeddingArtifactMismatchError(
                    f"Chunk scores include unknown participant {pid}"
                )

            expected_count = len(self._embeddings[pid])
            if not isinstance(chunks_data, list):
                raise EmbeddingArtifactMismatchError(
                    f"Invalid chunk scores for {pid}: expected list"
                )

            if len(chunks_data) != expected_count:
                raise EmbeddingArtifactMismatchError(
                    f"Chunk count mismatch for {pid}: "
                    f"embeddings={expected_count}, scores={len(chunks_data)}"
                )

            validated_participant_scores: list[dict[str, int | None]] = []
            for idx, chunk_score_map in enumerate(chunks_data):
                validated_participant_scores.append(
                    self._validate_chunk_score_map(
                        pid=pid,
                        chunk_idx=idx,
                        raw_map=chunk_score_map,
                        valid_keys=valid_keys,
                    )
                )

            validated_scores[pid] = validated_participant_scores

        return validated_scores

    def _load_chunk_scores(self) -> None:
        """Load and validate chunk scores sidecar."""
        if self._chunk_scores is not None:
            return

        scores_path = self._get_chunk_scores_path()
        if not scores_path.exists():
            self._chunk_scores = {}
            return

        # Ensure embeddings are loaded for validation context (chunk counts)
        if self._embeddings is None:
            self._load_embeddings()

        try:
            self._validate_chunk_scores_prompt_hash()

            with scores_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, dict):
                raise EmbeddingArtifactMismatchError(
                    f"Invalid chunk scores schema in {scores_path.name}: expected object"
                )

            self._chunk_scores = self._validate_chunk_scores_data(raw_data, set(PHQ8_ITEM_KEY_SET))
            logger.info("Chunk scores loaded", participants=len(self._chunk_scores))

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load chunk scores file",
                path=str(scores_path),
                error=str(e),
            )
            self._chunk_scores = {}

    def has_chunk_scores(self) -> bool:
        """Check if chunk-level scores are available.

        Returns:
            True if the sidecar was successfully loaded and is not empty.
        """
        self._load_chunk_scores()
        return bool(self._chunk_scores)

    def get_chunk_score(
        self,
        participant_id: int,
        chunk_index: int,
        item: PHQ8Item,
    ) -> int | None:
        """Get chunk-level PHQ-8 item score.

        Args:
            participant_id: Participant ID.
            chunk_index: Index of the chunk in the participant's list.
            item: PHQ-8 item.

        Returns:
            Score (0-3) or None if unavailable.
        """
        self._load_chunk_scores()
        if not self._chunk_scores:
            return None

        participant_scores = self._chunk_scores.get(participant_id)
        if not participant_scores:
            return None

        if not (0 <= chunk_index < len(participant_scores)):
            return None

        chunk_map = participant_scores[chunk_index]
        key = f"PHQ8_{item.value}"

        val = chunk_map.get(key)
        if isinstance(val, int) and 0 <= val <= 3:
            return val

        return None

    def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
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

        paths = self._resolve_embedding_paths()
        if paths is None:
            self._embeddings = {}

            return self._embeddings

        # Load and validate metadata (if present)
        if paths.meta.exists():
            try:
                with paths.meta.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if not isinstance(metadata, dict):
                    raise TypeError(f"Expected dict in {paths.meta}, got {type(metadata)}")
                self._validate_metadata(metadata)
            except (OSError, TypeError, ValueError) as e:
                logger.warning("Failed to load/validate metadata", error=str(e))
        elif self._embedding_backend.backend.value != "ollama":
            logger.warning(
                "Embeddings metadata not found; skipping artifact validation",
                meta_path=str(paths.meta),
            )

        logger.info(
            "Loading reference embeddings",
            npz_path=str(paths.npz),
            json_path=str(paths.json),
        )

        texts_data = self._load_texts_json(paths.json)
        npz_data = np.load(paths.npz, allow_pickle=False)

        # Spec 38: Skip-if-disabled, crash-if-broken.
        if self._embedding_settings.enable_item_tag_filter:
            self._load_tags(texts_data)
        else:
            self._tags = {}
            logger.debug("Tag filtering disabled, skipping tag loading")

        try:
            normalized = self._combine_and_normalize(texts_data, npz_data)
        finally:
            npz_data.close()

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

    def get_participant_tags(self, participant_id: int) -> list[list[str]]:
        """Get item tags for a specific participant's chunks.

        Args:
            participant_id: Participant ID.

        Returns:
            List of tag lists (one list of tags per chunk).
            Returns empty list if no tags available.
        """
        # Ensure loaded
        if self._tags is None:
            self._load_embeddings()

        if self._tags is None:
            return []

        return self._tags.get(participant_id, [])

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
