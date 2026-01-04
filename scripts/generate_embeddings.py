#!/usr/bin/env python3
"""Generate reference embeddings for few-shot prompting.

This script generates the pre-computed embeddings artifact required for
few-shot PHQ-8 assessment. It uses ONLY training data to avoid data leakage.

Output Format (NPZ + JSON sidecar):
    - {output_path}.npz: Embeddings as numpy arrays (key: "emb_{pid}")
    - {output_path}.json: Text chunks (key: str(pid) -> list[str])
    - {output_path}.meta.json: Provenance metadata
    - {output_path}.tags.json: Item tags (if --write-item-tags)
    - {output_path}.partial.json: Skip manifest (only in --allow-partial mode with skips)

Usage:
    # Generate embeddings (strict mode - crash on any failure)
    python scripts/generate_embeddings.py

    # Override backend
    python scripts/generate_embeddings.py --backend huggingface

    # Allow partial output for debugging (exit 2 if skips occur)
    python scripts/generate_embeddings.py --allow-partial

Exit Codes (Spec 40):
    0 - Success (all participants and chunks processed)
    1 - Failure (error in strict mode, or fatal error)
    2 - Partial success (--allow-partial mode with skips)

Paper Reference:
    - Section 2.4.2: Embedding-based few-shot prompting
    - Appendix D: Optimal hyperparameters (chunk_size=8, step_size=2, dim=4096)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    EmbeddingBackendSettings,
    EmbeddingSettings,
    LLMBackend,
    LoggingSettings,
    ModelSettings,
    Settings,
    get_settings,
    resolve_reference_embeddings_path,
)
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingRequest
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.infrastructure.validation import validate_embedding
from ai_psychiatrist.services.transcript import TranscriptService

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

logger = get_logger(__name__)

# Exit codes (Spec 40)
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PARTIAL = 2  # Partial success (some skips in --allow-partial mode)

PAPER_SPLITS_DIRNAME = "paper_splits"


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails in strict mode.

    Attributes:
        participant_id: The participant that failed.
        chunk_index: The chunk that failed (None for participant-level failures).
    """

    def __init__(self, message: str, participant_id: int, *, chunk_index: int | None = None):
        self.participant_id = participant_id
        self.chunk_index = chunk_index
        super().__init__(message)


class KeywordTagger:
    """Deterministic keyword-based tagger for PHQ-8 items."""

    def __init__(self, keywords_path: Path) -> None:
        if not keywords_path.exists():
            raise FileNotFoundError(f"Keywords file not found: {keywords_path}")

        with keywords_path.open("r", encoding="utf-8") as f:
            self._keywords_map: dict[str, list[str]] = yaml.safe_load(f)

        # Validate keys against PHQ8Item
        valid_items = {f"PHQ8_{item.value}" for item in PHQ8Item}
        for key in self._keywords_map:
            if key not in valid_items:
                logger.warning("Unknown item key in keywords file", key=key)

    def tag_chunk(self, text: str) -> list[str]:
        """Tag a chunk with PHQ-8 items based on keyword matches."""
        tags: list[str] = []
        text_lower = text.lower()

        for item_key, keywords in self._keywords_map.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    tags.append(item_key)
                    break  # One match per item is sufficient

        return sorted(tags)


@dataclass
class GenerationConfig:
    """Configuration for embedding generation."""

    data_settings: DataSettings
    embedding_settings: EmbeddingSettings
    backend_settings: EmbeddingBackendSettings
    model_settings: ModelSettings
    chunk_size: int
    step_size: int
    dimension: int
    min_chars: int
    model: str
    resolved_model: str
    output_path: Path
    split: str
    dry_run: bool
    # Spec 34: Tagging support
    write_item_tags: bool
    tagger_type: str
    keywords_path: Path
    # Spec 40: Fail-fast mode (default strict)
    allow_partial: bool = False


@dataclass
class GenerationResult:
    """Result of embedding generation."""

    embeddings: dict[int, list[tuple[str, list[float]]]]
    tags: dict[int, list[list[str]]]  # pid -> list of tags per chunk
    total_chunks: int
    # Spec 40: Skip tracking for --allow-partial mode
    skipped_participants: list[int] = field(default_factory=list)
    total_skipped_chunks: int = 0
    chunk_skip_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_skips(self) -> bool:
        """True if any participants or chunks were skipped."""
        return len(self.skipped_participants) > 0 or self.total_skipped_chunks > 0


@dataclass
class SkipReport:
    """Aggregated skip information for a single participant (partial mode only)."""

    skipped_chunks: int = 0
    chunk_skip_reasons: dict[str, int] = field(default_factory=dict)

    def record_chunk_skip(self, reason: str) -> None:
        self.skipped_chunks += 1
        self.chunk_skip_reasons[reason] = self.chunk_skip_reasons.get(reason, 0) + 1


def create_sliding_chunks(
    transcript_text: str,
    chunk_size: int = 8,
    step_size: int = 2,
) -> list[str]:
    """Split transcript into overlapping chunks.

    Args:
        transcript_text: Full transcript text.
        chunk_size: Number of lines per chunk (Paper Appendix D: 8).
        step_size: Sliding window step (Paper Appendix D: 2).

    Returns:
        List of chunk text strings.
    """
    lines = transcript_text.split("\n")

    # Remove empty trailing lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Handle empty or whitespace-only transcripts
    if not lines:
        return []

    if len(lines) <= chunk_size:
        return ["\n".join(lines)]

    chunks = []
    for i in range(0, len(lines) - chunk_size + 1, step_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        chunks.append(chunk)

    # Ensure last lines are included
    last_start = len(lines) - chunk_size
    if last_start > 0 and (last_start % step_size) != 0:
        final_chunk = "\n".join(lines[last_start:])
        if final_chunk not in chunks:
            chunks.append(final_chunk)

    return chunks


async def generate_embedding(
    client: EmbeddingClient,
    text: str,
    model: str,
    dimension: int,
) -> list[float]:
    """Generate L2-normalized embedding for text.

    Args:
        client: Embedding client.
        text: Text to embed.
        model: Embedding model name.
        dimension: Target embedding dimension.

    Returns:
        L2-normalized embedding vector.
    """
    response = await client.embed(
        EmbeddingRequest(
            text=text,
            model=model,
            dimension=dimension,
        )
    )
    return list(response.embedding)


def _paper_train_split_path(data_settings: DataSettings) -> Path:
    """Resolve paper-style train split CSV path."""
    return data_settings.base_dir / PAPER_SPLITS_DIRNAME / "paper_split_train.csv"


def get_participant_ids(data_settings: DataSettings, *, split: str) -> list[int]:
    """Get participant IDs for embedding generation.

    Uses ONLY training data to avoid data leakage in few-shot retrieval.

    Args:
        data_settings: Data path configuration.
        split: "avec-train" or "paper-train".

    Returns:
        List of training participant IDs.
    """
    import pandas as pd  # noqa: PLC0415

    if split == "avec-train":
        csv_path = data_settings.train_csv
    elif split == "paper-train":
        csv_path = _paper_train_split_path(data_settings)
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return sorted(df["Participant_ID"].astype(int).tolist())


def calculate_split_hash(data_settings: DataSettings, split: str) -> str:
    """Calculate hash of the split CSV for provenance."""

    if split == "avec-train":
        csv_path = data_settings.train_csv
    elif split == "paper-train":
        csv_path = _paper_train_split_path(data_settings)
    else:
        return "unknown"

    if not csv_path.exists():
        return "missing"

    # Hash the file content
    with csv_path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def calculate_split_ids_hash(data_settings: DataSettings, split: str) -> str:
    """Calculate hash of the participant IDs in the split for semantic provenance."""
    import pandas as pd  # noqa: PLC0415

    if split == "avec-train":
        csv_path = data_settings.train_csv
    elif split == "paper-train":
        csv_path = _paper_train_split_path(data_settings)
    else:
        return "unknown"

    if not csv_path.exists():
        return "missing"

    try:
        df = pd.read_csv(csv_path)
        ids = sorted(df["Participant_ID"].astype(int).tolist())
        # Canonical string representation: "1,2,3"
        ids_str = ",".join(map(str, ids))
        return hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:12]
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as e:
        logger.warning(f"Failed to calculate split_ids_hash: {e}")
        return "error"


def _classify_chunk_skip_reason(exc: Exception) -> str:
    """Classify chunk-level skip reasons for partial manifests (Spec 057/055)."""
    message = str(exc)
    if message.startswith("Embedding dimension mismatch: expected "):
        return "dimension_mismatch"
    if "NaN detected" in message:
        return "embedding_nan"
    if "Inf detected" in message:
        return "embedding_inf"
    if "All-zero" in message:
        return "embedding_zero"
    return "embedding_error"


async def process_participant(  # noqa: PLR0912
    client: EmbeddingClient,
    transcript_service: TranscriptService,
    participant_id: int,
    model: str,
    dimension: int,
    chunk_size: int,
    step_size: int,
    min_chars: int,
    *,
    tagger: KeywordTagger | None = None,
    allow_partial: bool = False,
) -> tuple[list[tuple[str, list[float]]], list[list[str]], SkipReport]:
    """Generate embeddings for all chunks of a participant's transcript.

    Spec 40: Fail-fast by default, with optional --allow-partial mode.

    Args:
        client: Embedding client.
        transcript_service: Transcript loading service.
        participant_id: Participant to process.
        model: Embedding model name.
        dimension: Target embedding dimension.
        chunk_size: Lines per chunk.
        step_size: Sliding window step.
        min_chars: Minimum characters for a chunk to be embedded.
        tagger: Optional tagger for generating item tags.
        allow_partial: If True, skip failures instead of crashing.

    Returns:
        Tuple of:
        - List of (chunk_text, embedding) pairs.
        - List of tag lists (one per chunk).
        - SkipReport with counts and reason codes (partial mode only).

    Raises:
        EmbeddingGenerationError: In strict mode (allow_partial=False), on any failure.
    """
    skip_report = SkipReport()

    # Load transcript (fail-fast unless allow_partial)
    try:
        transcript = transcript_service.load_transcript(participant_id)
    except Exception as e:
        if allow_partial:
            logger.warning(
                "Failed to load transcript (skipping participant)",
                participant_id=participant_id,
                error=str(e),
            )
            return [], [], skip_report
        raise EmbeddingGenerationError(
            f"Failed to load transcript for participant {participant_id}: {e}",
            participant_id=participant_id,
        ) from e

    # Create chunks
    chunks = create_sliding_chunks(transcript.text, chunk_size, step_size)

    # Empty transcript check (fail-fast unless allow_partial)
    if not chunks:
        if allow_partial:
            logger.warning(
                "No chunks produced for transcript (skipping participant)",
                participant_id=participant_id,
            )
            return [], [], skip_report
        raise EmbeddingGenerationError(
            f"No chunks produced for participant {participant_id} (empty transcript)",
            participant_id=participant_id,
        )

    results: list[tuple[str, list[float]]] = []
    chunk_tags: list[list[str]] = []

    for chunk_idx, chunk in enumerate(chunks):
        if len(chunk.strip()) < min_chars:
            continue

        # Embed chunk (fail-fast unless allow_partial)
        try:
            embedding = await generate_embedding(client, chunk, model, dimension)
            if len(embedding) != dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {dimension}, got {len(embedding)}"
                )
            validate_embedding(
                np.array(embedding, dtype=np.float32),
                context=f"generated embedding (participant {participant_id} chunk {chunk_idx})",
            )
            results.append((chunk, embedding))
        except Exception as e:
            if allow_partial:
                reason = _classify_chunk_skip_reason(e)
                logger.warning(
                    "Failed to embed chunk (skipping)",
                    participant_id=participant_id,
                    chunk_index=chunk_idx,
                    error=str(e),
                    reason=reason,
                )
                skip_report.record_chunk_skip(reason)
                continue
            raise EmbeddingGenerationError(
                f"Failed to embed chunk {chunk_idx} for participant {participant_id}: {e}",
                participant_id=participant_id,
                chunk_index=chunk_idx,
            ) from e

        # Tag chunk - ALWAYS fail-fast (tagging must be correct)
        if tagger:
            try:
                tags = tagger.tag_chunk(chunk)
            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Failed to tag chunk {chunk_idx} for participant {participant_id}: {e}",
                    participant_id=participant_id,
                    chunk_index=chunk_idx,
                ) from e
            chunk_tags.append(tags)
        else:
            chunk_tags.append([])

    # No embedded chunks check (fail-fast unless allow_partial)
    if not results:
        if allow_partial:
            logger.warning(
                "No chunks embedded for participant (skipping participant)",
                participant_id=participant_id,
            )
            return [], [], skip_report
        raise EmbeddingGenerationError(
            f"No chunks embedded for participant {participant_id}",
            participant_id=participant_id,
        )

    return results, chunk_tags, skip_report


def slugify_model(model: str) -> str:
    """Deterministic model name slugification."""
    # Qwen/Qwen3-Embedding-8B -> qwen3_8b
    # qwen3-embedding:8b -> qwen3_8b
    raw = model.split("/")[-1].lower()

    name_part, tag_part = raw, ""
    if ":" in raw:
        name_part, tag_part = raw.split(":", 1)

    base = name_part.replace("-embedding", "").replace("_embedding", "")
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    tag_part = re.sub(r"[^a-z0-9]+", "_", tag_part).strip("_")

    if tag_part and not base.endswith(f"_{tag_part}"):
        base = f"{base}_{tag_part}"

    return base


def get_output_filename(backend: str, model: str, split: str) -> str:
    """Generate standardized output filename.

    Format: {backend}_{model_slug}_{split}
    Example: huggingface_qwen3_8b_paper_train
    """
    model_slug = slugify_model(model)
    split_slug = split.replace("-", "_")
    return f"{backend}_{model_slug}_{split_slug}"


def prepare_config(args: argparse.Namespace, *, settings: Settings) -> GenerationConfig:
    """Prepare configuration from args and environment."""
    # Override backend if specified
    if args.backend:
        settings.embedding_config.backend = EmbeddingBackend(args.backend)

    data_settings = settings.data
    embedding_settings = settings.embedding
    model_settings = settings.model
    backend_settings = settings.embedding_config

    # Appendix D hyperparameters (baseline defaults)
    chunk_size = embedding_settings.chunk_size
    step_size = embedding_settings.chunk_step
    dimension = embedding_settings.dimension
    min_chars = embedding_settings.min_evidence_chars
    model = model_settings.embedding_model
    resolved_model = resolve_model_name(model, LLMBackend(backend_settings.backend.value))

    # Determine output path
    if args.output:
        output_path = args.output
    elif (
        "embeddings_path" in data_settings.model_fields_set
        or "embeddings_file" in embedding_settings.model_fields_set
    ):
        output_path = resolve_reference_embeddings_path(data_settings, embedding_settings)
    else:
        filename = get_output_filename(
            backend=backend_settings.backend.value,
            model=model,
            split=args.split,
        )
        output_path = data_settings.base_dir / "embeddings" / f"{filename}.npz"

    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")

    # Spec 34: Keywords path
    # Assuming standard project structure
    keywords_path = (
        Path(__file__).parent.parent / "src/ai_psychiatrist/resources/phq8_keywords.yaml"
    ).resolve()

    return GenerationConfig(
        data_settings=data_settings,
        embedding_settings=embedding_settings,
        backend_settings=backend_settings,
        model_settings=model_settings,
        chunk_size=chunk_size,
        step_size=step_size,
        dimension=dimension,
        min_chars=min_chars,
        model=model,
        resolved_model=resolved_model,
        output_path=output_path,
        split=args.split,
        dry_run=args.dry_run,
        write_item_tags=args.write_item_tags,
        tagger_type=args.tagger,
        keywords_path=keywords_path,
        allow_partial=args.allow_partial,
    )


async def run_generation_loop(
    config: GenerationConfig,
    client: EmbeddingClient,
    transcript_service: TranscriptService,
) -> GenerationResult:
    """Run the main generation loop.

    Spec 40: Fail-fast by default, tracks skips in --allow-partial mode.
    """
    # Get training participants only (avoid data leakage)
    try:
        participant_ids = get_participant_ids(config.data_settings, split=config.split)
        print(f"\nFound {len(participant_ids)} participants")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        if config.split == "paper-train":
            print("Hint: Run 'uv run python scripts/create_paper_split.py' first.")
        print("Please ensure DAIC-WOZ dataset is prepared.")
        raise  # Propagate to main to handle exit code

    # Initialize Tagger
    tagger: KeywordTagger | None = None
    if config.write_item_tags:
        try:
            tagger = KeywordTagger(config.keywords_path)
            print(f"Initialized KeywordTagger from {config.keywords_path}")
        except Exception as e:
            print(f"Failed to initialize tagger: {e}")
            raise

    # Process participants
    all_embeddings: dict[int, list[tuple[str, list[float]]]] = {}
    all_tags: dict[int, list[list[str]]] = {}
    total_chunks = 0

    # Spec 40: Track skips for partial mode
    skipped_participants: list[int] = []
    total_skipped_chunks = 0
    skip_reason_counts: dict[str, int] = {}

    print(f"\nProcessing {len(participant_ids)} participants...")
    for idx, pid in enumerate(participant_ids, 1):
        if idx % 10 == 0 or idx == len(participant_ids):
            print(f"  Progress: {idx}/{len(participant_ids)} participants...")

        results, chunk_tags, skip_report = await process_participant(
            client,
            transcript_service,
            pid,
            config.model,
            config.dimension,
            config.chunk_size,
            config.step_size,
            config.min_chars,
            tagger=tagger,
            allow_partial=config.allow_partial,
        )

        total_skipped_chunks += skip_report.skipped_chunks
        for reason, count in skip_report.chunk_skip_reasons.items():
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + count

        if results:
            all_embeddings[pid] = results
            all_tags[pid] = chunk_tags
            total_chunks += len(results)
        elif config.allow_partial:
            # Only reachable in partial mode (strict mode would have crashed)
            skipped_participants.append(pid)

    return GenerationResult(
        embeddings=all_embeddings,
        tags=all_tags,
        total_chunks=total_chunks,
        skipped_participants=skipped_participants,
        total_skipped_chunks=total_skipped_chunks,
        chunk_skip_reason_counts=skip_reason_counts,
    )


def save_embeddings(  # noqa: PLR0915
    result: GenerationResult,
    config: GenerationConfig,
) -> None:
    """Save embeddings, text chunks, and metadata.

    Spec 40: Uses atomic temp→rename pattern to prevent half-artifacts.
    """
    # Ensure output directory exists
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare NPZ and JSON data
    npz_arrays: dict[str, Any] = {}
    json_texts: dict[str, list[str]] = {}
    json_tags: dict[str, list[list[str]]] = {}

    for pid, pairs in result.embeddings.items():
        texts = [text for text, _ in pairs]
        embeddings = [emb for _, emb in pairs]

        json_texts[str(pid)] = texts
        npz_arrays[f"emb_{pid}"] = np.array(embeddings, dtype=np.float32)

    # Prepare tags JSON if enabled
    if config.write_item_tags and result.tags:
        for pid, tags in result.tags.items():
            json_tags[str(pid)] = tags

    actual_dimensions = [len(emb) for pairs in result.embeddings.values() for _text, emb in pairs]
    actual_dimension_min = min(actual_dimensions) if actual_dimensions else 0
    actual_dimension_max = max(actual_dimensions) if actual_dimensions else 0

    # Prepare metadata
    metadata = {
        "backend": config.backend_settings.backend.value,
        "model": config.resolved_model,
        "model_canonical": config.model,
        "dimension": config.dimension,
        "actual_dimension_min": actual_dimension_min,
        "actual_dimension_max": actual_dimension_max,
        "dimension_mismatch_count": result.chunk_skip_reason_counts.get("dimension_mismatch", 0),
        "chunk_size": config.chunk_size,
        "chunk_step": config.step_size,
        "min_evidence_chars": config.min_chars,
        "split": config.split,
        "participant_count": len(result.embeddings),
        "generated_at": datetime.now(UTC).isoformat(),
        "generator_script": "scripts/generate_embeddings.py",
        "split_csv_hash": calculate_split_hash(config.data_settings, config.split),
        "split_ids_hash": calculate_split_ids_hash(config.data_settings, config.split),
    }

    # Final paths
    npz_path = config.output_path
    json_path = config.output_path.with_suffix(".json")
    meta_path = config.output_path.with_suffix(".meta.json")
    tags_path = config.output_path.with_suffix(".tags.json")

    # Temp paths for atomic writes (Spec 40)
    tmp_npz_path = config.output_path.with_suffix(".tmp.npz")
    tmp_json_path = config.output_path.with_suffix(".tmp.json")
    tmp_meta_path = config.output_path.with_suffix(".tmp.meta.json")
    tmp_tags_path = config.output_path.with_suffix(".tmp.tags.json")

    print(f"\nSaving embeddings to {npz_path}...")
    print(f"Saving text chunks to {json_path}...")
    print(f"Saving metadata to {meta_path}...")
    if config.write_item_tags:
        print(f"Saving item tags to {tags_path}...")

    temp_paths = [tmp_npz_path, tmp_json_path, tmp_meta_path]
    if config.write_item_tags:
        temp_paths.append(tmp_tags_path)

    try:
        # Write all temp files first
        np.savez_compressed(str(tmp_npz_path), **npz_arrays)

        with tmp_json_path.open("w", encoding="utf-8") as f:
            json.dump(json_texts, f, indent=2)

        with tmp_meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if config.write_item_tags:
            with tmp_tags_path.open("w", encoding="utf-8") as f:
                json.dump(json_tags, f, indent=2)

        # All writes succeeded - atomically rename temp → final
        tmp_npz_path.replace(npz_path)
        tmp_json_path.replace(json_path)
        tmp_meta_path.replace(meta_path)
        if config.write_item_tags:
            tmp_tags_path.replace(tags_path)

    except Exception:
        # Clean up any temp files created
        for tmp_path in temp_paths:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        raise

    # Summary
    npz_size = npz_path.stat().st_size / (1024 * 1024)
    json_size = json_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Participants: {len(result.embeddings)}")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  NPZ file: {npz_path} ({npz_size:.2f} MB)")
    print(f"  JSON file: {json_path} ({json_size:.2f} MB)")
    if config.write_item_tags:
        tags_size = tags_path.stat().st_size / (1024 * 1024)
        print(f"  Tags file: {tags_path} ({tags_size:.2f} MB)")
    print("=" * 60)


async def main_async(args: argparse.Namespace) -> int:  # noqa: PLR0915
    """Async main entry point.

    Spec 40: Fail-fast by default, with exit code 2 for partial output.
    """
    setup_logging(LoggingSettings(level="INFO", format="console"))

    settings = get_settings()
    config = prepare_config(args, settings=settings)

    print("=" * 60)
    print("REFERENCE EMBEDDINGS GENERATOR")
    print("=" * 60)
    print(f"  Backend: {config.backend_settings.backend.value}")
    print(f"  Model: {config.model}")
    print(f"  Model (resolved): {config.resolved_model}")
    print(f"  Dimension: {config.dimension}")
    print(f"  Chunk size: {config.chunk_size} lines")
    print(f"  Step size: {config.step_size} lines")
    print(f"  Min chars: {config.min_chars}")
    print(f"  Split: {config.split}")
    print(f"  Output: {config.output_path}")
    print(f"  Write Tags: {config.write_item_tags}")
    if config.write_item_tags:
        print(f"  Tagger: {config.tagger_type}")
    mode_str = "PARTIAL (--allow-partial)" if config.allow_partial else "STRICT (fail-fast)"
    print(f"  Mode: {mode_str}")
    print("=" * 60)

    if config.dry_run:
        print("\n[DRY RUN] Would generate embeddings with above settings.")
        print("[DRY RUN] No files will be created.")
        return EXIT_SUCCESS

    transcript_service = TranscriptService(config.data_settings)
    client = create_embedding_client(settings)

    result: GenerationResult | None = None
    try:
        result = await run_generation_loop(config, client, transcript_service)

        # Spec 40: Check for complete failure (0 participants succeeded)
        if not result.embeddings:
            print(
                "\nERROR: No embeddings generated (0 participants succeeded).",
                file=sys.stderr,
            )
            return EXIT_FAILURE

        save_embeddings(result, config)

    except EmbeddingGenerationError as e:
        # Spec 40: Fail-fast error in strict mode
        print(f"\nERROR: {e}", file=sys.stderr)
        print(f"  Participant: {e.participant_id}", file=sys.stderr)
        if e.chunk_index is not None:
            print(f"  Chunk index: {e.chunk_index}", file=sys.stderr)
        print(
            "\nHint: Use --allow-partial to skip failures and continue.",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return EXIT_FAILURE

    finally:
        await client.close()

    # Spec 40: Report skips in partial mode (after saving artifacts)
    if config.allow_partial and result is not None and result.has_skips:
        manifest_path = config.output_path.with_suffix(".partial.json")
        manifest = {
            "output_npz": str(config.output_path),
            "skipped_participants": result.skipped_participants,
            "skipped_participant_count": len(result.skipped_participants),
            "skipped_chunks": result.total_skipped_chunks,
            "chunk_skip_reason_counts": result.chunk_skip_reason_counts,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 60)
        print("WARNING: PARTIAL OUTPUT (some data was skipped)")
        print("=" * 60)
        if result.skipped_participants:
            print(
                f"  Skipped participants ({len(result.skipped_participants)}): "
                f"{result.skipped_participants}"
            )
        if result.total_skipped_chunks > 0:
            print(f"  Skipped chunks: {result.total_skipped_chunks}")
        print(f"  Manifest: {manifest_path}")
        print("\nThis artifact is INCOMPLETE. Do not use for final evaluation.")
        return EXIT_PARTIAL

    return EXIT_SUCCESS


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for few-shot prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate embeddings (backend from env)
    python scripts/generate_embeddings.py

    # Override backend
    python scripts/generate_embeddings.py --backend huggingface

    # Generate with item tags
    python scripts/generate_embeddings.py --write-item-tags

Environment Variables:
    EMBEDDING_BACKEND: Backend to use (ollama or huggingface)
    OLLAMA_HOST: Ollama server host (default: 127.0.0.1)
    MODEL_EMBEDDING_MODEL: Model to use (default: qwen3-embedding:8b)
    EMBEDDING_DIMENSION: Vector dimension (default: 4096)
    EMBEDDING_CHUNK_SIZE: Lines per chunk (default: 8)
    EMBEDDING_CHUNK_STEP: Sliding window step (default: 2)
        """,
    )
    parser.add_argument(
        "--split",
        choices=["avec-train", "paper-train"],
        default="paper-train",
        help="Which training population to embed (paper-train requires create_paper_split.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output NPZ path (JSON sidecar is written alongside)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface"],
        default=None,
        help="Override embedding backend",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without generating embeddings",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help="Allow partial output on failures (exit 2). Default: strict mode.",
    )
    parser.add_argument(
        "--write-item-tags",
        action="store_true",
        help="Write <output>.tags.json sidecar aligned with <output>.json texts",
    )
    parser.add_argument(
        "--tagger",
        choices=["keyword"],
        default="keyword",
        help="Chunk tagger backend (only used when --write-item-tags is set)",
    )
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
