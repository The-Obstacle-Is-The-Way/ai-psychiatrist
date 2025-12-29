#!/usr/bin/env python3
"""Generate reference embeddings for few-shot prompting.

This script generates the pre-computed embeddings artifact required for
few-shot PHQ-8 assessment. It uses ONLY training data to avoid data leakage.

Output Format (NPZ + JSON sidecar):
    - {output_path}.npz: Embeddings as numpy arrays (key: "emb_{pid}")
    - {output_path}.json: Text chunks (key: str(pid) -> list[str])
    - {output_path}.meta.json: Provenance metadata

Usage:
    # Generate embeddings (backend from env)
    python scripts/generate_embeddings.py

    # Override backend
    python scripts/generate_embeddings.py --backend huggingface

Paper Reference:
    - Section 2.4.2: Embedding-based few-shot prompting
    - Appendix D: Optimal hyperparameters (chunk_size=8, step_size=2, dim=4096)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass
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
from ai_psychiatrist.domain.exceptions import DomainError
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingRequest
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services.transcript import TranscriptService

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

logger = get_logger(__name__)

PAPER_SPLITS_DIRNAME = "paper_splits"


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
                logger.warning(f"Unknown item key in keywords file: {key}")

    def tag_chunk(self, text: str) -> list[str]:
        """Tag a chunk with PHQ-8 items based on keyword matches."""
        tags: list[str] = []
        text_lower = text.lower()

        for item_key, keywords in self._keywords_map.items():
            for kw in keywords:
                if kw in text_lower:
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


@dataclass
class GenerationResult:
    """Result of embedding generation."""

    embeddings: dict[int, list[tuple[str, list[float]]]]
    tags: dict[int, list[list[str]]]  # pid -> list of tags per chunk
    total_chunks: int


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


async def process_participant(
    client: EmbeddingClient,
    transcript_service: TranscriptService,
    participant_id: int,
    model: str,
    dimension: int,
    chunk_size: int,
    step_size: int,
    min_chars: int,
    tagger: KeywordTagger | None = None,
) -> tuple[list[tuple[str, list[float]]], list[list[str]]]:
    """Generate embeddings for all chunks of a participant's transcript.

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

    Returns:
        Tuple of:
        - List of (chunk_text, embedding) pairs.
        - List of tag lists (one per chunk).
    """
    try:
        transcript = transcript_service.load_transcript(participant_id)
    except (DomainError, ValueError, OSError) as e:
        logger.warning(
            "Failed to load transcript",
            participant_id=participant_id,
            error=str(e),
        )
        return [], []

    chunks = create_sliding_chunks(transcript.text, chunk_size, step_size)
    results: list[tuple[str, list[float]]] = []
    chunk_tags: list[list[str]] = []

    for chunk in chunks:
        if len(chunk.strip()) < min_chars:
            continue

        try:
            embedding = await generate_embedding(client, chunk, model, dimension)
            results.append((chunk, embedding))

            if tagger:
                tags = tagger.tag_chunk(chunk)
                chunk_tags.append(tags)
            else:
                chunk_tags.append([])

        except (DomainError, ValueError, OSError) as e:
            logger.warning(
                "Failed to embed chunk",
                participant_id=participant_id,
                error=str(e),
            )
            continue

    return results, chunk_tags


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

    # Paper-optimal hyperparameters
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
    )


async def run_generation_loop(
    config: GenerationConfig,
    client: EmbeddingClient,
    transcript_service: TranscriptService,
) -> GenerationResult:
    """Run the main generation loop."""
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

    print(f"\nProcessing {len(participant_ids)} participants...")
    for idx, pid in enumerate(participant_ids, 1):
        if idx % 10 == 0 or idx == len(participant_ids):
            print(f"  Progress: {idx}/{len(participant_ids)} participants...")

        results, chunk_tags = await process_participant(
            client,
            transcript_service,
            pid,
            config.model,
            config.dimension,
            config.chunk_size,
            config.step_size,
            config.min_chars,
            tagger=tagger,
        )

        if results:
            all_embeddings[pid] = results
            all_tags[pid] = chunk_tags
            total_chunks += len(results)

    return GenerationResult(embeddings=all_embeddings, tags=all_tags, total_chunks=total_chunks)


def save_embeddings(
    result: GenerationResult,
    config: GenerationConfig,
) -> None:
    """Save embeddings, text chunks, and metadata."""
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

    # Prepare metadata
    metadata = {
        "backend": config.backend_settings.backend.value,
        "model": config.resolved_model,
        "model_canonical": config.model,
        "dimension": config.dimension,
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

    # Save files
    json_path = config.output_path.with_suffix(".json")
    meta_path = config.output_path.with_suffix(".meta.json")
    tags_path = config.output_path.with_suffix(".tags.json")

    print(f"\nSaving embeddings to {config.output_path}...")
    print(f"Saving text chunks to {json_path}...")
    print(f"Saving metadata to {meta_path}...")
    if config.write_item_tags:
        print(f"Saving item tags to {tags_path}...")

    np.savez_compressed(str(config.output_path), **npz_arrays)
    with json_path.open("w") as f:
        json.dump(json_texts, f, indent=2)
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    if config.write_item_tags:
        with tags_path.open("w") as f:
            json.dump(json_tags, f, indent=2)

    # Summary
    npz_size = config.output_path.stat().st_size / (1024 * 1024)
    json_size = json_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Participants: {len(result.embeddings)}")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  NPZ file: {config.output_path} ({npz_size:.2f} MB)")
    print(f"  JSON file: {json_path} ({json_size:.2f} MB)")
    if config.write_item_tags:
        tags_size = tags_path.stat().st_size / (1024 * 1024)
        print(f"  Tags file: {tags_path} ({tags_size:.2f} MB)")
    print("=" * 60)


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
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
    print("=" * 60)

    if config.dry_run:
        print("\n[DRY RUN] Would generate embeddings with above settings.")
        print("[DRY RUN] No files will be created.")
        return 0

    transcript_service = TranscriptService(config.data_settings)
    client = create_embedding_client(settings)

    try:
        result = await run_generation_loop(config, client, transcript_service)
        save_embeddings(result, config)
    except FileNotFoundError:
        return 1
    finally:
        await client.close()

    return 0


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
