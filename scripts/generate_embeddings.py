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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.config import (
    DataSettings,
    EmbeddingBackend,
    LLMBackend,
    LoggingSettings,
    get_settings,
)
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services.transcript import TranscriptService

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient

logger = get_logger(__name__)

PAPER_SPLITS_DIRNAME = "paper_splits"


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
    from ai_psychiatrist.infrastructure.llm.protocols import (  # noqa: PLC0415
        EmbeddingRequest,
    )

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


async def process_participant(
    client: EmbeddingClient,
    transcript_service: TranscriptService,
    participant_id: int,
    model: str,
    dimension: int,
    chunk_size: int,
    step_size: int,
    min_chars: int,
) -> list[tuple[str, list[float]]]:
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

    Returns:
        List of (chunk_text, embedding) pairs.
    """
    try:
        transcript = transcript_service.load_transcript(participant_id)
    except Exception as e:
        logger.warning(
            "Failed to load transcript",
            participant_id=participant_id,
            error=str(e),
        )
        return []

    chunks = create_sliding_chunks(transcript.text, chunk_size, step_size)
    results: list[tuple[str, list[float]]] = []

    for chunk in chunks:
        if len(chunk.strip()) < min_chars:
            continue

        try:
            embedding = await generate_embedding(client, chunk, model, dimension)
            results.append((chunk, embedding))
        except Exception as e:
            logger.warning(
                "Failed to embed chunk",
                participant_id=participant_id,
                error=str(e),
            )
            continue

    return results


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


async def main_async(args: argparse.Namespace) -> int:  # noqa: PLR0915
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))

    # Load settings
    settings = get_settings()

    # Override backend if specified
    if args.backend:
        settings.embedding_backend.backend = EmbeddingBackend(args.backend)

    data_settings = settings.data
    embedding_settings = settings.embedding
    model_settings = settings.model
    backend_settings = settings.embedding_backend

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
    else:
        filename = get_output_filename(
            backend=backend_settings.backend.value,
            model=model,
            split=args.split,
        )
        output_path = data_settings.base_dir / "embeddings" / f"{filename}.npz"

    print("=" * 60)
    print("REFERENCE EMBEDDINGS GENERATOR")
    print("=" * 60)
    print(f"  Backend: {backend_settings.backend.value}")
    print(f"  Model: {model}")
    print(f"  Model (resolved): {resolved_model}")
    print(f"  Dimension: {dimension}")
    print(f"  Chunk size: {chunk_size} lines")
    print(f"  Step size: {step_size} lines")
    print(f"  Min chars: {min_chars}")
    print(f"  Split: {args.split}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would generate embeddings with above settings.")
        print("[DRY RUN] No files will be created.")
        return 0

    # Get training participants only (avoid data leakage)
    try:
        participant_ids = get_participant_ids(data_settings, split=args.split)
        print(f"\nFound {len(participant_ids)} participants")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure DAIC-WOZ dataset is prepared.")
        return 1

    transcript_service = TranscriptService(data_settings)

    # Create client using factory
    client = create_embedding_client(settings)

    try:
        # Process participants
        all_embeddings: dict[int, list[tuple[str, list[float]]]] = {}
        total_chunks = 0

        print(f"\nProcessing {len(participant_ids)} participants...")
        for idx, pid in enumerate(participant_ids, 1):
            if idx % 10 == 0 or idx == len(participant_ids):
                print(f"  Progress: {idx}/{len(participant_ids)} participants...")

            results = await process_participant(
                client,
                transcript_service,
                pid,
                model,
                dimension,
                chunk_size,
                step_size,
                min_chars,
            )

            if results:
                all_embeddings[pid] = results
                total_chunks += len(results)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare NPZ and JSON data
        npz_arrays: dict[str, Any] = {}
        json_texts: dict[str, list[str]] = {}

        for pid, pairs in all_embeddings.items():
            texts = [text for text, _ in pairs]
            embeddings = [emb for _, emb in pairs]

            json_texts[str(pid)] = texts
            npz_arrays[f"emb_{pid}"] = np.array(embeddings, dtype=np.float32)

        # Prepare metadata
        metadata = {
            "backend": backend_settings.backend.value,
            "model": resolved_model,
            "model_canonical": model,
            "dimension": dimension,
            "chunk_size": chunk_size,
            "chunk_step": step_size,
            "split": args.split,
            "participant_count": len(all_embeddings),
            "generated_at": datetime.now(UTC).isoformat(),
            "generator_script": "scripts/generate_embeddings.py",
            "split_csv_hash": calculate_split_hash(data_settings, args.split),
        }

        # Save files
        json_path = output_path.with_suffix(".json")
        meta_path = output_path.with_suffix(".meta.json")

        print(f"\nSaving embeddings to {output_path}...")
        print(f"Saving text chunks to {json_path}...")
        print(f"Saving metadata to {meta_path}...")

        np.savez_compressed(str(output_path), **npz_arrays)
        with json_path.open("w") as f:
            json.dump(json_texts, f, indent=2)
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        # Summary
        npz_size = output_path.stat().st_size / (1024 * 1024)
        json_size = json_path.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Participants: {len(all_embeddings)}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  NPZ file: {output_path} ({npz_size:.2f} MB)")
        print(f"  JSON file: {json_path} ({json_size:.2f} MB)")
        print("=" * 60)

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
        default="avec-train",
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
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
