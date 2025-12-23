#!/usr/bin/env python3
"""Generate reference embeddings for few-shot prompting.

This script generates the pre-computed embeddings artifact required for
few-shot PHQ-8 assessment. It uses ONLY training data to avoid data leakage.

Output Format (NPZ + JSON sidecar):
    - {output_path}.npz: Embeddings as numpy arrays (key: "emb_{pid}")
    - {output_path}.json: Text chunks (key: str(pid) -> list[str])

This format replaces pickle for security (no arbitrary code execution).

Usage:
    # Generate embeddings (requires Ollama running)
    python scripts/generate_embeddings.py

    # Dry run to check configuration
    python scripts/generate_embeddings.py --dry-run

    # Use specific Ollama host
    OLLAMA_HOST=your-server python scripts/generate_embeddings.py

Paper Reference:
    - Section 2.4.2: Embedding-based few-shot prompting
    - Appendix D: Optimal hyperparameters (chunk_size=8, step_size=2, dim=4096)

Spec Reference: docs/specs/08_EMBEDDING_SERVICE.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.config import DataSettings, LoggingSettings, get_settings
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services.transcript import TranscriptService

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
    client: OllamaClient,
    text: str,
    model: str,
    dimension: int,
) -> list[float]:
    """Generate L2-normalized embedding for text.

    Args:
        client: Ollama client.
        text: Text to embed.
        model: Embedding model name.
        dimension: Target embedding dimension.

    Returns:
        L2-normalized embedding vector.
    """
    embedding = await client.simple_embed(
        text=text,
        model=model,
        dimension=dimension,
    )
    return list(embedding)


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


async def process_participant(
    client: OllamaClient,
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
        client: Ollama client.
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


async def main_async(args: argparse.Namespace) -> int:  # noqa: PLR0915
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))

    # Load settings
    settings = get_settings()
    data_settings = settings.data
    embedding_settings = settings.embedding
    model_settings = settings.model
    ollama_settings = settings.ollama

    # Paper-optimal hyperparameters
    chunk_size = embedding_settings.chunk_size
    step_size = embedding_settings.chunk_step
    dimension = embedding_settings.dimension
    min_chars = embedding_settings.min_evidence_chars
    model = model_settings.embedding_model

    default_output = data_settings.embeddings_path
    if args.split == "paper-train":
        default_output = data_settings.base_dir / "embeddings" / "paper_reference_embeddings.npz"
    output_path = args.output or default_output

    print("=" * 60)
    print("REFERENCE EMBEDDINGS GENERATOR")
    print("=" * 60)
    print(f"  Ollama: {ollama_settings.base_url}")
    print(f"  Model: {model}")
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

    async with OllamaClient(ollama_settings) as client:
        # Check Ollama connectivity and model availability
        print("\nChecking Ollama connectivity...")
        try:
            models = await client.list_models()
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama: {e}")
            print(f"Ensure Ollama is running at {ollama_settings.base_url}")
            return 1

        model_names = [m.get("name") for m in models if isinstance(m.get("name"), str)]
        if model not in model_names:
            print(f"WARNING: Model '{model}' not found in Ollama.")
            print(f"Available models: {model_names}")
            print(f"You may need to: ollama pull {model}")

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

        # Save as NPZ + JSON sidecar (replaces pickle for security)
        json_path = output_path.with_suffix(".json")

        print(f"\nSaving embeddings to {output_path}...")
        print(f"Saving text chunks to {json_path}...")

        np.savez_compressed(str(output_path), **npz_arrays)
        with json_path.open("w") as f:
            json.dump(json_texts, f, indent=2)

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

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for few-shot prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate embeddings (requires Ollama running)
    python scripts/generate_embeddings.py

    # Dry run to check configuration
    python scripts/generate_embeddings.py --dry-run

Environment Variables:
    OLLAMA_HOST: Ollama server host (default: 127.0.0.1)
    OLLAMA_PORT: Ollama server port (default: 11434)
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
        "--dry-run",
        action="store_true",
        help="Show configuration without generating embeddings",
    )
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
