#!/usr/bin/env python3
"""Generate offline PHQ-8 chunk scores.

This script implements Spec 35: Offline Chunk-Level PHQ-8 Scoring.
It iterates over existing reference chunks (JSON sidecar) and uses an LLM
to estimate PHQ-8 scores (0-3 or null) for each chunk.

Output Format (JSON sidecar):
    - {embeddings_path}.chunk_scores.json
    - {embeddings_path}.chunk_scores.meta.json

Usage:
    python scripts/score_reference_chunks.py \
        --embeddings-file huggingface_qwen3_8b_paper_train \
        --scorer-backend ollama \
        --scorer-model llama3:8b

Paper Reference:
    - Spec 35: Offline Chunk-Level PHQ-8 Scoring
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_psychiatrist.config import (
    LLMBackend,
    LoggingSettings,
    Settings,
    get_settings,
    resolve_reference_embeddings_path_from_embeddings_file,
)
from ai_psychiatrist.infrastructure.llm.factory import create_llm_client
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging
from ai_psychiatrist.services.chunk_scoring import (
    PHQ8_ITEM_KEY_SET,
    chunk_scoring_prompt_hash,
    render_chunk_scoring_prompt,
)

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

logger = get_logger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for chunk scoring."""

    embeddings_path: Path
    texts_path: Path
    output_path: Path
    meta_path: Path
    scorer_backend: LLMBackend
    scorer_model: str
    temperature: float = 0.0
    dry_run: bool = False
    limit: int | None = None  # Debug: limit chunks per participant


@dataclass
class ScoringResult:
    """Result of chunk scoring."""

    scores: dict[str, list[dict[str, int | None]]]
    chunks_processed: int
    errors: int


async def score_chunk(
    client: SimpleChatClient,
    chunk_text: str,
    model: str,
    temperature: float,
) -> dict[str, int | None] | None:
    """Score a single chunk using the LLM."""
    prompt = render_chunk_scoring_prompt(chunk_text=chunk_text)

    try:
        response = await client.simple_chat(
            user_prompt=prompt,
            system_prompt="You are a strict data labeling assistant. Output JSON only.",
            model=model,
            temperature=temperature,
        )

        content = response.strip()
        data = json.loads(content)

        if not isinstance(data, dict):
            logger.warning("Scorer output is not a dict", content=content[:50])
            return None

        key_set = set(data)
        if key_set != PHQ8_ITEM_KEY_SET:
            missing = sorted(PHQ8_ITEM_KEY_SET - key_set)
            extra = sorted(key_set - PHQ8_ITEM_KEY_SET)
            logger.warning(
                "Scorer output has invalid key set",
                missing=missing[:3],
                extra=extra[:3],
                content=content[:50],
            )
            return None

        validated: dict[str, int | None] = {}
        for key in PHQ8_ITEM_KEY_SET:
            val = data[key]
            if val is None:
                validated[key] = None
                continue

            # Reject bool explicitly (bool is a subclass of int).
            if isinstance(val, bool) or not isinstance(val, int) or not (0 <= val <= 3):
                logger.warning("Invalid value for key", key=key, value=val)
                return None

            validated[key] = val

        return validated

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Scoring failed: {e}", chunk_preview=chunk_text[:50])
        return None


async def process_participant(
    client: SimpleChatClient,
    _participant_id: str,
    chunks: list[str],
    config: ScoringConfig,
) -> list[dict[str, int | None]]:
    """Process all chunks for a participant."""
    results: list[dict[str, int | None]] = []

    # Process sequentially to avoid rate limits / context pollution
    for i, chunk in enumerate(chunks):
        if config.limit and i >= config.limit:
            break

        # Skip empty/short chunks (consistent with embedding generation logic)
        # Assuming chunks loaded from sidecar are already filtered/valid?
        # Actually sidecar chunks match embeddings 1:1.

        score_map = await score_chunk(client, chunk, config.scorer_model, config.temperature)

        if score_map is None:
            # Fallback: all nulls
            score_map = cast(
                "dict[str, int | None]",
                dict.fromkeys(PHQ8_ITEM_KEY_SET, None),
            )

        results.append(score_map)

    return results


def prepare_config(args: argparse.Namespace, settings: Settings) -> ScoringConfig:
    """Prepare configuration."""
    embeddings_path = resolve_reference_embeddings_path_from_embeddings_file(
        base_dir=settings.data.base_dir,
        embeddings_file=args.embeddings_file,
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    texts_path = embeddings_path.with_suffix(".json")
    if not texts_path.exists():
        raise FileNotFoundError(f"Texts sidecar not found: {texts_path}")

    output_path = embeddings_path.with_suffix(".chunk_scores.json")
    meta_path = embeddings_path.with_suffix(".chunk_scores.meta.json")

    backend = LLMBackend(args.scorer_backend)

    return ScoringConfig(
        embeddings_path=embeddings_path,
        texts_path=texts_path,
        output_path=output_path,
        meta_path=meta_path,
        scorer_backend=backend,
        scorer_model=args.scorer_model,
        temperature=0.0,  # Enforced by spec
        dry_run=args.dry_run,
        limit=args.limit,
    )


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))
    settings = get_settings()

    try:
        config = prepare_config(args, settings)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    print("=" * 60)
    print("PHQ-8 CHUNK SCORER (Spec 35)")
    print("=" * 60)
    print(f"  Embeddings: {config.embeddings_path}")
    print(f"  Texts: {config.texts_path}")
    print(f"  Output: {config.output_path}")
    print(f"  Backend: {config.scorer_backend.value}")
    print(f"  Model: {config.scorer_model}")
    print(f"  Temperature: {config.temperature}")
    print("=" * 60)

    if config.dry_run:
        print("\n[DRY RUN] Config valid. Exiting.")
        return 0

    if not args.allow_same_model and config.scorer_model == settings.model.quantitative_model:
        print(
            "\nERROR: scorer model matches quantitative assessment model "
            "(Spec 35 circularity risk)."
        )
        print("Set a different --scorer-model, or pass --allow-same-model to override.")
        return 2

    # Load texts
    with config.texts_path.open("r", encoding="utf-8") as f:
        texts_data = json.load(f)

    print(f"\nLoaded {len(texts_data)} participants from sidecar.")

    # Init client - override backend to match scorer settings
    client_settings = settings.model_copy(deep=True)
    client_settings.backend.backend = config.scorer_backend

    # LLMClient implementations have simple_chat, cast for type checker
    llm_client = create_llm_client(client_settings)
    client = cast("SimpleChatClient", llm_client)

    scores_data: dict[str, list[dict[str, int | None]]] = {}
    total_processed = 0

    try:
        keys = sorted(texts_data.keys(), key=int)
        for i, pid in enumerate(keys, 1):
            chunks = texts_data[pid]
            print(f"[{i}/{len(keys)}] Processing Participant {pid} ({len(chunks)} chunks)...")

            p_scores = await process_participant(client, pid, chunks, config)
            scores_data[pid] = p_scores
            total_processed += len(chunks)

            # Simple flush to disk every participant? No, atomic write at end is safer.

    finally:
        await llm_client.close()

    # Save output
    print(f"\nSaving scores to {config.output_path}...")
    with config.output_path.open("w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2)

    # Save metadata
    metadata = {
        "scorer_model": config.scorer_model,
        "scorer_backend": config.scorer_backend.value,
        "temperature": config.temperature,
        "prompt_hash": chunk_scoring_prompt_hash(),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_embeddings": config.embeddings_path.name,
        "total_chunks": total_processed,
    }

    print(f"Saving metadata to {config.meta_path}...")
    with config.meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDONE.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Score chunks for PHQ-8 evidence.")
    parser.add_argument(
        "--embeddings-file",
        required=True,
        help="Path or name of embeddings file (e.g. huggingface_qwen3_8b_paper_train)",
    )
    parser.add_argument(
        "--scorer-backend",
        choices=["ollama", "huggingface"],
        required=True,
        help="LLM backend to use for scoring",
    )
    parser.add_argument(
        "--scorer-model",
        required=True,
        help="Model name (e.g. llama3:8b)",
    )
    parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help=(
            "Allow using the same model as the quantitative assessment model "
            "(unsafe; enables circularity)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit chunks per participant (debug)",
    )

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
