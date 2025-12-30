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
import hashlib
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
)
from ai_psychiatrist.domain.enums import PHQ8Item
from ai_psychiatrist.infrastructure.llm.factory import create_llm_client
from ai_psychiatrist.infrastructure.logging import get_logger, setup_logging

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


SCORING_PROMPT_TEMPLATE = """\
You are labeling a single transcript chunk for PHQ-8 item frequency evidence.

Task:
- For each PHQ-8 item key below, output an integer 0-3 if the chunk \
explicitly supports that frequency.
- If the chunk does not mention the symptom or frequency is unclear, \
output null.
- Do not guess or infer beyond the text.

Keys (must be present exactly):
PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired,
PHQ8_Appetite, PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving

Chunk:
{chunk_text}

Return JSON only in this exact shape:
{
  "PHQ8_NoInterest": 0|1|2|3|null,
  "PHQ8_Depressed": 0|1|2|3|null,
  ...
}
"""


async def score_chunk(
    client: SimpleChatClient,
    chunk_text: str,
    model: str,
    temperature: float,
) -> dict[str, int | None] | None:
    """Score a single chunk using the LLM."""
    prompt = SCORING_PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    try:
        response = await client.simple_chat(
            user_prompt=prompt,
            system_prompt="You are a strict data labeling assistant. Output JSON only.",
            model=model,
            temperature=temperature,
        )

        # Clean markdown code blocks if present
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)

        if not isinstance(data, dict):
            logger.warning("Scorer output is not a dict", content=content[:50])
            return None

        # Validate schema
        valid_keys = {f"PHQ8_{item.value}" for item in PHQ8Item}
        validated: dict[str, int | None] = {}

        for key in valid_keys:
            if key not in data:
                # Missing key -> treat as failure to follow schema
                logger.warning(f"Scorer output missing key {key}", content=content[:50])
                return None

            val = data[key]
            if val is None:
                validated[key] = None
            elif isinstance(val, int) and 0 <= val <= 3:
                validated[key] = val
            else:
                # Invalid value -> schema violation
                logger.warning(f"Invalid value for {key}: {val}", content=content[:50])
                return None

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
            score_map = {f"PHQ8_{item.value}": None for item in PHQ8Item}

        results.append(score_map)

    return results


def prepare_config(args: argparse.Namespace, settings: Settings) -> ScoringConfig:
    """Prepare configuration."""
    # Resolve embeddings path
    # If args.embeddings_file looks like a path, use it.
    # Otherwise try to resolve via settings helper.

    # Temporarily patch settings with arg value to use resolver logic
    # or just replicate it.

    candidate = Path(args.embeddings_file)
    if candidate.exists() or candidate.is_absolute():
        embeddings_path = candidate
    else:
        # Resolve relative to data dir
        embeddings_path = settings.data.base_dir / "embeddings" / candidate
        if not embeddings_path.suffix:
            embeddings_path = embeddings_path.with_suffix(".npz")

    if not embeddings_path.exists():
        # Try appending .npz if missing
        if not embeddings_path.suffix and embeddings_path.with_suffix(".npz").exists():
            embeddings_path = embeddings_path.with_suffix(".npz")
        else:
            # Fallback to resolver if name provided
            pass  # Already handled roughly above

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
        "prompt_hash": hashlib.sha256(SCORING_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12],
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
