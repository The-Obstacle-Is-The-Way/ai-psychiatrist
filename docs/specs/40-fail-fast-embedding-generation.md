# Spec 40: Fail-Fast Embedding Generation

| Field | Value |
|-------|-------|
| **Status** | READY |
| **Priority** | HIGH |
| **Addresses** | BUG-042 (embedding generation silently skips participants/chunks) |
| **Effort** | ~0.5 day |
| **Impact** | Research reproducibility - guarantees complete embeddings or explicit failure |

---

## Problem Statement

`scripts/generate_embeddings.py` silently skips:

1. **Entire participants** when transcript loading fails (returns `[], []`)
2. **Individual chunks** when embedding fails (`continue` without error)

This produces "successful" embeddings artifacts that are **silently incomplete**. Downstream evaluation proceeds with missing data, corrupting research results without any indication.

### Code Evidence

**Participant-Level Silent Skip** (`scripts/generate_embeddings.py:315-323`):

```python
try:
    transcript = transcript_service.load_transcript(participant_id)
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to load transcript", participant_id=participant_id, error=str(e))
    return [], []  # SILENT SKIP
```

**Chunk-Level Silent Skip** (`scripts/generate_embeddings.py:333-349`):

```python
try:
    embedding = await generate_embedding(client, chunk, model, dimension)
    results.append((chunk, embedding))
    ...
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to embed chunk", participant_id=participant_id, error=str(e))
    continue  # SILENT SKIP
```

**Main Loop Ignores Empty Results** (`scripts/generate_embeddings.py:498-501`):

```python
if results:  # Silently drops failed participants
    all_embeddings[pid] = results
    all_tags[pid] = chunk_tags
    total_chunks += len(results)
```

### Why This Is Wrong

| Scenario | Current Behavior | Correct Behavior |
|----------|------------------|------------------|
| 1 of 107 transcripts missing | Log warning, output 106 participants, exit 0 | **CRASH** with clear error |
| Embedding API timeout on 1 chunk | Log warning, output incomplete participant, exit 0 | **CRASH** with clear error |
| All transcripts missing | Log 107 warnings, output empty file, exit 0 | **CRASH** with clear error |

A "successful" run with incomplete data is **infinitely worse** than a crash with a clear error.

---

## Correct Behavior (Fail-Fast Contract)

### Default: Strict Mode (No Flags Required)

1. **Any transcript load failure** → **CRASH** with participant ID and error
2. **Any chunk embedding failure** → **CRASH** with participant ID, chunk index, and error
3. Exit code: **non-zero** on any failure

### Opt-In: Partial Mode (`--allow-partial`)

If explicit best-effort is needed for debugging/development:

1. **Transcript failure** → Log warning, skip participant, continue
2. **Chunk failure** → Log warning, skip chunk, continue
3. **At end of run**:
   - Write a `{output}.partial.json` manifest (counts + skipped IDs + skipped chunk count)
   - Print a short summary + the manifest path
   - Exit code: **2** (partial success) if any skips occurred
   - Exit code: **0** only if zero skips

---

## Implementation Plan

### Step 1 — Add `--allow-partial` CLI Argument

**File**: `scripts/generate_embeddings.py`

**Location**: `main()` argument parser (around lines 654-688)

**Add**:

```python
parser.add_argument(
    "--allow-partial",
    action="store_true",
    default=False,
    help="Allow partial output on failures (exit 2). Default: strict mode (crash on any failure).",
)
```

**Update `GenerationConfig` dataclass** (around lines 85-121):

```python
@dataclass
class GenerationConfig:
    # ... existing fields ...
    allow_partial: bool = False
```

**Update `prepare_config()`** to pass through:

```python
allow_partial=args.allow_partial,
```

---

### Step 2 — Define Custom Exception

**File**: `scripts/generate_embeddings.py`

**Location**: After imports (around line 66)

**Add**:

```python
class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails in strict mode."""

    def __init__(self, message: str, participant_id: int, *, chunk_index: int | None = None):
        self.participant_id = participant_id
        self.chunk_index = chunk_index
        super().__init__(message)
```

---

### Step 3 — Refactor `process_participant()` for Fail-Fast

**File**: `scripts/generate_embeddings.py`

**Location**: `process_participant()` function (lines 286-351)

**Change signature** to accept `allow_partial`:

```python
async def process_participant(
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
    allow_partial: bool = False,  # NEW PARAMETER
) -> tuple[list[tuple[str, list[float]]], list[list[str]], int]:
```

**Add skip tracking** (local state in `process_participant()`):

```python
skipped_chunks = 0
```

**Replace transcript loading** (lines 315-323):

```python
# BEFORE (silent skip):
try:
    transcript = transcript_service.load_transcript(participant_id)
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to load transcript", participant_id=participant_id, error=str(e))
    return [], []

# AFTER (fail-fast or explicit skip):
try:
    transcript = transcript_service.load_transcript(participant_id)
except Exception as e:
    if allow_partial:
        logger.warning(
            "Failed to load transcript (skipping participant)",
            participant_id=participant_id,
            error=str(e),
        )
        return [], [], 0
    raise EmbeddingGenerationError(
        f"Failed to load transcript for participant {participant_id}: {e}",
        participant_id=participant_id,
    ) from e
```

**Add an explicit check for empty chunking** (immediately after `create_sliding_chunks(...)`):

```python
chunks = create_sliding_chunks(transcript.text, chunk_size, step_size)
if not chunks:
    if allow_partial:
        logger.warning(
            "No chunks produced for transcript (skipping participant)",
            participant_id=participant_id,
        )
        return [], [], 0
    raise EmbeddingGenerationError(
        f"No chunks produced for participant {participant_id} (empty transcript)",
        participant_id=participant_id,
    )
```

**Replace chunk embedding** (lines 333-349):

```python
# BEFORE (silent continue):
try:
    embedding = await generate_embedding(client, chunk, model, dimension)
    results.append((chunk, embedding))
    if tagger:
        tags = tagger.tag_chunk(chunk)
        chunk_tags.append(tags)
    else:
        chunk_tags.append([])
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to embed chunk", participant_id=participant_id, error=str(e))
    continue

# AFTER (fail-fast or explicit skip):
try:
    embedding = await generate_embedding(client, chunk, model, dimension)
    results.append((chunk, embedding))
except Exception as e:
    if allow_partial:
        logger.warning(
            "Failed to embed chunk (skipping)",
            participant_id=participant_id,
            chunk_index=chunk_idx,
            error=str(e),
        )
        skipped_chunks += 1
        continue
    raise EmbeddingGenerationError(
        f"Failed to embed chunk {chunk_idx} for participant {participant_id}: {e}",
        participant_id=participant_id,
        chunk_index=chunk_idx,
    ) from e

# Tagging should remain fail-fast even in --allow-partial mode.
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
```

**Note**: Add `enumerate()` to track `chunk_idx`:

```python
for chunk_idx, chunk in enumerate(chunks):
    if len(chunk.strip()) < min_chars:
        continue
    # ... rest of loop
```

**Add an explicit check for “no embedded chunks”** (right before the return):

```python
if not results:
    if allow_partial:
        logger.warning(
            "No chunks embedded for participant (skipping participant)",
            participant_id=participant_id,
        )
        return [], [], skipped_chunks
    raise EmbeddingGenerationError(
        f"No chunks embedded for participant {participant_id}",
        participant_id=participant_id,
    )

return results, chunk_tags, skipped_chunks
```

---

### Step 4 — Add Skip Tracking to `run_generation_loop()`

**File**: `scripts/generate_embeddings.py`

**Location**: `run_generation_loop()` function (lines 449-503)

**Add tracking state**:

```python
async def run_generation_loop(
    config: GenerationConfig,
    client: EmbeddingClient,
    transcript_service: TranscriptService,
) -> GenerationResult:
    # ... existing setup code ...

    # Track skips for partial mode reporting
    skipped_participants: list[int] = []
    total_skipped_chunks = 0

    print(f"\nProcessing {len(participant_ids)} participants...")
    for idx, pid in enumerate(participant_ids, 1):
        if idx % 10 == 0 or idx == len(participant_ids):
            print(f"  Progress: {idx}/{len(participant_ids)} participants...")

        results, chunk_tags, skipped_chunks = await process_participant(
            client,
            transcript_service,
            pid,
            config.model,
            config.dimension,
            config.chunk_size,
            config.step_size,
            config.min_chars,
            tagger=tagger,
            allow_partial=config.allow_partial,  # PASS THROUGH
        )

        total_skipped_chunks += skipped_chunks

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
        skipped_participants=skipped_participants,  # NEW FIELD
        total_skipped_chunks=total_skipped_chunks,  # NEW FIELD
    )
```

**Update `GenerationResult` dataclass** (around lines 124-132):

```python
@dataclass
class GenerationResult:
    embeddings: dict[int, list[tuple[str, list[float]]]]
    tags: dict[int, list[list[str]]]
    total_chunks: int
    skipped_participants: list[int] = field(default_factory=list)  # NEW
    total_skipped_chunks: int = 0  # NEW

    @property
    def has_skips(self) -> bool:
        """True if any participants or chunks were skipped."""
        return len(self.skipped_participants) > 0 or self.total_skipped_chunks > 0
```

**Also update imports** at the top of the script:

```python
from dataclasses import dataclass, field
```

---

### Step 5 — Update Exit Codes + Partial Manifest

**File**: `scripts/generate_embeddings.py`

**Location**: `main_async()` function (around lines 586-627)

**Define exit codes at module level**:

```python
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PARTIAL = 2  # Partial success (some skips in --allow-partial mode)
```

**Update `main_async()` to handle strict/partial mode**:

```python
async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    setup_logging(LoggingSettings(level="INFO", format="console"))

    settings = get_settings()
    config = prepare_config(args, settings=settings)

    # ... existing configuration printout ...

    if config.dry_run:
        return EXIT_SUCCESS

    transcript_service = TranscriptService(config.data_settings)
    client = create_embedding_client(settings)

    try:
        result = await run_generation_loop(config, client, transcript_service)
        if not result.embeddings:
            print("\nERROR: No embeddings generated (0 participants succeeded).", file=sys.stderr)
            return EXIT_FAILURE
        save_embeddings(result, config)
    except EmbeddingGenerationError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print(f"  Participant: {e.participant_id}", file=sys.stderr)
        if e.chunk_index is not None:
            print(f"  Chunk index: {e.chunk_index}", file=sys.stderr)
        print("\nHint: Use --allow-partial to skip failures and continue.", file=sys.stderr)
        return EXIT_FAILURE
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        await client.close()

    # Report skips in partial mode (after saving artifacts)
    if config.allow_partial and result.has_skips:
        manifest_path = config.output_path.with_suffix(".partial.json")
        manifest = {
            "output_npz": str(config.output_path),
            "skipped_participants": result.skipped_participants,
            "skipped_chunks": result.total_skipped_chunks,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 60)
        print("WARNING: PARTIAL OUTPUT (some data was skipped)")
        print("=" * 60)
        if result.skipped_participants:
            print(f"  Skipped participants ({len(result.skipped_participants)}): {result.skipped_participants}")
        if result.total_skipped_chunks > 0:
            print(f"  Skipped chunks: {result.total_skipped_chunks}")
        print(f"  Manifest: {manifest_path}")
        print("\nThis artifact is INCOMPLETE. Do not use for final evaluation.")
        return EXIT_PARTIAL

    return EXIT_SUCCESS
```

---

### Step 6 — Make `save_embeddings()` Atomic (No Half-Artifacts)

**File**: `scripts/generate_embeddings.py`

**Location**: `save_embeddings()` function (lines 506-584)

**Goal**: If a write fails, do not leave a misleading “valid-looking” partial artifact on disk.

**Change**: Write to temp files first, then rename into place only after all writes succeed:

```python
# Final paths
npz_path = config.output_path
json_path = config.output_path.with_suffix(".json")
meta_path = config.output_path.with_suffix(".meta.json")
tags_path = config.output_path.with_suffix(".tags.json")

# Temp paths (must end with the final suffix so libraries don’t append extensions)
tmp_npz_path = config.output_path.with_suffix(".tmp.npz")
tmp_json_path = config.output_path.with_suffix(".tmp.json")
tmp_meta_path = config.output_path.with_suffix(".tmp.meta.json")
tmp_tags_path = config.output_path.with_suffix(".tmp.tags.json")

# 1) Write all temp files
# 2) Replace() temp → final (atomic rename on most filesystems)
```

**On any exception** while writing:
- Delete any temp files that exist.
- Re-raise (exit code 1).

**Tags file rule**:
- If `--write-item-tags` is set, always write `{output}.tags.json` (even if empty `{}`) so the presence/absence of the sidecar is unambiguous.

---

### Step 7 — Update Script Docstring

**File**: `scripts/generate_embeddings.py`

**Location**: Module docstring (lines 1-22)

**Add to Usage section**:

```python
"""Generate reference embeddings for few-shot prompting.

...

Usage:
    # Generate embeddings (strict mode - crash on any failure)
    python scripts/generate_embeddings.py

    # Override backend
    python scripts/generate_embeddings.py --backend huggingface

    # Allow partial output for debugging (exit 2 if skips occur)
    python scripts/generate_embeddings.py --allow-partial

Exit Codes:
    0 - Success (all participants and chunks processed)
    1 - Failure (error in strict mode, or fatal error)
    2 - Partial success (--allow-partial mode with skips)

Sidecars (partial mode only):
    - `{output}.partial.json`: machine-readable summary of skipped items

...
"""
```

---

## Tests

### New Test File: `tests/unit/scripts/test_generate_embeddings_fail_fast.py`

Create a new test file so we don’t disturb existing helper tests in `tests/unit/scripts/test_generate_embeddings.py`:

```python
"""Unit tests for scripts/generate_embeddings.py fail-fast behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.generate_embeddings import (
    EmbeddingGenerationError,
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    process_participant,
)


class TestProcessParticipantStrictMode:
    """Tests for default strict mode (allow_partial=False)."""

    @pytest.mark.asyncio
    async def test_transcript_load_failure_raises(self) -> None:
        """Transcript load failure must raise in strict mode."""
        mock_client = AsyncMock()
        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.side_effect = FileNotFoundError("missing")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=50,
                allow_partial=False,  # STRICT MODE
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index is None
        assert "participant 100" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_transcript_raises(self) -> None:
        """Empty transcript must raise (otherwise participant is silently dropped)."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(text="")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=10,
                allow_partial=False,
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index is None

    @pytest.mark.asyncio
    async def test_embedding_failure_raises(self) -> None:
        """Chunk embedding failure must raise in strict mode."""
        mock_client = AsyncMock()
        mock_client.embed.side_effect = RuntimeError("API timeout")

        mock_transcript_service = MagicMock()
        mock_transcript = MagicMock()
        mock_transcript.text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\n" * 10
        mock_transcript_service.load_transcript.return_value = mock_transcript

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=2,
                min_chars=10,
                allow_partial=False,  # STRICT MODE
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index == 0  # First chunk
        assert "chunk 0" in str(exc_info.value)


class TestProcessParticipantPartialMode:
    """Tests for explicit partial mode (allow_partial=True)."""

    @pytest.mark.asyncio
    async def test_transcript_load_failure_returns_empty(self) -> None:
        """Transcript load failure returns empty in partial mode."""
        mock_client = AsyncMock()
        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.side_effect = FileNotFoundError("missing")

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=2,
            min_chars=50,
            allow_partial=True,  # PARTIAL MODE
        )

        assert results == []
        assert tags == []
        assert skipped_chunks == 0

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_empty(self) -> None:
        """Empty transcript skips participant in partial mode."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(text="")

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=2,
            min_chars=10,
            allow_partial=True,
        )

        assert results == []
        assert tags == []
        assert skipped_chunks == 0

    @pytest.mark.asyncio
    async def test_embedding_failure_skips_chunk(self) -> None:
        """Chunk embedding failure skips chunk in partial mode."""
        mock_client = AsyncMock()
        # First call fails, second succeeds
        mock_client.embed.side_effect = [
            RuntimeError("API timeout"),
            MagicMock(embedding=[1.0] * 3),
        ]

        mock_transcript_service = MagicMock()
        mock_transcript = MagicMock()
        # Create enough text for 2 chunks
        mock_transcript.text = ("Line content here.\n" * 8) * 2
        mock_transcript_service.load_transcript.return_value = mock_transcript

        results, tags, skipped_chunks = await process_participant(
            client=mock_client,
            transcript_service=mock_transcript_service,
            participant_id=100,
            model="test",
            dimension=3,
            chunk_size=8,
            step_size=8,  # No overlap for predictable chunks
            min_chars=10,
            allow_partial=True,  # PARTIAL MODE
        )

        # Should have 1 result (second chunk), first was skipped
        assert len(results) == 1
        assert skipped_chunks == 1

    @pytest.mark.asyncio
    async def test_tagger_failure_raises_even_in_partial(self) -> None:
        """Tagging failures must remain fail-fast when tagger is enabled."""
        mock_client = AsyncMock()
        mock_client.embed.return_value = MagicMock(embedding=[1.0] * 3)

        mock_transcript_service = MagicMock()
        mock_transcript_service.load_transcript.return_value = MagicMock(
            text=("Line content here.\n" * 8),
        )

        tagger = MagicMock()
        tagger.tag_chunk.side_effect = RuntimeError("tagger broke")

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            await process_participant(
                client=mock_client,
                transcript_service=mock_transcript_service,
                participant_id=100,
                model="test",
                dimension=3,
                chunk_size=8,
                step_size=8,
                min_chars=10,
                tagger=tagger,
                allow_partial=True,
            )

        assert exc_info.value.participant_id == 100
        assert exc_info.value.chunk_index == 0


class TestExitCodes:
    """Tests for correct exit codes."""

    def test_exit_codes_defined(self) -> None:
        """Exit codes must be correctly defined."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE == 1
        assert EXIT_PARTIAL == 2
```

---

## Verification Criteria

### Strict Mode (Default)

- [ ] Missing transcript file → Script crashes with `EmbeddingGenerationError`, exit 1
- [ ] Embedding API failure → Script crashes with `EmbeddingGenerationError`, exit 1
- [ ] If `--write-item-tags` is set and tagging fails → Script crashes (fail-fast; no partial tags)
- [ ] Error message includes participant ID
- [ ] Error message includes chunk index (for chunk failures)
- [ ] Error message suggests `--allow-partial` as workaround

### Partial Mode (`--allow-partial`)

- [ ] Missing transcript → Log warning, skip participant, continue
- [ ] Embedding failure → Log warning, skip chunk, continue
- [ ] Manifest written: `{output}.partial.json` includes skipped participant IDs + skipped chunk count
- [ ] Summary printed at end: skipped participant IDs, skipped chunk count, manifest path
- [ ] Exit code 2 if any skips occurred
- [ ] Exit code 0 only if zero skips

### Backwards Compatibility

- [ ] Existing successful runs (no failures) behave identically
- [ ] No new required arguments
- [ ] Default behavior is stricter (fail-fast), not looser

### Saving (All Modes)

- [ ] If any file write fails, `{output}.npz` / `{output}.json` / `{output}.meta.json` are not left in a half-written state (atomic temp→rename)
- [ ] If `--write-item-tags` is set, `{output}.tags.json` is always written (even if `{}`)

---

## Why This Is Correct for Research

1. **Explicit over implicit**: Failures are visible, not hidden in logs
2. **Reproducibility**: Same input always produces same output (or fails)
3. **No silent data corruption**: Incomplete artifacts are impossible in strict mode
4. **Opt-in flexibility**: `--allow-partial` exists for debugging, clearly marked as non-production
5. **Machine-readable**: Exit codes allow CI/scripts to detect partial success

---

## Related

- **Supersedes**: None (this is a new spec)
- **Implements**: BUG-042
- **Pattern**: Follows Spec 38 "Skip-If-Disabled, Crash-If-Broken" philosophy
