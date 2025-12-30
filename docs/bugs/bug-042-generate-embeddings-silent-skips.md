# BUG-042: Embedding Generation Silently Skips Participants/Chunks

| Field | Value |
|-------|-------|
| **Status** | NEW |
| **Severity** | MEDIUM |
| **Affects** | `scripts/generate_embeddings.py` (artifact correctness) |
| **Introduced** | Original design |
| **Discovered** | 2025-12-30 |
| **Root Cause** | Best-effort fallback pattern in a research pipeline |
| **Solution** | TBD (spec required) |

## Summary

`scripts/generate_embeddings.py` silently skips:
- an entire participant when transcript loading fails, and
- individual chunks when embedding fails.

This can produce a “successful” embeddings artifact that is missing participants and/or missing chunks, without failing the command.

For a research reproduction project, this is dangerous: downstream runs can silently use incomplete reference embeddings.

## Code Evidence

### Participant-Level Silent Skip

**File**: `scripts/generate_embeddings.py:315-324`

```python
try:
    transcript = transcript_service.load_transcript(participant_id)
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to load transcript", participant_id=participant_id, error=str(e))
    return [], []
```

### Chunk-Level Silent Skip

**File**: `scripts/generate_embeddings.py:333-350`

```python
try:
    embedding = await generate_embedding(client, chunk, model, dimension)
    results.append((chunk, embedding))
    ...
except (DomainError, ValueError, OSError) as e:
    logger.warning("Failed to embed chunk", participant_id=participant_id, error=str(e))
    continue
```

## Why This Is a Bug (Not Just a Design Choice)

- The script decides “training participants only” to avoid leakage, but then silently changes the effective set when failures occur.
- The output files (`.npz`, `.json`, optional `.tags.json`) don’t encode “participant/chunk coverage” as an error condition.
- Downstream evaluation will happily proceed with whatever embeddings exist, making comparisons non-reproducible.

If this behavior is desired for production robustness, it must be behind an explicit opt-in flag (e.g., `--allow-partial`) and the default must be strict for research runs.

## Expected Behavior (Fail-Fast Contract)

1. If any participant transcript fails to load → **CRASH** (default behavior).
2. If any chunk fails to embed → **CRASH** (default behavior).
3. If best-effort mode is needed:
   - Require explicit opt-in (`--allow-partial`)
   - Emit a machine-readable summary (counts + participant IDs) and exit non-zero unless explicitly overridden.

## Verification

- Run `scripts/generate_embeddings.py` with a missing transcript file and confirm it exits non-zero and surfaces the failing participant ID.
- Run with an embedding client stub that fails on one chunk and confirm the script exits non-zero (unless `--allow-partial` is enabled).
