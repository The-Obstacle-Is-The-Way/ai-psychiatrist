# BUG-034: Few-Shot Participant Count Regression

| Field | Value |
|-------|-------|
| **Status** | FIXED |
| **Severity** | CRITICAL |
| **Affects** | few_shot mode |
| **Introduced** | Between commits 5e62455 and be35e35 |
| **Discovered** | 2025-12-30 |
| **Root Cause** | BUG-033 (embedding timeouts) |
| **Solution** | [Spec 37: Batch Query Embedding](../specs/37-batch-query-embedding.md) |

## Summary

Few-shot mode participant success rate dropped from 41/41 (100%) to 32/41 (78%) after recent refactoring. All 9 failures were LLM timeout errors at 120 seconds.

**Adversarial note**: Run 4 (`77a2bdb8`) was recorded with `git_dirty=true` in the output JSON, so the exact code state is not perfectly identified by the commit hash alone. Treat the regression as “observed in the workspace state at the time”, not conclusively attributable to `be35e35` until a clean rerun is captured.

## Impact

| Run | Date | Commit | Mode | Success | Fail | Duration |
|-----|------|--------|------|---------|------|----------|
| Run 3 | Dec 29 | 5e62455 | few_shot | 41/41 | 0 | 5,681s |
| Run 4 | Dec 30 | be35e35 | few_shot | 32/41 | 9 | 11,783s |

**Performance degradation**: 107% slower (doubled execution time)

## Failed Participants

All 9 failures have identical error: `"LLM request timed out after 120s"`

| PID | Transcript Size |
|-----|-----------------|
| 345 | 13.6 KB |
| 357 | 6.9 KB |
| 385 | 7.7 KB |
| 390 | - |
| 413 | - |
| 417 | - |
| 422 | 25.0 KB |
| 451 | - |
| 487 | 20.4 KB |

Note: Transcript sizes are normal range. Larger transcripts (e.g., 314 @ 25.1 KB) succeeded.

## Root Cause Analysis

### Primary Cause: Runtime Embedding Timeouts (BUG-033)
Few-shot mode triggers **runtime query embeddings** (embedding backend = HuggingFace in Run 4). Those queries time out at **120s** because `EmbeddingService.embed_text()` constructs `EmbeddingRequest(...)` without overriding `timeout_seconds`, so it uses the hard-coded `EmbeddingRequest.timeout_seconds = 120` default (see BUG-033 for code-level details).

This is *not* fixed by setting `HF_DEFAULT_EMBED_TIMEOUT`, because the failing path does not use `HuggingFaceClient.simple_embed(...)`.

### Contributing Factors

1. **88% slower on successful runs**: Even excluding timeout penalty (9 × 120s = 1,080s), Dec 30 run was 88% slower

2. **Code changes between runs**:
   - Spec 34 item tag filtering logic added
   - Reference store initialization overhead
   - Additional validation in `_load_tags()`

3. **Probabilistic timing**: Some participants hit the 120s ceiling based on:
   - Evidence text length
   - Embedding dimension
   - System load

## Changes Between Runs

```bash
git diff 5e62455..be35e35 --stat
# 35 files changed, 849 insertions, 105 deletions

# Key changes:
src/ai_psychiatrist/services/reference_store.py  # +164 lines
src/ai_psychiatrist/services/embedding.py        # +22 lines (Spec 34 filtering)
src/ai_psychiatrist/config.py                    # +4 lines
scripts/generate_embeddings.py                   # +116 lines
```

## Reproduction

1. Checkout commit be35e35
2. Run: `uv run python scripts/reproduce_results.py --mode both --split paper`
3. Observe: few_shot failures with 120s timeout errors

## Fix

### Immediate
There is no env-only fix today because the timeout is a dataclass default. Mitigation requires code change (Spec 37).

### Long-term
Implement batch embedding to reduce API calls (see BUG-033 for details)

## Verification

After fix, verify:
- [ ] All 41 participants succeed in few_shot mode
- [ ] Duration is comparable to Dec 29 run (~5,600s)
- [ ] No timeout errors in logs

## Related

- BUG-033: Runtime query embedding timeouts
- BUG-035: Spec 34 tag validation bugs
- Spec 34: Item-Tagged Reference Embeddings
