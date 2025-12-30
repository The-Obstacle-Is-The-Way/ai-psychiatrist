# BUG-034: Few-Shot Participant Count Regression

| Field | Value |
|-------|-------|
| **Status** | OPEN |
| **Severity** | CRITICAL |
| **Affects** | few_shot mode |
| **Introduced** | Between commits 5e62455 and be35e35 |
| **Discovered** | 2025-12-30 |

## Summary

Few-shot mode participant success rate dropped from 41/41 (100%) to 32/41 (78%) after recent refactoring. All 9 failures were LLM timeout errors at 120 seconds.

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
The HuggingFace embedding client has a 120-second timeout that is too short.

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
Set `HF_DEFAULT_EMBED_TIMEOUT=300` in `.env`

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
