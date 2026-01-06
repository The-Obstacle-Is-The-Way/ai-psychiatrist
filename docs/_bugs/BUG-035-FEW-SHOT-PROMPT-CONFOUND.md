# BUG-035: Few-Shot vs Zero-Shot Prompt Confound

**Date**: 2026-01-06
**Status**: FIXED
**Severity**: HIGH - Invalidates few-shot vs zero-shot causal claims
**Discovered By**: External agent review (validation audit)
**Affects**: All historical runs comparing zero-shot vs few-shot
**Fixed In**: This commit (2026-01-06)

---

## Executive Summary

When few-shot retrieval returns **zero usable references**, the prompt still differs from zero-shot because `format_for_prompt()` returns a wrapper block:

```text
<Reference Examples>
No valid evidence found
</Reference Examples>
```

Zero-shot passes an empty string, which results in **no reference section at all**.

**Impact**: Any claim that "few-shot performs differently than zero-shot" cannot be cleanly attributed to retrieval benefit, because the prompts themselves differ even when retrieval contributes nothing.

---

## Root Cause

### File: `src/ai_psychiatrist/services/embedding.py`

```python
# Lines 112-115
if entries:
    return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n</Reference Examples>"

return "<Reference Examples>\nNo valid evidence found\n</Reference Examples>"  # <-- BUG
```

When no valid reference entries exist, this returns a non-empty wrapper instead of an empty string.

### File: `src/ai_psychiatrist/agents/prompts/quantitative.py`

```python
# Lines 102
reference_section = f"\n{reference_bundle}\n" if reference_bundle else ""
```

This correctly handles empty string (no section), but the embedding service never returns empty string.

### Result

| Condition | `reference_text` | Prompt includes |
|-----------|------------------|-----------------|
| Zero-shot | `""` | No reference section |
| Few-shot (with refs) | `<Reference Examples>...</Reference Examples>` | Reference section with examples |
| Few-shot (no refs) | `<Reference Examples>\nNo valid evidence found\n</Reference Examples>` | Reference section with "No valid evidence" message |

The third row is the bug. Few-shot with no refs should produce the same prompt as zero-shot.

---

## Impact Assessment

### Runs Affected

All runs that compare zero-shot vs few-shot modes:
- Run 12 (and prior runs)
- Any ablation claiming "few-shot helps/hurts"

### Claims Invalidated

From `docs/results/few-shot-analysis.md`:
> "Few-shot underperforms zero-shot (MAE 0.616 vs 0.572)"

This claim is **confounded**. The observed difference could be due to:
1. Actual retrieval effect (intended measurement)
2. Prompt difference from the "No valid evidence found" block (confound)
3. Interaction between the two

We cannot determine which without fixing the confound.

### Claims That Remain Valid

- Coverage statistics (abstention rates)
- Individual mode performance (not comparative)
- AURC/AUGRC within each mode

---

## Fix

### Option A: Return Empty String (Recommended)

```python
# src/ai_psychiatrist/services/embedding.py
def format_for_prompt(self) -> str:
    entries: list[str] = []
    # ... existing entry building ...

    if entries:
        return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n</Reference Examples>"

    return ""  # Was: "<Reference Examples>\nNo valid evidence found\n</Reference Examples>"
```

**Pros**: Clean experimental design. Few-shot with no refs = identical to zero-shot.
**Cons**: Loses observability that retrieval returned nothing (but this is logged elsewhere).

### Option B: Keep Wrapper, Add to Zero-Shot

Add the same empty wrapper to zero-shot prompts so both conditions have identical structure.

**Cons**: Adds unnecessary prompt tokens to zero-shot. Not recommended.

---

## Verification

After fix:

```bash
# Verify prompt equality when retrieval is empty
uv run python -c "
from ai_psychiatrist.services.embedding import ReferenceBundle

# Empty bundle
bundle = ReferenceBundle(item_references={})
assert bundle.format_for_prompt() == '', 'Should return empty string'
print('PASS: Empty bundle returns empty string')
"
```

---

## Related Documentation Updates Needed

1. **docs/results/few-shot-analysis.md**: Add caveat about confound in historical runs
2. **HYPOTHESES-EXPLAINED.md**: Note that "few-shot vs zero-shot" claims need revalidation post-fix
3. **MASTER_BUG_AUDIT.md**: Add this bug to the findings table
4. **docs/_specs/index.md**: No changes needed (specs 061-063 are independent of this confound)

---

## Testing Plan

1. Unit test: `ReferenceBundle.format_for_prompt()` returns `""` when no entries
2. Integration test: Verify prompt content is identical between zero-shot and few-shot-with-no-refs
3. Regression: Re-run comparison after fix to measure true retrieval effect

---

## Historical Note

This bug has existed since the initial few-shot implementation. It was discovered during an external validation audit on 2026-01-06. The original implementation may have intended the wrapper as an observability feature, but it creates a confound that invalidates comparative claims.

---

## References

- `src/ai_psychiatrist/services/embedding.py:119` (fix location: empty bundle returns empty string)
- `src/ai_psychiatrist/agents/prompts/quantitative.py:102` (prompt construction)
- `tests/unit/services/test_embedding.py:73` (unit regression test)
- `tests/unit/agents/test_quantitative.py:630` (prompt equality regression test)
- `docs/results/few-shot-analysis.md` (affected analysis)
- External validation prompt (discovery source)
