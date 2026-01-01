# BUG-035: Spec 34 Tag Loading Called Unconditionally

| Field | Value |
|-------|-------|
| **Status** | FIXED |
| **Severity** | HIGH |
| **Affects** | ReferenceStore initialization |
| **Introduced** | Commit ab5647e (Spec 34) |
| **Discovered** | 2025-12-30 |
| **Solution** | [Spec 38: Conditional Feature Loading](../specs/38-conditional-feature-loading.md) |

## Summary

`_load_tags()` is called unconditionally from `ReferenceStore._load_embeddings()`, regardless of whether `enable_item_tag_filter` is True or False. This causes:

1. **When disabled**: *Schema/validation* errors in `.tags.json` crash the system even though the feature is off
2. **When enabled**: Any validation error crashes (this is CORRECT behavior for research)

The problem is specifically case #1 - a disabled feature should not touch its resources.

---

## Root Cause

**File**: `src/ai_psychiatrist/services/reference_store.py:877`

```python
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    # ... load embeddings ...
    self._load_tags(texts_data)  # ALWAYS called, even when enable_item_tag_filter=False
```

---

## Failure Scenario

```
1. User sets enable_item_tag_filter=False (doesn't want tag filtering)
2. Tags file exists and parses as JSON, but fails schema/validation (e.g., tag count mismatch)
3. _load_tags() is called anyway
4. EmbeddingArtifactMismatchError raised
5. System CRASHES
6. User is confused: "I disabled tag filtering, why did tags crash my run?"
```

---

## Correct Fix (Per Revised Spec 38)

**DO NOT** add more exception catching with fallbacks. That would corrupt research results.

**DO** make loading conditional:

```python
def _load_embeddings(self) -> None:
    # ... load embeddings ...

    # Only load tags if tag filtering is ENABLED
    if self._embedding_settings.enable_item_tag_filter:
        self._load_tags(texts_data)  # Crash if invalid (correct for research)
    else:
        self._tags = {}  # Skip entirely
        logger.debug("Tag filtering disabled, skipping tag loading")
```

---

## What NOT to Do (Anti-Pattern)

The original Spec 38 proposed "graceful degradation":
```python
# WRONG - DO NOT DO THIS
except (json.JSONDecodeError, OSError, EmbeddingArtifactMismatchError) as e:
    logger.warning("...")
    self._tags = {}  # Silent fallback corrupts research results
```

This is **wrong** for research reproduction. If a user enables tag filtering and it fails, they should get an error, not silently-corrupted results.

---

## Existing Test is CORRECT

The test `test_mismatched_tags_length_raises` expects the exception to be raised. This is **correct behavior** when `enable_item_tag_filter=True`.

However, today the test constructs `EmbeddingSettings(dimension=2)` (defaults to `enable_item_tag_filter=False`) and still crashes because `_load_tags()` is called unconditionally. After Spec 38 is implemented, this test must be updated to set `enable_item_tag_filter=True` explicitly (so it continues to test the enabled/strict path).

---

## Verification

After fix:
- [ ] `enable_item_tag_filter=False` → `_load_tags()` NOT called
- [ ] `enable_item_tag_filter=False` with corrupt tags file → No crash (file not touched)
- [ ] `enable_item_tag_filter=True` with corrupt tags file → CRASH (correct for research)
- [ ] Existing test `test_mismatched_tags_length_raises` still passes

---

## Related

- Spec 38: Conditional Feature Loading (the correct solution)
- BUG-037: Silent Fallbacks That Corrupt Research Results
- BUG-038: Unconditional Optional Feature Loading
