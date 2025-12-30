# BUG-038: Optional Features Loaded Unconditionally

| Field | Value |
|-------|-------|
| **Status** | FIXED |
| **Severity** | HIGH |
| **Affects** | Startup, initialization |
| **Introduced** | Spec 34 (ab5647e) |
| **Discovered** | 2025-12-30 |
| **Related** | BUG-035, BUG-037 |
| **Solution** | [Spec 38: Conditional Feature Loading](../specs/38-conditional-feature-loading.md) |

## Summary

Optional feature **tag filtering** is loaded and validated unconditionally during `ReferenceStore` embedding load, even when the feature is disabled. This:
1. Wastes resources loading unused data
2. Can crash the system on validation errors for disabled features
3. Violates the principle that disabled features should be invisible

---

## Root Cause

**File**: `src/ai_psychiatrist/services/reference_store.py:877`

```python
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    # ... load embeddings ...
    self._load_tags(texts_data)  # ALWAYS called, regardless of enable_item_tag_filter
```

The loading methods are called unconditionally. Whether the feature is enabled is checked later during retrieval, but by then the damage is done:
- Validation errors have already crashed the system
- Resources have already been loaded into memory

---

## Impact

### Scenario A: Tag File Has Validation Error, Feature Disabled

```
1. User sets enable_item_tag_filter=False (doesn't want tag filtering)
2. Tags file exists but has a schema error
3. _load_tags() is called anyway
4. EmbeddingArtifactMismatchError raised
5. System CRASHES
6. User is confused: "I disabled tag filtering, why did tags crash my run?"
```

### Scenario B: Unnecessary Resource Usage

```
1. User sets enable_item_tag_filter=False
2. Tags file exists and is valid (10MB of data)
3. _load_tags() loads and validates everything
4. self._tags is populated but never used
5. Memory wasted, startup slowed
```

---

## Code Evidence

### Tag Loading (Always Called)

```python
# reference_store.py:877
self._load_tags(texts_data)
```

No conditional. Compare to how it SHOULD work:
```python
if self._embedding_settings.enable_item_tag_filter:
    self._load_tags(texts_data)
else:
    self._tags = {}  # Skip entirely
```

### Chunk Scores Loading (Always Called)

Chunk scores are **not** loaded in `_load_embeddings()` today. They are loaded lazily via:
- `ReferenceStore.has_chunk_scores()` → `ReferenceStore._load_chunk_scores()`
- `ReferenceStore.get_chunk_score()` → `ReferenceStore._load_chunk_scores()`

---

## Fix

### Principle: Skip What's Not Needed

```python
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    # ... load embeddings ...

    # Only load tags if tag filtering is enabled
    if self._embedding_settings.enable_item_tag_filter:
        self._load_tags(texts_data)
    else:
        self._tags = {}
        logger.debug("Tag filtering disabled, skipping tag loading")
```

### No Silent Fallbacks in Loading Methods

If loading is attempted (because feature is enabled) and fails:
- **CRASH** - don't fall back to empty dict
- The current `except (json.JSONDecodeError, OSError)` blocks should be removed
- Validation errors should propagate

---

## Relationship to BUG-035

BUG-035 documented that `EmbeddingArtifactMismatchError` is not caught. The proposed Spec 38 "fix" was to catch it and fall back to empty dict.

**That fix is WRONG.**

The correct fix is:
1. Don't call `_load_tags()` if `enable_item_tag_filter=False`
2. If `enable_item_tag_filter=True`, let validation errors crash (feature is broken)

---

## Verification

After fix:
- [ ] `enable_item_tag_filter=False` → `_load_tags()` not called
- [ ] `enable_item_tag_filter=False` with corrupt tags file → No crash (file not touched)
- [ ] `enable_item_tag_filter=True` with corrupt tags file → CRASH (feature enabled but broken)
