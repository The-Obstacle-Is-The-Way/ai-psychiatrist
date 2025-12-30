# BUG-035: Spec 34 Tag Validation Uncaught Exceptions

| Field | Value |
|-------|-------|
| **Status** | OPEN |
| **Severity** | CRITICAL |
| **Affects** | ReferenceStore with `enable_item_tag_filter=True` |
| **Introduced** | Commit ab5647e (Spec 34) |
| **Discovered** | 2025-12-30 |

## Summary

The `_load_tags()` method raises `EmbeddingArtifactMismatchError` during validation but only catches `(json.JSONDecodeError, OSError)` in its except clause. When `enable_item_tag_filter=True` and tags have ANY validation error, the system crashes with no fallback.

## Root Cause

```python
# reference_store.py lines 600-602
except (json.JSONDecodeError, OSError) as e:  # BUG: Missing EmbeddingArtifactMismatchError
    logger.warning("Failed to load tags file", path=str(tags_path), error=str(e))
    self._tags = {}
```

The except clause is too narrow. `EmbeddingArtifactMismatchError` is raised by:
- `_validate_tags_top_level()` at lines 494, 502
- `_validate_participant_tags()` at line 553
- `_validate_chunk_tags()` at line 531

## Failure Scenarios

### Scenario A: Missing Participants in Tags
```
1. Tags file has fewer participants than embeddings
2. _validate_tags_top_level() raises at line 494
3. Exception NOT caught
4. System crash
```

### Scenario B: Tag Count Mismatch
```
1. Participant has 50 chunks in texts, 48 in tags
2. _validate_participant_tags() raises at line 553
3. Exception NOT caught
4. System crash
```

### Scenario C: Invalid Tag Value
```
1. Tag contains "PHQ8_Sleepp" (typo)
2. _validate_chunk_tags() raises at line 531
3. Exception NOT caught
4. System crash
```

## Call Stack on Error

```
_load_tags() raises EmbeddingArtifactMismatchError
  → _load_embeddings() (line 877) does not catch
    → ReferenceStore initialization fails
      → EmbeddingService __init__ fails
        → Application crash
          → ALL participants fail
```

## Impact

When `enable_item_tag_filter=True`:
- System cannot start if tags file has ANY validation error
- No fallback to unfiltered retrieval
- 0% participant success rate
- Total system failure

## Comparison to Similar Code

The `_load_chunk_scores()` method (line 750) uses a similar pattern but includes a fallback mechanism via `_raise_or_warn_chunk_scores_provenance()` which allows graceful degradation. Tags loading lacks this.

## Fix

### Option 1: Expand except clause
```python
# In _load_tags() line 600
except (json.JSONDecodeError, OSError, EmbeddingArtifactMismatchError) as e:
    logger.warning("Failed to load tags file", path=str(tags_path), error=str(e))
    self._tags = {}
```

### Option 2: Graceful degradation pattern
```python
def _load_tags(self, texts_data: dict[str, Any]) -> None:
    try:
        # ... validation ...
    except EmbeddingArtifactMismatchError as e:
        if self._embedding_settings.enable_item_tag_filter:
            # Fall back to unfiltered retrieval
            logger.warning("Tag validation failed, falling back to unfiltered", error=str(e))
            self._tags = {}
        else:
            raise  # Re-raise if filtering was required
```

## Test Gap

Existing test `test_load_tags_and_validate_length` (line 1077) validates that exceptions ARE raised:
```python
with pytest.raises(EmbeddingArtifactMismatchError, match="Tag count mismatch"):
    store._load_embeddings()
```

This passes because the exception IS raised, but doesn't test graceful handling.

## Verification

After fix:
- [ ] System starts even with invalid tags file when `enable_item_tag_filter=True`
- [ ] Warning logged about falling back to unfiltered retrieval
- [ ] Retrieval works without item filtering
- [ ] No participants fail due to tag validation

## Related

- Spec 34: Item-Tagged Reference Embeddings
- BUG-032: Spec 34 visibility gap (different bug, same feature)
- BUG-033: Runtime query embedding timeouts
