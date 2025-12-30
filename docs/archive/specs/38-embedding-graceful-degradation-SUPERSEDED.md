# Spec 38: Graceful Degradation for Item-Tag Sidecar Failures

| Field | Value |
|-------|-------|
| **Status** | READY |
| **Priority** | HIGH |
| **Addresses** | BUG-035 (tag validation crash) |
| **Effort** | ~0.5 day |
| **Impact** | Prevents startup/run failures due to invalid `.tags.json` sidecar |

---

## Problem Statement

`ReferenceStore._load_tags()` validates the `.tags.json` sidecar and can raise `EmbeddingArtifactMismatchError`, but it only catches `(json.JSONDecodeError, OSError)`. This makes the system **fail hard** (crash) when:
- `.tags.json` exists but is malformed JSON, or
- `.tags.json` schema/contents fail validation (missing participants, chunk count mismatch, unknown tags, etc.)

This is unacceptable because the tags sidecar is **optional metadata**: retrieval must still work without tags (fallback to unfiltered retrieval).

**SSOT evidence**:
- Code: `src/ai_psychiatrist/services/reference_store.py:570-602`
- `_load_tags()` is called unconditionally from `_load_embeddings()` (so the crash can happen even if `enable_item_tag_filter=False`).

---

## Desired Behavior (Spec Contract)

1. If `.tags.json` is missing:
   - `ReferenceStore` must load successfully.
   - Tag filtering is effectively disabled (unfiltered retrieval).
   - If `enable_item_tag_filter=True`, log a warning that filtering is requested but tags are missing.

2. If `.tags.json` exists but is invalid:
   - `ReferenceStore` must load successfully.
   - Tag filtering must be disabled (unfiltered retrieval).
   - If `enable_item_tag_filter=True`, log a warning that filtering is degraded due to invalid tags.
   - If `enable_item_tag_filter=False`, do **not** crash; log at INFO or DEBUG (no warnings).

3. If `.tags.json` is valid:
   - Load tags and allow filtering when enabled.

4. Visibility:
   - Expose degraded state programmatically (so scripts/runs can report it).

---

## Implementation Plan (Exact Changes)

### Step 1 — Add degraded-state tracking to ReferenceStore

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `ReferenceStore.__init__` (near `self._tags` initialization, ~`src/ai_psychiatrist/services/reference_store.py:104`)

**Add field**

```python
# Degradation tracking (Spec 38)
self._item_tag_filter_degraded_reason: str | None = None
```

**Add properties** (place near other `@property` methods, e.g. near `is_loaded`)

```python
@property
def is_item_tag_filter_degraded(self) -> bool:
    """True when tag filtering was requested but could not be enabled."""
    return (
        self._embedding_settings.enable_item_tag_filter
        and self._item_tag_filter_degraded_reason is not None
    )


@property
def item_tag_filter_degraded_reason(self) -> str | None:
    """Reason for degraded tag filtering (None when not degraded)."""
    if not self._embedding_settings.enable_item_tag_filter:
        return None
    return self._item_tag_filter_degraded_reason
```

---

### Step 2 — Widen exception handling in `_load_tags()` (fail open)

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `_load_tags()` (currently `src/ai_psychiatrist/services/reference_store.py:570-602`)

**Required changes**
1. Reset degraded state at the start of `_load_tags()`:
   - `self._item_tag_filter_degraded_reason = None`
2. Catch validation exceptions and fall back to empty tags:
   - Catch: `json.JSONDecodeError`, `OSError`, `EmbeddingArtifactMismatchError`, `ValueError`
3. Logging level:
   - If `enable_item_tag_filter=True`: WARNING
   - Else: INFO (or DEBUG)

**Before** (current behavior crashes on validation errors):

```python
except (json.JSONDecodeError, OSError) as e:
    logger.warning("Failed to load tags file", path=str(tags_path), error=str(e))
    self._tags = {}
```

**After**:

```python
except (json.JSONDecodeError, OSError, EmbeddingArtifactMismatchError, ValueError) as e:
    # The tags sidecar is optional metadata; never allow it to break loading.
    if self._embedding_settings.enable_item_tag_filter:
        self._item_tag_filter_degraded_reason = f"{type(e).__name__}: {e}"
        logger.warning(
            "tags_validation_failed_falling_back_to_unfiltered",
            path=str(tags_path),
            error=str(e),
            error_type=type(e).__name__,
        )
    else:
        logger.info(
            "tags_ignored_due_to_validation_error",
            path=str(tags_path),
            error=str(e),
            error_type=type(e).__name__,
        )
    self._tags = {}
```

---

### Step 3 — Missing tags file should mark degraded when filtering requested

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `_load_tags()` missing-file branch (currently `src/ai_psychiatrist/services/reference_store.py:572-576`)

**Required change**
- When `enable_item_tag_filter=True` and tags file is missing, set:
  - `self._item_tag_filter_degraded_reason = "missing_tags_sidecar"`
  - keep the existing warning (`_warn_missing_tags`).

Rationale: missing tags means “requested feature cannot be enabled”, which is degraded operation.

---

## Tests (Implementation-Ready, Deterministic)

**File**: `tests/unit/services/test_embedding.py` (existing `TestReferenceStoreTags`)

### Replace crash test with degrade test

Current test expects a raise:
- `TestReferenceStoreTags.test_mismatched_tags_length_raises`

Replace it with:
1) invalid tags do **not** raise, and tags fall back to empty.
2) when filtering is enabled, degraded state is set.

**New tests (copy/pasteable)**

```python
def test_mismatched_tags_length_falls_back_to_empty(self, tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = tmp_path / "embeddings.npz"

    raw_data = {
        "100": [("chunk1", [1.0, 0.0]), ("chunk2", [0.0, 1.0])],
    }
    _create_npz_embeddings(embeddings_path, raw_data)

    # Invalid: 2 chunks but only 1 tag entry
    tags_data = {"100": [["PHQ8_Sleep"]]}
    embeddings_path.with_suffix(".tags.json").write_text(json.dumps(tags_data))

    data_settings = DataSettings(
        base_dir=tmp_path,
        transcripts_dir=transcripts_dir,
        embeddings_path=embeddings_path,
    )
    embed_settings = EmbeddingSettings(dimension=2)

    store = ReferenceStore(data_settings, embed_settings)
    store._load_embeddings()

    assert store.get_participant_tags(100) == []


def test_mismatched_tags_length_sets_degraded_when_filter_enabled(self, tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = tmp_path / "embeddings.npz"

    raw_data = {
        "100": [("chunk1", [1.0, 0.0]), ("chunk2", [0.0, 1.0])],
    }
    _create_npz_embeddings(embeddings_path, raw_data)

    tags_data = {"100": [["PHQ8_Sleep"]]}
    embeddings_path.with_suffix(".tags.json").write_text(json.dumps(tags_data))

    data_settings = DataSettings(
        base_dir=tmp_path,
        transcripts_dir=transcripts_dir,
        embeddings_path=embeddings_path,
    )
    embed_settings = EmbeddingSettings(dimension=2, enable_item_tag_filter=True)

    store = ReferenceStore(data_settings, embed_settings)
    store._load_embeddings()

    assert store.is_item_tag_filter_degraded is True
    assert store.item_tag_filter_degraded_reason is not None
```

---

## Verification Criteria

- Unit tests pass:
  - `uv run pytest tests/unit/services/test_embedding.py -v`
- Manual validation:
  - Corrupt `.tags.json` should no longer brick the pipeline; it should log a warning and continue with unfiltered retrieval.
