# Spec 38: Conditional Feature Loading (Skip-If-Disabled, Crash-If-Broken)

| Field | Value |
|-------|-------|
| **Status** | READY |
| **Priority** | HIGH |
| **Addresses** | BUG-035 (tag validation crash), BUG-037 (silent fallbacks), BUG-038 (unconditional loading) |
| **Effort** | ~0.5 day |
| **Impact** | Correct fail-fast behavior for research reproduction |

---

## Problem Statement

The previous Spec 38 proposed "graceful degradation" - catching validation errors and falling back to empty data. **This is fundamentally wrong for a research reproduction project.**

### Why Graceful Degradation Is Wrong

| Scenario | "Graceful Degradation" | Correct Behavior |
|----------|------------------------|------------------|
| Tag filtering enabled, tags invalid | Log warning, return unfiltered results | **CRASH** - user requested a feature that's broken |
| Tag filtering disabled, tags invalid | Log warning (or nothing) | **Skip loading entirely** - don't touch the file |
| Tag filtering enabled, tags missing | Log warning, return unfiltered results | **CRASH** - user requested a feature that can't work |

**Silent fallbacks corrupt research results.** A "successful" run with wrong results is infinitely worse than a crash with a clear error.

---

## Correct Behavior (Spec Contract)

### Principle: Skip If Disabled, Crash If Broken

1. **If feature is DISABLED** → Don't load its resources at all
   - No file I/O
   - No validation
   - No warnings (feature is off, nothing to warn about)

2. **If feature is ENABLED** → Load and validate strictly
   - File missing → **CRASH** with clear error
   - Validation fails → **CRASH** with clear error
   - No fallbacks, no degradation

---

## Implementation Plan

### Step 1 — Make Tag Loading Conditional

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `_load_embeddings()` method (`src/ai_psychiatrist/services/reference_store.py:831`)

**Before**:
```python
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    # ... load embeddings ...
    self._load_tags(texts_data)  # ALWAYS called
```

**After**:
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

---

### Step 2 — Remove Fallback Exception Handling in _load_tags()

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `_load_tags()` method (`src/ai_psychiatrist/services/reference_store.py:570`)

**Before** (wrong - catches errors and falls back):
```python
def _load_tags(self, texts_data: dict[str, Any]) -> None:
    tags_path = self._get_tags_path()
    if not tags_path.exists():
        self._warn_missing_tags(tags_path)
        self._tags = {}
        return

    try:
        # ... validation ...
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load tags file", ...)
        self._tags = {}  # WRONG: silent fallback
```

**After** (correct - crash if broken):
```python
def _load_tags(self, texts_data: dict[str, Any]) -> None:
    """Load and validate item tags sidecar.

    Called only when enable_item_tag_filter=True.
    Raises on any error (no fallbacks).
    """
    tags_path = self._get_tags_path()

    # File MUST exist when feature is enabled
    if not tags_path.exists():
        raise EmbeddingArtifactMismatchError(
            f"Tag filtering enabled but tags file missing: {tags_path}. "
            f"Either disable tag filtering (EMBEDDING_ENABLE_ITEM_TAG_FILTER=false) "
            f"or regenerate embeddings with tags."
        )

    # Parse JSON - let errors propagate
    with tags_path.open("r", encoding="utf-8") as f:
        tags_data = json.load(f)

    # Validate - let errors propagate
    tags_data = self._validate_tags_top_level(tags_data, texts_data, tags_path)

    valid_tags = {f"PHQ8_{item.value}" for item in PHQ8Item}
    validated_tags: dict[int, list[list[str]]] = {}
    for pid_str, texts in texts_data.items():
        pid = int(pid_str)
        validated_tags[pid] = self._validate_participant_tags(
            pid=pid,
            raw_p_tags=tags_data[pid_str],
            expected_len=len(texts),
            tags_path=tags_path,
            valid_tags=valid_tags,
        )

    self._tags = validated_tags
    logger.info("Item tags loaded", participants=len(self._tags))
```

**Key changes**:
1. No `try/except` with fallback
2. Missing file raises clear error with fix instructions
3. All validation errors propagate

---

### Step 3 — Remove _warn_missing_tags Helper

**File**: `src/ai_psychiatrist/services/reference_store.py`

**Location**: `_warn_missing_tags()` (`src/ai_psychiatrist/services/reference_store.py:468`)

The `_warn_missing_tags()` method is no longer needed. When feature is disabled, we skip loading entirely. When feature is enabled and file is missing, we crash.

**Delete**:
```python
def _warn_missing_tags(self, tags_path: Path) -> None:
    # ... this method is no longer needed
```

---

### Step 4 — Make Reference Validation Fail-Fast When Enabled

**File**: `src/ai_psychiatrist/services/reference_validation.py`

Reference validation is an optional feature (Spec 36). When enabled, it must either work correctly or crash. Returning `"unsure"` on exceptions is a silent fallback that changes run behavior (BUG-037).

**Location**:
- `LLMReferenceValidator.validate()` (`src/ai_psychiatrist/services/reference_validation.py:71-87`)
- `LLMReferenceValidator._parse_decision()` (`src/ai_psychiatrist/services/reference_validation.py:108-128`)

**Before** (wrong - catches all exceptions and returns `"unsure"`):
```python
try:
    response = await self._client.simple_chat(...)
    return self._parse_decision(response)
except Exception as e:
    logger.warning("Reference validation failed", error=str(e))
    return "unsure"
```

**After** (correct - crash if broken):
- Remove the broad `except Exception` fallback.
- If the model returns invalid JSON (or missing/invalid `"decision"`), raise `LLMResponseParseError` (do not silently return `"unsure"`).

Rationale: if the user enabled reference validation, a run with silently-disabled (or silently-rejecting) validation is scientifically corrupted.

---

## Tests

### Keep Existing Crash Test

**File**: `tests/unit/services/test_embedding.py`

The test `test_mismatched_tags_length_raises` is **CORRECT** - it expects the system to crash on validation errors when `enable_item_tag_filter=True`.

**Required update** (once Spec 38 is implemented): explicitly set `EmbeddingSettings(enable_item_tag_filter=True, ...)` in:
- `test_load_tags_and_validate_length`
- `test_mismatched_tags_length_raises`

### Add New Tests

```python
def test_tags_not_loaded_when_filter_disabled(self, tmp_path: Path) -> None:
    """When enable_item_tag_filter=False, tags file should not be touched."""
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = tmp_path / "embeddings.npz"

    raw_data = {"100": [("chunk1", [1.0, 0.0])]}
    _create_npz_embeddings(embeddings_path, raw_data)

    # Create invalid tags file - should NOT cause error because feature is disabled
    tags_path = embeddings_path.with_suffix(".tags.json")
    tags_path.write_text("INVALID JSON {{{")

    data_settings = DataSettings(
        base_dir=tmp_path,
        transcripts_dir=transcripts_dir,
        embeddings_path=embeddings_path,
    )
    embed_settings = EmbeddingSettings(
        dimension=2,
        enable_item_tag_filter=False,  # Feature disabled
    )

    store = ReferenceStore(data_settings, embed_settings)
    store._load_embeddings()  # Should NOT crash

    assert store._tags == {}  # Empty because skipped


def test_tags_crash_when_filter_enabled_and_missing(self, tmp_path: Path) -> None:
    """When enable_item_tag_filter=True and tags missing, must crash."""
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = tmp_path / "embeddings.npz"

    raw_data = {"100": [("chunk1", [1.0, 0.0])]}
    _create_npz_embeddings(embeddings_path, raw_data)

    # NO tags file

    data_settings = DataSettings(
        base_dir=tmp_path,
        transcripts_dir=transcripts_dir,
        embeddings_path=embeddings_path,
    )
    embed_settings = EmbeddingSettings(
        dimension=2,
        enable_item_tag_filter=True,  # Feature enabled
    )

    store = ReferenceStore(data_settings, embed_settings)

    with pytest.raises(EmbeddingArtifactMismatchError, match="tags file missing"):
        store._load_embeddings()  # MUST crash
```

---

### Add Validator Tests (Reference Validation)

**File**: `tests/unit/services/test_reference_validation.py` (new)

Add unit tests for `LLMReferenceValidator`:

1. **Exceptions propagate (no silent fallback)**:
   - Mock `SimpleChatClient.simple_chat` to raise `RuntimeError("boom")`
   - Assert `await validator.validate(...)` raises `RuntimeError` (not `"unsure"`)

2. **Invalid JSON crashes**:
   - Mock `SimpleChatClient.simple_chat` to return `"not json"`
   - Assert `await validator.validate(...)` raises `LLMResponseParseError`

---

## Verification Criteria

- [ ] `enable_item_tag_filter=False` → `_load_tags()` not called, no file I/O
- [ ] `enable_item_tag_filter=False` with corrupt/missing tags → No crash
- [ ] `enable_item_tag_filter=True` with missing tags → CRASH with clear error
- [ ] `enable_item_tag_filter=True` with invalid tags → CRASH with clear error
- [ ] Existing test `test_mismatched_tags_length_raises` still passes
- [ ] With reference validation enabled, any validator failure raises (no `"unsure"` fallback)

---

## Why This Is Correct for Research

1. **No silent behavior changes**: If you enable a feature, it works or crashes
2. **Clear errors with fix instructions**: User knows exactly what's wrong
3. **Reproducibility**: Same config always produces same behavior
4. **No hidden data corruption**: Results are correct or run fails

---

## Supersedes

This spec supersedes the original "Graceful Degradation" version of Spec 38. The "graceful degradation" approach was inappropriate for a research reproduction project.
