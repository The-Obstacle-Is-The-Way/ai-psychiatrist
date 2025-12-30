# BUG-037: Silent Fallbacks That Corrupt Research Results

| Field | Value |
|-------|-------|
| **Status** | SPEC'd |
| **Severity** | CRITICAL |
| **Affects** | ALL modes (zero-shot, few-shot) |
| **Introduced** | Original design |
| **Discovered** | 2025-12-30 |
| **Root Cause** | "Fail-safe" pattern misapplied to research reproduction |
| **Solution** | [Spec 38: Conditional Feature Loading](../specs/38-conditional-feature-loading.md) |

## Summary

The codebase contains multiple places where errors are silently suppressed and fallback behavior is applied. **For a research reproduction project, this is fundamentally wrong.** Silent fallbacks can produce results that look valid but are scientifically corrupted.

**Principle**: If a requested feature fails, the run should **CRASH** with a clear error, not silently produce different results.

---

## Critical Improper Fallbacks

### 1. Tag Loading Falls Back to Empty Dict

**File**: `src/ai_psychiatrist/services/reference_store.py:600-602`

```python
except (json.JSONDecodeError, OSError) as e:
    logger.warning("Failed to load tags file", path=str(tags_path), error=str(e))
    self._tags = {}  # SILENT FALLBACK: Filtering silently disabled
```

**Problem**: If `enable_item_tag_filter=True` but tags fail to load, the system:
- Logs a warning (easily missed)
- Silently disables tag filtering
- Produces unfiltered results that look like filtered results

**Impact**: Research results are now DIFFERENT from what the user requested, but the output looks normal.

**Correct Behavior**:
- If `enable_item_tag_filter=True` and tags fail → **CRASH**
- If `enable_item_tag_filter=False` → Don't attempt to load tags at all

---

### 2. Missing Tags File Falls Back to Empty Dict

**File**: `src/ai_psychiatrist/services/reference_store.py:573-576`

```python
if not tags_path.exists():
    self._warn_missing_tags(tags_path)
    self._tags = {}  # SILENT FALLBACK: Filtering silently disabled
    return
```

**Problem**: Same as above. If tag filtering is enabled but file is missing, the system silently produces unfiltered results.

**Correct Behavior**:
- If `enable_item_tag_filter=True` and tags missing → **CRASH** with clear error
- If `enable_item_tag_filter=False` → Skip tag loading entirely

---

### 3. Reference Validation Falls Back to "Unsure"

**File**: `src/ai_psychiatrist/services/reference_validation.py:84-86`

```python
except Exception as e:
    logger.warning("Reference validation failed", error=str(e))
    return "unsure"  # Fail safe -> treated as reject by default logic
```

**Problem**: If validation is enabled but the LLM call fails (timeout, connection error, etc.):
- Silently treats all references as "unsure" → rejected
- User thinks validation is working but it's failing silently
- Results have fewer references than they should

**Impact**: The comment literally says "Fail safe" but **THIS IS WRONG FOR RESEARCH**. A run with validation enabled should produce validated results, not silently-rejected results.

**Correct Behavior**:
- If `enable_reference_validation=True` and validation fails → **CRASH**
- If `enable_reference_validation=False` → Skip validation entirely

---

### 4. Ground Truth Score Parsing Silent Fallback

**File**: `src/ai_psychiatrist/services/ground_truth.py:134-138`

```python
if "PHQ8_Score" in row.columns:
    try:
        return int(row["PHQ8_Score"].iloc[0])
    except (ValueError, TypeError):
        pass  # SILENT FALLBACK: Fall through to calculation
```

**Problem**: If the PHQ8_Score column exists but contains invalid data:
- Silently falls back to calculating from individual items
- No log, no warning, nothing
- User doesn't know the primary data source failed

**Impact**: Ground truth might be calculated differently than expected with no indication.

**Correct Behavior**:
- If PHQ8_Score column exists but parsing fails → **CRASH** or at minimum log an ERROR
- Data corruption should be visible, not hidden

---

### 5. Transcripts Directory Missing Returns Empty List

**File**: `src/ai_psychiatrist/services/transcript.py:114-119`

```python
if not self._transcripts_dir.exists():
    logger.warning(
        "Transcripts directory not found",
        path=str(self._transcripts_dir),
    )
    return []  # SILENT FALLBACK: Returns no participants
```

**Problem**: If the transcripts directory doesn't exist:
- Logs a warning
- Returns empty list
- Script proceeds with 0 participants

**Impact**: A misconfigured `DATA_TRANSCRIPTS_DIR` produces a "successful" run with 0 results instead of crashing.

**Correct Behavior**:
- If transcripts directory not found → **CRASH** with clear error about misconfiguration

---

## Architectural Issue

The codebase has "fail-safe" patterns appropriate for a production web service, but this is a **research reproduction project**. The correct paradigm is:

| Production Service | Research Reproduction |
|--------------------|----------------------|
| Keep running if possible | Fail fast on ANY anomaly |
| Silent degradation OK | Silent degradation = CORRUPT DATA |
| User sees "something" | User sees error or correct results |
| Availability > Correctness | Correctness > Everything |

---

## Fix Requirements

### Principle: Requested Features Must Work or Crash

1. **If a feature is disabled** → Don't load its resources, don't validate, skip entirely
2. **If a feature is enabled** → It MUST work correctly or the run MUST fail
3. **No silent fallbacks** that change research behavior
4. **Errors must be visible** in the output, not just warnings in logs

### Specific Fixes

| Location | Current | Fix |
|----------|---------|-----|
| `reference_store.py:573-576` | Warn + empty dict | Crash if `enable_item_tag_filter=True` |
| `reference_store.py:600-602` | Warn + empty dict | Crash if `enable_item_tag_filter=True` |
| `reference_validation.py:84-86` | Return "unsure" | Crash if `enable_reference_validation=True` |
| `ground_truth.py:134-138` | Silent pass | Log ERROR (don't crash, but visible) |
| `transcript.py:114-119` | Warn + empty list | Crash |

---

## Relationship to Other Bugs

- **BUG-035**: Documents that `_load_tags()` is called unconditionally. Fix: Skip entirely if disabled.
- **Spec 38 (proposed)**: Proposes "graceful degradation" - **THIS IS WRONG**. Should be revised to "skip if disabled, crash if enabled and fails".

---

## Verification

After fix:
- [ ] Misconfigured `enable_item_tag_filter=True` without tags file → CRASH
- [ ] Corrupt tags file with `enable_item_tag_filter=True` → CRASH
- [ ] `enable_item_tag_filter=False` → Tags not loaded at all
- [ ] Misconfigured `enable_reference_validation=True` with LLM failure → CRASH
- [ ] `enable_reference_validation=False` → Validation not attempted
- [ ] Missing transcripts directory → CRASH
- [ ] Invalid ground truth data → Visible ERROR (not silent)
