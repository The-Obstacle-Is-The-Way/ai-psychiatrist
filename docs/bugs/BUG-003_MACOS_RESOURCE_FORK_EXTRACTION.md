# BUG-003: macOS Resource Fork Extracted Instead of Transcript

**Severity**: MEDIUM (P2)
**Status**: RESOLVED (code fix), DATA RE-EXTRACT REQUIRED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/04A_DATA_ORGANIZATION.md`, `docs/specs/05_TRANSCRIPT_SERVICE.md`

---

## Executive Summary

`prepare_dataset.py` extracted the **first** zip entry matching `_TRANSCRIPT.csv`. On macOS-created zips, this can be the **AppleDouble resource fork** (e.g., `__MACOSX/._<file>`), not the real transcript. This produced a **binary, non-CSV** file for participant 487, causing UTF-8 decode errors and breaking transcript loading for that participant.

---

## Evidence (Pre-Fix)

- `scripts/prepare_dataset.py` used `_read_first_matching()` which returned the **first** matching suffix, including `__MACOSX/._...` entries.
- `data/transcripts/487_P/487_TRANSCRIPT.csv` contains AppleDouble metadata, not CSV:
  - Starts with `Mac OS X` header bytes
  - No `speaker` / `value` columns present
  - Fails `pd.read_csv(..., sep="\t")` with `UnicodeDecodeError`

---

## Impact

- Transcript for participant **487** is invalid and cannot be loaded.
- Any full-dataset run that touches participant 487 fails at transcript loading.
- Data integrity issue during preparation, not in the transcript service itself.

---

## Resolution

1. Updated `_read_first_matching()` to **skip macOS resource fork entries** (`__MACOSX/` or `._*`).
2. Added unit test to ensure resource forks are ignored during extraction.
3. Updated Spec 04A code listing to reflect the fix.

**Note:** Existing extracted data must be **re-extracted** after this fix to repair any invalid transcripts (e.g., participant 487).

---

## Verification

```bash
python -m pytest tests/unit/scripts/test_prepare_dataset.py
```

---

## Files Changed

- `scripts/prepare_dataset.py`
- `tests/unit/scripts/test_prepare_dataset.py`
- `docs/specs/04A_DATA_ORGANIZATION.md`
