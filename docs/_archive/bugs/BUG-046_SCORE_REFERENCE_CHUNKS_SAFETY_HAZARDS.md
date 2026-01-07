# BUG-046: score_reference_chunks Safety Hazards (Privacy Leak + `--limit` Truthiness Trap)

**Date**: 2026-01-07
**Status**: FIXED
**Severity**: P1 (privacy risk; can waste hours)
**Affects**: `scripts/score_reference_chunks.py` (Spec 35 chunk scoring utility)

---

## Summary

`scripts/score_reference_chunks.py` has two issues that violate repo safety and research hygiene expectations:

1. **Privacy leak**: logs may include raw DAIC-WOZ chunk text and/or LLM response content via `chunk_preview` / `response_preview`.
2. **Truthiness trap**: `--limit 0` is treated as “unset”, causing the script to process the full set of chunks (potentially hours of unnecessary work/cost).

---

## Impact

### A) Privacy leak (P1)

The script currently logs:

- `chunk_preview=chunk_text[:50]` (raw chunk text)
- `response_preview=response[:50]` (raw LLM output, which may echo chunk content)

If the user pipes logs to a file (`2>&1 | tee ...`) or shares run logs, this can leak restricted transcript content.

### B) `--limit` truthiness trap (P2)

The script uses checks like:

- `if config.limit ...`
- `if not config.limit ...`

So `--limit 0` behaves like `--limit` was not provided, unexpectedly running the full workload.

---

## Root Cause

**File**: `scripts/score_reference_chunks.py`

- Logging emits preview fields derived from sensitive text.
- `limit` is not validated and is used in boolean contexts, conflating `None` and `0`.

---

## Fix

### 1) Privacy-safe logging

- Remove `chunk_preview` / `response_preview` from logs.
- Replace with:
  - `chunk_hash` / `chunk_chars`
  - `response_hash` / `response_chars` (when a response exists)
- Ensure no raw `chunk_text`/`response` substrings are logged.

### 2) Fail-fast `--limit` validation + explicit checks

- Reject `--limit < 1` with a clear error.
- Replace truthiness checks with `is None` / `is not None`.

---

## Verification

- Unit tests added:
  - `tests/unit/scripts/test_score_reference_chunks.py::test_validate_cli_args_rejects_limit_zero`
  - `tests/unit/scripts/test_score_reference_chunks.py::test_score_chunk_logs_are_privacy_safe_on_invalid_json`
- CI:
  - `make ci` passes.
