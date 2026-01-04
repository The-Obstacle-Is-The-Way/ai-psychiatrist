# BUG-033: JSON Parse Failures Persisting After Control-Char Fixups (Run 11)

**Status**: ✅ Resolved (Implemented)
**Severity**: P0 (Inference failures / retries exhausted)
**Filed**: 2026-01-04
**Resolved**: 2026-01-04
**Component**: `src/ai_psychiatrist/infrastructure/llm/responses.py`
**Observed In**: Run 11 (`data/outputs/run11_confidence_suite_20260103_215102.log`)

---

## Summary

Run 11 contained repeated JSON parsing failures of the form:

- `json.JSONDecodeError: Expecting property name enclosed in double quotes`
- `SyntaxError: unexpected character after line continuation character`

These failures were deterministic for identical `text_hash` values and could exhaust retry budgets.

---

## Root Cause

Run 11 started **before** the Spec 059 patch (`json-repair` fallback) was in effect. The run therefore used the older parsing stack that could not recover from:

- Python-literal dicts mixed with stray backslashes
- Unquoted keys / trailing text / truncated structures
- Invalid escape sequences that break strict JSON parsing

---

## Resolution

### 1) Defense-in-depth parser includes json-repair (Spec 059)

`parse_llm_json()` now falls back to `json_repair.loads()` after:

1. `tolerant_json_fixups()`
2. `json.loads()`
3. `ast.literal_eval()` (after literal conversion)

### 2) Suppress noisy SyntaxWarnings during Python-literal fallback

Some malformed escape sequences (e.g. `\q`) can trigger `SyntaxWarning` during `ast.literal_eval()`.
We explicitly suppress `SyntaxWarning` during that fallback path to keep runs clean and deterministic.

### 3) Improve privacy-safe failure logging

When all repair attempts fail (including json-repair), we now include:

- `json_error_type`, `json_error_lineno`, `json_error_colno`, `json_error_pos`
- `repair_error_type`
- `text_hash`, `text_length`

No raw text is logged (DAIC-WOZ licensing constraint).

---

## Regression Tests

See:

- `tests/unit/infrastructure/llm/test_tolerant_json_fixups.py::TestJsonRepairFallback`
  - truncated JSON
  - unquoted keys
  - trailing text
  - python-literal + stray backslash (Run 11 pattern)
  - invalid escape sequences inside strings

---

## Notes

If you still observe parse failures after these fixes, file a new bug with:

- `text_hash`
- `json_error_type/lineno/colno/pos`
- `repair_error_type`
- the run id / mode / participant id (counts only)
