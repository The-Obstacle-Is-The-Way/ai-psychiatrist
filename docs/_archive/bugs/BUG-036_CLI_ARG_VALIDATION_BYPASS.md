# BUG-036: CLI Arg Validation Bypass (Invalid/Unsupported Runs)

**Date**: 2026-01-07
**Status**: FIXED
**Severity**: P1 (Wastes hours; can invalidate results)
**Affects**: `scripts/reproduce_results.py` runs with CLI overrides
**Discovered By**: Senior agent audit (post-Spec 061/062 merge)

---

## Executive Summary

Several CLI flags in `scripts/reproduce_results.py` could silently create **invalid or unsupported**
run configurations:

- `--limit 0` unintentionally ran the **full split** (truthiness bug).
- Numeric overrides (`--total-min-coverage`, `--binary-threshold`, `--consistency-samples`,
  `--consistency-temperature`) accepted out-of-range values without fail-fast checks.
- `--prediction-mode binary --binary-strategy direct|ensemble` passed `--dry-run` and only failed
  later (per-participant), wasting run time and producing misleading artifacts.

This violates repo policy: *prefer loud failures over best-effort fallbacks*.

---

## Root Causes

### 1) Truthiness bug for `--limit`

**File**: `scripts/reproduce_results.py`

The participant truncation logic used a truthiness check:

- `if limit:` treats `0` as “not set”
- Result: `--limit 0` ran all participants

### 2) Missing range validation for CLI overrides

Pydantic validates `.env` values, but CLI overrides can bypass constraints:

- Prediction overrides are assigned directly to settings fields (no range checks).
- Consistency overrides are applied to local variables (no range checks).

### 3) Unsupported strategy allowed to proceed in dry-run

`BINARY_STRATEGY=direct|ensemble` is explicitly deferred, but `--dry-run` did not flag it. Users
could start multi-hour runs that deterministically fail.

---

## Fix

### Code changes

- Added `validate_cli_args()` to enforce ranges and reject invalid CLI combinations.
- Added `validate_prediction_settings()` to fail fast when `prediction_mode="binary"` and
  `binary_strategy!="threshold"`.
- Called both validators in `main()` and `main_async()` so `--dry-run` is also protected.
- Fixed participant truncation to use `if limit is not None:` (no truthiness trap).

---

## Verification

- New unit tests:
  - `tests/unit/scripts/test_reproduce_results.py` covers invalid `--limit`, consistency overrides,
    prediction overrides, and unsupported binary strategy.
- Manual check:
  - `uv run python scripts/reproduce_results.py --split dev --dry-run --prediction-mode binary --binary-strategy direct`
    now fails immediately with an error.
- CI:
  - `make ci` passes.
