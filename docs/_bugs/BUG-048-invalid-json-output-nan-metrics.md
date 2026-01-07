# BUG-048: Invalid JSON Output When Metrics Are NaN

**Date**: 2026-01-07
**Status**: FIXED (pending senior review)
**Severity**: P1 (Breaks tooling; can invalidate downstream analysis)
**Affects**: `scripts/reproduce_results.py` JSON outputs
**Discovered By**: Senior audit (post-Spec 061/062 merge)

---

## Executive Summary

Some `scripts/reproduce_results.py` runs wrote **invalid JSON** by emitting `NaN` literals for
aggregate metrics when there were **zero evaluated subjects** (e.g., `--limit 1` selecting a
participant where all 8 items are `N/A`).

This breaks strict JSON parsers and common tooling (notably `jq`), and can silently derail
downstream analysis pipelines.

---

## Evidence

Example artifact containing `NaN` literals (invalid JSON):

- `data/outputs/both_paper-test_20260103_182316.json` (contains `"item_mae_weighted": NaN`, etc.)

---

## Root Cause

1. `compute_item_level_metrics()` used `float("nan")` as a placeholder when metrics were undefined
   (no evaluated subjects / no errors).
2. `save_results()` used `json.dump(..., allow_nan=True)` (Python default), which serializes
   non-finite floats as `NaN`/`Infinity` literals that are **not valid JSON**.

---

## Fix

### Code

- Sanitized non-finite aggregate metrics in `ExperimentResults.to_dict()`:
  - `item_mae_weighted`, `item_mae_by_item`, `item_mae_by_subject`, `prediction_coverage`
  - `per_item[*].coverage`
  - Non-finite values now serialize as `null`.
- Enforced strict JSON output:
  - `save_results()` now calls `json.dump(..., allow_nan=False)` so unexpected `NaN`/`Inf` fails
    loudly.

### Tests

- Added regression test:
  - `tests/unit/scripts/test_reproduce_results.py::test_experiment_results_to_dict_is_strict_json_when_item_metrics_nan`

---

## Verification

- Unit tests cover serialization of undefined metrics without emitting `NaN`.
- `make ci` should pass after the fix.
