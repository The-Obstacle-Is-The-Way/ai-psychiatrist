# BUG-002: DRY Violation in Oracle Items Creation

**Severity**: P4 (Code Style / Maintenance)
**Status**: Resolved (Already Fixed)
**Resolved**: 2026-01-03
**Resolution**: Code already has `_compute_optimal_metric` helper at `selective_prediction.py:318-327`
**Created**: 2026-01-03
**File**: `src/ai_psychiatrist/metrics/selective_prediction.py`

## Description

The functions `compute_aurc_optimal` and `compute_augrc_optimal` share identical logic for creating the "Oracle" CSF items (sorting by loss and assigning synthetic confidence). This logic is encapsulated in `_create_oracle_items`, which is good.

However, the calculation of the optimal metrics themselves:
```python
def compute_aurc_optimal(...):
    oracle_items = _create_oracle_items(items, loss=loss)
    if not oracle_items: return 0.0
    return compute_aurc(oracle_items, loss=loss)

def compute_augrc_optimal(...):
    oracle_items = _create_oracle_items(items, loss=loss)
    if not oracle_items: return 0.0
    return compute_augrc(oracle_items, loss=loss)
```

And similarly for `compute_eaurc` / `compute_eaugrc`.

While minor, there is a pattern of "create oracle items -> compute metric" that is repeated.

## Impact

Low impact. Increases boilerplate slightly.

## Recommended Fix

No immediate action required. If more metrics (e.g., `compute_f1_optimal`) are added, consider a generic `compute_optimal_metric(items, metric_fn)` helper.
