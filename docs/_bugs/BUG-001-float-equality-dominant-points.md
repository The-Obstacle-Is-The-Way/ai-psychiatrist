# BUG-001: Float Equality in `_compute_dominant_points`

**Status**: Fixed (2026-01-03)
**Severity**: Critical
**Component**: `src/ai_psychiatrist/metrics/selective_prediction.py`
**Discovered**: 2026-01-03
**Related Spec**: Spec 052 (Excess AURC/AUGRC)

## Summary

The `_compute_dominant_points` function uses float tuple membership in a set for matching, which is fragile due to floating point precision issues.

## Root Cause

Lines 425-431 of `selective_prediction.py`:

```python
# We need to return a mask matching the original inputs.
dominant_set = set(lower)

# We match by value (float equality is risky but these come from same source)
mask = []
for c, r in zip(coverages, risks, strict=True):
    mask.append((c, r) in dominant_set)
```

The comment acknowledges the risk ("float equality is risky") but proceeds anyway. When floats undergo arithmetic operations, precision loss can cause `(c, r)` to fail membership tests even when the values are mathematically equal.

## Symptoms

- `compute_aurc_achievable()` may incorrectly exclude valid dominant points
- The convex hull mask may not match all expected points
- Results are non-deterministic across platforms/Python versions

## Expected Behavior

The dominant point detection should use index-based tracking instead of value-based matching to avoid floating point comparison issues.

## Proposed Fix

Replace value-based matching with index-based tracking:

```python
def _compute_dominant_points(coverages: list[float], risks: list[float]) -> list[bool]:
    """Compute mask for dominant points (lower convex hull)."""
    if not coverages:
        return []

    # Create indexed points and sort by coverage
    indexed_points = sorted(enumerate(zip(coverages, risks, strict=True)), key=lambda x: x[1][0])

    # Monotone Chain algorithm for lower hull - track original indices
    lower: list[tuple[int, tuple[float, float]]] = []
    for idx, point in indexed_points:
        while len(lower) >= 2 and _cross_product(lower[-2][1], lower[-1][1], point) <= 0:
            lower.pop()
        lower.append((idx, point))

    # Create mask using indices (avoids float equality issues)
    dominant_indices = {idx for idx, _ in lower}
    return [i in dominant_indices for i in range(len(coverages))]
```

## Test Coverage

The existing test `test_aurc_achievable_realistic` may not catch this bug because:
1. It uses synthetic data that doesn't trigger precision issues
2. The test values may coincidentally pass equality checks

## Notes

This bug was noted in a previous session's summary as "fixed" but the fix was never applied to the codebase. The current code still contains the fragile implementation.
