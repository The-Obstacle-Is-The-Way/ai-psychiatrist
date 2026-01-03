# BUG-021: Float Equality in Dominant Points (Convex Hull)

**Severity**: P3 (Minor / Robustness)
**Status**: Resolved (Already Fixed)
**Resolved**: 2026-01-03
**Resolution**: Code already contains fix at `selective_prediction.py:388-397` with `epsilon = 1e-10`
**Created**: 2026-01-03
**File**: `src/ai_psychiatrist/metrics/selective_prediction.py`

## Description

The `_compute_dominant_points` function implements a Monotone Chain algorithm to find the lower convex hull of the risk-coverage curve. It uses a cross-product check to determine orientation:

```python
def _cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

...
while len(lower) >= 2 and _cross_product(lower[-2][1], lower[-1][1], point) <= 0:
    lower.pop()
```

The check `<= 0` relies on exact floating point comparison. In geometric algorithms, floating point precision errors can cause strictly collinear points to appear slightly non-collinear (or vice-versa), leading to:
1.  Jittery hulls (including/excluding points unpredictably).
2.  Potential infinite loops (unlikely here due to loop structure, but possible in other hull algos).
3.  Inconsistency across platforms.

## Impact

The `achievable_aurc` metric might vary slightly due to precision noise. While likely negligible for high-level metrics, it is not robust.

## Recommended Fix

Introduce an epsilon for the comparison:

```python
EPSILON = 1e-10

# ...
while len(lower) >= 2 and _cross_product(..., point) <= EPSILON:
    lower.pop()
```

Or ensure `_cross_product` logic explicitly handles "close to zero" as zero.
