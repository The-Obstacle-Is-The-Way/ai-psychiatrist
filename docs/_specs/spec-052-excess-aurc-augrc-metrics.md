# Spec 052: Excess AURC/AUGRC and Optimal Reference Metrics

**Status**: Ready to Implement
**Priority**: Low (metric enrichment)
**Depends on**: None (standalone)
**Estimated effort**: Low
**Research basis**: [fd-shifts (NeurIPS 2024)](https://arxiv.org/abs/2407.01032), [AsymptoticAURC (ICML 2025)](https://arxiv.org/abs/2410.15361)

## 0. Problem Statement

Our current metrics (AURC, AUGRC) measure **absolute** risk-coverage performance. However, they don't tell us:

1. **How much room for improvement exists?** (distance from optimal)
2. **Is the CSF providing value?** (vs. random ranking)
3. **What's the theoretical lower bound?** (oracle CSF)

The fd-shifts benchmark computes **excess metrics** (e-AURC, e-AUGRC) that subtract the optimal baseline:

```text
e-AURC = AURC - AURC_optimal
```

Where `AURC_optimal` is achieved when the CSF perfectly ranks predictions by their correctness (oracle).

This enables:
- **Absolute comparison**: e-AURC = 0 means perfect CSF
- **Method comparison**: e-AURC differences are interpretable
- **Progress tracking**: "We reduced e-AUGRC by 30%"

## 1. Goals / Non-Goals

### 1.1 Goals

- Implement **AURC_optimal** and **AUGRC_optimal** computation
- Add **e-AURC** (excess AURC) and **e-AUGRC** (excess AUGRC) metrics
- Add **achievable AURC** (convex hull of dominant points)
- Integrate into `evaluate_selective_prediction.py` output
- Document interpretation in metrics documentation

### 1.2 Non-Goals

- AURC as a loss function for training (future work)
- Novel AURC estimators from AsymptoticAURC (future work)

## 2. Background: Optimal and Excess Metrics

### 2.1 AURC_optimal (Binary Residuals)

From fd-shifts `rc_stats.py`:

```python
def aurc_optimal_binary(accuracy: float) -> float:
    """Optimal AURC for binary correctness."""
    err = 1 - accuracy
    if err == 0:
        return 0.0
    return err + (1 - err) * np.log(1 - err)
```

**Interpretation**: This is achieved when:
1. All correct predictions have higher confidence than all incorrect predictions
2. The CSF perfectly separates correct from incorrect

### 2.2 AUGRC_optimal (Binary Residuals)

From fd-shifts:

```python
def augrc_optimal_binary(error_rate: float) -> float:
    """Optimal AUGRC for binary correctness."""
    return 0.5 * error_rate ** 2
```

**Interpretation**: AUGRC_optimal is always lower than AURC_optimal because generalized risk penalizes abstention differently.

### 2.3 Excess Metrics

```text
e-AURC = AURC - AURC_optimal
e-AUGRC = AUGRC - AUGRC_optimal
```

**Interpretation**:
- e-AURC = 0 → Perfect CSF (oracle)
- e-AURC > 0 → CSF has room for improvement
- e-AURC >> 0 → CSF is far from optimal

### 2.4 Achievable AURC (Convex Hull)

The **achievable AURC** considers only the **dominant points** on the risk-coverage curve (convex hull vertices). This removes suboptimal working points that no rational user would choose.

From fd-shifts:

```python
def compute_dominant_points(coverages, risks):
    """Find convex hull vertices in ROC space."""
    from scipy.spatial import ConvexHull
    # Transform to ROC space, compute hull, transform back
    ...
```

## 3. Proposed Solution

### 3.1 Extend `selective_prediction.py`

Add to `src/ai_psychiatrist/metrics/selective_prediction.py`:

```python
def compute_aurc_optimal(items: Sequence[ItemPrediction]) -> float:
    """Compute optimal AURC (oracle CSF baseline)."""
    predicted = [i for i in items if i.pred is not None]
    if not predicted:
        return 0.0

    # Compute error rate (for binary correctness)
    n_correct = sum(1 for i in predicted if i.pred == i.gt)
    n_total = len(predicted)
    error_rate = 1 - (n_correct / n_total)

    if error_rate == 0:
        return 0.0

    # Formula for binary residuals
    return error_rate + (1 - error_rate) * np.log(1 - error_rate)


def compute_augrc_optimal(items: Sequence[ItemPrediction]) -> float:
    """Compute optimal AUGRC (oracle CSF baseline)."""
    predicted = [i for i in items if i.pred is not None]
    if not predicted:
        return 0.0

    n_correct = sum(1 for i in predicted if i.pred == i.gt)
    n_total = len(predicted)
    error_rate = 1 - (n_correct / n_total)

    return 0.5 * error_rate ** 2


def compute_eaurc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Compute excess AURC (distance from optimal)."""
    aurc = compute_aurc(items, loss=loss)
    aurc_opt = compute_aurc_optimal(items)
    return aurc - aurc_opt


def compute_eaugrc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Compute excess AUGRC (distance from optimal)."""
    augrc = compute_augrc(items, loss=loss)
    augrc_opt = compute_augrc_optimal(items)
    return augrc - augrc_opt
```

### 3.2 Non-Binary Residuals

For regression (our case with absolute error), the optimal baseline requires sorting by residual:

```python
def compute_aurc_optimal_regression(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Optimal AURC for regression residuals (oracle ranking by error)."""
    predicted = [i for i in items if i.pred is not None]
    if not predicted:
        return 0.0

    # Sort by loss (ascending = oracle ranking)
    losses = [_compute_loss(i, loss) for i in predicted]
    sorted_items = [x for _, x in sorted(zip(losses, predicted, strict=True))]

    # Create oracle CSF (rank = confidence, lower loss = higher confidence)
    oracle_items = [
        ItemPrediction(
            participant_id=i.participant_id,
            item_index=i.item_index,
            pred=i.pred,
            gt=i.gt,
            confidence=1 - rank / len(sorted_items),  # Higher confidence for lower loss
        )
        for rank, i in enumerate(sorted_items)
    ]

    return compute_aurc(oracle_items, loss=loss)
```

### 3.3 Achievable AURC (Dominant Points)

```python
def compute_aurc_achievable(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> float:
    """Compute achievable AURC using only dominant points (convex hull)."""
    curve = compute_risk_coverage_curve(items, loss=loss)
    dominant_mask = _compute_dominant_points(curve.coverage, curve.selective_risk)

    dominant_coverages = [c for c, m in zip(curve.coverage, dominant_mask) if m]
    dominant_risks = [r for r, m in zip(curve.selective_risk, dominant_mask) if m]

    return _integrate_curve(dominant_coverages, dominant_risks, curve.cmax, mode="aurc")
```

### 3.4 Evaluation Script Output

Update `scripts/evaluate_selective_prediction.py` to include:

```json
{
  "metrics": {
    "aurc": 0.135,
    "aurc_optimal": 0.082,
    "eaurc": 0.053,
    "aurc_achievable": 0.128,

    "augrc": 0.031,
    "augrc_optimal": 0.011,
    "eaugrc": 0.020,

    "cmax": 0.53
  },
  "interpretation": {
    "aurc_gap_pct": 39.3,
    "augrc_gap_pct": 64.5,
    "achievable_gain_pct": 5.2
  }
}
```

**Interpretation fields**:
- `aurc_gap_pct`: `(eaurc / aurc_optimal) * 100` — % room for improvement
- `augrc_gap_pct`: `(eaugrc / augrc_optimal) * 100`
- `achievable_gain_pct`: `((aurc - aurc_achievable) / aurc) * 100` — gain from better working point selection

## 4. Implementation Plan

### Phase 1: Core Metrics

1. Implement `compute_aurc_optimal`, `compute_augrc_optimal`
2. Implement `compute_eaurc`, `compute_eaugrc`
3. Handle both binary and regression residuals

### Phase 2: Achievable AURC

4. Implement `_compute_dominant_points` (convex hull)
5. Implement `compute_aurc_achievable`

### Phase 3: Integration

6. Add to `evaluate_selective_prediction.py` output
7. Add interpretation fields
8. Update documentation

## 5. Test Plan

### 5.1 Unit Tests

- `test_aurc_optimal_perfect`: All correct → optimal = 0
- `test_aurc_optimal_all_wrong`: All wrong → optimal = 1
- `test_eaurc_bounds`: 0 <= e-AURC <= AURC
- `test_achievable_leq_aurc`: achievable AURC <= AURC

### 5.2 Validation

Compare our optimal calculations against fd-shifts reference:

```python
from fd_shifts.analysis.rc_stats import RiskCoverageStats

# Our implementation
our_optimal = compute_aurc_optimal(items)

# fd-shifts reference
fd_stats = RiskCoverageStats(confids=confids, residuals=residuals)
their_optimal = fd_stats.aurc_optimal

assert np.isclose(our_optimal, their_optimal, rtol=1e-3)
```

## 6. Expected Outcomes

Based on Run 9 results:

| Metric | Value | Optimal | Excess | Gap % |
|--------|-------|---------|--------|-------|
| AURC | 0.135 | ~0.08 | ~0.055 | ~69% |
| AUGRC | 0.031 | ~0.01 | ~0.021 | ~68% |

**Interpretation**: Our CSF is capturing ~31% of the theoretical maximum ranking quality. There's significant room for improvement.

## 7. Acceptance Criteria

- [ ] `compute_aurc_optimal`, `compute_augrc_optimal` implemented
- [ ] `compute_eaurc`, `compute_eaugrc` implemented
- [ ] `compute_aurc_achievable` implemented (convex hull)
- [ ] `evaluate_selective_prediction.py` outputs optimal and excess metrics
- [ ] Documentation in `docs/statistics/metrics-and-evaluation.md`
- [ ] Tests pass: `make ci`

## 8. File Changes

### Modified Files

- `src/ai_psychiatrist/metrics/selective_prediction.py` (add optimal/excess/achievable)
- `scripts/evaluate_selective_prediction.py` (output new metrics)
- `docs/statistics/metrics-and-evaluation.md` (document interpretation)

### Test Files

- `tests/unit/metrics/test_selective_prediction.py` (add optimal tests)

## 9. References

- [fd-shifts rc_stats.py](../../_reference/fd-shifts/fd_shifts/analysis/rc_stats.py)
- [fd-shifts NeurIPS 2024](https://arxiv.org/abs/2407.01032)
- [AsymptoticAURC ICML 2025](https://arxiv.org/abs/2410.15361)
- [Geifman & El-Yaniv 2017 - SelectiveNet](https://arxiv.org/abs/1901.09192)
