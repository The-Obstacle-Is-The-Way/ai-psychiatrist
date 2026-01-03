# BUG-002: DRY Violation - Duplicate Oracle Construction

**Status**: Fixed (2026-01-03)
**Severity**: Medium
**Component**: `src/ai_psychiatrist/metrics/selective_prediction.py`
**Discovered**: 2026-01-03
**Related Spec**: Spec 052 (Excess AURC/AUGRC)

## Summary

The oracle CSF construction logic is duplicated between `compute_aurc_optimal` and `compute_augrc_optimal`. This violates the DRY (Don't Repeat Yourself) principle and increases maintenance burden.

## Root Cause

Lines 260-313 (`compute_aurc_optimal`) and lines 316-362 (`compute_augrc_optimal`) contain nearly identical code:

```python
# In compute_aurc_optimal:
predicted = [i for i in items if i.pred is not None]
abstained = [i for i in items if i.pred is None]

if not predicted:
    return 0.0

losses = [_compute_loss(i, loss) for i in predicted]
sorted_pairs = sorted(zip(losses, predicted, strict=True), key=lambda x: x[0])

n_predicted = len(sorted_pairs)
oracle_items: list[ItemPrediction] = []

for rank, (_, original_item) in enumerate(sorted_pairs):
    conf = 1.0 - (rank / n_predicted)
    oracle_items.append(...)

for item in abstained:
    oracle_items.append(...)

return compute_aurc(oracle_items, loss=loss)
```

The same pattern is repeated in `compute_augrc_optimal`, with only the final return statement differing.

## Symptoms

- Code duplication (~50 lines duplicated)
- If bug fixes are needed in oracle construction, they must be applied in two places
- Higher risk of divergent behavior if one function is updated but not the other

## Expected Behavior

A helper function `_create_oracle_items()` should be extracted and shared between both functions.

## Proposed Fix

Extract a shared helper:

```python
def _create_oracle_items(
    items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]
) -> list[ItemPrediction]:
    """Create oracle-ranked items with synthetic confidence based on loss.

    Items are ranked by loss (ascending), so lowest-loss items get highest confidence.
    Abstained items are preserved with confidence=-inf to maintain correct n.
    """
    predicted = [i for i in items if i.pred is not None]
    abstained = [i for i in items if i.pred is None]

    if not predicted:
        return []

    losses = [_compute_loss(i, loss) for i in predicted]
    sorted_pairs = sorted(zip(losses, predicted, strict=True), key=lambda x: x[0])

    n_predicted = len(sorted_pairs)
    oracle_items: list[ItemPrediction] = []

    for rank, (_, original_item) in enumerate(sorted_pairs):
        conf = 1.0 - (rank / n_predicted)
        oracle_items.append(
            ItemPrediction(
                participant_id=original_item.participant_id,
                item_index=original_item.item_index,
                pred=original_item.pred,
                gt=original_item.gt,
                confidence=conf,
            )
        )

    for item in abstained:
        oracle_items.append(
            ItemPrediction(
                participant_id=item.participant_id,
                item_index=item.item_index,
                pred=None,
                gt=item.gt,
                confidence=float("-inf"),
            )
        )

    return oracle_items


def compute_aurc_optimal(
    items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]
) -> float:
    """Compute optimal AURC (oracle CSF baseline)."""
    oracle_items = _create_oracle_items(items, loss=loss)
    if not oracle_items:
        return 0.0
    return compute_aurc(oracle_items, loss=loss)


def compute_augrc_optimal(
    items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]
) -> float:
    """Compute optimal AUGRC (oracle CSF baseline)."""
    oracle_items = _create_oracle_items(items, loss=loss)
    if not oracle_items:
        return 0.0
    return compute_augrc(oracle_items, loss=loss)
```

## Test Coverage

Existing tests should continue to pass after refactoring since behavior is unchanged.

## Notes

This issue was noted in a previous session's summary as "fixed" but the refactoring was never applied to the codebase.
