# BUG-045: Quantitative Severity Underestimates When Items Are N/A

**Status**: CLOSED
**Severity**: P1 (High)
**Discovered**: 2026-01-02
**Fixed**: 2026-01-02 (commit 5ae1d2a)

## The Issue

`PHQ8Assessment.total_score` treats **N/A** (unknown) item scores as **0**, and `PHQ8Assessment.severity`
is derived from that total. This can surface **misleadingly low severity labels** whenever any PHQ-8
items are unscored/unknown.

This is especially problematic because the quantitative prompt explicitly instructs the model to emit
`N/A` rather than assume absence (score `0`) when there is insufficient evidence.

This is not just a “display bug”: the API currently returns a single `severity` label in `/assess/quantitative`
and `/full_pipeline`, which is easy to misinterpret as a confident classification.

## Code Evidence

- `src/ai_psychiatrist/domain/value_objects.py`: `ItemAssessment.score_value` returns `0` when `score is None`.
- `src/ai_psychiatrist/domain/entities.py`: `PHQ8Assessment.total_score` sums `score_value` across all items.
- `src/ai_psychiatrist/domain/entities.py`: `PHQ8Assessment.severity` is computed from `total_score`.
- `server.py`: `/assess/quantitative` and `/full_pipeline` return `QuantitativeResult.severity` and `total_score`
  based on the above.
- `src/ai_psychiatrist/agents/quantitative.py`: logs `severity=assessment.severity.name` alongside `na_count`.

## Existing Test That Demonstrates the Bug

- `tests/unit/agents/test_quantitative.py` currently expects `SeverityLevel.MILD` even when 2 PHQ-8 items are `N/A`.
  This is the underestimation failure mode in unit-test form (min-total scoring treated as full-total severity).

## Repro (Deterministic)

```python
from ai_psychiatrist.domain.entities import PHQ8Assessment
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment

items = {
    item: ItemAssessment(item=item, evidence="", reason="", score=(3 if item == PHQ8Item.DEPRESSED else None))
    for item in PHQ8Item
}
assessment = PHQ8Assessment(items=items, mode=AssessmentMode.ZERO_SHOT, participant_id=1)

assert assessment.total_score == 3
assert assessment.na_count == 7
assert assessment.severity.name == "MINIMAL"  # Misleading: 7 items are unknown
```

## Impact Scope

- **API / UI risk**: Consumers may interpret `severity="MINIMAL"` as “no depression” even though the model abstained
  on most items and the true severity may be higher.
- **Logging / debugging**: Logs emit a single severity label even when `na_count > 0`.
- **Offline evaluation**: Paper-parity metrics are item-level MAE and do not require severity, but run outputs
  include `predicted_total`, which is currently a lower bound and should be labeled as such for auditability.

## Why This Matters (Psychiatry / Measurement)

- PHQ-8 severity bands are defined over a **complete** 8-item total. When items are unknown, severity is
  **not identified**, only bounded.
- Returning a single label (e.g., `"MINIMAL"`) can be misinterpreted as “low risk / no symptoms” even when
  evidence exists for high-frequency symptoms and the rest of the scale is simply unobserved.

## Proposed Fix (Design)

The fix is to represent partial PHQ-8 scoring as **bounds** rather than a single severity label.

1. Clarify semantics of totals:
   - Treat the existing `total_score` as a **lower bound** (`min_total_score`).
   - Add `max_total_score` (treat N/A as `3` per item maximum).
2. Add explicit severity bounds:
   - `severity_lower_bound = SeverityLevel.from_total_score(min_total_score)`
   - `severity_upper_bound = SeverityLevel.from_total_score(max_total_score)`
3. API output + logging:
   - Always return bounds (`*_lower_bound`, `*_upper_bound`).
   - Only return a single `severity` label when it is **determinate** (bounds are equal).

See implementation plan: `docs/_specs/spec-045-quantitative-severity-bounds.md`.

## Notes

- This is an **interpretation/safety correctness bug**, not a parsing or infrastructure issue.
- Any change here is likely to require updates to `server.py` response models and downstream scripts that
  consume `total_score`/`severity`.
