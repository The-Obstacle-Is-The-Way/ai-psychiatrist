# BUG-045: Quantitative Severity Underestimates When Items Are N/A

**Status**: RESOLVED
**Severity**: P1 (High; clinical interpretation risk)
**Discovered**: 2026-01-02
**Fixed**: 2026-01-02
**Spec**: `docs/_specs/spec-045-quantitative-severity-bounds.md`
**Verification**: `uv run pytest tests/ --tb=short` (2026-01-02)

---

## Summary

Historically, `PHQ8Assessment.severity` was derived from `total_score` where N/A (unknown) items were treated as `0`.
This produced **misleadingly low single-label severities** for partial assessments.

The system now reports:

- **Total score bounds**: `min_total_score` (N/A→0) and `max_total_score` (N/A→3)
- **Severity bounds**: `severity_lower_bound` and `severity_upper_bound`
- A single `severity` label **only when determinate**; otherwise `severity=None`

## The Issue

This bug was that the domain model and API surfaced a single severity label derived from a lower-bound total score.
When any items are unknown (`N/A`), the severity is **not identified**, only bounded.

This is especially problematic because the quantitative prompt explicitly instructs the model to emit
`N/A` rather than assume absence (score `0`) when there is insufficient evidence.

This is not just a “display bug”: the API used to return a single `severity` label in `/assess/quantitative`
and `/full_pipeline`, which is easy to misinterpret as a confident classification.

## Code Evidence

- Domain:
  - `src/ai_psychiatrist/domain/entities.py`: adds `min_total_score`, `max_total_score`, and severity bounds.
  - `src/ai_psychiatrist/domain/entities.py`: `PHQ8Assessment.severity` is now `SeverityLevel | None`.
- API:
  - `server.py`: returns `total_score_min`, `total_score_max`, `severity_lower_bound`, `severity_upper_bound`,
    and nullable `severity`.
- Reproduction outputs:
  - `scripts/reproduce_results.py`: writes `predicted_total_min/max` and severity bounds to run artifacts.

## Regression Coverage

- `tests/unit/domain/test_entities.py`: validates score bounds and severity determinacy rules.
- `tests/unit/agents/test_quantitative.py`: asserts partial assessments return `severity is None` and exposes bounds.

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

assert assessment.na_count == 7
assert assessment.min_total_score == 3
assert assessment.max_total_score == 24
assert assessment.severity is None
assert assessment.severity_lower_bound.name == "MINIMAL"
assert assessment.severity_upper_bound.name == "SEVERE"
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
  This is resolved by reporting bounds and only emitting a determinate `severity` label when valid.
