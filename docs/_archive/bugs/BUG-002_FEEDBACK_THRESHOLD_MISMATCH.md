# BUG-002: Feedback Threshold Mismatch (Scores <=2 vs Paper <4)

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/02_CORE_DOMAIN.md`, `docs/specs/03_CONFIG_LOGGING.md`, `docs/specs/07_JUDGE_AGENT.md`

---

## Executive Summary

The domain model treated **evaluation scores <= 2** as "low" and therefore eligible for feedback loop refinement. The paper specifies **scores below 4** should trigger refinement. Config (`FEEDBACK_SCORE_THRESHOLD=3`) already matched the paper, so the domain logic was out of sync with both the paper and the configured threshold.

**Result:** A score of **3** would incorrectly be treated as acceptable in the domain, reducing feedback loop activation and deviating from paper behavior.

---

## Evidence (Pre-Fix)

- `EvaluationScore.is_low` returned `self.score <= 2`.
- `QualitativeEvaluation.low_scores` used `EvaluationScore.is_low`, so it **excluded score 3**.
- Config and paper both specify **trigger when score < 4 (i.e., <= 3)**.

Paper reference:
- Section 2.3.1: "original evaluation score **below four**" triggers feedback loop.

---

## Impact

- Feedback loop would **not run** for score **3**, even though the paper requires refinement for any score below 4.
- Under-refinement can reduce assessment quality and paper-alignment.

---

## Resolution

Aligned domain logic and specs with paper threshold:

1. Updated `EvaluationScore.is_low` to return `score <= 3`.
2. Updated `QualitativeEvaluation.low_scores` docstring to match.
3. Updated domain tests to treat score **3** as low.
4. Updated Spec 02 examples and comments to match paper threshold.

---

## Verification

```bash
python -m pytest tests/unit/domain/test_value_objects.py tests/unit/domain/test_entities.py
```

---

## Files Changed

- `src/ai_psychiatrist/domain/value_objects.py`
- `src/ai_psychiatrist/domain/entities.py`
- `tests/unit/domain/test_value_objects.py`
- `tests/unit/domain/test_entities.py`
- `docs/specs/02_CORE_DOMAIN.md`
