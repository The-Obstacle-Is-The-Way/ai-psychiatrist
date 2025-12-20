# BUG-007: Feedback Threshold Inconsistent with low_scores

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/07_JUDGE_AGENT.md`, `docs/specs/07.5_INTEGRATION_CHECKPOINT_QUALITATIVE.md`

---

## Executive Summary

`FeedbackLoopSettings.score_threshold` is configurable, but `QualitativeEvaluation.low_scores` was **hard-coded** to treat scores <= 3 as low. As a result, when a custom threshold is used, the feedback loop's **control logic** used the configured threshold while **logging and feedback generation** still used the fixed <=3 rule. This created inconsistent behavior and confusing feedback for non-default thresholds.

---

## Evidence (Pre-Fix)

- `FeedbackLoopService._needs_improvement()` used `score_threshold` (configurable).
- `evaluation.low_scores` used `EvaluationScore.is_low` (fixed <=3).
- `FeedbackLoopService` and `JudgeAgent.get_feedback_for_low_scores()` relied on `evaluation.low_scores` for logging/feedback.

Result: With `score_threshold=2`, the loop might **not** run (correct), but logs/feedback would still treat score 3 as low (incorrect).

---

## Impact

- Incorrect feedback and logging when using non-default thresholds.
- Harder to tune feedback loop behavior and reason about results.

---

## Resolution

1. Added `QualitativeEvaluation.low_scores_for_threshold(threshold)`.
2. Updated `FeedbackLoopService` to use configured threshold for logging and feedback.
3. Updated `JudgeAgent.get_feedback_for_low_scores()` to accept an optional threshold.
4. Added tests for threshold-respecting behavior.

---

## Verification

```bash
pytest tests/unit/agents/test_judge.py -v --no-cov
pytest tests/unit/domain/test_entities.py -v --no-cov
```

---

## Files Changed

- `src/ai_psychiatrist/domain/entities.py`
- `src/ai_psychiatrist/agents/judge.py`
- `src/ai_psychiatrist/services/feedback_loop.py`
- `tests/unit/agents/test_judge.py`
- `tests/unit/domain/test_entities.py`
