# BUG-017: FeedbackLoop Not Wired in API

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-20
**Spec Reference**: `docs/specs/11_FULL_PIPELINE.md`, `docs/specs/07_JUDGE_AGENT.md`

## Resolution

Wired FeedbackLoopService into server.py as specified:
- Added `FeedbackLoopService` initialization in lifespan
- Added `get_feedback_loop_service()` dependency getter
- Updated `/full_pipeline` to use `feedback_loop.run(transcript)` instead of direct agent call
- Response now includes `EvaluationResult` with all 4 quality metrics
- Pipeline follows Paper Section 2.3 order: qualitative (with loop) → quantitative → meta-review

---

## Executive Summary

The `FeedbackLoopService` exists and is tested (`src/ai_psychiatrist/services/feedback_loop.py`), but `server.py` calls `QualitativeAssessmentAgent.assess()` directly instead of using the feedback loop. Per the paper (Section 2.3.1-2.3.2), the qualitative assessment should go through an iterative refinement loop with the `JudgeAgent` until quality thresholds are met.

---

## Evidence

- **Spec 11 shows correct usage**:
  ```python
  # Qualitative with feedback loop
  loop_result = await feedback_loop.run(transcript)
  ```

- **server.py current implementation** (`assess_qualitative` endpoint):
  ```python
  agent = QualitativeAssessmentAgent(llm_client=ollama)
  assessment = await agent.assess(transcript)  # Direct call, no feedback loop
  ```

- **FeedbackLoopService** exists at `src/ai_psychiatrist/services/feedback_loop.py`

- **JudgeAgent** exists at `src/ai_psychiatrist/agents/judge.py`

---

## Impact

- **Quality Degradation**: Qualitative assessments may not meet quality thresholds
- **Paper Fidelity**: Section 2.3.1-2.3.2 describes iterative refinement, not single-pass
- **Unused Code**: `JudgeAgent` and `FeedbackLoopService` are tested but not used in production

---

## Scope & Disposition

- **Code Path**: Modern API (`server.py`)
- **Fix Category**: Integration wiring
- **Priority**: P2 - Affects quality but not a blocker

---

## Recommended Fix

1. Add `FeedbackLoopService` to server.py lifespan initialization
2. Replace direct `QualitativeAssessmentAgent.assess()` call with `FeedbackLoopService.run()`
3. Update response models to include evaluation metrics (iteration count, scores)

### Before:
```python
agent = QualitativeAssessmentAgent(llm_client=ollama)
assessment = await agent.assess(transcript)
```

### After:
```python
feedback_loop = FeedbackLoopService(
    qualitative_agent=qual_agent,
    judge_agent=judge_agent,
    settings=settings.feedback,
)
loop_result = await feedback_loop.run(transcript)
assessment = loop_result.final_assessment
evaluation = loop_result.final_evaluation
```

---

## Files Involved

- `server.py` (UPDATE - wire FeedbackLoopService)
- Response models may need updating to expose evaluation metrics

---

## Related Bugs

- **BUG-016**: MetaReviewAgent not implemented (same wiring pattern)
- **BUG-012**: Server used legacy agents - RESOLVED
