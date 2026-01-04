# BUG-025: PydanticAI TextOutput Fails on Python-Literal “JSON”

**Severity**: P1 (Reliability)
**Status**: Resolved
**Resolved**: 2026-01-03
**Created**: 2026-01-03
**Files**:
- `src/ai_psychiatrist/agents/extractors.py`
- `src/ai_psychiatrist/agents/quantitative.py`

## Description

During quantitative scoring (including consistency mode), the model occasionally emits a **Python-literal dict** instead of strict JSON inside `<answer>` tags (e.g., single-quoted keys/strings, `None`, mixed `null`), which causes:

- `json.loads(...)` to raise `JSONDecodeError` (often: “Expecting property name enclosed in double quotes”)
- The `extract_quantitative()` TextOutput validator to raise `ModelRetry`
- After `PYDANTIC_AI_RETRIES` attempts, PydanticAI to raise:
  `pydantic_ai.exceptions.UnexpectedModelBehavior: Exceeded maximum retries (3) for output validation`

In **consistency mode**, a single failed sample previously aborted the entire participant, amplifying failure probability proportional to `n_samples`.

## Impact

- Sporadic participant failures in run logs (e.g., PID 383).
- Higher failure rate in consistency runs due to multiple independent scoring calls.

## Root Cause

The structured-output contract is “JSON in `<answer>` tags”, but the model sometimes produces *JSON-like* output that is valid as a Python literal and invalid as JSON.

## Fix

1. **Tolerant parsing fallback** in `extract_quantitative()`:
   - Attempt `json.loads(...)` first.
   - On `JSONDecodeError`, fall back to `ast.literal_eval(...)` after converting JSON literals (`true/false/null`) to Python (`True/False/None`) **outside strings**.

2. **Consistency-mode resilience** in `QuantitativeAssessmentAgent.assess_with_consistency()`:
   - Collect `n_samples` successful samples with a bounded number of extra attempts.
   - If some samples fail, continue when possible and log structured warnings.

## Verification

- Added unit coverage:
  - `tests/unit/agents/test_pydantic_ai_extractors.py::test_extract_quantitative_python_literal_valid`
  - `tests/unit/agents/test_quantitative.py::TestQuantitativeAssessmentAgent::test_assess_with_consistency_recovers_from_sample_failure`

## Notes

- This is a **parsing robustness** issue, not an orchestration/agent-graph bug.
- Increasing `PYDANTIC_AI_RETRIES` can still help for other transient failures, but is no longer the primary mitigation for python-literal outputs.
