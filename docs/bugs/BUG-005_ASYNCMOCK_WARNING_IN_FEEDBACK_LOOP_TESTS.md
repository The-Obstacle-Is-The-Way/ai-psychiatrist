# BUG-005: AsyncMock Warning in Feedback Loop Tests

**Severity**: LOW (P3)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/07.5_INTEGRATION_CHECKPOINT_QUALITATIVE.md`, `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

`tests/unit/services/test_feedback_loop.py` used an `AsyncMock` for `JudgeAgent`, but `get_feedback_for_low_scores()` is **synchronous**. The service calls it without `await`, which caused runtime warnings: **"coroutine AsyncMockMixin._execute_mock_call was never awaited"**. This is a test hygiene bug that can mask real issues and fails stricter warning policies.

---

## Evidence (Pre-Fix)

- `make check` emitted RuntimeWarning: `coroutine 'AsyncMockMixin._execute_mock_call' was never awaited`.
- `FeedbackLoopService` calls `get_feedback_for_low_scores()` synchronously.
- The test fixture used `AsyncMock()` for the entire judge agent, making the sync method awaitable.

---

## Impact

- No functional production impact, but noisy test runs and potential CI failures under strict warning-as-error policies.

---

## Scope & Disposition

- **Code Path**: Tests only (`tests/unit/...`).
- **Fix Category**: Test hygiene (non-production).
- **Recommended Action**: Resolved; no further work unless tests change.

---

## Resolution

- Replaced `get_feedback_for_low_scores` with a **synchronous** `Mock` in the judge fixture.

---

## Verification

```bash
pytest tests/unit/services/test_feedback_loop.py -v --no-cov
```

---

## Files Changed

- `tests/unit/services/test_feedback_loop.py`
