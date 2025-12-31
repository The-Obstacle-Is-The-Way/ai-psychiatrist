# Spec 39: Preserve Exception Types (Stop Masking Errors)

| Field | Value |
|-------|-------|
| **Status** | COMPLETE |
| **Priority** | MEDIUM |
| **Addresses** | BUG-039 (exception handlers mask error types) |
| **Effort** | ~2 hours |
| **Impact** | Proper error handling, debuggability, targeted retry logic |

---

## Problem Statement

All four agent classes catch `Exception` and convert it to `ValueError`, losing the original exception type. This makes it impossible to:

1. Distinguish timeout errors from validation errors
2. Implement targeted retry logic
3. Debug the actual root cause of failures

**This is technical debt that violates the fail-fast principle.**

---

## Current (Wrong) Pattern

```python
# In all 4 agents: quantitative.py, qualitative.py, judge.py, meta_review.py
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error("Pydantic AI call failed", error=str(e))
    raise ValueError(f"Pydantic AI scoring failed: {e}") from e
```

**Problem**: any error type (timeouts, HTTP errors, parsing errors, etc.) becomes `ValueError`. Callers cannot use `isinstance()` checks for targeted handling, and debugging loses the original exception class.

---

## Correct Pattern

### Option A: Re-raise Exceptions Unchanged (Recommended)

```python
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error("Agent failed", error=str(e), error_type=type(e).__name__)
    raise
```

### Option B: Agent-Specific Exception Wrappers

```python
# domain/exceptions.py - add new exceptions
class AgentError(DomainError):
    """Base for agent-related errors."""

class AssessmentError(AgentError):
    """Assessment failed."""

class ScoringError(AgentError):
    """Scoring failed."""

# In agent code
except LLMTimeoutError as e:
    raise ScoringError(f"Scoring timed out: {e}") from e
except LLMError as e:
    raise ScoringError(f"LLM error during scoring: {e}") from e
```

**Recommendation**: Option A is simpler and sufficient. Option B adds more structure if we need agent-specific error handling later.

---

## Implementation Plan

### Step 1 — Update QuantitativeAssessmentAgent

**File**: `src/ai_psychiatrist/agents/quantitative.py`

**Location**: `_score_items()` exception handler (`src/ai_psychiatrist/agents/quantitative.py:297-305`)

**Before**:
```python
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error("Pydantic AI call failed during scoring", error=str(e))
    raise ValueError(f"Pydantic AI scoring failed: {e}") from e
```

**After**:
```python
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error(
        "Pydantic AI call failed during scoring",
        error=str(e),
        error_type=type(e).__name__,
        prompt_chars=len(prompt),
        temperature=temperature,
    )
    raise
```

No new imports required.

---

### Step 2 — Update QualitativeAssessmentAgent

**File**: `src/ai_psychiatrist/agents/qualitative.py`

**Locations**:
- `assess()` exception handler (`src/ai_psychiatrist/agents/qualitative.py:146-154`)
- `refine()` exception handler (`src/ai_psychiatrist/agents/qualitative.py:205-213`)

Replace `raise ValueError(...) from e` with `raise` (after logging), and include `error_type=type(e).__name__` in the log fields.

---

### Step 3 — Update JudgeAgent

**File**: `src/ai_psychiatrist/agents/judge.py`

**Location**: `evaluate_metric()` exception handler (`src/ai_psychiatrist/agents/judge.py:165-174`)

Replace `raise ValueError(...) from e` with `raise` (after logging), and include `error_type=type(e).__name__` in the log fields.

---

### Step 4 — Update MetaReviewAgent

**File**: `src/ai_psychiatrist/agents/meta_review.py`

**Location**: `review()` exception handler (`src/ai_psychiatrist/agents/meta_review.py:156-165`)

Replace `raise ValueError(...) from e` with `raise` (after logging), and include `error_type=type(e).__name__` in the log fields.

---

## Verification

After implementation:

- [ ] Exceptions raised by Pydantic AI calls propagate with original type (not converted to `ValueError`)
- [ ] `asyncio.CancelledError` still propagates (no swallowing)
- [ ] Logs include `error_type` for rapid diagnosis
- [ ] All existing tests pass
- [ ] Callers can use `isinstance()` for targeted handling

---

## Tests

### Unit Tests: Exception Types Preserved

Add one test per agent verifying that an exception raised by the underlying Pydantic AI agent is **not** converted to `ValueError`.

Use the existing mocking pattern in each agent test module (they already patch `ai_psychiatrist.agents.pydantic_agents.create_*_agent`).

**Examples (copy/paste patterns; adapt per agent):**

- `tests/unit/agents/test_quantitative.py`: patch `create_quantitative_agent` so `mock_agent.run.side_effect = RuntimeError("boom")`, then assert `await agent.assess(...)` raises `RuntimeError`, not `ValueError`.
- `tests/unit/agents/test_qualitative.py`: patch `create_qualitative_agent` so `mock_agent.run.side_effect = RuntimeError("boom")`, then assert `await agent.assess(...)` raises `RuntimeError`, not `ValueError`.
- `tests/unit/agents/test_judge.py`: patch `create_judge_metric_agent` so `mock_agent.run.side_effect = RuntimeError("boom")`, then assert `await agent.evaluate_metric(...)` raises `RuntimeError`, not `ValueError`.
- `tests/unit/agents/test_meta_review.py`: patch `create_meta_review_agent` so `mock_agent.run.side_effect = RuntimeError("boom")`, then assert `await agent.review(...)` raises `RuntimeError`, not `ValueError`.

```python
@pytest.mark.asyncio
async def test_agent_run_error_not_masked(...) -> None:
    """Exceptions from the Pydantic AI agent should not be converted to ValueError."""
    ...
```

---

## Related

- BUG-039: Exception Handlers Mask Original Error Types
- Spec 38: Conditional Feature Loading (related fail-fast principle)
