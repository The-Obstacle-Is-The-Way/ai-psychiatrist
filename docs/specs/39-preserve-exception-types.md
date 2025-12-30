# Spec 39: Preserve Exception Types (Stop Masking Errors)

| Field | Value |
|-------|-------|
| **Status** | READY |
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

**Problem**: `LLMTimeoutError` becomes `ValueError`. Callers cannot use `isinstance()` to handle different error types.

---

## Correct Pattern

### Option A: Re-raise Domain Exceptions Unchanged (Recommended)

```python
except asyncio.CancelledError:
    raise
except (LLMError, EmbeddingError, DomainError) as e:
    # Log and re-raise domain exceptions unchanged
    logger.error("Agent failed", error=str(e), error_type=type(e).__name__)
    raise
except Exception as e:
    # Only convert truly unexpected exceptions
    logger.error("Unexpected error in agent", error=str(e), error_type=type(e).__name__)
    raise RuntimeError(f"Unexpected error in agent: {e}") from e
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

**Location**: Lines ~297-300

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
except (LLMError, EmbeddingError, DomainError) as e:
    logger.error(
        "Scoring failed",
        error=str(e),
        error_type=type(e).__name__,
        participant_id=transcript.participant_id,
    )
    raise
except Exception as e:
    logger.error(
        "Unexpected error during scoring",
        error=str(e),
        error_type=type(e).__name__,
        participant_id=transcript.participant_id,
    )
    raise RuntimeError(f"Unexpected error during scoring: {e}") from e
```

**Add import**:
```python
from ai_psychiatrist.infrastructure.llm.exceptions import LLMError
from ai_psychiatrist.services.embedding import EmbeddingError  # if exists
```

---

### Step 2 — Update QualitativeAssessmentAgent

**File**: `src/ai_psychiatrist/agents/qualitative.py`

**Locations**: Lines ~146-149 and ~205-208

Apply same pattern as Step 1.

---

### Step 3 — Update JudgeAgent

**File**: `src/ai_psychiatrist/agents/judge.py`

**Location**: Lines ~165-168

Apply same pattern as Step 1.

---

### Step 4 — Update MetaReviewAgent

**File**: `src/ai_psychiatrist/agents/meta_review.py`

**Location**: Lines ~156-159

Apply same pattern as Step 1.

---

## Verification

After implementation:

- [ ] `LLMTimeoutError` propagates as `LLMTimeoutError` (not wrapped)
- [ ] `EmbeddingArtifactMismatchError` propagates with original type
- [ ] Truly unexpected exceptions become `RuntimeError` (not `ValueError`)
- [ ] Exception chain (`__cause__`) preserved for debugging
- [ ] All existing tests pass
- [ ] Callers can use `isinstance()` for targeted handling

---

## Tests

### Unit Test: Exception Types Preserved

```python
@pytest.mark.asyncio
async def test_timeout_error_not_masked() -> None:
    """LLMTimeoutError should propagate unchanged, not as ValueError."""
    agent = QuantitativeAssessmentAgent(...)

    # Mock LLM to raise timeout
    agent._llm_client.chat = AsyncMock(
        side_effect=LLMTimeoutError(timeout_seconds=120)
    )

    with pytest.raises(LLMTimeoutError):  # NOT ValueError
        await agent.assess(transcript)
```

---

## Related

- BUG-039: Exception Handlers Mask Original Error Types
- Spec 38: Conditional Feature Loading (related fail-fast principle)
