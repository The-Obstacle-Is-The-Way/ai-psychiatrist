# BUG-039: Exception Handlers Mask Original Error Types

| Field | Value |
|-------|-------|
| **Status** | DEFERRED |
| **Severity** | MEDIUM |
| **Affects** | All agents, debugging |
| **Introduced** | Original design |
| **Discovered** | 2025-12-30 |
| **Priority** | Low (does not corrupt results, affects debugging only) |

## Summary

All four agent classes catch `Exception` and convert it to `ValueError`, losing the original exception type. This makes it impossible to:
1. Distinguish timeout errors from validation errors
2. Implement targeted retry logic
3. Debug the actual root cause of failures

---

## Affected Code

### Pattern (All 4 Agents)

**QuantitativeAssessmentAgent** (`agents/quantitative.py:297-300`)
```python
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error("Pydantic AI call failed during scoring", error=str(e))
    raise ValueError(f"Pydantic AI scoring failed: {e}") from e
```

**QualitativeAssessmentAgent** (`agents/qualitative.py:146-149`, `205-208`)
```python
except asyncio.CancelledError:
    raise
except Exception as e:
    logger.error("Pydantic AI call failed", error=str(e), ...)
    raise ValueError(f"Pydantic AI assessment failed: {e}") from e
```

**JudgeAgent** (`agents/judge.py:165-168`)
```python
except Exception as e:
    logger.error("Pydantic AI call failed", error=str(e), ...)
    raise ValueError(f"Pydantic AI evaluation failed: {e}") from e
```

**MetaReviewAgent** (`agents/meta_review.py:156-159`)
```python
except Exception as e:
    logger.error("Pydantic AI call failed", error=str(e), ...)
    raise ValueError(f"Pydantic AI meta-review failed: {e}") from e
```

---

## Problem

### 1. Exception Type Lost

```python
# Original exception
LLMTimeoutError(timeout_seconds=120)

# After catching and converting
ValueError("Pydantic AI scoring failed: LLM request timed out after 120s")
```

Callers cannot check `isinstance(e, LLMTimeoutError)` because it's now a `ValueError`.

### 2. Can't Implement Targeted Handling

```python
# This is now IMPOSSIBLE:
try:
    await agent.assess(transcript)
except LLMTimeoutError:
    # Retry with longer timeout
    pass
except EmbeddingArtifactMismatchError:
    # Skip this participant, embeddings broken
    pass
except LLMError:
    # General LLM issue
    pass
```

### 3. Debugging Is Harder

The original exception has structured data (e.g., `LLMTimeoutError.timeout_seconds`). After conversion to `ValueError`, this is only available as a substring in the message.

---

## Root Cause

The pattern was likely intended to provide a uniform error type for callers. But `ValueError` is too generic and hides information.

---

## Fix

### Option 1: Re-raise Domain Exceptions As-Is

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
    raise RuntimeError(f"Unexpected error: {e}") from e
```

### Option 2: Create Agent-Specific Exceptions

```python
# domain/exceptions.py
class AgentError(DomainError):
    """Base for agent-related errors."""

class AssessmentError(AgentError):
    """Assessment agent failed."""

class ScoringError(AgentError):
    """Scoring agent failed."""
```

Then:
```python
except LLMTimeoutError as e:
    raise ScoringError(f"Scoring timed out: {e}") from e
except LLMError as e:
    raise ScoringError(f"LLM error during scoring: {e}") from e
```

This preserves the exception chain while adding context.

---

## Related

- BUG-033: Timeouts are hard to handle because they're converted to ValueError
- BUG-037: Silent fallbacks (a different type of exception masking)

---

## Verification

After fix:
- [ ] `LLMTimeoutError` propagates as `LLMTimeoutError` (or wrapped in domain-specific error)
- [ ] `EmbeddingArtifactMismatchError` propagates with original type
- [ ] Callers can use `isinstance()` checks for targeted handling
- [ ] Exception chain (`__cause__`) is preserved for debugging
