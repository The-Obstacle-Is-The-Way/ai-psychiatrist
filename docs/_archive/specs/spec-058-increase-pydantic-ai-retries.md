# Spec 058: Increase PydanticAI Default Retries

**Status**: ✅ Implemented (2026-01-04)
**Canonical Docs**: `docs/configs/configuration.md`, `docs/_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md`
**Priority**: High
**Risk**: Very Low
**Effort**: Trivial

---

## Problem

Run 10 showed 2 participants (383, 427) failing after `Exceeded maximum retries (3) for output validation`.

The current default retry count of 3 is insufficient for:
1. Non-deterministic LLM output variance
2. Complex 8-item PHQ-8 JSON structures (~2KB)
3. Consistency sampling with `temperature=0.3`

## Solution

Increase the default `PYDANTIC_AI_RETRIES` from 3 to 5.

## Rationale

From [ANALYSIS-026](../../_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md):
> "3 retries is too few for complex structured output"

5 retries provides:
- 66% more attempts (5 vs 3)
- Better coverage of non-deterministic failures
- Still reasonable runtime impact (each retry adds ~10-30s)

## Implementation

### Change

```python
# src/ai_psychiatrist/config.py
class PydanticAISettings(BaseSettings):
    retries: int = Field(
        default=5,  # Changed from 3
        ge=0,
        le=10,
        description="Retry count for validation failures (0 disables retries).",
    )
```

### Tests

None required - this is a config default change. Existing tests use the configured value.

### Documentation

Update `.env.example` to document the new default.

## Acceptance Criteria

- [x] Default retries changed from 3 to 5 in `src/ai_psychiatrist/config.py`
- [x] `.env.example` updated (default documented)
- [x] `make ci` passes

---

## References

- ANALYSIS-026: JSON Parsing Architecture Audit
- Run 10 failures: PIDs 383, 427
