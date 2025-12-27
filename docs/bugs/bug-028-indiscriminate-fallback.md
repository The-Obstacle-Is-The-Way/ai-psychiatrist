# BUG-028: Indiscriminate Fallback Wastes Time on Timeouts

**Status**: Open
**Severity**: Medium
**Discovered**: 2025-12-27
**Component**: ALL agent files with Pydantic AI integration

## Summary

The Pydantic AI fallback catches **all exceptions** and triggers legacy code, including for timeout errors. This is wasteful because:

1. Timeout means the LLM is overloaded
2. Legacy calls the SAME LLM
3. Legacy will ALSO timeout
4. Result: Double the wasted time

## Scope: This Affects ALL FOUR Agents

| Agent | File | Line | Same Pattern |
| ----- | ---- | ---- | ------------ |
| Quantitative | `quantitative.py` | 296 | Yes |
| Qualitative | `qualitative.py` | 157, 230 | Yes |
| Judge | `judge.py` | 168 | Yes |
| Meta Review | `meta_review.py` | 160 | Yes |

All have the same pattern:
```python
except Exception as e:  # Catches EVERYTHING
    logger.error("Pydantic AI call failed; falling back to legacy", ...)
```

## Root Cause Analysis

### Is This Just Backward Compatibility Cruft?

**YES.** The fallback exists because:

1. **Historical**: OllamaClient (`ollama.py`) was the original implementation
2. **Pydantic AI added later**: As a wrapper for structured output
3. **Defensive**: Kept legacy "just in case" Pydantic AI fails

But the defensive approach is **wrong for timeouts** - the most common failure.

### The Logic Error

```python
# Current code (quantitative.py:290-298)
except Exception as e:  # Catches EVERYTHING
    logger.error("Pydantic AI call failed; falling back to legacy", error=str(e))
    # Falls through to legacy
```

This assumes:
- If Pydantic AI fails, legacy might succeed
- **WRONG for timeouts**: Same LLM, same problem

### Timeline of Participant 390 Failure

```text
04:05:06 - Started (2371-word transcript)
04:37:13 - Pydantic AI timeout (wasted ~32 min)
04:42:13 - Legacy timeout (wasted another 5 min)
04:42:13 - Failed

Total wasted: 37 minutes
If we had failed fast: 32 minutes
Waste from fallback: 5 minutes
```

## Should We Remove the Fallback Entirely?

### Arguments FOR Removing

| Reason | Details |
| ------ | ------- |
| **Timeouts are common** | Most failures are LLM overload - fallback won't help |
| **Connection errors are fatal** | If Ollama is down, legacy won't help |
| **Validation errors are rare** | Pydantic AI extractors handle most cases |
| **Library bugs are very rare** | pydantic_ai is stable |
| **Complexity** | Two code paths to maintain |
| **Confusion** | Makes it look like recovery when it's not |

### Arguments AGAINST Removing

| Reason | Details |
| ------ | ------- |
| **Edge cases** | Some validation failures might work with legacy |
| **Library bugs** | If pydantic_ai has a bug, we have a fallback |
| **Risk aversion** | Don't break what works |

### Recommendation

**Option 1 (Conservative)**: Don't fallback for timeouts only

```python
except asyncio.TimeoutError:
    raise  # Don't waste time
except Exception as e:
    # Other errors might benefit from fallback
    logger.error("Pydantic AI error; trying legacy")
```

**Option 2 (Clean)**: Remove fallback entirely

```python
result = await self._scoring_agent.run(prompt)
return self._from_quantitative_output(result.output)
# No fallback - Pydantic AI is the only path
```

**Option 3 (Pragmatic)**: Keep but mark for deprecation

```python
# TODO: Remove legacy fallback in next major version
# See BUG-028 for analysis
except Exception as e:
    logger.warning("Legacy fallback triggered (to be removed)", error=str(e))
```

## Files Affected

1. `src/ai_psychiatrist/agents/quantitative.py:283-309` - Main fallback logic
2. `src/ai_psychiatrist/agents/judge.py:150-170` - Similar pattern
3. `src/ai_psychiatrist/agents/meta_review.py:132-160` - Similar pattern
4. `src/ai_psychiatrist/agents/qualitative.py` - Check for similar pattern

## Connection to Spec 21

The spec `docs/specs/21-broad-exception-handling.md` discusses this pattern but **doesn't question whether the fallback itself makes sense**.

From Spec 21 (line 94):
> **Decision Needed**: Is graceful degradation to legacy parsing acceptable for any exception, or should we only catch Pydantic AI / LLM specific errors?

The spec recommends keeping broad exception handling, but the **real question should be**:

> Is graceful degradation to legacy parsing **ever useful** when the root cause is LLM timeout?

**Answer: NO.** The fallback is backward compatibility cruft that should be removed or at minimum made smarter.

## Recommended Path Forward

### Phase 1: Immediate (Don't fallback for timeouts)

Change all 5 fallback sites to NOT fallback for timeout errors:

```python
except asyncio.TimeoutError:
    raise  # Don't waste time - LLM is overloaded
except Exception as e:
    logger.error("Pydantic AI error; trying legacy")
```

### Phase 2: Medium-term (Consider removing fallback entirely)

If Pydantic AI proves stable after several months:
1. Mark the legacy code path as deprecated
2. Add feature flag to disable fallback: `PYDANTIC_AI_FALLBACK_ENABLED=false`
3. Eventually remove the legacy code paths

### Phase 3: Long-term (Clean architecture)

Remove all legacy OllamaClient usage from agents. Keep OllamaClient only for:
- Scripts that don't need Pydantic AI
- Testing / debugging

## Related

- `docs/pydantic-ai-fallback-mechanism.md` - Full analysis
- `docs/bugs/bug-027-pydantic-ai-timeout-mismatch.md` - Timeout configuration bug
- `docs/specs/21-broad-exception-handling.md` - Exception handling spec (needs update)
