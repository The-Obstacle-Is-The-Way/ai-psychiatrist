# Pydantic AI Fallback Mechanism

This document explains the fallback behavior in the quantitative assessment agent and critically analyzes whether it makes sense.

## Architecture Overview

The quantitative assessment agent has two paths:

```text
┌─────────────────────────────────────────────────────────────────┐
│   PRIMARY PATH (Pydantic AI)                                    │
│   Python → pydantic_ai.Agent → OllamaProvider → Ollama API      │
├─────────────────────────────────────────────────────────────────┤
│   FALLBACK PATH (Legacy)                                        │
│   Python → httpx.AsyncClient ──────────────────→ Ollama API     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Ollama Server     │
                    │   Gemma 3 27B       │  ◄── SAME MODEL
                    │   /api/chat         │  ◄── SAME ENDPOINT
                    └─────────────────────┘
```

**CRITICAL POINT**: Both paths call the **SAME LLM** (Gemma 3 27B) through the **SAME API**.

---

## First Principles: Does This Fallback Make Sense?

### Why Does the Fallback Exist?

The fallback exists because:

1. **Historical reason**: The legacy code existed before Pydantic AI was added
2. **Backward compatibility**: Don't break what was working
3. **Defensive coding**: In case the pydantic_ai library has bugs

### What Can Cause Pydantic AI to Fail?

| Failure Type | Root Cause | Frequency |
| ------------ | ---------- | --------- |
| **LLM Timeout** | Ollama too slow, GPU throttling | COMMON |
| **Connection Error** | Ollama server unreachable | RARE |
| **Validation Error** | LLM output doesn't match Pydantic model | RARE |
| **Library Bug** | Bug in pydantic_ai library | VERY RARE |

### Can the Fallback Fix These Failures?

| Failure Type | Does Fallback Help? | Why? |
| ------------ | ------------------- | ---- |
| **LLM Timeout** | **NO** | Legacy calls SAME LLM → will ALSO timeout |
| **Connection Error** | **NO** | Legacy calls SAME server → will ALSO fail |
| **Validation Error** | **MAYBE** | Legacy has tolerant JSON parsing |
| **Library Bug** | **YES** | Legacy doesn't use pydantic_ai |

### Critical Analysis

**For the MOST COMMON failure (timeout), the fallback is USELESS.**

When Pydantic AI times out, it means:
- The LLM (Gemma 3 27B) is overloaded or slow
- GPU is throttling, or transcript is too large

Falling back to legacy will just:
1. Call the SAME overloaded LLM
2. Wait another 300 seconds
3. Timeout again
4. Waste time

**The fallback ONLY helps for rare edge cases:**
- Pydantic AI validation rejects valid JSON that legacy can parse
- A bug in the pydantic_ai library causes failure

---

## Is This Bad Code?

**Partially yes, partially no.**

### What's Reasonable

- Having a fallback for library bugs makes sense
- Logging the failure and continuing to next participant is correct
- Not crashing the entire run is correct

### What's Problematic

1. **Timeout-specific fallback is wasteful**: If we know the failure is a timeout, we should NOT try the fallback - the LLM is overloaded.

2. **Timeout mismatch bug (BUG-027)**: Pydantic AI has hardcoded 600s timeout, legacy has configurable 300s. They're not synchronized.

3. **Misleading design**: The fallback looks like a recovery mechanism, but for timeouts (the common case), it just wastes time.

---

## What SHOULD Happen?

### Current Behavior (Suboptimal)

```python
try:
    result = await self._scoring_agent.run(prompt)  # Pydantic AI
    return result
except Exception as e:  # ANY exception → fallback
    logger.error("Pydantic AI failed; falling back to legacy")
    # Falls through to legacy path
```

### Better Behavior (Proposed)

```python
try:
    result = await self._scoring_agent.run(prompt)
    return result
except asyncio.TimeoutError:
    # DON'T fallback for timeout - LLM is overloaded
    raise
except ValidationError:
    # Fallback makes sense - legacy parsing might work
    logger.warning("Validation failed; trying legacy parser")
    # Falls through to legacy path
except Exception as e:
    # Library bug - fallback makes sense
    logger.error("Pydantic AI error; falling back to legacy")
    # Falls through to legacy path
```

---

## What Happened to Participant 390

### Timeline

```text
04:05:06 - Started (2371-word transcript)
04:37:13 - Pydantic AI timeout after ~32 minutes
          (retries × timeout = wasted time)
04:42:13 - Legacy fallback timeout after 300s
04:42:13 - Participant marked as failed
```

### Analysis

1. The LLM was overloaded (large transcript + GPU throttling)
2. Pydantic AI timed out (root cause: LLM slow)
3. Fallback was triggered (wrong decision for timeout)
4. Legacy also timed out (predictable - same LLM)
5. **Total wasted time: ~37 minutes instead of ~32 minutes**

The fallback added 5 minutes of wasted waiting.

---

## Actual Bugs Found

### BUG-027: Timeout Mismatch

| Path | Timeout | Configurable? |
| ---- | ------- | ------------- |
| Pydantic AI | 600s | NO (hardcoded) |
| Legacy | 300s | YES (`OLLAMA_TIMEOUT_SECONDS`) |

See `docs/bugs/bug-027-pydantic-ai-timeout-mismatch.md` for details.

### BUG-028: Indiscriminate Fallback (NEW)

The current code falls back for ANY exception, including timeouts. This wastes time for the most common failure case.

**Location**: `src/ai_psychiatrist/agents/quantitative.py:290-298`

```python
except Exception as e:  # Too broad
    logger.error("Pydantic AI call failed; falling back to legacy", error=str(e))
```

**Fix**: Discriminate by exception type - don't fallback for timeouts.

---

## Recommendations

### The Real Fix: Infinite/Long Timeout

**The root cause is GPU limitation, not a code bug.**

For research runs, we should give the LLM as much time as it needs:

```python
# In pydantic_agents.py - pass custom httpx client
import httpx

http_client = httpx.AsyncClient(timeout=None)  # Infinite timeout
# OR
http_client = httpx.AsyncClient(timeout=3600)  # 1 hour

provider = OllamaProvider(
    base_url='http://localhost:11434/v1',
    http_client=http_client,
)
```

### Short Term (Workaround)

Increase legacy timeout (doesn't fix Pydantic AI path, but helps if fallback triggers):

```bash
export OLLAMA_TIMEOUT_SECONDS=3600  # 1 hour
```

### Medium Term (Bug Fixes)

1. **Fix BUG-027**: Add `timeout_seconds` to `PydanticAISettings` and pass to OllamaProvider
2. **Fix BUG-028**: Don't fallback for timeout errors (they'll just timeout again)

### Long Term (Architecture)

Consider removing the fallback entirely if Pydantic AI proves stable. The fallback adds complexity for minimal benefit.

---

## Summary

| Question | Answer |
| -------- | ------ |
| Does fallback use a different model? | **NO** - same Gemma 3 27B |
| Does fallback help for timeouts? | **NO** - wastes time |
| Does fallback help for validation errors? | **MAYBE** - legacy parsing is more tolerant |
| Is the fallback architecture correct? | **PARTIALLY** - wrong for timeouts |
| Should we remove the fallback? | **CONSIDER IT** - adds complexity for rare benefits |

---

## Related Files

- `src/ai_psychiatrist/agents/quantitative.py:275-309` - Fallback logic
- `src/ai_psychiatrist/agents/pydantic_agents.py` - Pydantic AI agent factories
- `src/ai_psychiatrist/infrastructure/llm/ollama.py` - Legacy OllamaClient
- `docs/bugs/bug-027-pydantic-ai-timeout-mismatch.md` - Timeout configuration bug
- `docs/specs/21-broad-exception-handling.md` - Exception handling specification
