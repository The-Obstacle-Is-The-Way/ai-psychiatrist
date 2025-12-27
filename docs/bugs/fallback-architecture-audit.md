# Fallback Architecture Audit

**Date**: 2025-12-27
**Status**: Analyzed - action items identified
**Scope**: Runtime fallbacks affecting paper reproduction correctness and reproducibility

---

## Executive Summary

Both independent investigations converged on the same findings:

1. **The fallback does NOT switch models.** Both paths call the same LLM (Gemma 3 27B) via Ollama. The difference is the Python wrapper layer and parsing/repair behavior.

2. **The fallback is backward compatibility cruft.** Legacy code existed before Pydantic AI. The fallback was kept "just in case" but is rarely helpful.

3. **For timeouts (the common failure), the fallback is USELESS.** It calls the same overloaded LLM and will also timeout, wasting time.

4. **The real research risk is unrecorded pipeline divergence.** Per-participant path differences (Pydantic AI vs legacy vs repair ladder) can cause run-to-run drift.

5. **Timeouts ARE configurable.** Pydantic AI accepts `model_settings={"timeout": ...}`. We just don't pass it today.

---

## Architecture: Two Paths to Same LLM

```text
┌─────────────────────────────────────────────────────────────────┐
│   PRIMARY PATH (Pydantic AI)                                    │
│   Python → pydantic_ai.Agent → OllamaProvider → /v1/chat/completions
├─────────────────────────────────────────────────────────────────┤
│   FALLBACK PATH (Legacy)                                        │
│   Python → httpx.AsyncClient → OllamaClient → /api/chat         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Ollama Server     │
                    │   Gemma 3 27B       │  ◄── SAME MODEL
                    └─────────────────────┘
```

**Key point**: "Legacy" is not "non-LLM." It's just our older wrapper hitting a different Ollama endpoint.

---

## Inventory: All Runtime Fallbacks

### 1. Pydantic AI → Legacy (All 4 Agents)

**Pattern**: Try Pydantic AI, catch any exception, fall back to legacy.

| Agent | Location |
| ----- | -------- |
| Quantitative | `quantitative.py::_score_items()` |
| Qualitative | `qualitative.py::assess()` and `refine()` |
| Judge | `judge.py::_evaluate_metric()` |
| Meta-review | `meta_review.py::review()` |

**When helpful**: Library bugs, validation failures where legacy parsing succeeds.
**When useless**: Timeouts, connection errors (same LLM/server).

### 2. Quantitative Parsing Repair Ladder

In the legacy quantitative path:
1. Parse directly (strip tags, tolerant fixups)
2. **LLM repair prompt** (`_llm_repair`) - different prompt!
3. Fallback skeleton

**Location**: `quantitative.py::_parse_response()` + `_llm_repair()`

**Research risk**: Stage 2 uses a different prompt, potentially shifting outputs.

### 3. Meta-review Severity Fallback

If legacy response can't parse to integer 0-4, falls back to quantitative-derived severity.

**Location**: `meta_review.py::_parse_response()`

**Research risk**: Genuine semantic fallback - different severity source.

### 4. Judge Default Score on Failure

If judge LLM call fails, returns `score=3` (triggers refinement thresholds).

**Location**: `judge.py::_evaluate_metric()`

**Research risk**: Influences feedback loop behavior.

### 5. Batch Continue-on-Error

Participant evaluation failures → `success=False` → run continues.

**Location**: `reproduce_results.py::evaluate_participant()`

**Research risk**: Acceptable, but must record which participants failed.

---

## The Timeout Problem

### Current Mismatch

| Path | Default Timeout | Configurable? |
| ---- | --------------- | ------------- |
| Pydantic AI | 600s | YES via `model_settings={"timeout": ...}` |
| Legacy | 300s | YES via `OLLAMA_TIMEOUT_SECONDS` |

**Problem**: We don't pass timeout to Pydantic AI today, and the defaults differ.

### What Happened to Participant 390

```text
04:05:06 - Started (2371-word transcript)
04:37:13 - Pydantic AI timeout (~32 min with retries)
04:42:13 - Legacy fallback timeout (300s)
04:42:13 - Participant marked as failed
```

**Result**: Fallback added 5 minutes of wasted waiting. Both paths timed out because the LLM was slow (GPU throttling).

### The Fix: Long/Infinite Timeout

For GPU-limited research runs, give the LLM as much time as it needs:

```python
# Option 1: Via model_settings (preferred)
result = await agent.run(prompt, model_settings={"timeout": 3600})  # 1 hour

# Option 2: Custom httpx client
http_client = httpx.AsyncClient(timeout=None)  # Infinite
provider = OllamaProvider(base_url=..., http_client=http_client)
```

---

## What's Wrong with Current Fallback

### The Indiscriminate Exception Catch

```python
# Current (all agents)
except Exception as e:  # Catches EVERYTHING including timeouts
    logger.error("Pydantic AI call failed; falling back to legacy")
```

**Problem**: Timeouts trigger fallback, which will also timeout.

### Proposed Fix

```python
except asyncio.TimeoutError:
    raise  # Don't waste time - LLM is overloaded
except (ValidationError, ModelRetry) as e:
    logger.warning("Validation failed; trying legacy parser")
    # Fallback makes sense here
except Exception as e:
    logger.error("Pydantic AI error; falling back to legacy")
    # Library bug - fallback makes sense
```

---

## Recommendations

### A. Immediate: Unify Timeout Configuration

Add `timeout_seconds` to `PydanticAISettings` and pass to agents:

```python
class PydanticAISettings(BaseSettings):
    enabled: bool = True
    retries: int = 3
    timeout_seconds: float | None = Field(default=None)  # None = infinite
```

### B. Short-term: Don't Fallback on Timeouts

Discriminate by exception type. Timeouts should fail fast, not trigger useless fallback.

### C. Medium-term: Record Pipeline Path

Log per-participant which path was used:
- `pydantic_ai_primary`
- `legacy_primary`
- `legacy_fallback_after_pydantic_failure`
- `legacy_llm_repair_used`

### D. Long-term: Remove Legacy Fallback

If Pydantic AI proves stable, mark legacy code for deprecation:
1. Add feature flag: `PYDANTIC_AI_FALLBACK_ENABLED=false`
2. Run experiments without fallback
3. Remove legacy code paths from agents

---

## Backward Compatibility Shims (Good Ones)

These are fine and should be kept:

| Shim | Location | Purpose |
| ---- | -------- | ------- |
| Embedding metadata semantic hash | `reference_store.py` | Avoid false positives from CSV rewrites |
| API mode accepts 0/1 integers | `server.py` | Backward compat for clients |
| PHQ8Item enum values | `enums.py` | Match legacy artifact format |

---

## Summary Table

| Question | Answer |
| -------- | ------ |
| Does fallback use different model? | **NO** - same Gemma 3 27B |
| Does fallback help for timeouts? | **NO** - wastes time |
| Does fallback help for validation errors? | **MAYBE** - legacy parsing more tolerant |
| Is fallback backward compat cruft? | **YES** - legacy code predates Pydantic AI |
| Should we remove fallback? | **EVENTUALLY** - after Pydantic AI proves stable |
| Is timeout configurable? | **YES** - via `model_settings={"timeout": ...}` |

---

## Files Affected

- `src/ai_psychiatrist/agents/quantitative.py` - Main fallback + repair ladder
- `src/ai_psychiatrist/agents/qualitative.py` - Fallback in assess/refine
- `src/ai_psychiatrist/agents/judge.py` - Fallback + default score
- `src/ai_psychiatrist/agents/meta_review.py` - Fallback + severity fallback
- `src/ai_psychiatrist/agents/pydantic_agents.py` - Agent factories (need timeout)
- `src/ai_psychiatrist/config.py` - Add timeout to PydanticAISettings
