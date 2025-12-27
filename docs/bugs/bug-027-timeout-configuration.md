# BUG-027: Timeout Configuration Gaps

**Status**: Open
**Severity**: Medium
**Discovered**: 2025-12-27
**Component**: `src/ai_psychiatrist/agents/pydantic_agents.py`, `src/ai_psychiatrist/config.py`

---

## Summary

The Pydantic AI agents don't pass a configurable timeout, causing:
1. Default 600s timeout (hardcoded in pydantic_ai library)
2. Mismatch with legacy path (300s default)
3. No way to set infinite timeout for GPU-limited research runs

---

## Root Cause

**This is primarily a GPU/compute limitation, not a code bug.**

Large transcripts + GPU throttling = slow inference. The fix is to give the LLM as much time as it needs.

### Current State

| Path | Timeout | Source |
| ---- | ------- | ------ |
| Pydantic AI | 600s | `pydantic_ai.models.cached_async_http_client(timeout=600)` |
| Legacy | 300s | `OllamaSettings.timeout_seconds` |

### The Gap

Pydantic AI supports `model_settings={"timeout": ...}` but we don't pass it.

---

## Proposed Fix

### Step 1: Add Timeout to Config

```python
# src/ai_psychiatrist/config.py
class PydanticAISettings(BaseSettings):
    enabled: bool = True
    retries: int = 3
    timeout_seconds: float | None = Field(
        default=None,  # None = infinite (wait as long as needed)
        ge=0,
        description="Timeout for Pydantic AI LLM calls. None = infinite.",
    )
```

### Step 2: Pass Timeout to Agents

```python
# src/ai_psychiatrist/agents/quantitative.py (and other agents)
result = await self._scoring_agent.run(
    prompt,
    model_settings={
        "temperature": temperature,
        "timeout": self._pydantic_ai.timeout_seconds,  # Add this
    },
)
```

### Step 3: Sync Legacy Timeout

Ensure both paths use the same timeout source:

```python
# Both should read from same config or at least be synchronized
OLLAMA_TIMEOUT_SECONDS = PYDANTIC_AI_TIMEOUT_SECONDS
```

---

## Workaround (Until Fixed)

For current research runs, increase legacy timeout:

```bash
export OLLAMA_TIMEOUT_SECONDS=3600  # 1 hour
```

This doesn't fix Pydantic AI (still 600s default) but helps if fallback triggers.

---

## Files to Modify

1. `src/ai_psychiatrist/config.py` - Add `timeout_seconds` to `PydanticAISettings`
2. `src/ai_psychiatrist/agents/quantitative.py` - Pass timeout in `model_settings`
3. `src/ai_psychiatrist/agents/qualitative.py` - Pass timeout in `model_settings`
4. `src/ai_psychiatrist/agents/judge.py` - Pass timeout in `model_settings`
5. `src/ai_psychiatrist/agents/meta_review.py` - Pass timeout in `model_settings`

---

## Related

- `docs/bugs/fallback-architecture-audit.md` - Full architecture analysis
- `docs/specs/21-broad-exception-handling.md` - Exception handling spec
