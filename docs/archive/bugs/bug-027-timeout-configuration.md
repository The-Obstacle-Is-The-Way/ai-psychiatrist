# BUG-027: Timeout Configuration Gaps

**Status**: IMPLEMENTED
**Severity**: Medium
**Discovered**: 2025-12-27
**Implemented**: 2025-12-28
**Component**: `src/ai_psychiatrist/config.py`, all 4 agents

---

## Summary

The Pydantic AI agents don't pass a configurable timeout, causing:
1. Default 600s timeout (hardcoded in pydantic_ai library)
2. Mismatch with legacy path (historically 300s default)
3. Confusing timeout behavior during fallback

---

## Root Cause

**This is primarily a GPU/compute limitation, not a code bug.**

Large transcripts + GPU throttling = slow inference. The fix is to give the LLM as much time as it needs.

### Current State

| Path | Timeout | Source |
| ---- | ------- | ------ |
| Pydantic AI | 600s (default) / configurable | `model_settings={"timeout": ...}` |
| Legacy | configurable | `OllamaSettings.timeout_seconds` |

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
        default=None,  # None = use pydantic_ai library default (600s)
        ge=0,
        description="Timeout for Pydantic AI LLM calls. None = use library default.",
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
# Keep legacy and Pydantic AI timeouts aligned by default:
# - If only one of {OLLAMA_TIMEOUT_SECONDS, PYDANTIC_AI_TIMEOUT_SECONDS} is set, propagate to the other.
# - If both are set and differ, emit a warning (fallback path may timeout sooner).
# - If both are unset, use defaults (Pydantic AI: 600s library default; Ollama: 600s project default).
```

---

## Workaround (Until Fixed)

For long-running research runs, increase the timeout (either env var works; Settings will sync if the other is unset):

```bash
export PYDANTIC_AI_TIMEOUT_SECONDS=3600
# or:
export OLLAMA_TIMEOUT_SECONDS=3600
```

---

## Files to Modify

1. `src/ai_psychiatrist/config.py` - Add `timeout_seconds` to `PydanticAISettings`
2. `src/ai_psychiatrist/agents/quantitative.py` - Pass timeout in `model_settings`
3. `src/ai_psychiatrist/agents/qualitative.py` - Pass timeout in `model_settings`
4. `src/ai_psychiatrist/agents/judge.py` - Pass timeout in `model_settings`
5. `src/ai_psychiatrist/agents/meta_review.py` - Pass timeout in `model_settings`

---

## Implementation Notes (2025-12-28)

Steps 1 and 2 were implemented:

1. **config.py**: Added `timeout_seconds: float | None = Field(default=None, ge=0, ...)` to `PydanticAISettings`
2. **All 4 agents**: Pass timeout via `model_settings` using conditional spread syntax:
   ```python
   timeout = self._pydantic_ai.timeout_seconds
   model_settings={
       "temperature": temperature,
       **({"timeout": timeout} if timeout is not None else {}),
   }
   ```

Step 3 (sync legacy timeout) was NOT implemented because:
- Legacy timeout is already configurable via `OLLAMA_TIMEOUT_SECONDS`
- Users who need long timeouts can set both env vars
- Forcing them in sync would limit flexibility

**Usage**: Set `PYDANTIC_AI_TIMEOUT_SECONDS=3600` for 1-hour timeout, or leave unset for library default (600s).

### Follow-up (2025-12-29)

Step 3 was implemented in a flexible form:

- Defaults are now aligned (Ollama default timeout increased to 600s).
- If only one timeout env var is set, it propagates to the other at `Settings` construction time.
- If both are set and differ, a warning is emitted.

---

## Related

- `docs/bugs/fallback-architecture-audit.md` - Full architecture analysis
- `docs/specs/21-broad-exception-handling.md` - Exception handling spec
