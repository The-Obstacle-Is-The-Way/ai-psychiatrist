# BUG-043: Pydantic AI Structured Output Bypasses Tolerant JSON Fixups

**Status**: OPEN
**Severity**: P2 (Medium - ~2.4% of participants affected)
**Discovered**: 2026-01-01 (Run 7), 2026-01-02 (Run 8)
**Related Issue**: [GitHub #84](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/84)

---

## Summary

The `tolerant_json_fixups()` function that repairs malformed LLM JSON (missing commas, smart quotes, etc.) is **correctly implemented** but **not applied** to the Pydantic AI structured output path used for PHQ-8 scoring.

---

## Affected Runs

| Run | Participant | Mode | Error |
|-----|-------------|------|-------|
| Run 7 | 339 | zero-shot | `JSONDecodeError: Expecting ',' delimiter` |
| Run 8 | 383 | few-shot | `JSONDecodeError: Expecting ',' delimiter` |

---

## Root Cause Analysis

### The Fix That Already Exists

`tolerant_json_fixups()` in `src/ai_psychiatrist/infrastructure/llm/responses.py:30-91` handles missing commas:

```python
# Lines 20-22: Regex patterns
_MISSING_COMMA_AFTER_PRIMITIVE_RE = re.compile(r'("|\d|true|false|null)\s*\n\s*"([^"]+)"\s*:')
_MISSING_COMMA_AFTER_CONTAINER_RE = re.compile(r'([}\]])\s*\n\s*"([^"]+)"\s*:')

# Lines 65-72: Fix applied
missing_commas_fixed = _MISSING_COMMA_AFTER_PRIMITIVE_RE.sub(r'\1,\n"\2":', fixed)
missing_commas_fixed = _MISSING_COMMA_AFTER_CONTAINER_RE.sub(r'\1,\n"\2":', missing_commas_fixed)
```

### Where The Fix IS Applied

**Evidence extraction** (`quantitative.py:354-358`) calls `tolerant_json_fixups()`:

```python
# quantitative.py:354-358
clean = self._strip_json_block(raw)
clean = tolerant_json_fixups(clean)  # ✅ Fix applied
obj = json.loads(clean)
```

### Where The Fix IS NOT Applied

**PHQ-8 scoring** (`quantitative.py:294-303`) uses Pydantic AI structured output:

```python
# quantitative.py:294-303
result = await self._scoring_agent.run(
    prompt,
    model_settings={...},
)
return self._from_quantitative_output(result.output)  # ❌ Pydantic AI parses internally
```

Pydantic AI's internal JSON parsing:
1. Receives raw LLM response
2. Attempts to parse as JSON → **FAILS** (missing comma)
3. Sends retry prompt to LLM: "Invalid JSON, please fix"
4. LLM at temp=0 produces **identical malformed output**
5. After 3 retries, raises `UnexpectedModelBehavior`

**The failure loop:**
```
LLM → malformed JSON → Pydantic AI parse fails → retry → LLM (temp=0) → same malformed JSON → ...
```

---

## Why Pydantic AI Retries Don't Help

At `temperature=0.0`, the LLM is **deterministic**. Asking it to "fix the JSON" produces the exact same output because:

1. The same prompt + transcript → same activations
2. temp=0 → greedy decoding (most likely token always selected)
3. No randomness to escape the malformed output

---

## Proposed Fixes (Ranked by Effort/Risk)

### Option A: Custom Result Validator (Recommended)

Inject `tolerant_json_fixups()` into Pydantic AI's result validation pipeline:

```python
from pydantic_ai import Agent
from pydantic_ai.result import ResultData

class TolerantQuantitativeAgent(Agent):
    def _validate_result(self, raw_text: str) -> ResultData:
        # Apply tolerant fixups before standard validation
        fixed_text = tolerant_json_fixups(raw_text)
        return super()._validate_result(fixed_text)
```

**Effort**: Medium
**Risk**: Low (additive, doesn't change core logic)

### Option B: Pre-Processing Wrapper

Wrap the LLM client to apply fixups to all responses:

```python
class TolerantLLMWrapper:
    def __init__(self, wrapped_client):
        self._client = wrapped_client

    async def chat(self, *args, **kwargs) -> str:
        response = await self._client.chat(*args, **kwargs)
        return tolerant_json_fixups(response)
```

**Effort**: Low
**Risk**: Medium (affects all LLM calls, may have unintended side effects)

### Option C: Fallback to Manual Parsing

If Pydantic AI structured output fails, fall back to manual `_score_items_fallback()`:

```python
async def _score_items(...):
    try:
        return await self._score_items_pydantic_ai(...)
    except UnexpectedModelBehavior:
        return await self._score_items_manual(...)  # Uses tolerant_json_fixups
```

**Effort**: High (duplicate scoring logic)
**Risk**: Low (graceful degradation)

### Option D: Increase Retry Temperature

Add slight temperature on retries to escape deterministic failure:

```python
model_settings = {
    "temperature": 0.0,
    "retry_temperature": 0.1,  # Hypothetical Pydantic AI feature
}
```

**Effort**: Depends on Pydantic AI support
**Risk**: Medium (introduces non-determinism)

---

## Immediate Workaround

Currently, failed participants are simply skipped:
- Run 7: 40/41 zero-shot success (PID 339 failed)
- Run 8: 40/41 few-shot success (PID 383 failed)

This is acceptable for research runs (~2.4% failure rate) but should be fixed for production.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/ai_psychiatrist/agents/quantitative.py` | Hook tolerant_json_fixups into Pydantic AI path |
| `src/ai_psychiatrist/infrastructure/llm/responses.py` | Already has fix (no change needed) |

---

## Test Cases

1. Unit test: Verify `tolerant_json_fixups()` repairs missing commas ✅ (already works)
2. Integration test: Verify Pydantic AI scoring path applies tolerant fixups ❌ (need to add)
3. E2E test: Run participant 339/383 and verify success after fix

---

## References

- [Pydantic AI Result Validation](https://ai.pydantic.dev/results/)
- GitHub Issue #84: Original bug report
- Run 7/8 logs: `data/outputs/repro_post_preprocessing_20260101_183533.log`
