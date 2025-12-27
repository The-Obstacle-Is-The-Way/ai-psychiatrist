# BUG-027: Pydantic AI Timeout Not Configurable

**Status**: Open
**Severity**: Medium
**Discovered**: 2025-12-27
**Component**: `src/ai_psychiatrist/agents/pydantic_agents.py`

## Summary

The Pydantic AI agent path uses a **hardcoded 600s timeout** from the `pydantic_ai` library, while the legacy fallback path uses a **configurable 300s timeout** from `OLLAMA_TIMEOUT_SECONDS`. This mismatch causes:

1. Inconsistent behavior between primary and fallback paths
2. No way to increase Pydantic AI timeout for large transcripts
3. Potential for cascading failures when both paths timeout

## Reproduction

Observed during few-shot reproduction run (participant 390):

```text
04:05:06 - Started quantitative assessment (2371 words)
04:37:13 - Pydantic AI times out: "Request timed out."
04:42:13 - Legacy fallback times out after 300s
04:42:13 - Participant marked as failed
```

## Root Cause

### Pydantic AI Path (Hardcoded Timeout)

```python
# src/ai_psychiatrist/agents/pydantic_agents.py:42-43
model = OpenAIChatModel(
    model_name, provider=OllamaProvider(base_url=_ollama_v1_base_url(base_url))
)
```

`OllamaProvider` internally uses `cached_async_http_client(timeout=600)` - this 600s value is **not configurable** through our code.

### Legacy Fallback Path (Configurable Timeout)

```python
# src/ai_psychiatrist/infrastructure/llm/ollama.py:83
self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._default_timeout))
```

This uses `OllamaSettings.timeout_seconds` (default=300, configurable via `OLLAMA_TIMEOUT_SECONDS`).

### Configuration Gap

```python
# src/ai_psychiatrist/config.py:332-355
class PydanticAISettings(BaseSettings):
    enabled: bool = True
    retries: int = 3
    # NO timeout setting!
```

## Impact

1. **Research runs**: Large transcripts may timeout on Pydantic AI, trigger fallback, then timeout again
2. **Configuration confusion**: Setting `OLLAMA_TIMEOUT_SECONDS=600` only affects the fallback path
3. **Unnecessary failures**: Participants that would succeed with longer timeout are marked as failed

## Proposed Fix

### Option A: Pass Custom HTTP Client to OllamaProvider (Recommended)

```python
# src/ai_psychiatrist/agents/pydantic_agents.py
import httpx

def create_quantitative_agent(
    *,
    model_name: str,
    base_url: str,
    retries: int,
    system_prompt: str,
    timeout_seconds: int = 300,  # Add parameter
) -> Agent[None, QuantitativeOutput]:
    # Create custom client with configurable timeout
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    model = OpenAIChatModel(
        model_name,
        provider=OllamaProvider(
            base_url=_ollama_v1_base_url(base_url),
            http_client=http_client,  # Pass custom client
        )
    )
    # ...
```

### Option B: Add Timeout to PydanticAISettings

```python
# src/ai_psychiatrist/config.py
class PydanticAISettings(BaseSettings):
    enabled: bool = True
    retries: int = 3
    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=1200,
        description="Timeout for Pydantic AI LLM calls",
    )
```

Then use this value when creating agents.

## Files to Modify

1. `src/ai_psychiatrist/config.py` - Add `timeout_seconds` to `PydanticAISettings`
2. `src/ai_psychiatrist/agents/pydantic_agents.py` - Accept and use timeout parameter
3. `src/ai_psychiatrist/agents/quantitative.py` - Pass timeout when creating agent
4. `src/ai_psychiatrist/agents/judge.py` - Pass timeout when creating agent
5. `src/ai_psychiatrist/agents/meta_review.py` - Pass timeout when creating agent

## Workaround (Until Fixed)

Increase the legacy timeout to reduce cascading failures:

```bash
export OLLAMA_TIMEOUT_SECONDS=600
```

This doesn't fix the Pydantic AI timeout (still hardcoded at 600s) but ensures the fallback has matching timeout.

## Related

- `docs/pydantic-ai-fallback-mechanism.md` - Fallback behavior documentation
- `docs/specs/21-broad-exception-handling.md` - Exception handling specification
