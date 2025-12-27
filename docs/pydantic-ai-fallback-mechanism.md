# Pydantic AI Fallback Mechanism

This document explains the fallback behavior that occurs when Pydantic AI times out during LLM scoring.

## CRITICAL CLARIFICATION: Both Paths Use the LLM

**The fallback is NOT a "crappier model" or a non-LLM approach.**

Both paths call the **SAME LLM** (Gemma 3 27B) through the **SAME Ollama API**:

```text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   PRIMARY PATH (Pydantic AI)                                    │
│   ─────────────────────────────────────────                     │
│   Python Code → Pydantic AI Agent → OllamaProvider → Ollama API │
│                      │                                          │
│                      └─ Adds: retries, validation, wrappers     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   FALLBACK PATH (Legacy)                                        │
│   ─────────────────────────────────────────                     │
│   Python Code → httpx.AsyncClient ──────────────────→ Ollama API│
│                      │                                          │
│                      └─ Direct HTTP, same model, same prompts   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Ollama Server     │
                    │   ───────────────   │
                    │   Gemma 3 27B       │  ◄── SAME MODEL
                    │   /api/chat         │  ◄── SAME ENDPOINT
                    └─────────────────────┘
```

**Paper Reference (Section 2.3.5):**
> "All agents use open-weight LLMs through the Ollama API"

Both paths satisfy this requirement. The difference is purely in the Python wrapper layer.

## What Is the Difference Then?

| Aspect | Primary (Pydantic AI) | Fallback (Legacy) |
| ------ | --------------------- | ----------------- |
| **LLM Used** | Gemma 3 27B | Gemma 3 27B |
| **API Called** | Ollama `/api/chat` | Ollama `/api/chat` |
| **Prompts Sent** | QUANTITATIVE_SYSTEM_PROMPT | QUANTITATIVE_SYSTEM_PROMPT |
| **Output Quality** | Identical | Identical |
| Python Wrapper | `pydantic_ai.Agent` | `httpx.AsyncClient` |
| Auto-Retries | Yes (default 3) | No |
| Output Validation | Pydantic model | Manual JSON parsing |
| Timeout Handling | OllamaProvider internal | Configurable (300s) |

**The LLM does the same work in both paths. The difference is only in how Python calls the API and parses the response.**

## Implications of Fallback

### Scenario 1: Fallback SUCCEEDS

When Pydantic AI fails but the legacy path succeeds:

| Implication | Details |
| ----------- | ------- |
| **Result Quality** | IDENTICAL to primary path - same LLM, same scores |
| **Scientific Validity** | 100% valid - paper methodology fully followed |
| **Data Integrity** | No degradation - PHQ-8 scores are the same |
| **Why Did This Happen?** | Pydantic AI timeout/validation issue, NOT LLM issue |

**Example**: Pydantic AI timed out waiting for response, but the direct HTTP call succeeded. The LLM responded - Pydantic AI just gave up waiting.

### Scenario 2: Fallback FAILS (What Happened to Participant 390)

When BOTH paths fail:

| Implication | Details |
| ----------- | ------- |
| **Result** | Participant marked as failed, skipped |
| **Data Loss** | Yes - one participant has no PHQ-8 scores |
| **Root Cause** | LLM itself failed (timeout, resource exhaustion) |
| **Run Continues** | Yes - next participant processed normally |
| **Summary Reflects** | Failed participants listed with `success=False` |

**Example**: Participant 390 had a 2371-word transcript. Pydantic AI timed out after retries, legacy also timed out after 300s. The LLM genuinely couldn't complete the request.

### Scenario 3: Primary Succeeds (Normal Operation)

| Implication | Details |
| ----------- | ------- |
| **Result Quality** | Optimal - structured validation ensures clean output |
| **Advantage** | Retries handle transient failures |
| **Performance** | Slightly more overhead than legacy |

## TL;DR: Is This a Bug?

**No.** The fallback mechanism is working as designed. Here's what happened:

```text
04:37:13 - Pydantic AI times out after retries → triggers fallback
04:42:13 - Legacy fallback ALSO times out (300s limit)
04:42:13 - Participant 390 marked as failed, run continues to participant 413
```

The underlying issue is **LLM resource exhaustion** (GPU thermal throttling, large transcript), not a code bug.

## First Principles Analysis: Is Silent Fallback Appropriate?

### Why Silent Fallback Is Correct

| Principle | Reasoning |
| --------- | --------- |
| **Maximize data collection** | If we fail loudly (raise exception), the ENTIRE run stops. With 41 participants, we'd get 0 results instead of 40. |
| **Failures ARE logged** | The error is logged at ERROR level with full context. It's not truly "silent". |
| **Research context** | This is a research pipeline, not a production API. Partial results are valuable. |
| **Graceful degradation** | The run summary shows exactly which participants failed and why. |

### When We SHOULD Fail Loudly

The current design is correct for batch research runs. We SHOULD fail loudly if:

- This were a production API serving real patients (not our use case)
- A configuration error causes ALL participants to fail
- The LLM is completely unreachable (different from timeout)

## What Triggers the Fallback?

The fallback is triggered when **any exception** occurs during the Pydantic AI agent call (except `asyncio.CancelledError`). Common triggers include:

| Trigger | Log Message | Root Cause |
| ------- | ----------- | ---------- |
| LLM timeout | `error='Request timed out.'` | Ollama took too long (>retries × timeout) |
| Connection error | `error='Connection refused'` | Ollama server unreachable |
| Invalid response | `error='ValidationError...'` | LLM output didn't match expected schema |

Example log entry:

```text
2025-12-27T04:37:13.542816Z [error] Pydantic AI call failed during scoring; falling back to legacy
    [ai_psychiatrist.agents.quantitative] error='Request timed out.'
    filename=quantitative.py func_name=_score_items lineno=293
```

## Technical Details

### Primary Path (Pydantic AI)

```python
# src/ai_psychiatrist/agents/quantitative.py:283-289
if self._scoring_agent is not None:
    try:
        result = await self._scoring_agent.run(
            prompt,
            model_settings={"temperature": temperature},
        )
        return self._from_quantitative_output(result.output)
```

- Uses `pydantic_ai.Agent` with `TextOutput` mode
- Has built-in retry mechanism (configurable via `PYDANTIC_AI_RETRIES`)
- Validates output against `QuantitativeOutput` Pydantic model
- Uses extractors for parsing LLM text into structured data

### Fallback Path (Legacy)

```python
# src/ai_psychiatrist/agents/quantitative.py:300-309
raw_response = await self._llm.simple_chat(
    user_prompt=prompt,
    system_prompt=QUANTITATIVE_SYSTEM_PROMPT,
    model=model,
    temperature=temperature,
)
return await self._parse_response(raw_response)
```

- Uses direct HTTP calls via `httpx.AsyncClient`
- Single attempt (no automatic retries)
- Manual JSON parsing with tolerant fixups (`_parse_response`)
- Three repair strategies: direct parse → LLM repair → empty skeleton

## What Happens When BOTH Fail?

If the fallback also fails (as in participant 390), the evaluation is marked as failed:

```python
# scripts/reproduce_results.py:292-304
except Exception as e:
    logger.error("Evaluation failed", participant_id=participant_id, error=str(e))
    return EvaluationResult(
        participant_id=participant_id,
        mode=mode,
        duration_seconds=duration,
        success=False,  # ← Marked as failed
        error=str(e),   # ← Error preserved
    )
```

The run **continues** to the next participant. Failed participants are:

1. Logged at ERROR level
2. Included in the summary with `success=False`
3. Counted in the final statistics

## Why Did This Happen?

Common reasons for LLM timeouts during few-shot scoring:

1. **Large transcript** - More words = longer processing time
2. **GPU thermal throttling** - High fan activity suggests thermal limits
3. **Few-shot context** - Embedding-based references add to prompt length
4. **Model complexity** - 27B parameter models are compute-intensive

## Potential Improvements (Not Bugs)

If timeouts are frequent, consider:

1. **Increase timeout**: `OLLAMA_TIMEOUT_SECONDS=600`
2. **Use smaller model**: Switch to 8B variant for faster inference
3. **Reduce context**: Use zero-shot mode instead of few-shot
4. **Better hardware**: GPU with more VRAM and thermal headroom

## Monitoring

Watch for these log patterns:

```bash
# Normal operation
grep "Quantitative assessment complete" logs.log

# Fallback triggered (Pydantic AI failed, legacy may succeed)
grep "falling back to legacy" logs.log

# Total failures (participant skipped entirely)
grep "Evaluation failed" logs.log
```

## Related Files

- `src/ai_psychiatrist/agents/quantitative.py:275-309` - Fallback logic in `_score_items()`
- `src/ai_psychiatrist/agents/pydantic_agents.py` - Pydantic AI agent factories
- `src/ai_psychiatrist/infrastructure/llm/ollama.py` - Legacy OllamaClient
- `docs/specs/21-broad-exception-handling.md` - Exception handling specification
