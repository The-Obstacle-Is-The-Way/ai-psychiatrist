# BUG-043: JSON Missing Comma Repair Needed

**Status**: Closed (Implemented)
**Severity**: Low (affects ~2% of participants)
**Discovered**: 2026-01-01
**Affected Component**: `src/ai_psychiatrist/agents/extractors.py`
**Spec**: `docs/_specs/spec-043-json-missing-comma-repair.md`

---

## Summary

Participant 339 consistently fails in zero-shot mode due to the LLM (Gemma 3 27B) generating malformed JSON with missing comma delimiters. The current `_tolerant_fixups()` function handles trailing commas but not missing commas.

## Error

```
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 8 column 19 (char 404)
```

Full trace:
```
pydantic_ai.exceptions.ModelRetry: Invalid JSON in <answer>: Expecting ',' delimiter: line 8 column 19 (char 404). Please ensure <answer> contains valid JSON.
...
pydantic_ai.exceptions.UnexpectedModelBehavior: Exceeded maximum retries (3) for output validation
```

## Reproduction

```bash
# Participant 339 fails consistently in zero-shot mode
uv run python scripts/reproduce_results.py --split paper-test --zero-shot-only

# Or direct test:
uv run python -c "
import asyncio
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.services.transcript import TranscriptService
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.config import get_settings

async def test():
    settings = get_settings()
    llm = OllamaClient(settings.ollama)
    agent = QuantitativeAssessmentAgent(
        llm,
        model_settings=settings.model,
        pydantic_ai_settings=settings.pydantic_ai,
        ollama_base_url=settings.ollama.base_url,
    )
    ts = TranscriptService(settings.data)
    transcript = ts.load_transcript(339)
    result = await agent.assess(transcript)  # Will fail
    print(result)

asyncio.run(test())
"
```

## Historical Pattern

| Run | Zero-shot 339 | Few-shot 339 |
|-----|--------------|--------------|
| 2025-12-28 | ✅ Success | ✅ Success |
| 2025-12-29 (early) | ✅ Success | ✅ Success |
| 2025-12-29 (late) | ❌ Fail | ✅ Success |
| 2025-12-30 | ❌ Fail | ✅ Success |
| 2026-01-01 | ❌ Fail | ✅ Success |

Few-shot mode always succeeds for 339 (with 3 items scored: Depressed=1, Sleep=1, Concentrating=1).

## Root Cause Analysis

1. **LLM generates malformed JSON**: Gemma 3 27B produces JSON with missing commas for this specific participant in zero-shot mode
2. **Deterministic failure at temp=0**: With temperature=0, the model produces the same malformed output on every retry
3. **Pydantic AI exhausts retries**: After 3 identical failures, it gives up
4. **Few-shot prompt changes model behavior**: The additional context from reference examples changes the output enough to produce valid JSON

## Current Mitigation

`_tolerant_fixups()` in `extractors.py` handles:
- Smart quotes → regular quotes ✅
- Trailing commas → removed ✅
- Missing commas → **NOT HANDLED** ❌

## Proposed Fix

Add missing comma repair to `_tolerant_fixups()`:

```python
def _tolerant_fixups(json_str: str) -> str:
    """Apply tolerant fixups to common LLM JSON mistakes."""
    # Replace smart quotes
    json_str = (
        json_str.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # Remove trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # FIX: Add missing commas between JSON object entries
    # Pattern: "value"\n"key": -> "value",\n"key":
    # This handles cases where LLM forgets comma between object fields
    json_str = re.sub(
        r'("|\d|true|false|null)\s*\n\s*"([^"]+)"\s*:',
        r'\1,\n"\2":',
        json_str,
    )

    return json_str
```

## Alternative Solutions

1. **Increase retries**: Unlikely to help at temp=0 (deterministic)
2. **Add temperature jitter on retry**: Could help break out of deterministic failure
3. **Use `json-repair` library**: More robust but adds dependency
4. **Accept ~2% failure rate**: Document as known limitation

## Impact

- **Affected**: ~1/41 participants (2.4%) in paper-test split for zero-shot mode
- **Workaround**: Few-shot mode succeeds for this participant
- **Research validity**: Minor impact; participant can be excluded from zero-shot analysis or results can note the failure

## Related

- `docs/_archive/bugs/pydantic-ai-fallback-architecture.md` - Documents the retry mechanism
- `docs/_archive/bugs/investigation-025-json-parsing-edge-cases.md` - Previous JSON parsing investigation
- `src/ai_psychiatrist/agents/extractors.py:52-65` - Current `_tolerant_fixups()`
