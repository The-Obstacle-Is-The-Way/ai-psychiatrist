# ANALYSIS-026: JSON Parsing Architecture Deep Audit

**Date**: 2026-01-03
**Status**: 🔍 INVESTIGATION (NOT A BUG FIX - ANALYSIS ONLY)
**Severity**: HIGH - Systemic issue causing recurrent failures
**Triggered By**: Run10 errors showing "Exceeded maximum retries (3) for output validation"

---

## Executive Summary

This document audits the entire JSON parsing and error handling chain in the codebase to determine if our approach is fundamentally sound or if we're missing industry-standard solutions.

**Key Finding**: The recent "fix" (BUG-025) was **not applied** to Run10 because the run started with commit `064ed30` (10:58 AM) while the fix was committed at `f67443b` (1:52 PM). The running process has OLD code.

---

## Part 1: Run10 Error Analysis

### Error Pattern Observed
```
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 34 column 178 (char 2280)
pydantic_ai.exceptions.UnexpectedModelBehavior: Exceeded maximum retries (3) for output validation
```

### Why Fix Didn't Help
| Timing | Event |
|--------|-------|
| 10:58:31 | Commit `064ed30` (docs update) |
| 13:52:20 | Commit `f67443b` (JSON parsing fix) |
| 16:20:01 | Run10 started with `064ed30` + uncommitted changes |

**Conclusion**: Run10 is using pre-fix code. The fix has NOT been tested yet.

---

## Part 2: Current Architecture

### JSON Parsing Call Chain (Post-Fix)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM Response (raw text)                         │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. _extract_answer_json(text)                                      │
│     - Regex for <answer>...</answer> tags                           │
│     - Fallback to ```json...``` fences                              │
│     - Fallback to first {...} block                                 │
│     - Returns: raw JSON string or raises ModelRetry                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. tolerant_json_fixups(json_str)                                  │
│     - Smart quotes → ASCII quotes                                   │
│     - Remove zero-width spaces                                      │
│     - Insert missing commas at newlines                             │
│     - Escape unescaped quotes                                       │
│     - Remove trailing commas                                        │
│     - Returns: fixed JSON string                                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. _parse_json_object_like(json_str)  [NEW IN FIX]                 │
│     - Try json.loads() first                                        │
│     - On failure: convert true/false/null → True/False/None         │
│     - Try ast.literal_eval() on Python-ized string                  │
│     - Returns: dict or raises JSONDecodeError                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. QuantitativeOutput.model_validate(data)                         │
│     - Pydantic schema validation                                    │
│     - On failure: raises ModelRetry                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. PydanticAI Agent                                                │
│     - max_result_retries=3 (default)                                │
│     - On 3 failures: raises UnexpectedModelBehavior                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Current Retry Configuration
- **PydanticAI default**: `max_result_retries=3`
- **Our override**: None (using default)
- **Industry recommendation**: 5-10 retries for structured output

---

## Part 3: Industry Comparison

### Libraries We're NOT Using

| Library | Purpose | Downloads | Status |
|---------|---------|-----------|--------|
| [`json-repair`](https://github.com/mangiucugna/json_repair) | Fix malformed LLM JSON | 3M+/month | ❌ Not installed |
| [`instructor`](https://python.useinstructor.com/) | Structured output with auto-retry | 3M+/month | ❌ Not installed |
| [`outlines`](https://github.com/outlines-dev/outlines) | Constrained generation | 1M+/month | ❌ Not installed |

### What `json-repair` Does That We Don't
- Handles truncated JSON (incomplete objects)
- Fixes missing closing brackets
- Handles mixed quote styles (`'` vs `"`)
- Handles unquoted keys
- Handles trailing text after JSON
- 0.55.0 is battle-tested across millions of LLM outputs

### What `instructor` Does That We Don't
- **Automatic self-correction**: When validation fails, sends error back to LLM with instruction to fix
- **Semantic validation**: Uses LLM to validate natural language criteria
- **10+ retry recommendation**: Built-in support for higher retry counts
- **Pydantic-native**: Same stack we're already using

---

## Part 4: What We're Doing vs Best Practices

### ✅ What We Do Well
1. **Type-safe output schemas** with Pydantic
2. **Extraction fallbacks** (XML tags, code fences, raw JSON)
3. **Basic JSON repair** (smart quotes, trailing commas)
4. **Consistency sampling** (multiple samples for confidence)
5. **Sample-level failure resilience** (continues collecting samples on error)

### ⚠️ Anti-Patterns / Gaps

| Gap | Our Approach | Industry Best Practice |
|-----|--------------|------------------------|
| JSON repair | Custom `tolerant_json_fixups` (~100 LOC) | Use `json-repair` library (battle-tested) |
| Retry count | 3 retries (PydanticAI default) | 5-10 retries recommended |
| Error feedback | No feedback to LLM on parse failure | `instructor` sends error back to LLM for self-correction |
| Retry visibility | Limited logging on retry | Rich telemetry with retry context |
| Constrained output | None | `outlines` guarantees valid JSON via FSM |

### Specific Code Smell: Reinventing Wheels

**File**: `src/ai_psychiatrist/infrastructure/llm/responses.py:163-230` (68 lines)
**File**: `src/ai_psychiatrist/agents/extractors.py:32-95` (64 lines)

~130 lines of custom JSON repair code that could be replaced with:
```python
import json_repair
data = json_repair.loads(raw_text)
```

---

## Part 5: Why Errors Keep Happening

### Root Causes

1. **Model behavior variance**: Gemma3:27b occasionally outputs Python-style dicts (`True` not `true`)
2. **Retry exhaustion**: 3 retries is too few for complex structured output
3. **No self-correction**: Failed parses don't inform the model what went wrong
4. **Temperature >0**: Consistency sampling uses temp=0.3, increasing output variance
5. **Output length**: PHQ-8 JSON is ~2KB with 8 nested objects - more opportunity for errors

### Specific Failure Modes Not Handled

1. **Truncated output**: Model hits token limit mid-JSON
2. **Explanation after JSON**: Model adds "I hope this helps!" after closing brace
3. **Nested quote escaping**: Deep nesting breaks quote escape logic
4. **Unicode in values**: Non-ASCII characters in evidence quotes
5. **Streaming corruption**: Partial responses during network issues

---

## Part 6: Recommendations

### Quick Wins (Low Risk)
1. **Increase `max_result_retries` to 5-10** in PydanticAI agent config
2. **Add `json-repair` to dependencies** and use as primary parser
3. **Log raw LLM output on parse failure** for debugging

### Medium-Term (Moderate Effort)
4. **Consider `instructor` library** for auto-retry with error feedback
5. **Add structured output mode** if using OpenAI-compatible API
6. **Implement output length validation** - warn if approaching token limit

### Long-Term (Architecture Change)
7. **Consider LangGraph** for complex multi-agent workflows with explicit error handling nodes
8. **Consider `outlines` for constrained generation** - guarantees valid JSON at generation time
9. **Add retry telemetry dashboard** for monitoring failure patterns

---

## Part 7: Risk Assessment

### If We Do Nothing
- **Expected failure rate**: ~2-5% of participants (observed in Run10: 1/15 = 6.7%)
- **Impact**: Dropped participants reduce coverage, bias evaluation
- **Debugging**: Each failure requires manual log analysis

### If We Adopt json-repair + instructor
- **Expected failure rate**: <0.5% (based on library benchmarks)
- **Effort**: ~1-2 hours to integrate
- **Risk**: New dependency, but both are mature and well-maintained

---

## Part 8: Immediate Action Items

### Before Next Run
- [ ] Verify Run10 completes (or cancel and restart with fix)
- [ ] Confirm `f67443b` fix is included in codebase
- [ ] Increase `max_result_retries` to at least 5

### This Sprint
- [ ] Evaluate `json-repair` library integration
- [ ] Add logging of raw LLM output on parse failure
- [ ] Create test cases for all known failure modes

### Future
- [ ] Evaluate `instructor` library for self-correcting structured output
- [ ] Consider LangGraph for workflow-level error handling
- [ ] Add retry telemetry metrics

---

## Appendix A: Code Locations

| File | Line | Function | Purpose |
|------|------|----------|---------|
| `agents/extractors.py` | 208 | `extract_quantitative` | Main extraction entry point |
| `agents/extractors.py` | 81 | `_parse_json_object_like` | JSON/Python literal parsing |
| `agents/extractors.py` | 32 | `_replace_json_literals_for_python` | true→True conversion |
| `infrastructure/llm/responses.py` | 163 | `tolerant_json_fixups` | Smart quote/comma repair |
| `agents/quantitative.py` | 349 | `_collect_consistency_samples` | Sample-level retry loop |
| `agents/quantitative.py` | 454 | `_score_items` | PydanticAI agent call |

## Appendix B: Related Bug Reports

- [BUG-025](../_archive/bugs/BUG-025_PYDANTIC_AI_JSON_PYTHON_LITERAL_FALLBACK.md) - Python literal fallback fix (archived)

## Appendix C: Sources

### LLM Structured Output Best Practices
- [Pydantic AI Output Documentation](https://ai.pydantic.dev/output/)
- [Machine Learning Mastery: Pydantic for LLM Outputs](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)
- [Instructor Library](https://python.useinstructor.com/)
- [json-repair PyPI](https://pypi.org/project/json-repair/)

### Agent Framework Comparisons
- [ZenML: Pydantic AI vs LangGraph](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)
- [LangWatch: Best AI Agent Frameworks 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
- [Langfuse: AI Agent Comparison](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)

---

*This is an analysis document. No code changes have been made.*
