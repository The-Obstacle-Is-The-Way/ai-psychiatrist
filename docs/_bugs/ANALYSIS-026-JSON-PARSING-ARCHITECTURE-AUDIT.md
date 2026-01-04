# ANALYSIS-026: JSON Parsing Architecture Deep Audit

**Date**: 2026-01-03
**Status**: ✅ RESOLVED
**Severity**: HIGH - Systemic issue causing recurrent failures
**Triggered By**: Run10 errors showing "Exceeded maximum retries (3) for output validation"
**Resolution Date**: 2026-01-03

---

## Resolution Summary

**Root cause validated and fixed.** The investigation identified three systemic issues:

1. **JSON parsing was fragmented** across 3 call sites with inconsistent behavior
2. **`_extract_evidence()` silently dropped evidence** on parse failure (data corruption)
3. **Ollama `format:json` not used** for constrained generation

### Fixes Applied

| Issue | Fix | Files Changed |
|-------|-----|---------------|
| Fragmented parsing | Created canonical `parse_llm_json()` function | `responses.py` |
| Silent fallback | Removed silent `{}` fallback, now raises | `quantitative.py` |
| No format constraint | Added `format="json"` to evidence extraction | `quantitative.py`, `ollama.py`, `protocols.py` |
| Mock client outdated | Updated `MockLLMClient` for `format` param | `mock_llm.py` |

### Test Results
- `make ci` ✅ (ruff, mypy, pytest, coverage)
- `pytest`: 838 passed, 7 skipped (as of 2026-01-03)
- Coverage: 83.58% (≥ 80% threshold)

---

## Original Executive Summary

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

**Conclusion**: Run10 used pre-fix code. The fix is now merged and validated via `make ci`; re-run is required to confirm runtime impact on long-running tmux jobs.

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
│  2. parse_llm_json(json_str)                                        │
│     - tolerant_json_fixups(): smart quotes, missing commas, etc.     │
│     - Try json.loads()                                              │
│     - Fallback: ast.literal_eval() after literal conversion          │
│     - Returns: dict or raises JSONDecodeError (NO SILENT FALLBACKS)  │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. QuantitativeOutput.model_validate(data)                         │
│     - Pydantic schema validation                                    │
│     - On failure: raises ModelRetry                                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. PydanticAI Agent                                                │
│     - max_result_retries=3 (default)                                │
│     - On 3 failures: raises UnexpectedModelBehavior                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Current Retry Configuration
- **PydanticAI default**: `max_result_retries=3`
- **Our override**: None (using default)
- **Practical note**: If you observe repeated transient parse/validation failures, increasing retries can help, but it increases runtime. Prefer generation-time constraints (Ollama JSON mode/schema) when possible.

---

## Part 3: Industry Comparison

### Libraries We're NOT Using

| Library | Purpose | Status |
|---------|---------|--------|
| [`json-repair`](https://github.com/mangiucugna/json_repair) | Post-hoc repair of malformed JSON | ❌ Not installed |
| [`instructor`](https://python.useinstructor.com/) | Structured output with auto-retry + error feedback | ❌ Not installed |
| [`outlines`](https://github.com/outlines-dev/outlines) | Grammar / FSM constrained generation | ❌ Not installed |

### What `json-repair` Does That We Don't
- Handles truncated JSON (incomplete objects)
- Fixes missing closing brackets
- Handles mixed quote styles (`'` vs `"`)
- Handles unquoted keys
- Handles trailing text after JSON

### What `instructor` Does That We Don't
- **Automatic self-correction**: When validation fails, sends error back to LLM with instruction to fix
- **Semantic validation**: Uses LLM to validate natural language criteria
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
| Retry count | 3 retries (PydanticAI default) | Tune retries to observed failure rate; prefer generation-time constraints when available |
| Error feedback | No feedback to LLM on parse failure | `instructor` sends error back to LLM for self-correction |
| Retry visibility | Limited logging on retry | Rich telemetry with retry context |
| Constrained output | **Ollama JSON mode for evidence extraction** | Prefer generation-time constraints (JSON mode / schema) |

### Specific Code Smell: “Custom Repair” Maintenance Burden

We now intentionally centralize all tolerant parsing in:

- `src/ai_psychiatrist/infrastructure/llm/responses.py` (`tolerant_json_fixups`, `parse_llm_json`)

This reduces whack-a-mole fixes, but it does mean we own the maintenance burden of custom repair logic.

If failures persist (e.g., truncated JSON, unquoted keys), evaluate whether adopting `json-repair` as a first-pass parser would reduce maintenance risk:
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
3. **Improve failure capture without transcript leakage**: log stable hashes + lengths; optionally write raw output to a local-only quarantine file behind an explicit flag (never default).

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
- **Failure rate is workload-dependent**: observed failures were enough to drop participants and invalidate mode comparisons when silent fallbacks existed.
- **Impact**: Dropped participants reduce coverage, bias evaluation
- **Debugging**: Each failure requires manual log analysis

### If We Adopt json-repair + instructor
- **Potential**: Lower parse-failure rate via post-hoc repair + self-correction loops
- **Effort**: Moderate (new dependency + integration + test updates)
- **Risk**: New dependency, but both are mature and well-maintained

---

## Part 8: Action Items

### ✅ Completed (2026-01-03)
- [x] Created canonical `parse_llm_json()` function in `responses.py`
- [x] Removed silent fallback in `_extract_evidence()` - now raises on failure
- [x] Added `format="json"` to Ollama API for evidence extraction
- [x] Updated `ChatRequest` to support `format` parameter
- [x] Updated all extractors to use canonical parser
- [x] Fixed test that expected old silent fallback behavior
- [x] `make ci` passes (ruff, mypy, pytest, coverage)

### Post-Resolution Fixes (2026-01-04)
- [x] Added control character sanitization to `tolerant_json_fixups()` - fixes "Invalid control character" errors from Run 10 (PIDs 383, 427)
- [x] Increased `PYDANTIC_AI_RETRIES` default from 3 to 5 (Spec 058)
- [x] Added `json-repair` library as fallback in `parse_llm_json()` (Spec 059)

### Still Recommended (Future)
- [ ] Evaluate `instructor` library for self-correcting structured output
- [ ] Add retry telemetry metrics

---

## Appendix A: Code Locations (Updated)

| File | Line | Function | Purpose |
|------|------|----------|---------|
| `infrastructure/llm/responses.py` | 216 | `parse_llm_json()` | **NEW** Canonical JSON parser (SSOT) |
| `infrastructure/llm/responses.py` | 163 | `_replace_json_literals_for_python()` | **MOVED** true→True conversion |
| `infrastructure/llm/responses.py` | 275 | `tolerant_json_fixups()` | Smart quote/comma repair |
| `infrastructure/llm/protocols.py` | 67 | `ChatRequest.format` | **NEW** Format constraint field |
| `infrastructure/llm/ollama.py` | 170 | `chat()` | Uses format parameter |
| `agents/extractors.py` | 142 | `extract_quantitative()` | Uses canonical parser |
| `agents/quantitative.py` | 571 | `_extract_evidence()` | Uses format="json", raises on failure |

## Appendix B: Related Bug Reports

- [BUG-025](../_archive/bugs/BUG-025_PYDANTIC_AI_JSON_PYTHON_LITERAL_FALLBACK.md) - Python literal fallback fix (archived)

## Appendix C: Sources

### LLM Structured Output Best Practices
- [Pydantic AI Output Documentation](https://ai.pydantic.dev/output/)
- [Machine Learning Mastery: Pydantic for LLM Outputs](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)
- [Instructor Library](https://python.useinstructor.com/)
- [json-repair PyPI](https://pypi.org/project/json-repair/)
- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)

### Agent Framework Comparisons
- [ZenML: Pydantic AI vs LangGraph](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)
- [LangWatch: Best AI Agent Frameworks 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
- [Langfuse: AI Agent Comparison](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)

---

## Appendix D: Key Design Decisions

### Why NO Silent Fallbacks in Research Code

The old behavior in `_extract_evidence()` was:
```python
except (json.JSONDecodeError, ValueError):
    logger.warning("Failed to parse evidence JSON, using empty evidence")
    obj = {}  # <-- SILENT DEGRADATION
```

**This is data corruption.** When evidence parsing fails:
1. Few-shot mode silently degrades to ~zero-shot behavior
2. Retrieval quality collapses without any indication
3. Research metrics are corrupted
4. The researcher has no way to know this happened

**New behavior:** Raise on failure. The caller decides retry/fail policy.

### ⚠️ CRITICAL: Zero-Shot vs Few-Shot Mode Isolation

**Zero-shot and few-shot are INDEPENDENT RESEARCH METHODOLOGIES.** They must be completely isolated.

The silent fallback bug violated this isolation:

```
FEW-SHOT MODE (BROKEN):
  _extract_evidence() → {} (silent failure)
    ↓
  build_reference_bundle({}) → empty bundle
    ↓
  reference_text = "" (no references)
    ↓
  make_scoring_prompt(transcript, "") → SAME AS ZERO-SHOT!
    ↓
  Research results corrupted without indication
```

**Why this matters:**
- Zero-shot and few-shot are distinct experimental conditions
- If few-shot silently becomes zero-shot, comparative analysis is invalid
- Published results claiming "few-shot performance" could be partially zero-shot
- This is a fundamental methodological error

**The fix ensures:**
- `_extract_evidence()` raises on failure instead of returning `{}`
- Few-shot mode fails loudly if it can't build proper references
- Mode isolation is maintained throughout the pipeline

### Why Ollama `format:"json"` Matters

From [Ollama docs](https://docs.ollama.com/capabilities/structured-outputs):
> "JSON mode outputs are always a well-formed JSON object"

This is **grammar-level enforcement**, not post-hoc repair. The LLM's output is constrained at token generation time to only produce valid JSON.

---

*This analysis document has been resolved. Code changes were made to fix the identified issues.*
