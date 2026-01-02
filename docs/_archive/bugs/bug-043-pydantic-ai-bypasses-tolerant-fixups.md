# BUG-043: Deterministic Quantitative Scoring Failures from Malformed JSON

**Status**: RESOLVED
**Severity**: P2 (Medium — deterministic failure for specific participants)
**Discovered**: 2026-01-01 / 2026-01-02
**Fixed**: 2026-01-02
**Related Issue**: [GitHub #84](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/84)

---

## Summary

This bug was **not** caused by Pydantic AI “bypassing” tolerant JSON repair. The Pydantic AI scoring path uses `TextOutput(extract_quantitative)`, and `extract_quantitative()` already calls `tolerant_json_fixups()` before `json.loads()`.

The actual issue was that the original fixups were **insufficient** for a real-world deterministic malformed-output variant:

1. **Unescaped quotes inside string values** (common when the model quotes transcript excerpts) → `JSONDecodeError` that often looks like “missing comma”.
2. **Stray comma-delimited string fragments** after a string value (e.g., `"evidence": "a", "b", ...`) → `JSONDecodeError: Expecting ':' delimiter`.
3. **Missing non-critical fields** like `reason` in one item object → strict validation failure and deterministic retries.

---

## Evidence (Wiring)

The scoring agent is wired as:

- `src/ai_psychiatrist/agents/pydantic_agents.py` uses `TextOutput(extract_quantitative)`
- `src/ai_psychiatrist/agents/extractors.py#extract_quantitative` does:
  - `_extract_answer_json(...)`
  - `tolerant_json_fixups(...)`
  - `json.loads(...)`
  - `QuantitativeOutput.model_validate(...)`

So tolerant fixups were already on the structured-output path.

---

## Root Cause (What Actually Failed)

For participant 383 (few-shot), the model produced valid-ish JSON *shape* but:

- emitted **unescaped `\"`** in evidence excerpts (invalid JSON)
- emitted **multiple quoted fragments** after `"evidence":` separated by commas (invalid object syntax)
- omitted `PHQ8_Depressed.reason` entirely (valid JSON after repair, but invalid schema)

Because `temperature=0.0`, Pydantic AI retries reproduced the same structural defects and failed after 3 attempts.

---

## Fix (Implemented)

1. `src/ai_psychiatrist/infrastructure/llm/responses.py`
   - `tolerant_json_fixups()` now additionally:
     - Escapes unescaped quotes inside strings (`unescaped_quotes`)
     - Joins stray comma-delimited string fragments in value position (`string_fragments`)

2. `src/ai_psychiatrist/agents/extractors.py`
   - `extract_quantitative()` now fills missing non-critical fields before Pydantic validation:
     - `reason`: `"Auto-filled: missing reason"`
     - `evidence`: `"No relevant evidence found"`
   - `score` remains critical; invalid/missing scores still trigger `ModelRetry`.

---

## Tests (Regression Coverage)

- `tests/unit/infrastructure/llm/test_tolerant_json_fixups.py`
  - Unescaped quotes repaired
  - Leading accidental quotes repaired
  - Stray string fragments joined
- `tests/unit/agents/test_pydantic_ai_extractors.py`
  - Missing `reason` is filled and validation succeeds

---

## Verification

- `make ci` passes (ruff format/check, mypy, full pytest suite).
- Manual reproduction: participant 383 (few-shot) no longer fails in the Pydantic AI path after repair.
