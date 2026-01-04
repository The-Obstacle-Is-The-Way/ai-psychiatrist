# Spec 059: json-repair Library as Last-Resort Fallback

**Status**: ✅ Implemented (2026-01-04)
**Canonical Docs**: `docs/_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md`, `docs/pipeline-internals/evidence-extraction.md`
**Priority**: Medium
**Risk**: Low
**Effort**: Low

---

## Problem

Run 10 had structural JSON errors (`Expecting property name enclosed in double quotes`) that our `tolerant_json_fixups()` doesn't handle:
- Unquoted keys (e.g., `{foo: "bar"}`)
- Truncated JSON (incomplete objects)
- Missing closing brackets
- Trailing text after JSON
- “Python-literal + stray backslash” artifacts that break `ast.literal_eval()` (e.g., `"{'a': 1}\\ foo"` → `SyntaxError: unexpected character after line continuation character`)

## Solution

Add [`json-repair`](https://pypi.org/project/json-repair/) (v0.55.0+) as a last-resort fallback in `parse_llm_json()`.

### Why json-repair?

| Criteria | Value |
|----------|-------|
| Maturity | v0.55.0, actively maintained |
| Purpose | Specifically designed for LLM output |
| API | Drop-in replacement for `json.loads()` |
| Dependencies | Zero (pure Python) |
| Usage | Used by many LLM projects |

### What it handles that we don't

From the [documentation](https://pypi.org/project/json-repair/):
- Missing quotation marks
- Improperly formatted values (true, false, null)
- Corrupted key-value structures
- Incomplete/broken arrays/objects
- Extra non-JSON characters (comments, trailing text)

**Concrete example (matches the run log pattern)**:

```python
broken = \"{'a': 1}\\\\ foo\"
# json.loads(broken) -> JSONDecodeError: Expecting property name enclosed in double quotes
# ast.literal_eval(broken) -> SyntaxError: unexpected character after line continuation character
json_repair.loads(broken)  # -> {'a': 1}
```

## Implementation

### 1. Add dependency

```toml
# pyproject.toml
dependencies = [
  # ...
  "json-repair>=0.55.0",
]
```

### 2. Update parse_llm_json()

```python
# src/ai_psychiatrist/infrastructure/llm/responses.py

import json_repair

def parse_llm_json(text: str) -> dict[str, Any]:
    """Canonical JSON parser with defense-in-depth fallbacks.

    Parse order:
    1. Apply tolerant_json_fixups() for smart quotes, control chars, etc.
    2. Try json.loads()
    3. If that fails, try ast.literal_eval() with Python literal conversion
    4. If that fails, try json_repair.loads() as last resort (Spec 059)
    5. RAISE on failure - never silently degrade
    """
    fixed = tolerant_json_fixups(text)

    # Step 1: Try standard JSON
    try:
        result = json.loads(fixed)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("Expected JSON object", text, 0)
        return result
    except json.JSONDecodeError as json_error:
        # Step 2: Try Python literal
        pythonish = _replace_json_literals_for_python(fixed)
        try:
            result = ast.literal_eval(pythonish)
            if isinstance(result, dict):
                return result
        except (SyntaxError, ValueError):
            pass

        # Step 3: Try json-repair as last resort (Spec 059)
        result = json_repair.loads(fixed)
        if isinstance(result, dict):
            # Observability only (Spec 060): record that the json-repair path was needed.
            record_telemetry(
                TelemetryCategory.JSON_REPAIR_FALLBACK,
                text_hash=_stable_text_hash(text),
                text_length=len(text),
            )
            return result

        # Step 4: Give up
        raise json_error
```

### Design Decisions

1. **Fallback, not replacement**: Our `tolerant_json_fixups()` runs first because:
   - It's more predictable (we know exactly what it does)
   - It handles control characters (Run 10 fix)
   - json-repair only activates on failure

2. **Required dependency**: `json-repair` is installed by default (not optional) so reproduction runs cannot silently degrade into per-participant failures due to missing repair tooling.

3. **Telemetry (Spec 060)**: We record a privacy-safe telemetry event whenever the json-repair fallback is used. This avoids relying on brittle log scraping.

## Tests

Implemented as unit tests in:
- `tests/unit/infrastructure/llm/test_tolerant_json_fixups.py` (truncated JSON, unquoted keys, trailing text, missing closing bracket, etc.)
- `tests/unit/infrastructure/llm/test_responses.py` (integration coverage for canonical parser behavior)

## Acceptance Criteria

- [x] `json-repair>=0.55.0` added to dependencies
- [x] `parse_llm_json()` updated with json-repair fallback
- [x] Logging added for json-repair recovery events
- [x] Unit tests for fallback scenarios
- [x] `make ci` passes

---

## References

- [json-repair on PyPI](https://pypi.org/project/json-repair/)
- [GitHub: mangiucugna/json_repair](https://github.com/mangiucugna/json_repair/)
- [Tutorial on json_repair for LLM output](https://medium.com/@yanxingyang/tutorial-on-using-json-repair-in-python-easily-fix-invalid-json-returned-by-llm-8e43e6c01fa0)
- ANALYSIS-026: JSON Parsing Architecture Audit
