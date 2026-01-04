# Spec 059: json-repair Library as Last-Resort Fallback

**Status**: Ready for Implementation
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

## Implementation

### 1. Add dependency

```toml
# pyproject.toml
[project.dependencies]
json-repair = ">=0.55.0"
```

### 2. Update parse_llm_json()

```python
# src/ai_psychiatrist/infrastructure/llm/responses.py

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

def parse_llm_json(text: str) -> dict[str, Any]:
    """Canonical JSON parser with defense-in-depth fallbacks.

    Parse order:
    1. Apply tolerant_json_fixups() for smart quotes, control chars, etc.
    2. Try json.loads()
    3. If that fails, try ast.literal_eval() with Python literal conversion
    4. If that fails AND json-repair is available, try json_repair.loads()
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

        # Step 3: Try json-repair as last resort
        if _HAS_JSON_REPAIR:
            try:
                result = json_repair.loads(fixed)
                if isinstance(result, dict):
                    logger.info(
                        "json-repair recovered malformed JSON",
                        component="json_parser",
                        text_hash=_stable_text_hash(text),
                    )
                    return result
            except Exception as repair_error:
                logger.warning(
                    "json-repair fallback also failed",
                    component="json_parser",
                    repair_error=str(repair_error),
                )

        # Step 4: Give up
        raise json_error
```

### Design Decisions

1. **Fallback, not replacement**: Our `tolerant_json_fixups()` runs first because:
   - It's more predictable (we know exactly what it does)
   - It handles control characters (Run 10 fix)
   - json-repair only activates on failure

2. **Optional import**: json-repair is optional. If not installed:
   - Parse behavior is unchanged
   - No import error
   - Tests still pass

3. **Logging**: We log when json-repair succeeds to track how often it's needed.

## Tests

```python
# tests/unit/infrastructure/llm/test_json_repair_fallback.py

def test_json_repair_recovers_truncated_json():
    """json-repair should recover truncated JSON."""
    truncated = '{"score": 2, "reason": "incomplete'
    result = parse_llm_json(truncated)
    assert result["score"] == 2

def test_json_repair_recovers_unquoted_keys():
    """json-repair should recover unquoted keys."""
    broken = '{score: 2, reason: "valid"}'
    result = parse_llm_json(broken)
    assert result["score"] == 2

def test_json_repair_recovers_trailing_text():
    """json-repair should recover JSON with trailing text."""
    broken = '{"score": 2} I hope this helps!'
    result = parse_llm_json(broken)
    assert result["score"] == 2
```

## Acceptance Criteria

- [ ] `json-repair>=0.55.0` added to dependencies
- [ ] `parse_llm_json()` updated with json-repair fallback
- [ ] Logging added for json-repair recovery events
- [ ] Tests for fallback scenarios
- [ ] All existing tests pass

---

## References

- [json-repair on PyPI](https://pypi.org/project/json-repair/)
- [GitHub: josdejong/jsonrepair](https://github.com/josdejong/jsonrepair)
- [Tutorial on json_repair for LLM output](https://medium.com/@yanxingyang/tutorial-on-using-json-repair-in-python-easily-fix-invalid-json-returned-by-llm-8e43e6c01fa0)
- ANALYSIS-026: JSON Parsing Architecture Audit
