# Spec 054: Strict Evidence Schema Validation

**Status**: Ready to Implement
**Priority**: High
**Complexity**: Low
**Related**: PIPELINE-BRITTLENESS.md, ANALYSIS-026

---

## Problem Statement

When the LLM returns malformed evidence JSON, non-list values are silently coerced to empty arrays:

```python
# Current behavior in _extract_evidence()
evidence[key] = [str(q).strip() for q in obj.get(key, []) if str(q).strip()]
#                                          ^^^^^^^^^^^^^^^^
#                                          If obj[key] is a string, this iterates
#                                          over characters, then filters to empty
```

**Example of silent corruption**:
```json
{
    "PHQ8_Sleep": "Patient mentioned trouble sleeping",  // String, not array
    "PHQ8_Tired": ["I feel exhausted"]  // Correct
}
```

Result: `PHQ8_Sleep` becomes `[]` silently, losing the evidence.

---

## Current Behavior

```python
# quantitative.py - _extract_evidence()
obj = parse_llm_json(clean)
return {
    key: [str(q).strip() for q in obj.get(key, []) if str(q).strip()]
    for key in PHQ8_DOMAIN_KEYS
}
```

Problems:
1. If `obj[key]` is a string, `for q in "some string"` iterates over characters
2. Each character gets `.strip()`, likely becomes empty or single char
3. Result is `[]` or garbage like `["P", "a", "t"]`
4. No error raised, no warning logged

---

## Proposed Solution

Add explicit type validation immediately after JSON parsing, before any processing.

---

## Implementation

### Schema Validation Function

```python
# New: src/ai_psychiatrist/infrastructure/llm/schema_validation.py

from typing import Any
from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EvidenceSchemaError(ValueError):
    """Raised when evidence JSON does not match expected schema."""

    def __init__(self, message: str, violations: dict[str, str]):
        super().__init__(message)
        self.violations = violations


def validate_evidence_schema(obj: dict[str, Any]) -> dict[str, list[str]]:
    """Validate and normalize evidence extraction JSON schema.

    Expected schema:
    {
        "PHQ8_NoInterest": ["quote1", "quote2", ...],
        "PHQ8_Depressed": [...],
        ...
    }

    Args:
        obj: Parsed JSON object from LLM

    Returns:
        Validated dict with all keys present and values as list[str]

    Raises:
        EvidenceSchemaError: If any value is not a list or contains non-strings
    """
    violations: dict[str, str] = {}
    validated: dict[str, list[str]] = {}

    for key in PHQ8_DOMAIN_KEYS:
        value = obj.get(key)

        # Case 1: Key missing - acceptable, use empty list
        if value is None:
            validated[key] = []
            continue

        # Case 2: Not a list - VIOLATION
        if not isinstance(value, list):
            violations[key] = f"Expected list, got {type(value).__name__}: {str(value)[:100]}"
            continue

        # Case 3: List with non-string elements - normalize with warning
        normalized = []
        for i, item in enumerate(value):
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    normalized.append(stripped)
            elif item is not None:
                # Non-string, non-null in list - log warning but convert
                logger.warning(
                    "evidence_list_item_not_string",
                    key=key,
                    index=i,
                    item_type=type(item).__name__,
                    item_value=str(item)[:50],
                )
                converted = str(item).strip()
                if converted:
                    normalized.append(converted)

        validated[key] = normalized

    # If any violations, raise with details
    if violations:
        raise EvidenceSchemaError(
            f"Evidence schema violations in {len(violations)} fields",
            violations=violations,
        )

    return validated
```

### Integration

```python
# quantitative.py - modify _extract_evidence()

from ai_psychiatrist.infrastructure.llm.schema_validation import (
    validate_evidence_schema,
    EvidenceSchemaError,
)

async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
    """Extract evidence quotes for each PHQ-8 domain."""
    # ... existing LLM call ...

    obj = parse_llm_json(clean)

    # NEW: Strict schema validation
    try:
        evidence = validate_evidence_schema(obj)
    except EvidenceSchemaError as e:
        logger.error(
            "evidence_schema_validation_failed",
            violations=e.violations,
            raw_response_preview=clean[:500],
        )
        raise  # Propagate - fail loudly, don't silently degrade

    return evidence
```

---

## Error Handling Strategy

When schema validation fails:

| Option | Behavior | Recommendation |
|--------|----------|----------------|
| **Fail loudly** | Raise exception, participant marked as failed | ✅ Default |
| **Retry** | Trigger LLM retry with corrective prompt | Consider for Phase 2 |
| **Repair** | Attempt to fix (e.g., wrap string in list) | ❌ Too risky |

**Recommendation**: Fail loudly. This matches ANALYSIS-026 principle of no silent degradation.

---

## Testing

```python
# tests/unit/infrastructure/llm/test_schema_validation.py

import pytest
from ai_psychiatrist.infrastructure.llm.schema_validation import (
    validate_evidence_schema,
    EvidenceSchemaError,
)


def test_valid_schema_passes():
    obj = {
        "PHQ8_NoInterest": ["quote 1", "quote 2"],
        "PHQ8_Depressed": [],
        "PHQ8_Sleep": ["quote 3"],
        "PHQ8_Tired": [],
        "PHQ8_Appetite": [],
        "PHQ8_Failure": [],
        "PHQ8_Concentrating": [],
        "PHQ8_Moving": [],
    }
    result = validate_evidence_schema(obj)
    assert result["PHQ8_NoInterest"] == ["quote 1", "quote 2"]
    assert result["PHQ8_Depressed"] == []


def test_missing_keys_filled_with_empty():
    obj = {"PHQ8_NoInterest": ["quote"]}  # Only one key
    result = validate_evidence_schema(obj)
    assert result["PHQ8_NoInterest"] == ["quote"]
    assert result["PHQ8_Depressed"] == []  # Missing key → []


def test_string_instead_of_list_raises():
    obj = {
        "PHQ8_NoInterest": "This is a string, not a list",
        "PHQ8_Depressed": [],
    }
    with pytest.raises(EvidenceSchemaError) as exc_info:
        validate_evidence_schema(obj)

    assert "PHQ8_NoInterest" in exc_info.value.violations
    assert "Expected list" in exc_info.value.violations["PHQ8_NoInterest"]


def test_dict_instead_of_list_raises():
    obj = {
        "PHQ8_Sleep": {"quote": "nested object"},
    }
    with pytest.raises(EvidenceSchemaError) as exc_info:
        validate_evidence_schema(obj)

    assert "PHQ8_Sleep" in exc_info.value.violations


def test_number_instead_of_list_raises():
    obj = {"PHQ8_Tired": 42}
    with pytest.raises(EvidenceSchemaError) as exc_info:
        validate_evidence_schema(obj)

    assert "int" in exc_info.value.violations["PHQ8_Tired"]


def test_null_value_treated_as_missing():
    obj = {"PHQ8_Appetite": None}
    result = validate_evidence_schema(obj)
    assert result["PHQ8_Appetite"] == []


def test_whitespace_only_strings_filtered():
    obj = {"PHQ8_Failure": ["valid", "   ", "", "also valid"]}
    result = validate_evidence_schema(obj)
    assert result["PHQ8_Failure"] == ["valid", "also valid"]


def test_non_string_list_items_converted_with_warning(caplog):
    obj = {"PHQ8_Concentrating": ["valid", 123, True]}
    result = validate_evidence_schema(obj)
    assert result["PHQ8_Concentrating"] == ["valid", "123", "True"]
    assert "evidence_list_item_not_string" in caplog.text


def test_multiple_violations_collected():
    obj = {
        "PHQ8_NoInterest": "string 1",
        "PHQ8_Depressed": "string 2",
        "PHQ8_Sleep": [],  # Valid
    }
    with pytest.raises(EvidenceSchemaError) as exc_info:
        validate_evidence_schema(obj)

    assert len(exc_info.value.violations) == 2
    assert "PHQ8_NoInterest" in exc_info.value.violations
    assert "PHQ8_Depressed" in exc_info.value.violations
```

---

## Impact Analysis

### Before This Spec

```
LLM returns: {"PHQ8_Sleep": "trouble sleeping"}
Result: evidence["PHQ8_Sleep"] = []  # SILENT LOSS
```

### After This Spec

```
LLM returns: {"PHQ8_Sleep": "trouble sleeping"}
Result: EvidenceSchemaError raised
        Participant marked as failed
        Error logged with full context
        We KNOW something went wrong
```

---

## Migration

No migration needed. This is a strictness improvement that will cause some existing implicit failures to become explicit.

**Expected during rollout**: Some participants that previously "succeeded" with corrupted data will now fail explicitly. This is the intended behavior.

---

## Rollout Plan

1. **Phase 1**: Implement and deploy
2. **Phase 2**: Run evaluation, collect failure rate
3. **Phase 3**: If failure rate is high (>5%), investigate LLM prompt improvements
4. **Phase 4**: Consider retry logic if failures are transient

---

## Success Criteria

1. Zero silent type coercion in evidence processing
2. All schema violations logged with full context
3. Test coverage for all edge cases
4. No performance regression (<1ms overhead)

---

## Relationship to Other Specs

- **Spec 053 (Hallucination Detection)**: Runs AFTER this spec validates schema
- **ANALYSIS-026**: This extends the "no silent fallbacks" principle to schema validation
