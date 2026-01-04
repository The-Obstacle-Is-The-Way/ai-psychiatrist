# Spec 054: Strict Evidence Schema Validation

**Status**: Implemented (PR #92, 2026-01-03)
**Priority**: High
**Complexity**: Low
**Related**: PIPELINE-BRITTLENESS.md, ANALYSIS-026

---

## SSOT (Implemented)

- Code: `src/ai_psychiatrist/services/evidence_validation.py` (`validate_evidence_schema()`, `EvidenceSchemaError`)
- Wire-up: `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._extract_evidence()`)
- Tests: `tests/unit/services/test_evidence_validation.py`, `tests/unit/agents/test_quantitative.py`

## Problem Statement

When the LLM returns malformed evidence JSON, non-list values are silently coerced to empty arrays:

```python
# Current behavior in QuantitativeAssessmentAgent._extract_evidence()
arr = obj.get(key, []) if isinstance(obj, dict) else []
if not isinstance(arr, list):
    arr = []  # silently coerced
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

## Previous Behavior (Fixed)

```python
# src/ai_psychiatrist/agents/quantitative.py - _extract_evidence()
obj = parse_llm_json(clean)
evidence_dict: dict[str, list[str]] = {}
for key in PHQ8_DOMAIN_KEYS:
    arr = obj.get(key, []) if isinstance(obj, dict) else []
    if not isinstance(arr, list):
        arr = []  # silent coercion (bug)
    evidence_dict[key] = list({str(q).strip() for q in arr if str(q).strip()})
```

Problems:
1. Non-list values (e.g., string/object/number) are silently treated as `[]`.
2. If the model returns a valid JSON object but wrong types, the run “succeeds” with corrupted evidence.
3. In few-shot mode, this can silently reduce retrieval quality and distort confidence signals.

---

## Implemented Solution

Add explicit type validation immediately after JSON parsing, before any processing.

---

## Implementation

### Schema Validation Function

```python
# New (shared with Spec 053): src/ai_psychiatrist/services/evidence_validation.py

from typing import Any
from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EvidenceSchemaError(ValueError):
    """Raised when evidence JSON does not match expected schema."""

    def __init__(self, message: str, violations: dict[str, str]):
        super().__init__(message)
        self.violations = violations


def validate_evidence_schema(obj: object) -> dict[str, list[str]]:
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
        EvidenceSchemaError: If the top-level is not an object, or any value is not a list[str].
    """
    if not isinstance(obj, dict):
        raise EvidenceSchemaError(
            f"Expected JSON object at top level, got {type(obj).__name__}",
            violations={"__root__": f"Expected object, got {type(obj).__name__}"},
        )

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

        # Case 3: List must contain only strings
        normalized: list[str] = []
        for i, item in enumerate(value):
            if not isinstance(item, str):
                violations[key] = (
                    f"Expected list[str] but element {i} was {type(item).__name__}: "
                    f"{str(item)[:100]}"
                )
                break
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)

        if key in violations:
            continue

        # Preserve order while de-duping.
        seen: set[str] = set()
        deduped: list[str] = []
        for quote in normalized:
            if quote in seen:
                continue
            seen.add(quote)
            deduped.append(quote)

        validated[key] = deduped

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
# src/ai_psychiatrist/agents/quantitative.py - modify _extract_evidence()

from ai_psychiatrist.services.evidence_validation import validate_evidence_schema, EvidenceSchemaError

async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
    """Extract evidence quotes for each PHQ-8 domain."""
    # ... existing LLM call ...

    obj = parse_llm_json(clean)

    # NEW: Strict schema validation
    try:
        evidence = validate_evidence_schema(obj)
    except EvidenceSchemaError as e:
        import hashlib  # stdlib; used for privacy-safe hashing (no transcript text)

        logger.error(
            "evidence_schema_validation_failed",
            violations=e.violations,
            response_hash=hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12],
            response_len=len(clean),
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
# tests/unit/services/test_evidence_validation.py

import pytest
from ai_psychiatrist.services.evidence_validation import validate_evidence_schema, EvidenceSchemaError


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


def test_non_string_list_items_raises():
    obj = {"PHQ8_Concentrating": ["valid", 123, True]}
    with pytest.raises(EvidenceSchemaError) as exc_info:
        validate_evidence_schema(obj)
    assert "PHQ8_Concentrating" in exc_info.value.violations


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
        Error logged with violations + hashes (no transcript text)
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
2. All schema violations logged with *privacy-safe* context (violations + hashes only)
3. Test coverage for all edge cases
4. No performance regression (<1ms overhead)

---

## Relationship to Other Specs

- **Spec 053 (Hallucination Detection)**: Runs AFTER this spec validates schema
- **ANALYSIS-026**: This extends the "no silent fallbacks" principle to schema validation
