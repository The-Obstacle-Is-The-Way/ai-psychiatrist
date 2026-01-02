"""Unit tests for tolerant_json_fixups() — BUG-043 regression prevention."""

from __future__ import annotations

import json

import pytest

from ai_psychiatrist.infrastructure.llm.responses import tolerant_json_fixups

pytestmark = pytest.mark.unit


class TestTolerantJsonFixups:
    """Tests for tolerant_json_fixups()."""

    # --- Pattern A: Missing comma after primitive ---

    def test_missing_comma_after_string_value(self) -> None:
        """BUG-043: Missing comma between string value and next key."""
        broken = '{\n  "score": "high"\n  "confidence": 3\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)  # Must not raise
        assert parsed == {"score": "high", "confidence": 3}

    def test_missing_comma_after_number_value(self) -> None:
        """Missing comma between number value and next key."""
        broken = '{\n  "score": 2\n  "confidence": 3\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"score": 2, "confidence": 3}

    def test_missing_comma_after_boolean_true(self) -> None:
        """Missing comma after true literal."""
        broken = '{\n  "enabled": true\n  "count": 5\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"enabled": True, "count": 5}

    def test_missing_comma_after_boolean_false(self) -> None:
        """Missing comma after false literal."""
        broken = '{\n  "enabled": false\n  "count": 5\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"enabled": False, "count": 5}

    def test_missing_comma_after_null(self) -> None:
        """Missing comma after null literal."""
        broken = '{\n  "value": null\n  "other": 1\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"value": None, "other": 1}

    # --- Pattern B: Missing comma after object/array close ---

    def test_missing_comma_after_nested_object(self) -> None:
        """Missing comma after nested object closes."""
        broken = '{\n  "inner": {"a": 1}\n  "outer": 2\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"inner": {"a": 1}, "outer": 2}

    def test_missing_comma_after_array(self) -> None:
        """Missing comma after array closes."""
        broken = '{\n  "items": [1, 2, 3]\n  "count": 3\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"items": [1, 2, 3], "count": 3}

    # --- Smart quotes ---

    def test_smart_quotes_replaced(self) -> None:
        """Smart quotes must be replaced with ASCII quotes."""
        broken = "{\u201cscore\u201d: 1}"
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"score": 1}

    # --- Trailing commas ---

    def test_trailing_comma_in_object_removed(self) -> None:
        """Trailing comma before } must be removed."""
        broken = '{"a": 1, "b": 2,}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"a": 1, "b": 2}

    def test_trailing_comma_in_array_removed(self) -> None:
        """Trailing comma before ] must be removed."""
        broken = "[1, 2, 3,]"
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == [1, 2, 3]

    # --- Zero-width spaces ---

    def test_zero_width_spaces_removed(self) -> None:
        """Zero-width spaces must be removed."""
        broken = '{"key\u200b": "value\u200b"}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"key": "value"}

    # --- Idempotence and no-op guarantees ---

    def test_idempotent(self) -> None:
        """Applying fixups twice must yield same result."""
        broken = '{\n  "a": 1\n  "b": 2\n}'
        once = tolerant_json_fixups(broken)
        twice = tolerant_json_fixups(once)
        assert once == twice

    def test_valid_json_unchanged(self) -> None:
        """Valid JSON must pass through unchanged."""
        valid = '{"a": 1, "b": 2}'
        fixed = tolerant_json_fixups(valid)
        assert fixed == valid

    def test_complex_valid_json_unchanged(self) -> None:
        """Complex valid JSON must pass through unchanged."""
        valid = '{\n  "items": [\n    {"id": 1},\n    {"id": 2}\n  ],\n  "count": 2\n}'
        fixed = tolerant_json_fixups(valid)
        assert fixed == valid

    # --- Multiple missing commas ---

    def test_multiple_missing_commas(self) -> None:
        """Multiple missing commas in sequence."""
        broken = '{\n  "a": 1\n  "b": 2\n  "c": 3\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"a": 1, "b": 2, "c": 3}

    # --- Unescaped quotes inside string values ---

    def test_unescaped_quote_inside_string_value_escaped(self) -> None:
        """Unescaped quotes inside string values must be escaped for valid JSON."""
        broken = '{\n  "evidence": "He said "hi" yesterday",\n  "reason": "ok",\n  "score": 0\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"evidence": 'He said "hi" yesterday', "reason": "ok", "score": 0}

    def test_unescaped_leading_quote_in_string_value_escaped(self) -> None:
        """Leading accidental double-quote in a value must not break JSON parsing."""
        broken = (
            '{\n  "evidence": ""quoted text without escapes",\n  "reason": "ok",\n  "score": 0\n}'
        )
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {
            "evidence": '"quoted text without escapes',
            "reason": "ok",
            "score": 0,
        }
