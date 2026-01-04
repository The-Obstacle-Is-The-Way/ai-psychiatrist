"""Unit tests for tolerant_json_fixups() and parse_llm_json() — BUG-043 regression prevention."""

from __future__ import annotations

import json

import pytest

from ai_psychiatrist.infrastructure.llm.responses import parse_llm_json, tolerant_json_fixups

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

    def test_stray_string_fragments_after_string_value_joined(self) -> None:
        """Stray comma-delimited string fragments must be joined into one string value."""
        broken = '{\n  "evidence": "first", "second",\n  "reason": "ok",\n  "score": 0\n}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"evidence": "first\nsecond", "reason": "ok", "score": 0}

    # --- Control characters inside string values (Run 10 failure mode) ---

    def test_raw_newline_in_string_value_escaped(self) -> None:
        """Raw newline inside string value causes 'Invalid control character' - must be escaped."""
        # This simulates the Run 10 failure: LLM output with raw newline in string
        broken = '{"reason": "line 1\nline 2", "score": 1}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"reason": "line 1\nline 2", "score": 1}

    def test_raw_tab_in_string_value_escaped(self) -> None:
        """Raw tab inside string value must be escaped."""
        broken = '{"reason": "before\tafter", "score": 1}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"reason": "before\tafter", "score": 1}

    def test_raw_carriage_return_in_string_value_escaped(self) -> None:
        """Raw carriage return inside string value must be escaped."""
        broken = '{"reason": "before\rafter", "score": 1}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"reason": "before\rafter", "score": 1}

    def test_null_byte_in_string_value_escaped(self) -> None:
        """Null byte (0x00) inside string value must be escaped to \\u0000."""
        broken = '{"reason": "has\x00null", "score": 1}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"reason": "has\x00null", "score": 1}

    def test_form_feed_in_string_value_escaped(self) -> None:
        """Form feed (0x0C) inside string value must be escaped."""
        broken = '{"reason": "has\x0cff", "score": 1}'
        fixed = tolerant_json_fixups(broken)
        parsed = json.loads(fixed)
        assert parsed == {"reason": "has\x0cff", "score": 1}

    def test_control_chars_outside_strings_preserved(self) -> None:
        """Control characters outside strings (structural whitespace) must be preserved."""
        # Newlines and tabs as JSON structural whitespace are valid
        valid = '{\n\t"a": 1,\n\t"b": 2\n}'
        fixed = tolerant_json_fixups(valid)
        assert fixed == valid  # No changes
        parsed = json.loads(fixed)
        assert parsed == {"a": 1, "b": 2}

    def test_already_escaped_control_chars_unchanged(self) -> None:
        """Already escaped control chars (\\n, \\t) must not be double-escaped."""
        valid = '{"reason": "line 1\\nline 2", "score": 1}'
        fixed = tolerant_json_fixups(valid)
        assert fixed == valid  # No changes
        parsed = json.loads(fixed)
        assert parsed == {"reason": "line 1\nline 2", "score": 1}

    def test_control_chars_idempotent(self) -> None:
        """Control char escaping must be idempotent."""
        broken = '{"reason": "has\ttab", "score": 1}'
        once = tolerant_json_fixups(broken)
        twice = tolerant_json_fixups(once)
        assert once == twice
        parsed = json.loads(twice)
        assert parsed == {"reason": "has\ttab", "score": 1}


class TestJsonRepairFallback:
    """Tests for json-repair fallback in parse_llm_json() (Spec 059)."""

    def test_json_repair_recovers_truncated_json(self) -> None:
        """json-repair should recover truncated JSON (missing closing brace)."""
        truncated = '{"score": 2, "reason": "incomplete'
        result = parse_llm_json(truncated)
        assert result["score"] == 2

    def test_json_repair_recovers_unquoted_keys(self) -> None:
        """json-repair should recover unquoted keys."""
        broken = '{score: 2, reason: "valid"}'
        result = parse_llm_json(broken)
        assert result["score"] == 2
        assert result["reason"] == "valid"

    def test_json_repair_recovers_trailing_text(self) -> None:
        """json-repair should recover JSON with trailing text."""
        broken = '{"score": 2, "reason": "ok"} I hope this helps!'
        result = parse_llm_json(broken)
        assert result == {"score": 2, "reason": "ok"}

    def test_json_repair_recovers_missing_closing_bracket(self) -> None:
        """json-repair should recover missing closing brackets."""
        broken = '{"items": [1, 2, 3, "score": 2}'
        result = parse_llm_json(broken)
        assert "items" in result or "score" in result  # At least partial recovery

    def test_json_repair_does_not_activate_for_valid_json(self) -> None:
        """json-repair should not be needed for valid JSON."""
        valid = '{"score": 2, "reason": "ok"}'
        result = parse_llm_json(valid)
        assert result == {"score": 2, "reason": "ok"}

    def test_json_repair_raises_for_completely_invalid(self) -> None:
        """parse_llm_json should still raise for completely invalid input."""
        garbage = "this is not json at all"
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json(garbage)
