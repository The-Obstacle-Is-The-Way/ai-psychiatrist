"""Tests for LLM response parsing utilities.

Tests verify robust parsing of JSON, XML, and score extraction from
various LLM output formats.

NOTE: MockLLMClient lives in tests/fixtures/ per BUG-001 (test/prod separation).
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.infrastructure.llm.responses import (
    extract_json_from_response,
    extract_score_from_text,
    extract_xml_tags,
    repair_json_with_llm,
)
from tests.fixtures.mock_llm import MockLLMClient

pytestmark = pytest.mark.unit


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_clean_json(self) -> None:
        """Should parse clean JSON."""
        raw = '{"key": "value"}'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_nested_json(self) -> None:
        """Should parse nested JSON structures."""
        raw = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json_from_response(raw)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_json_in_answer_tags(self) -> None:
        """Should extract JSON from answer tags."""
        raw = 'Some text\n<answer>{"key": "value"}</answer>\nMore text'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_json_in_answer_tags_nested(self) -> None:
        """Should extract nested JSON from answer tags."""
        raw = '<answer>\n{"outer": {"inner": 1}}\n</answer>'
        result = extract_json_from_response(raw)
        assert result == {"outer": {"inner": 1}}

    def test_markdown_code_block_json(self) -> None:
        """Should strip markdown json fences."""
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_markdown_code_block_generic(self) -> None:
        """Should strip generic markdown fences."""
        raw = '```\n{"key": "value"}\n```'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_smart_quotes_double(self) -> None:
        """Should handle smart double quotes."""
        raw = "{\u201ckey\u201d: \u201cvalue\u201d}"
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_smart_quotes_single(self) -> None:
        """Should handle smart single quotes in strings."""
        raw = "{\u201ckey\u201d: \u201cit\u2019s working\u201d}"
        result = extract_json_from_response(raw)
        assert result == {"key": "it's working"}

    def test_trailing_comma_object(self) -> None:
        """Should handle trailing comma in object."""
        raw = '{"key": "value",}'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_trailing_comma_array(self) -> None:
        """Should handle trailing comma in array."""
        raw = '{"items": [1, 2, 3,]}'
        result = extract_json_from_response(raw)
        assert result == {"items": [1, 2, 3]}

    def test_zero_width_space(self) -> None:
        """Should remove zero-width spaces."""
        raw = '{\u200b"key": "value"}'
        result = extract_json_from_response(raw)
        assert result == {"key": "value"}

    def test_json_embedded_in_prose(self) -> None:
        """Should extract JSON embedded in explanatory text."""
        raw = """Here is the assessment:
        {"score": 3, "reason": "evidence found"}
        That's the result."""
        result = extract_json_from_response(raw)
        assert result == {"score": 3, "reason": "evidence found"}

    def test_no_json_raises(self) -> None:
        """Should raise when no JSON found."""
        raw = "This is just plain text without any JSON."
        with pytest.raises(LLMResponseParseError, match="No JSON object found"):
            extract_json_from_response(raw)

    def test_invalid_json_recovered_by_json_repair(self) -> None:
        """json-repair should recover malformed JSON like unquoted values (Spec 059)."""
        raw = '{"key": value_without_quotes}'
        # json-repair recovers this as {"key": "value_without_quotes"} or similar
        result = extract_json_from_response(raw)
        assert "key" in result  # Successfully recovered

    def test_empty_object(self) -> None:
        """Should handle empty JSON object."""
        raw = "{}"
        result = extract_json_from_response(raw)
        assert result == {}

    def test_multiple_json_objects_uses_outer(self) -> None:
        """Should use outer braces when multiple objects present."""
        raw = '{"outer": {"nested": "value"}}'
        result = extract_json_from_response(raw)
        assert result == {"outer": {"nested": "value"}}


class TestExtractXmlTags:
    """Tests for XML tag extraction."""

    def test_single_tag(self) -> None:
        """Should extract single tag content."""
        raw = "<assessment>Test content</assessment>"
        result = extract_xml_tags(raw, ["assessment"])
        assert result == {"assessment": "Test content"}

    def test_multiple_tags(self) -> None:
        """Should extract multiple tags."""
        raw = "<a>First</a><b>Second</b>"
        result = extract_xml_tags(raw, ["a", "b"])
        assert result == {"a": "First", "b": "Second"}

    def test_missing_tag_returns_empty(self) -> None:
        """Should return empty string for missing tags."""
        raw = "<a>First</a>"
        result = extract_xml_tags(raw, ["a", "b"])
        assert result == {"a": "First", "b": ""}

    def test_multiline_content(self) -> None:
        """Should extract multiline content."""
        raw = """<assessment>
Line 1
Line 2
Line 3
</assessment>"""
        result = extract_xml_tags(raw, ["assessment"])
        assert "Line 1" in result["assessment"]
        assert "Line 2" in result["assessment"]
        assert "Line 3" in result["assessment"]

    def test_case_insensitive(self) -> None:
        """Should match tags case-insensitively."""
        raw = "<Assessment>Content</Assessment>"
        result = extract_xml_tags(raw, ["assessment"])
        assert result == {"assessment": "Content"}

    def test_content_with_whitespace_trimmed(self) -> None:
        """Should trim whitespace from content."""
        raw = "<tag>  \n  content  \n  </tag>"
        result = extract_xml_tags(raw, ["tag"])
        assert result == {"tag": "content"}

    def test_nested_tags(self) -> None:
        """Should extract outer tag content including inner tags."""
        raw = "<outer><inner>nested</inner></outer>"
        result = extract_xml_tags(raw, ["outer"])
        assert "<inner>nested</inner>" in result["outer"]

    def test_empty_tags_list(self) -> None:
        """Should return empty dict for empty tags list."""
        raw = "<a>content</a>"
        result = extract_xml_tags(raw, [])
        assert result == {}


class TestExtractScore:
    """Tests for score extraction from evaluation text."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Score: 4", 4),
            ("score: 3", 3),
            ("SCORE: 5", 5),
            ("Score:4", 4),  # No space
            ("Score : 2", 2),  # Space before colon
        ],
    )
    def test_score_keyword_patterns(self, text: str, expected: int) -> None:
        """Should extract score from 'Score:' patterns."""
        assert extract_score_from_text(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Rating: 5", 5),
            ("rating: 3", 3),
            ("Rating:4", 4),
        ],
    )
    def test_rating_keyword_patterns(self, text: str, expected: int) -> None:
        """Should extract score from 'Rating:' patterns."""
        assert extract_score_from_text(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("4/5", 4),
            ("3 / 5", 3),
            ("5/5", 5),
            ("4 out of 5", 4),
            ("3 out of 5 stars", 3),
        ],
    )
    def test_fraction_patterns(self, text: str, expected: int) -> None:
        """Should extract score from fraction patterns."""
        assert extract_score_from_text(text) == expected

    def test_score_at_start(self) -> None:
        """Should extract number at start of text."""
        assert extract_score_from_text("4\nThe assessment is good.") == 4

    def test_score_embedded_in_text(self) -> None:
        """Should extract score from explanatory text."""
        text = "Based on the evidence, I rate this assessment a score of 4."
        assert extract_score_from_text(text) == 4

    def test_out_of_range_high_returns_none(self) -> None:
        """Should return None for score > 5."""
        assert extract_score_from_text("Score: 6") is None

    def test_out_of_range_zero_returns_none(self) -> None:
        """Should return None for score = 0."""
        assert extract_score_from_text("Score: 0") is None

    def test_no_score_returns_none(self) -> None:
        """Should return None when no score found."""
        assert extract_score_from_text("No score here") is None

    def test_multiple_numbers_uses_first_match(self) -> None:
        """Should use first matching score pattern."""
        text = "Score: 4. Earlier score was 2."
        assert extract_score_from_text(text) == 4

    def test_decimal_not_matched(self) -> None:
        """Should not match decimal numbers as scores."""
        # This tests edge case - should find valid integer score
        text = "Score: 3.5 is invalid, but the real Score: 3 is valid"
        assert extract_score_from_text(text) == 3


class TestRepairJsonWithLlm:
    """Tests for repair_json_with_llm."""

    @pytest.mark.asyncio
    async def test_repair_json_success(self) -> None:
        """Should return repaired JSON from LLM output."""
        mock = MockLLMClient(
            chat_responses=['```json\\n{"a": {"evidence": "x", "reason": "y", "score": 1}}\\n```']
        )

        result = await repair_json_with_llm(
            mock,
            broken_json='{"a": {"evidence": "x", "reason": "y", "score": 1,}}',
            expected_keys=["a"],
        )

        assert result == {"a": {"evidence": "x", "reason": "y", "score": 1}}

    @pytest.mark.asyncio
    async def test_repair_json_invalid_raises(self) -> None:
        """Should raise when LLM output is not valid JSON."""
        mock = MockLLMClient(chat_responses=["not json"])

        with pytest.raises(LLMResponseParseError):
            await repair_json_with_llm(
                mock,
                broken_json='{"a": [}',
                expected_keys=["a"],
            )
