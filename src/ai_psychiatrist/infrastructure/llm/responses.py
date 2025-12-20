"""Utilities for parsing LLM responses.

This module provides robust parsing for LLM outputs, handling common issues
like markdown code blocks, smart quotes, and malformed JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class SimpleChatClient(Protocol):
    """Protocol for LLM clients with simple_chat method."""

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.8,
        response_format: str | None = None,
    ) -> str:
        """Send a simple chat prompt and return response."""
        ...


def extract_json_from_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM response.

    Handles common issues like markdown code blocks, smart quotes,
    and trailing commas.

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        LLMResponseParseError: If no valid JSON found.
    """
    # Try extracting from <answer> tags first
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = answer_match.group(1) if answer_match else raw

    # Strip markdown code blocks
    text = _strip_markdown_fences(text)

    # Normalize quotes and fix trailing commas
    text = _normalize_json_text(text)

    # Extract JSON object
    text = _extract_json_object(text)

    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed", error=str(e), text_preview=text[:200])
        raise LLMResponseParseError(raw, str(e)) from e


def extract_xml_tags(raw: str, tags: list[str]) -> dict[str, str]:
    """Extract content from XML-style tags.

    Args:
        raw: Raw text with XML tags.
        tags: List of tag names to extract.

    Returns:
        Dictionary mapping tag names to content.
    """
    result: dict[str, str] = {}
    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            result[tag] = match.group(1).strip()
        else:
            result[tag] = ""
    return result


def extract_score_from_text(text: str) -> int | None:
    """Extract numeric score from evaluation text.

    Args:
        text: Text containing score.

    Returns:
        Extracted score (1-5) or None if not found.
    """
    patterns = [
        r"score\s*[:\s]\s*(\d+)",  # Score: 4, score : 3, etc.
        r"score\s+of\s+(\d+)",  # score of 4
        r"rating\s*[:\s]\s*(\d+)",  # Rating: 5, rating: 3, etc.
        r"(\d+)\s*[/\s]\s*(?:out of\s*)?5",  # 4/5, 3 out of 5
        r"^(\d+)\b",  # Number at start
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

    return None


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code block fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _normalize_json_text(text: str) -> str:
    """Normalize JSON text by fixing common issues."""
    # Replace smart quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove zero-width spaces
    text = text.replace("\u200b", "")

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


def _extract_json_object(text: str) -> str:
    """Extract JSON object boundaries from text."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        raise LLMResponseParseError(text, "No JSON object found")

    return text[start : end + 1]


async def repair_json_with_llm(
    llm_client: SimpleChatClient,
    broken_json: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    """Attempt to repair malformed JSON using LLM.

    Args:
        llm_client: LLM client implementing SimpleChatClient protocol.
        broken_json: Malformed JSON string.
        expected_keys: Expected keys in output.

    Returns:
        Repaired JSON as dictionary.

    Raises:
        LLMResponseParseError: If repair fails.
    """
    value_template = '{"evidence": <string>, "reason": <string>, "score": <int 0-3 or "N/A">}'
    default_value = (
        '{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}'
    )
    repair_prompt = (
        "You will be given malformed JSON. Output ONLY a valid JSON object with these EXACT keys:\n"
        f"{', '.join(expected_keys)}\n\n"
        f"Each value must be an object: {value_template}.\n"
        f"If something is missing, fill with {default_value}.\n\n"
        "Malformed JSON:\n"
        f"{broken_json}\n\n"
        "Return only the fixed JSON. No prose, no markdown, no tags."
    )

    response = await llm_client.simple_chat(repair_prompt)
    return extract_json_from_response(response)
