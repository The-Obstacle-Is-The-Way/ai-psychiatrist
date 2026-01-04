"""Utilities for parsing LLM responses.

This module provides robust parsing for LLM outputs, handling common issues
like markdown code blocks, smart quotes, and malformed JSON.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Protocol, runtime_checkable

import json_repair

from ai_psychiatrist.domain.exceptions import LLMResponseParseError
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


_MISSING_COMMA_AFTER_PRIMITIVE_RE = re.compile(r'("|\d|true|false|null)\s*\n\s*"([^"]+)"\s*:')
_MISSING_COMMA_AFTER_CONTAINER_RE = re.compile(r'([}\]])\s*\n\s*"([^"]+)"\s*:')
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_STRAY_STRING_FRAGMENT_RE = re.compile(
    r'(:\s*)("(?:\\.|[^"\\])*")\s*,\s*("(?:\\.|[^"\\])*")(?=\s*[,}])',
    flags=re.DOTALL,
)


def _looks_like_json_object_key(text: str, start: int) -> bool:
    """Return True if `text[start:]` begins with a JSON object key (`\"...\"\\s*:`)."""
    if start >= len(text) or text[start] != '"':
        return False

    escaped = False
    i = start + 1
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == '"':
            i += 1
            break
        i += 1
    else:
        return False

    while i < len(text) and text[i].isspace():
        i += 1

    return i < len(text) and text[i] == ":"


def _escape_unescaped_quotes_in_strings(text: str) -> str:
    """Escape unescaped quotes inside JSON strings (best-effort).

    LLMs occasionally emit unescaped double quotes inside string values
    (e.g., `"evidence": ""quoted text"` or `"He said "hi""`), which breaks JSON
    parsing with errors like `Expecting ',' delimiter`.

    This is a conservative, index-stable pass that:
    - Leaves already-escaped quotes (`\\\"`) untouched.
    - Keeps valid string terminators (followed by `,`, `}`, `]`, or `:`).
    - Treats quotes followed by a plausible next object key (`\"...\"\\s*:`) as terminators.
    - Otherwise, escapes the quote in-place.
    """
    out: list[str] = []
    in_string = False
    escaped = False

    i = 0
    while i < len(text):
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue

        # Inside a string literal
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue

        if ch != '"':
            out.append(ch)
            i += 1
            continue

        # Potential string terminator or internal quote.
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1

        if j >= len(text):
            out.append('"')
            in_string = False
            i += 1
            continue

        next_non_ws = text[j]
        if next_non_ws in {",", "}", "]", ":"} or _looks_like_json_object_key(text, j):
            out.append('"')
            in_string = False
            i += 1
            continue

        # Treat as an internal quote and escape it.
        out.append('\\"')
        i += 1

    return "".join(out)


def _join_stray_string_fragments(text: str) -> str:
    """Join stray comma-delimited string fragments into a single string value.

    Some LLMs emit multiple quoted fragments for a single string field, e.g.:
        `"evidence": "quote 1", "quote 2", "reason": ...`

    This is invalid JSON (the second fragment is interpreted as an object key, so the
    parser raises `Expecting ':' delimiter`). This pass conservatively joins adjacent
    string literals *in a value position* (immediately after `:`) when the next token
    after the second string is `,` or `}` (i.e., it's not a key).

    Joined fragments are separated by the JSON escape sequence `\\n`.
    """

    def _join(match: re.Match[str]) -> str:
        prefix = match.group(1)
        first = match.group(2)
        second = match.group(3)
        return f"{prefix}{first[:-1]}\\n{second[1:]}"

    fixed = text
    for _ in range(50):
        merged = _STRAY_STRING_FRAGMENT_RE.sub(_join, fixed)
        if merged == fixed:
            break
        fixed = merged
    return fixed


def _stable_text_hash(text: str) -> str:
    """Return a short, stable hash for logging (no raw text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _replace_json_literals_for_python(text: str) -> str:
    """Convert JSON literals (true/false/null) to Python (True/False/None) outside strings.

    This handles the common LLM failure mode where models output Python-style dicts
    instead of JSON (e.g., `True` instead of `true`).
    """
    out: list[str] = []
    in_string: str | None = None
    escaped = False

    i = 0
    while i < len(text):
        ch = text[i]

        if in_string is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in {'"', "'"}:
            in_string = ch
            out.append(ch)
            i += 1
            continue

        if ch.isalpha():
            start = i
            while i < len(text) and text[i].isalpha():
                i += 1
            token = text[start:i]
            lowered = token.lower()
            if lowered == "true":
                out.append("True")
            elif lowered == "false":
                out.append("False")
            elif lowered == "null":
                out.append("None")
            else:
                out.append(token)
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Canonical JSON parser for all LLM outputs with defense-in-depth fallbacks.

    This is the single source of truth for parsing JSON from LLM responses.
    All call sites MUST use this function instead of raw json.loads().

    Parse order:
    1. Apply tolerant_json_fixups() for smart quotes, control chars, missing commas, etc.
    2. Try json.loads()
    3. If that fails, convert Python literals and try ast.literal_eval()
    4. If that fails, try json_repair.loads() as last resort (Spec 059)
    5. RAISE on failure - never silently degrade

    Args:
        text: Raw JSON string from LLM output.

    Returns:
        Parsed dictionary.

    Raises:
        json.JSONDecodeError: If parsing fails after all repair attempts.
    """

    # Step 1: Apply tolerant fixups (smart quotes, control chars, trailing commas, etc.)
    fixed = tolerant_json_fixups(text)

    # Step 2: Try standard JSON parsing
    try:
        result = json.loads(fixed)
        if not isinstance(result, dict):
            raise json.JSONDecodeError("Expected JSON object", text, 0)
        return result
    except json.JSONDecodeError as json_error:
        # Step 3: Try Python literal parsing (handles True/False/None)
        pythonish = _replace_json_literals_for_python(fixed)
        try:
            result = ast.literal_eval(pythonish)
            if not isinstance(result, dict):
                raise json.JSONDecodeError("Expected JSON object", text, 0)

            logger.debug(
                "Parsed LLM JSON via Python literal fallback",
                component="json_parser",
                before_hash=_stable_text_hash(text),
            )
            return result
        except (SyntaxError, ValueError):
            pass

        # Step 4: Try json-repair as last resort (Spec 059)
        try:
            result = json_repair.loads(fixed)
            if not isinstance(result, dict):
                raise json.JSONDecodeError("Expected JSON object", text, 0)

            logger.info(
                "json-repair recovered malformed LLM JSON",
                component="json_parser",
                text_hash=_stable_text_hash(text),
                text_length=len(text),
            )
            return result
        except Exception as repair_error:
            # Step 5: RAISE - all fallbacks exhausted
            logger.warning(
                "Failed to parse LLM JSON after all repair attempts including json-repair",
                component="json_parser",
                json_error=str(json_error),
                repair_error=str(repair_error),
                text_length=len(text),
                text_hash=_stable_text_hash(text),
            )
            raise json_error from repair_error


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape unescaped control characters inside JSON string values.

    Control characters (0x00-0x1F except \t, \n, \r which we convert to escapes)
    cause "Invalid control character" JSON parse errors. This pass escapes them
    inside string literals only, preserving structural whitespace outside strings.
    """
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue

        # Inside a string literal
        if escaped:
            out.append(ch)
            escaped = False
            continue

        if ch == "\\":
            out.append(ch)
            escaped = True
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            continue

        # Check for control characters (0x00-0x1F)
        code = ord(ch)
        if code < 0x20:
            # Convert common whitespace to JSON escapes
            if ch == "\t":
                out.append("\\t")
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            else:
                # Other control chars: use Unicode escape
                out.append(f"\\u{code:04x}")
        else:
            out.append(ch)

    return "".join(out)


def tolerant_json_fixups(text: str) -> str:
    """Apply tolerant fixups to common LLM JSON mistakes.

    Repairs (in order):
    1. Smart quotes → ASCII quotes
    2. Zero-width spaces → removed
    3. Missing commas between object entries → inserted
    4. Unescaped quotes inside strings → escaped
    5. Control characters in strings → escaped (fixes "Invalid control character")
       NOTE: Must run AFTER unescaped quote escaping so string boundaries are accurate
    6. Stray string fragments → joined
    7. Trailing commas before } or ] → removed

    Properties:
    - Idempotent: fixups(fixups(x)) == fixups(x)
    - No-op on clean JSON strings
    - Conservative: only inserts commas at newline boundaries
    """

    applied_fixes: list[str] = []
    fixed = text

    # 1) Smart quotes → ASCII quotes
    smart_quotes_fixed = (
        fixed.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    if smart_quotes_fixed != fixed:
        applied_fixes.append("smart_quotes")
        fixed = smart_quotes_fixed

    # 2) Remove zero-width spaces
    zero_width_fixed = fixed.replace("\u200b", "")
    if zero_width_fixed != fixed:
        applied_fixes.append("zero_width_spaces")
        fixed = zero_width_fixed

    # 3) Missing commas between object entries (newline-boundary only)
    missing_commas_fixed = _MISSING_COMMA_AFTER_PRIMITIVE_RE.sub(r'\1,\n"\2":', fixed)
    missing_commas_fixed = _MISSING_COMMA_AFTER_CONTAINER_RE.sub(
        r'\1,\n"\2":', missing_commas_fixed
    )
    if missing_commas_fixed != fixed:
        applied_fixes.append("missing_commas")
        fixed = missing_commas_fixed

    # 4) Escape unescaped quotes inside JSON strings (best-effort)
    escaped_quotes_fixed = _escape_unescaped_quotes_in_strings(fixed)
    if escaped_quotes_fixed != fixed:
        applied_fixes.append("unescaped_quotes")
        fixed = escaped_quotes_fixed

    # 5) Escape control characters in string values
    # NOTE: Must run AFTER unescaped quote escaping so string boundaries are accurate
    control_chars_fixed = _escape_control_chars_in_strings(fixed)
    if control_chars_fixed != fixed:
        applied_fixes.append("control_chars")
        fixed = control_chars_fixed

    # 6) Join stray comma-delimited string fragments in value position
    joined_fragments_fixed = _join_stray_string_fragments(fixed)
    if joined_fragments_fixed != fixed:
        applied_fixes.append("string_fragments")
        fixed = joined_fragments_fixed

    # 7) Remove trailing commas before } or ]
    trailing_commas_fixed = _TRAILING_COMMA_RE.sub(r"\1", fixed)
    if trailing_commas_fixed != fixed:
        applied_fixes.append("trailing_commas")
        fixed = trailing_commas_fixed

    if applied_fixes:
        logger.debug(
            "Applied tolerant JSON fixups",
            component="json_fixups",
            applied_fixes=applied_fixes,
            before_length=len(text),
            after_length=len(fixed),
            before_hash=_stable_text_hash(text),
            after_hash=_stable_text_hash(fixed),
        )

    return fixed


@runtime_checkable
class SimpleChatClient(Protocol):
    """Protocol for LLM clients with simple_chat method."""

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.0,
        format: str | None = None,
    ) -> str:
        """Send a simple chat prompt and return response.

        Args:
            user_prompt: User message content.
            system_prompt: Optional system message.
            model: Model to use.
            temperature: Sampling temperature.
            format: Output format constraint. Use "json" for guaranteed well-formed
                JSON output via Ollama's grammar-level constraints.

        Returns:
            Generated response content.
        """
        ...


def extract_json_from_response(raw: str) -> dict[str, Any]:
    """Extract JSON object from LLM response.

    Handles common issues like markdown code blocks, smart quotes,
    and trailing commas. Uses the canonical parse_llm_json() function
    for consistent parsing across all call sites.

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

    # Extract JSON object boundaries
    text = _extract_json_object(text)

    try:
        # Use canonical parser - NO SILENT FALLBACKS
        return parse_llm_json(text)
    except json.JSONDecodeError as e:
        logger.warning(
            "JSON parse failed in extract_json_from_response",
            component="json_parser",
            error=str(e),
            text_length=len(text),
            text_hash=_stable_text_hash(text),
        )
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
    return tolerant_json_fixups(text)


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
