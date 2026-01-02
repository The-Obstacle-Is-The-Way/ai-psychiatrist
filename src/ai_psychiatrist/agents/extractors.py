"""TextOutput extraction functions for Pydantic AI agents."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError
from pydantic_ai import ModelRetry

from ai_psychiatrist.agents.output_models import (
    JudgeMetricOutput,
    MetaReviewOutput,
    QualitativeOutput,
    QuantitativeOutput,
)
from ai_psychiatrist.infrastructure.llm.responses import (
    extract_score_from_text,
    extract_xml_tags,
    tolerant_json_fixups,
)
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


_DEFAULT_QUANT_REASON = "Auto-filled: missing reason"
_DEFAULT_QUANT_EVIDENCE = "No relevant evidence found"


def _fill_missing_quantitative_fields(data: object) -> object:
    """Fill non-critical missing fields in QuantitativeOutput payloads.

    We treat `score` as critical (must be present and parseable), but allow LLMs to
    occasionally omit `evidence`/`reason` while still producing a usable score.

    This prevents deterministic failures in strict TextOutput validation when the
    model returns valid JSON but misses a non-critical field.
    """
    if not isinstance(data, dict):
        return data

    fixed: dict[str, object] = dict(data)
    for item_key in QuantitativeOutput.model_fields:
        raw_item = fixed.get(item_key)
        if not isinstance(raw_item, dict):
            continue

        if "reason" in raw_item and "evidence" in raw_item:
            continue

        patched_item: dict[str, object] = dict(raw_item)
        patched_item.setdefault("evidence", _DEFAULT_QUANT_EVIDENCE)
        patched_item.setdefault("reason", _DEFAULT_QUANT_REASON)
        fixed[item_key] = patched_item

    return fixed


def _find_answer_json(text: str, *, allow_unwrapped_object: bool) -> str | None:
    """Find JSON inside <answer> tags, code fences, or (optionally) the first {...} block."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    if allow_unwrapped_object:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0).strip()

    return None


def _extract_answer_json(text: str) -> str:
    """Extract required JSON from <answer>...</answer> tags (preferred) or fallbacks."""
    json_str = _find_answer_json(text, allow_unwrapped_object=True)
    if json_str is None:
        raise ModelRetry(
            "Could not find structured output. "
            "Please wrap your JSON response in <answer>...</answer> tags."
        )
    return json_str


def _clean_quote_line(line: str) -> str:
    """Normalize a quote line from an exact_quotes block."""
    cleaned = line.strip()
    if not cleaned or cleaned == "-":
        return ""
    if cleaned[0] in {"-", "*", "•"}:
        cleaned = cleaned[1:].strip()
    return cleaned


def extract_qualitative(text: str) -> QualitativeOutput:
    """Extract and validate qualitative assessment output from a raw LLM response.

    Expects XML tags for assessment domains and optional exact_quotes.
    """
    required_tags = [
        "assessment",
        "PHQ8_symptoms",
        "social_factors",
        "biological_factors",
        "risk_factors",
    ]
    extracted = extract_xml_tags(text, [*required_tags, "exact_quotes"])

    # Validation: Check for missing required tags
    missing = [tag for tag in required_tags if not extracted.get(tag)]
    if missing:
        raise ModelRetry(
            f"Missing required XML tags: {', '.join(missing)}. "
            "Please ensure all assessment sections are provided in XML tags."
        )

    # Parse quotes
    quotes: list[str] = []
    raw_quotes = extracted.get("exact_quotes", "")
    if raw_quotes:
        quotes = [_clean_quote_line(line) for line in raw_quotes.split("\n")]
        quotes = [q for q in quotes if q]

    try:
        return QualitativeOutput(
            assessment=extracted["assessment"],
            phq8_symptoms=extracted["PHQ8_symptoms"],
            social_factors=extracted["social_factors"],
            biological_factors=extracted["biological_factors"],
            risk_factors=extracted["risk_factors"],
            exact_quotes=quotes,
        )
    except ValidationError as e:
        raise ModelRetry(f"Output validation failed: {e}") from e


def extract_quantitative(text: str) -> QuantitativeOutput:
    """Extract and validate quantitative scoring output from a raw LLM response."""
    try:
        json_str = tolerant_json_fixups(_extract_answer_json(text))
        data = json.loads(json_str)
        return QuantitativeOutput.model_validate(_fill_missing_quantitative_fields(data))
    except json.JSONDecodeError as e:
        raise ModelRetry(
            f"Invalid JSON in <answer>: {e}. Please ensure <answer> contains valid JSON."
        ) from e
    except (ValidationError, ValueError) as e:
        raise ModelRetry(
            f"Response validation failed: {e}. Please ensure all PHQ-8 items are present and valid."
        ) from e


def extract_judge_metric(text: str) -> JudgeMetricOutput:
    """Extract and validate judge metric output from a raw LLM response."""
    json_str = _find_answer_json(text, allow_unwrapped_object=False)
    if json_str is not None:
        try:
            data = json.loads(tolerant_json_fixups(json_str))
            return JudgeMetricOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            raise ModelRetry(f"Invalid judge output JSON: {e}") from e

    score = extract_score_from_text(text)
    if score is None:
        raise ModelRetry(
            "Could not extract judge score. Please include a line like 'Score: 4' "
            "with a number between 1 and 5."
        )

    try:
        return JudgeMetricOutput.model_validate({"score": score, "explanation": text.strip()})
    except ValidationError as e:
        raise ModelRetry(f"Invalid judge output: {e}") from e


def extract_meta_review(text: str) -> MetaReviewOutput:
    """Extract and validate meta-review output from a raw LLM response."""
    json_str = _find_answer_json(text, allow_unwrapped_object=False)
    if json_str is not None:
        try:
            data = json.loads(tolerant_json_fixups(json_str))
            return MetaReviewOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            raise ModelRetry(f"Invalid meta-review output JSON: {e}") from e

    tags = extract_xml_tags(text, ["severity", "explanation"])
    severity_raw = tags.get("severity", "").strip()
    if not severity_raw:
        raise ModelRetry(
            "Could not find <severity> tag. Please include <severity>0-4</severity> "
            "and <explanation>...</explanation> in your response."
        )

    try:
        severity_value = int(severity_raw)
    except ValueError as e:
        raise ModelRetry(f"Invalid severity value: {severity_raw!r}. Please provide 0-4.") from e

    if not (0 <= severity_value <= 4):
        logger.warning(
            "Clamping out-of-range severity",
            raw_value=severity_value,
            clamped_to=max(0, min(4, severity_value)),
        )
    severity_value = max(0, min(4, severity_value))
    explanation = tags.get("explanation", "").strip() or text.strip()

    try:
        return MetaReviewOutput.model_validate(
            {"severity": severity_value, "explanation": explanation}
        )
    except ValidationError as e:
        raise ModelRetry(f"Invalid meta-review output: {e}") from e
