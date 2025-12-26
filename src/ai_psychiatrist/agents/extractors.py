"""TextOutput extraction functions for Pydantic AI agents."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError
from pydantic_ai import ModelRetry

from ai_psychiatrist.agents.output_models import (
    JudgeMetricOutput,
    MetaReviewOutput,
    QuantitativeOutput,
)
from ai_psychiatrist.infrastructure.llm.responses import extract_score_from_text, extract_xml_tags
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


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


def _tolerant_fixups(json_str: str) -> str:
    """Apply tolerant fixups to common LLM JSON mistakes."""
    # Replace smart quotes
    json_str = (
        json_str.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # Remove trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json_str


def extract_quantitative(text: str) -> QuantitativeOutput:
    """Extract and validate quantitative scoring output from a raw LLM response."""
    try:
        json_str = _tolerant_fixups(_extract_answer_json(text))
        data = json.loads(json_str)
        return QuantitativeOutput.model_validate(data)
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
            data = json.loads(_tolerant_fixups(json_str))
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
            data = json.loads(_tolerant_fixups(json_str))
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
