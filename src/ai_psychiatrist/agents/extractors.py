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


def _extract_answer_json(text: str) -> str:
    """Extract JSON from <answer>...</answer> tags (preferred) or code fences."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0).strip()

    raise ModelRetry(
        "Could not find structured output. "
        "Please wrap your JSON response in <answer>...</answer> tags."
    )


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
    try:
        json_str = _tolerant_fixups(_extract_answer_json(text))
        data = json.loads(json_str)
        return JudgeMetricOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ModelRetry(f"Invalid judge output: {e}") from e


def extract_meta_review(text: str) -> MetaReviewOutput:
    """Extract and validate meta-review output from a raw LLM response."""
    try:
        json_str = _tolerant_fixups(_extract_answer_json(text))
        data = json.loads(json_str)
        return MetaReviewOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ModelRetry(f"Invalid meta-review output: {e}") from e
