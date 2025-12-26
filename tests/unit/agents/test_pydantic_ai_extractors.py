"""Unit tests for Pydantic AI TextOutput extractors."""

from __future__ import annotations

import pytest
from pydantic_ai import ModelRetry

from ai_psychiatrist.agents.extractors import extract_quantitative
from ai_psychiatrist.agents.output_models import QuantitativeOutput

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:Data directory does not exist.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Few-shot enabled but embeddings not found.*:UserWarning"),
]


def test_extract_quantitative_valid() -> None:
    response = """
<thinking>
analysis...
</thinking>
<answer>
{
  "PHQ8_NoInterest": {"evidence": "a", "reason": "b", "score": 2},
  "PHQ8_Depressed": {"evidence": "a", "reason": "b", "score": 1},
  "PHQ8_Sleep": {"evidence": "a", "reason": "b", "score": 0},
  "PHQ8_Tired": {"evidence": "a", "reason": "b", "score": 3},
  "PHQ8_Appetite": {"evidence": "a", "reason": "b", "score": "N/A"},
  "PHQ8_Failure": {"evidence": "a", "reason": "b", "score": "0"},
  "PHQ8_Concentrating": {"evidence": "a", "reason": "b", "score": 1},
  "PHQ8_Moving": {"evidence": "a", "reason": "b", "score": null}
}
</answer>
"""
    parsed = extract_quantitative(response)
    assert isinstance(parsed, QuantitativeOutput)
    assert parsed.PHQ8_NoInterest.score == 2
    assert parsed.PHQ8_Appetite.score is None
    assert parsed.PHQ8_Failure.score == 0
    assert parsed.PHQ8_Moving.score is None


def test_extract_quantitative_missing_answer_tags_retries() -> None:
    with pytest.raises(ModelRetry):
        extract_quantitative("no answer tags here")


def test_extract_quantitative_invalid_json_retries() -> None:
    with pytest.raises(ModelRetry):
        extract_quantitative("<answer>{not valid json}</answer>")


def test_extract_quantitative_invalid_score_range_retries() -> None:
    response = """
<answer>
{
  "PHQ8_NoInterest": {"evidence": "a", "reason": "b", "score": 99},
  "PHQ8_Depressed": {"evidence": "a", "reason": "b", "score": 1},
  "PHQ8_Sleep": {"evidence": "a", "reason": "b", "score": 0},
  "PHQ8_Tired": {"evidence": "a", "reason": "b", "score": 3},
  "PHQ8_Appetite": {"evidence": "a", "reason": "b", "score": "N/A"},
  "PHQ8_Failure": {"evidence": "a", "reason": "b", "score": 0},
  "PHQ8_Concentrating": {"evidence": "a", "reason": "b", "score": 1},
  "PHQ8_Moving": {"evidence": "a", "reason": "b", "score": null}
}
</answer>
"""
    with pytest.raises(ModelRetry):
        extract_quantitative(response)
