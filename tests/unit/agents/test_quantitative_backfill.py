"""Tests for quantitative agent backfill toggle (SPEC-003)."""

from __future__ import annotations

import json

import pytest

from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import ModelSettings, QuantitativeSettings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import NAReason, PHQ8Item
from tests.fixtures.mock_llm import MockLLMClient


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Fixture providing a MockLLMClient."""
    return MockLLMClient()


@pytest.mark.unit
async def test_backfill_disabled_no_enrichment(
    mock_llm_client: MockLLMClient,
) -> None:
    """With backfill disabled, no keyword evidence should be added."""
    # Configure mock to return empty evidence from LLM
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))
    # Add scoring response (all N/A) to handle subsequent call
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=False)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    # Transcript contains "exhausted" which would match Tired keywords
    transcript = Transcript(participant_id=999, text="I feel exhausted all the time.")
    result = await agent.assess(transcript)

    # With backfill OFF, Tired should be N/A (LLM returned empty, no backfill)
    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.score is None
    assert tired_item.na_reason == NAReason.LLM_ONLY_MISSED
    assert tired_item.evidence_source is None
    assert tired_item.llm_evidence_count == 0
    assert tired_item.keyword_evidence_count == 0


@pytest.mark.unit
async def test_backfill_enabled_adds_evidence(
    mock_llm_client: MockLLMClient,
) -> None:
    """With backfill explicitly enabled, keyword evidence should be added."""
    # Configure mock to return empty evidence from LLM
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))
    # Add scoring response (doesn't matter what, just valid JSON)
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=True)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    # Transcript contains "exhausted" which matches Tired keywords
    transcript = Transcript(participant_id=999, text="I feel exhausted all the time.")
    result = await agent.assess(transcript)

    # With backfill ON, Tired should have evidence from keywords
    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.keyword_evidence_count > 0
    assert tired_item.llm_evidence_count == 0
    assert tired_item.evidence_source == "keyword"
    assert tired_item.na_reason == NAReason.KEYWORDS_INSUFFICIENT


@pytest.mark.unit
async def test_na_reason_no_mention(mock_llm_client: MockLLMClient) -> None:
    """When neither LLM nor keywords find evidence, reason should be NO_MENTION."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Appetite": []}))
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=True)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    # Transcript has nothing about appetite
    transcript = Transcript(participant_id=999, text="I sleep well and feel good.")
    result = await agent.assess(transcript)

    appetite_item = result.items[PHQ8Item.APPETITE]
    assert appetite_item.score is None
    assert appetite_item.na_reason == NAReason.NO_MENTION
    assert appetite_item.evidence_source is None


@pytest.mark.unit
async def test_track_na_reasons_disabled_does_not_populate_na_reason(
    mock_llm_client: MockLLMClient,
) -> None:
    """When track_na_reasons is off, na_reason should always be None."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=False, track_na_reasons=False)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    transcript = Transcript(participant_id=999, text="I feel exhausted all the time.")
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.score is None
    assert tired_item.na_reason is None


@pytest.mark.unit
async def test_na_reason_score_na_with_evidence_llm_only(
    mock_llm_client: MockLLMClient,
) -> None:
    """If LLM evidence exists but score is N/A, use SCORE_NA_WITH_EVIDENCE."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Sleep": ["I can't sleep at night."]}))
    mock_llm_client.add_chat_response(
        json.dumps(
            {
                "PHQ8_Sleep": {
                    "evidence": "I can't sleep at night.",
                    "reason": "Insufficient detail to score frequency.",
                    "score": "N/A",
                }
            }
        )
    )

    settings = QuantitativeSettings(enable_keyword_backfill=False, track_na_reasons=True)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    transcript = Transcript(participant_id=999, text="I can't sleep at night.")
    result = await agent.assess(transcript)

    sleep_item = result.items[PHQ8Item.SLEEP]
    assert sleep_item.score is None
    assert sleep_item.na_reason == NAReason.SCORE_NA_WITH_EVIDENCE
    assert sleep_item.evidence_source == "llm"
    assert sleep_item.llm_evidence_count > 0
    assert sleep_item.keyword_evidence_count == 0


@pytest.mark.unit
async def test_evidence_source_both_llm_and_keyword(
    mock_llm_client: MockLLMClient,
) -> None:
    """If LLM evidence exists and keywords add more, evidence_source should be both."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": ["I feel exhausted."]}))
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=True, keyword_backfill_cap=2)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    transcript = Transcript(
        participant_id=999,
        text="I feel exhausted. I feel drained.",
    )
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.evidence_source == "both"
    assert tired_item.llm_evidence_count > 0
    assert tired_item.keyword_evidence_count > 0


@pytest.mark.unit
async def test_keyword_backfill_cap_respected(mock_llm_client: MockLLMClient) -> None:
    """Backfill should not add more than keyword_backfill_cap evidence items."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))
    mock_llm_client.add_chat_response(json.dumps({}))

    settings = QuantitativeSettings(enable_keyword_backfill=True, keyword_backfill_cap=1)

    agent = QuantitativeAssessmentAgent(
        llm_client=mock_llm_client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
    )

    transcript = Transcript(
        participant_id=999,
        text="I feel exhausted. I feel drained. I feel worn out.",
    )
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.keyword_evidence_count == 1
