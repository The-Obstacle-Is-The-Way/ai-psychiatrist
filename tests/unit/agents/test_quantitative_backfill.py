"""Tests for quantitative agent backfill toggle (SPEC-003)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.output_models import EvidenceOutput, QuantitativeOutput
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import ModelSettings, PydanticAISettings, QuantitativeSettings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import NAReason, PHQ8Item
from tests.fixtures.mock_llm import MockLLMClient


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Fixture providing a MockLLMClient."""
    return MockLLMClient()


@pytest.fixture
def mock_quantitative_output() -> QuantitativeOutput:
    """Create valid QuantitativeOutput object."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=0),
    )


@pytest.fixture
def mock_agent_factory(
    mock_quantitative_output: QuantitativeOutput,
) -> Generator[AsyncMock, None, None]:
    """Patch create_quantitative_agent to return a mock agent."""
    mock_agent = AsyncMock(spec_set=Agent)
    mock_agent.run.return_value = AsyncMock(output=mock_quantitative_output)

    patcher = patch(
        "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
        return_value=mock_agent,
    )
    mock = patcher.start()
    yield mock
    patcher.stop()


def create_agent(
    client: MockLLMClient, settings: QuantitativeSettings
) -> QuantitativeAssessmentAgent:
    """Helper to create agent with Pydantic AI enabled."""
    return QuantitativeAssessmentAgent(
        llm_client=client,
        model_settings=ModelSettings(),
        quantitative_settings=settings,
        pydantic_ai_settings=PydanticAISettings(enabled=True),
        ollama_base_url="http://mock-ollama:11434",
    )


async def test_backfill_disabled_no_enrichment(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """With backfill disabled, no keyword evidence should be added."""
    # Configure mock to return empty evidence from LLM (Step 1)
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))

    # Configure Pydantic Agent to return N/A for Tired (to verify na_reason)
    # We override the default mock output for this test
    mock_output = QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Sleep=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Tired=EvidenceOutput(evidence="", reason="", score=None),  # N/A
        PHQ8_Appetite=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Failure=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="", reason="", score=0),
    )
    mock_agent = mock_agent_factory.return_value
    mock_agent.run.return_value = AsyncMock(output=mock_output)

    settings = QuantitativeSettings(enable_keyword_backfill=False)
    agent = create_agent(mock_llm_client, settings)

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


async def test_backfill_enabled_adds_evidence(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """With backfill explicitly enabled, keyword evidence should be added."""
    # Configure mock to return empty evidence from LLM
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))

    settings = QuantitativeSettings(enable_keyword_backfill=True)
    agent = create_agent(mock_llm_client, settings)

    # Transcript contains "exhausted" which matches Tired keywords
    transcript = Transcript(participant_id=999, text="I feel exhausted all the time.")
    result = await agent.assess(transcript)

    # With backfill ON, Tired should have evidence from keywords
    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.keyword_evidence_count > 0
    assert tired_item.llm_evidence_count == 0
    # evidence_source is determined by counts, not by what Pydantic AI returned
    assert tired_item.evidence_source == "keyword"
    # na_reason is only set if score is None. Here mock output has score=0.


async def test_na_reason_no_mention(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """When neither LLM nor keywords find evidence, reason should be NO_MENTION."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Appetite": []}))

    mock_output = QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Sleep=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Tired=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Appetite=EvidenceOutput(evidence="", reason="", score=None),  # N/A
        PHQ8_Failure=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="", reason="", score=0),
    )
    mock_agent_factory.return_value.run.return_value = AsyncMock(output=mock_output)

    settings = QuantitativeSettings(enable_keyword_backfill=True)
    agent = create_agent(mock_llm_client, settings)

    # Transcript has nothing about appetite
    transcript = Transcript(participant_id=999, text="I sleep well and feel good.")
    result = await agent.assess(transcript)

    appetite_item = result.items[PHQ8Item.APPETITE]
    assert appetite_item.score is None
    assert appetite_item.na_reason == NAReason.NO_MENTION
    assert appetite_item.evidence_source is None


async def test_track_na_reasons_disabled_does_not_populate_na_reason(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """When track_na_reasons is off, na_reason should always be None."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))

    mock_output = QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Sleep=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Tired=EvidenceOutput(evidence="", reason="", score=None),  # N/A
        PHQ8_Appetite=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Failure=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="", reason="", score=0),
    )
    mock_agent_factory.return_value.run.return_value = AsyncMock(output=mock_output)

    settings = QuantitativeSettings(enable_keyword_backfill=False, track_na_reasons=False)
    agent = create_agent(mock_llm_client, settings)

    transcript = Transcript(participant_id=999, text="I feel exhausted all the time.")
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.score is None
    assert tired_item.na_reason is None


async def test_na_reason_score_na_with_evidence_llm_only(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """If LLM evidence exists but score is N/A, use SCORE_NA_WITH_EVIDENCE."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Sleep": ["I can't sleep at night."]}))

    mock_output = QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Sleep=EvidenceOutput(
            evidence="I can't sleep at night.", reason="Insufficient detail", score=None
        ),  # N/A
        PHQ8_Tired=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Appetite=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Failure=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="", reason="", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="", reason="", score=0),
    )
    mock_agent_factory.return_value.run.return_value = AsyncMock(output=mock_output)

    settings = QuantitativeSettings(enable_keyword_backfill=False, track_na_reasons=True)
    agent = create_agent(mock_llm_client, settings)

    transcript = Transcript(participant_id=999, text="I can't sleep at night.")
    result = await agent.assess(transcript)

    sleep_item = result.items[PHQ8Item.SLEEP]
    assert sleep_item.score is None
    assert sleep_item.na_reason == NAReason.SCORE_NA_WITH_EVIDENCE
    assert sleep_item.evidence_source == "llm"
    assert sleep_item.llm_evidence_count > 0
    assert sleep_item.keyword_evidence_count == 0


async def test_evidence_source_both_llm_and_keyword(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """If LLM evidence exists and keywords add more, evidence_source should be both."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": ["I feel exhausted."]}))

    settings = QuantitativeSettings(enable_keyword_backfill=True, keyword_backfill_cap=2)
    agent = create_agent(mock_llm_client, settings)

    transcript = Transcript(
        participant_id=999,
        text="I feel exhausted. I feel drained.",
    )
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.evidence_source == "both"
    assert tired_item.llm_evidence_count > 0
    assert tired_item.keyword_evidence_count > 0


async def test_keyword_backfill_cap_respected(
    mock_llm_client: MockLLMClient,
    mock_agent_factory: AsyncMock,
) -> None:
    """Backfill should not add more than keyword_backfill_cap evidence items."""
    mock_llm_client.add_chat_response(json.dumps({"PHQ8_Tired": []}))

    settings = QuantitativeSettings(enable_keyword_backfill=True, keyword_backfill_cap=1)
    agent = create_agent(mock_llm_client, settings)

    transcript = Transcript(
        participant_id=999,
        text="I feel exhausted. I feel drained. I feel worn out.",
    )
    result = await agent.assess(transcript)

    tired_item = result.items[PHQ8Item.TIRED]
    assert tired_item.keyword_evidence_count == 1
