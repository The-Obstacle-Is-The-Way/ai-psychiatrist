"""Additional coverage tests for QuantitativeAssessmentAgent."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.output_models import EvidenceOutput, QuantitativeOutput
from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import AssessmentMode
from tests.fixtures.mock_llm import MockLLMClient

pytestmark = pytest.mark.unit

SAMPLE_EVIDENCE_RESPONSE = json.dumps({k: ["evidence"] for k in PHQ8_DOMAIN_KEYS})


class TestQuantitativeCoverage:
    """Tests targeting specific code branches for 100% coverage."""

    @pytest.fixture
    def transcript(self) -> Transcript:
        return Transcript(participant_id=1, text="Test transcript.")

    @pytest.fixture
    def mock_quantitative_output(self) -> QuantitativeOutput:
        """Create valid QuantitativeOutput for mocking."""
        return QuantitativeOutput(
            PHQ8_NoInterest=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Depressed=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Sleep=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Tired=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Appetite=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Failure=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Concentrating=EvidenceOutput(evidence="e", reason="r", score=0),
            PHQ8_Moving=EvidenceOutput(evidence="e", reason="r", score=0),
        )

    @pytest.fixture
    def mock_agent_factory(
        self, mock_quantitative_output: QuantitativeOutput
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

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_extract_evidence_json_error(self, transcript: Transcript) -> None:
        """Test _extract_evidence handling of completely invalid JSON."""
        # Evidence response is invalid JSON - should log warning and use empty dict
        client = MockLLMClient(chat_responses=["NOT JSON AT ALL"])

        agent = QuantitativeAssessmentAgent(
            client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        # This should log a warning for evidence parsing and still complete
        result = await agent.assess(transcript)
        assert result.total_score is not None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_strip_json_block_variations(self) -> None:
        """Test _strip_json_block with different formats."""
        client = MockLLMClient()
        agent = QuantitativeAssessmentAgent(
            client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        # 1. Markdown with "json" language identifier
        resp1 = """```json
{"PHQ8_NoInterest": {"score": 1}}
```"""
        assert agent._strip_json_block(resp1) == '{"PHQ8_NoInterest": {"score": 1}}'

        # 2. Markdown without language identifier
        resp2 = """```
{"PHQ8_NoInterest": {"score": 2}}
```"""
        assert agent._strip_json_block(resp2) == '{"PHQ8_NoInterest": {"score": 2}}'

        # 3. Block at very start
        resp3 = """```json{"PHQ8_NoInterest": {"score": 3}}```"""
        assert agent._strip_json_block(resp3) == '{"PHQ8_NoInterest": {"score": 3}}'

        # 4. With <answer> tags
        resp4 = """<answer>
{"PHQ8_NoInterest": {"score": 4}}
</answer>"""
        assert agent._strip_json_block(resp4) == '{"PHQ8_NoInterest": {"score": 4}}'
