"""Tests for qualitative assessment agent.

Tests verify the agent correctly:
- Generates assessments across all 4 clinical domains
- Returns supporting quotes from Pydantic AI output
- Supports feedback-based refinement
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.output_models import QualitativeOutput
from ai_psychiatrist.agents.prompts.qualitative import (
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient
from tests.fixtures.mock_llm import MockLLMClient

SAMPLE_TRANSCRIPT_TEXT = """Ellie: How are you feeling today?
Participant: Not great, honestly. I don't really enjoy things anymore.
Ellie: Can you tell me more about that?
Participant: I'm always tired, can't seem to get enough sleep.
Ellie: How is your family life?
Participant: Things have been tense at home."""


class TestQualitativeAssessmentAgent:
    """Tests for QualitativeAssessmentAgent."""

    @pytest.fixture
    def mock_qualitative_output(self) -> QualitativeOutput:
        """Create valid QualitativeOutput for mocking."""
        return QualitativeOutput(
            assessment="Patient shows moderate depression symptoms with evident sleep disturbances.",
            phq8_symptoms="Reports feeling down, low energy, and significant fatigue.",
            social_factors="Lives with partner. Reports strained relationship.",
            biological_factors="Family history of depression mentioned.",
            risk_factors="Expresses passive ideation. No active plan.",
            exact_quotes=[
                "I don't really enjoy things anymore.",
                "I'm always tired, can't seem to get enough sleep.",
                "Things have been tense at home.",
            ],
        )

    @pytest.fixture
    def mock_agent_factory(
        self, mock_qualitative_output: QualitativeOutput
    ) -> Generator[AsyncMock, None, None]:
        """Patch create_qualitative_agent to return a mock agent."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_qualitative_output)

        patcher = patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        )
        mock = patcher.start()
        yield mock
        patcher.stop()

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text=SAMPLE_TRANSCRIPT_TEXT,
        )

    @pytest.mark.asyncio
    async def test_pydantic_agent_run_error_not_masked(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Exceptions from Pydantic AI should not be converted to ValueError (Spec 39)."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.side_effect = RuntimeError("boom")

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        ):
            agent = QualitativeAssessmentAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            with pytest.raises(RuntimeError, match="boom"):
                await agent.assess(sample_transcript)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_returns_all_domains(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should include all required domains."""
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        result = await agent.assess(sample_transcript)

        assert isinstance(result, QualitativeAssessment)
        assert result.overall
        assert result.phq8_symptoms
        assert result.social_factors
        assert result.biological_factors
        assert result.risk_factors
        assert result.participant_id == 123

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_extracts_quotes(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should contain supporting quotes from Pydantic AI output."""
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        result = await agent.assess(sample_transcript)

        assert len(result.supporting_quotes) > 0
        assert any("tired" in q.lower() for q in result.supporting_quotes)

    @pytest.mark.asyncio
    async def test_assess_calls_pydantic_agent(
        self,
        mock_agent_factory: AsyncMock,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should call the Pydantic AI agent."""
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        await agent.assess(sample_transcript)

        mock_agent_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_refine_calls_pydantic_agent(
        self,
        mock_agent_factory: AsyncMock,
        sample_transcript: Transcript,
    ) -> None:
        """Refinement should call the Pydantic AI agent."""
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        initial = await agent.assess(sample_transcript)

        feedback = {
            "completeness": "Missing analysis of sleep symptoms",
            "evidence": "Need more quotes to support claims",
        }
        await agent.refine(initial, feedback, sample_transcript)

        # Agent should be called twice (assess + refine)
        mock_agent = mock_agent_factory.return_value
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_full_text_property(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment full_text should contain all sections."""
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        result = await agent.assess(sample_transcript)

        full_text = result.full_text
        assert "Overall Assessment:" in full_text
        assert "PHQ-8 Symptoms:" in full_text
        assert "Social Factors:" in full_text
        assert "Biological Factors:" in full_text
        assert "Risk Factors:" in full_text

    @pytest.mark.asyncio
    async def test_pydantic_ai_path_success(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should use Pydantic AI agent when enabled and configured."""
        mock_output = QualitativeOutput(
            assessment="Test Assessment",
            phq8_symptoms="Test Symptoms",
            social_factors="Test Social",
            biological_factors="Test Bio",
            risk_factors="Test Risk",
            exact_quotes=["Test Quote"],
        )
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_output)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        ) as mock_factory:
            agent = QualitativeAssessmentAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True, timeout_seconds=123.0),
                ollama_base_url="http://localhost:11434",
            )
            result = await agent.assess(sample_transcript)

        mock_factory.assert_called_once()
        mock_agent.run.assert_called_once()
        assert mock_agent.run.call_args.kwargs["model_settings"]["timeout"] == 123.0
        assert result.overall == "Test Assessment"
        assert result.phq8_symptoms == "Test Symptoms"

    @pytest.mark.asyncio
    async def test_pydantic_ai_failure_raises(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should preserve the original exception type when Pydantic AI fails (Spec 39)."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.side_effect = RuntimeError("LLM timeout")

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        ):
            agent = QualitativeAssessmentAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            with pytest.raises(RuntimeError, match="LLM timeout"):
                await agent.assess(sample_transcript)

    @pytest.mark.asyncio
    async def test_refine_failure_raises(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should preserve the original exception type when refinement fails (Spec 39)."""
        mock_output = QualitativeOutput(
            assessment="Test",
            phq8_symptoms="Test",
            social_factors="Test",
            biological_factors="Test",
            risk_factors="Test",
            exact_quotes=[],
        )
        mock_agent = AsyncMock(spec_set=Agent)
        # First call succeeds (assess), second fails (refine)
        mock_agent.run.side_effect = [
            AsyncMock(output=mock_output),
            RuntimeError("Refinement failed"),
        ]

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        ):
            agent = QualitativeAssessmentAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            initial = await agent.assess(sample_transcript)

            with pytest.raises(RuntimeError, match="Refinement failed"):
                await agent.refine(initial, {"test": "feedback"}, sample_transcript)

    def test_init_without_ollama_url_raises(self) -> None:
        """Should raise ValueError when Pydantic AI enabled but no ollama_base_url."""
        with pytest.raises(ValueError, match="ollama_base_url"):
            QualitativeAssessmentAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url=None,
            )

    @pytest.mark.asyncio
    async def test_assess_without_agent_raises(self) -> None:
        """Should raise ValueError when agent not initialized."""
        # Create agent with Pydantic AI disabled
        agent = QualitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=False),
        )
        transcript = Transcript(participant_id=1, text="Test")

        with pytest.raises(ValueError, match="Pydantic AI agent not initialized"):
            await agent.assess(transcript)


class TestQualitativePrompts:
    """Tests for prompt template functions."""

    def test_make_qualitative_prompt_includes_transcript(self) -> None:
        """Prompt should include transcript text."""
        transcript = "Ellie: Hello\nParticipant: Hi there"
        prompt = make_qualitative_prompt(transcript)

        assert transcript in prompt
        assert "Ellie" in prompt
        assert "Participant" in prompt

    def test_make_qualitative_prompt_includes_domains(self) -> None:
        """Prompt should mention all assessment domains."""
        prompt = make_qualitative_prompt("test transcript")

        assert "PHQ-8" in prompt
        assert "social" in prompt.lower()
        assert "biological" in prompt.lower()
        assert "risk" in prompt.lower()

    def test_make_qualitative_prompt_includes_xml_format(self) -> None:
        """Prompt should specify expected XML format."""
        prompt = make_qualitative_prompt("test")

        assert "<assessment>" in prompt
        assert "<PHQ8_symptoms>" in prompt
        assert "<social_factors>" in prompt
        assert "<biological_factors>" in prompt
        assert "<risk_factors>" in prompt
        assert "<exact_quotes>" in prompt

    def test_make_qualitative_prompt_includes_examples(self) -> None:
        """Prompt should include example guidance."""
        prompt = make_qualitative_prompt("test")

        assert "Examples" in prompt

    def test_make_feedback_prompt_includes_feedback(self) -> None:
        """Feedback prompt should include all feedback metrics."""
        feedback = {
            "completeness": "Missing details",
            "accuracy": "Some inaccuracies",
        }
        prompt = make_feedback_prompt("original", feedback, "transcript")

        assert "COMPLETENESS" in prompt
        assert "Missing details" in prompt
        assert "ACCURACY" in prompt
        assert "Some inaccuracies" in prompt

    def test_make_feedback_prompt_includes_original(self) -> None:
        """Feedback prompt should include original assessment."""
        original = "This is the original assessment"
        prompt = make_feedback_prompt(original, {}, "transcript")

        assert original in prompt
        assert "ORIGINAL ASSESSMENT" in prompt

    def test_make_feedback_prompt_includes_transcript(self) -> None:
        """Feedback prompt should include transcript."""
        transcript = "Interview transcript here"
        prompt = make_feedback_prompt("original", {}, transcript)

        assert transcript in prompt
        assert "TRANSCRIPT" in prompt


class TestAgentProtocol:
    """Tests for SimpleChatClient protocol compatibility."""

    def test_mock_client_implements_protocol(self) -> None:
        """MockLLMClient should implement SimpleChatClient protocol."""
        client = MockLLMClient()
        assert isinstance(client, SimpleChatClient)
