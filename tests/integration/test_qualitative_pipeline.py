"""Integration tests for the full qualitative assessment pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.output_models import JudgeMetricOutput, QualitativeOutput
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.config import FeedbackLoopSettings, PydanticAISettings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService
from tests.fixtures.mock_llm import MockLLMClient

pytestmark = pytest.mark.integration


def make_qualitative_output(is_refined: bool = False) -> QualitativeOutput:
    """Create a QualitativeOutput for mocking."""
    if is_refined:
        return QualitativeOutput(
            assessment="Patient shows clear signs of major depression with specific symptoms.",
            phq8_symptoms="Depressed mood: Nearly every day (Score 3). Anhedonia: Several days.",
            social_factors="Social isolation due to recent divorce.",
            biological_factors="Family history of depression (Mother).",
            risk_factors="No current suicidal ideation.",
            exact_quotes=["I feel sad"],
        )
    return QualitativeOutput(
        assessment="Patient seems depressed.",
        phq8_symptoms="Depressed mood: Frequent",
        social_factors="Isolated.",
        biological_factors="None.",
        risk_factors="None.",
        exact_quotes=[],
    )


def make_judge_output(score: int, explanation: str) -> JudgeMetricOutput:
    """Create a JudgeMetricOutput for mocking."""
    return JudgeMetricOutput(score=score, explanation=explanation)


class TestQualitativePipeline:
    """Integration tests for qualitative pipeline."""

    @pytest.fixture
    def mock_qualitative_agent(self) -> Generator[AsyncMock, None, None]:
        """Mock qualitative agent that returns initial then refined assessment."""
        mock_agent = AsyncMock(spec_set=Agent)
        # First call returns initial assessment, subsequent calls return refined
        call_count = [0]

        async def mock_run(*args: object, **kwargs: object) -> AsyncMock:
            result = AsyncMock()
            if call_count[0] == 0:
                result.output = make_qualitative_output(is_refined=False)
            else:
                result.output = make_qualitative_output(is_refined=True)
            call_count[0] += 1
            return result

        mock_agent.run = mock_run

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
            return_value=mock_agent,
        ):
            yield mock_agent

    @pytest.fixture
    def mock_judge_agent(self) -> Generator[AsyncMock, None, None]:
        """Mock judge agent: first 4 calls low score, next 4 high score."""
        mock_agent = AsyncMock(spec_set=Agent)
        call_count = [0]

        async def mock_run(*args: object, **kwargs: object) -> AsyncMock:
            result = AsyncMock()
            # First 4 calls (initial evaluation) return low score
            # Next 4 calls (after refinement) return high score
            if call_count[0] < 4:
                result.output = make_judge_output(2, "Vague assessment")
            else:
                result.output = make_judge_output(5, "Excellent assessment")
            call_count[0] += 1
            return result

        mock_agent.run = mock_run

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ):
            yield mock_agent

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_qualitative_agent", "mock_judge_agent")
    async def test_feedback_loop_refinement(self) -> None:
        """Verify full feedback loop flow: Assess -> Judge -> Refine -> Judge."""
        client = MockLLMClient()

        qual_agent = QualitativeAssessmentAgent(
            client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        judge_agent = JudgeAgent(
            llm_client=client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        settings = FeedbackLoopSettings(
            enabled=True,
            max_iterations=5,
            score_threshold=3,
        )

        service = FeedbackLoopService(qual_agent, judge_agent, settings)

        transcript = Transcript(
            participant_id=123,
            text="Ellie: Hello.\nParticipant: Hi, I feel sad.",
        )

        result = await service.run(transcript)

        assert result.iterations_used == 1
        assert result.improved
        assert result.final_evaluation.average_score == 5.0
        assert len(result.history) == 2  # Initial + 1 refinement

        # Verify initial was bad
        initial_eval = result.history[0][1]
        assert initial_eval.average_score == 2.0

        # Verify final was good
        final_eval = result.final_evaluation
        assert final_eval.all_acceptable

    @pytest.mark.asyncio
    async def test_feedback_loop_max_iterations(self) -> None:
        """Verify feedback loop respects max_iterations."""
        # Always return low scores to force max iterations
        mock_qual_agent = AsyncMock(spec_set=Agent)
        mock_qual_agent.run.return_value = AsyncMock(
            output=make_qualitative_output(is_refined=False)
        )

        mock_judge_agent = AsyncMock(spec_set=Agent)
        mock_judge_agent.run.return_value = AsyncMock(output=make_judge_output(2, "Still vague"))

        with (
            patch(
                "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
                return_value=mock_qual_agent,
            ),
            patch(
                "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
                return_value=mock_judge_agent,
            ),
        ):
            client = MockLLMClient()

            qual_agent = QualitativeAssessmentAgent(
                client,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            judge_agent = JudgeAgent(
                llm_client=client,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )

            settings = FeedbackLoopSettings(
                enabled=True,
                max_iterations=2,
                score_threshold=3,
            )

            service = FeedbackLoopService(qual_agent, judge_agent, settings)

            transcript = Transcript(
                participant_id=456,
                text="Ellie: How are you?\nParticipant: Not great.",
            )

            result = await service.run(transcript)

            # Should hit max iterations (2) since scores never improve
            assert result.iterations_used == 2
            assert not result.improved
            # Initial + 2 refinement attempts = 3 total
            assert len(result.history) == 3

    @pytest.mark.asyncio
    async def test_feedback_loop_immediate_pass(self) -> None:
        """Verify feedback loop exits early if initial assessment passes."""
        mock_qual_agent = AsyncMock(spec_set=Agent)
        mock_qual_agent.run.return_value = AsyncMock(
            output=make_qualitative_output(is_refined=True)
        )

        mock_judge_agent = AsyncMock(spec_set=Agent)
        mock_judge_agent.run.return_value = AsyncMock(
            output=make_judge_output(5, "Excellent from the start")
        )

        with (
            patch(
                "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
                return_value=mock_qual_agent,
            ),
            patch(
                "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
                return_value=mock_judge_agent,
            ),
        ):
            client = MockLLMClient()

            qual_agent = QualitativeAssessmentAgent(
                client,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            judge_agent = JudgeAgent(
                llm_client=client,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )

            settings = FeedbackLoopSettings(
                enabled=True,
                max_iterations=5,
                score_threshold=3,
            )

            service = FeedbackLoopService(qual_agent, judge_agent, settings)

            transcript = Transcript(
                participant_id=789,
                text="Ellie: Hello.\nParticipant: Hi.",
            )

            result = await service.run(transcript)

            # Should pass immediately with 0 iterations
            assert result.iterations_used == 0
            assert not result.improved  # No improvement needed
            assert len(result.history) == 1  # Just initial assessment
            assert result.final_evaluation.all_acceptable
