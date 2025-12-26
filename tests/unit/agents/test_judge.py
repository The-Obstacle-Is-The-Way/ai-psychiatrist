"""Tests for judge agent and feedback loop."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.output_models import JudgeMetricOutput
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.domain.enums import EvaluationMetric
from ai_psychiatrist.domain.exceptions import LLMError
from tests.fixtures.mock_llm import MockLLMClient

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.protocols import ChatRequest


class TestJudgeAgent:
    """Tests for JudgeAgent."""

    @pytest.fixture
    def mock_high_score_response(self) -> str:
        """Response indicating high score."""
        return """
Explanation: The assessment is highly specific.
Score: 5
"""

    @pytest.fixture
    def mock_low_score_response(self) -> str:
        """Response indicating low score."""
        return """
Explanation: The assessment is too vague.
Score: 2
"""

    @pytest.fixture
    def sample_assessment(self) -> QualitativeAssessment:
        """Create sample assessment."""
        return QualitativeAssessment(
            overall="Patient shows moderate depression symptoms.",
            phq8_symptoms="Multiple symptoms present.",
            social_factors="Financial stress mentioned.",
            biological_factors="History of depression.",
            risk_factors="Previous suicide attempt.",
            participant_id=123,
        )

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text="Ellie: How are you?\nParticipant: Not well.",
        )

    @pytest.mark.asyncio
    async def test_evaluate_all_metrics(
        self,
        mock_high_score_response: str,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should evaluate all 4 metrics."""
        # 4 responses for 4 metrics
        mock_client = MockLLMClient(chat_responses=[mock_high_score_response] * 4)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        assert len(evaluation.scores) == 4
        assert EvaluationMetric.COHERENCE in evaluation.scores
        assert EvaluationMetric.COMPLETENESS in evaluation.scores
        assert EvaluationMetric.SPECIFICITY in evaluation.scores
        assert EvaluationMetric.ACCURACY in evaluation.scores

    @pytest.mark.asyncio
    async def test_extracts_scores_correctly(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should extract correct numeric scores."""

        # Mix of high and low scores
        def response_by_metric(request: ChatRequest) -> str:
            # Check the user prompt (last message) for the metric name
            last_msg = request.messages[-1].content.lower()
            if "coherence" in last_msg:
                return "Explanation: Good\nScore: 5"
            if "completeness" in last_msg:
                return "Explanation: Bad\nScore: 2"
            if "specificity" in last_msg:
                return "Explanation: Good\nScore: 5"
            if "accuracy" in last_msg:
                return "Explanation: Bad\nScore: 2"
            return "Explanation: Default\nScore: 3"

        mock_client = MockLLMClient(chat_function=response_by_metric)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        # We know exactly which metric got which score
        assert evaluation.scores[EvaluationMetric.COHERENCE].score == 5
        assert evaluation.scores[EvaluationMetric.COMPLETENESS].score == 2
        assert evaluation.scores[EvaluationMetric.SPECIFICITY].score == 5
        assert evaluation.scores[EvaluationMetric.ACCURACY].score == 2

        assert evaluation.needs_improvement
        assert len(evaluation.low_scores) == 2

    @pytest.mark.asyncio
    async def test_get_feedback_for_low_scores(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should return feedback only for low scores."""

        # Setup specific low score for one metric
        def response_by_metric(request: ChatRequest) -> str:
            last_msg = request.messages[-1].content.lower()
            if "coherence" in last_msg:
                return "Explanation: Good\nScore: 5"
            if "completeness" in last_msg:
                return "Explanation: Bad\nScore: 2"
            if "specificity" in last_msg:
                return "Explanation: Good\nScore: 5"
            if "accuracy" in last_msg:
                return "Explanation: Good\nScore: 5"
            return "Explanation: Default\nScore: 3"

        mock_client = MockLLMClient(chat_function=response_by_metric)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        feedback = agent.get_feedback_for_low_scores(evaluation)

        assert len(feedback) == 1
        # The low score one
        low_metric = evaluation.low_scores[0]
        assert low_metric == EvaluationMetric.COMPLETENESS
        assert low_metric.value in feedback
        assert "Scored 2/5" in feedback[low_metric.value]
        assert "Bad" in feedback[low_metric.value]

    @pytest.mark.asyncio
    async def test_get_feedback_respects_threshold(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should honor custom threshold for low scores."""

        def response_by_metric(request: ChatRequest) -> str:
            last_msg = request.messages[-1].content.lower()
            if "coherence" in last_msg:
                return "Explanation: OK\nScore: 4"
            if "completeness" in last_msg:
                return "Explanation: Low\nScore: 3"
            if "specificity" in last_msg:
                return "Explanation: Low\nScore: 2"
            if "accuracy" in last_msg:
                return "Explanation: Good\nScore: 5"
            return "Explanation: Default\nScore: 3"

        mock_client = MockLLMClient(chat_function=response_by_metric)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        feedback = agent.get_feedback_for_low_scores(evaluation, threshold=2)

        assert len(feedback) == 1
        low_metric = evaluation.low_scores_for_threshold(2)[0]
        assert low_metric == EvaluationMetric.SPECIFICITY
        assert low_metric.value in feedback

    @pytest.mark.asyncio
    async def test_default_score_on_failure(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should default to 3 if score extraction fails."""
        mock_client = MockLLMClient(chat_responses=["I am confused"] * 4)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        for score in evaluation.scores.values():
            assert score.score == 3

    @pytest.mark.asyncio
    async def test_default_score_on_llm_error(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should default to 3 and use safe explanation on LLM errors."""

        class FailingClient:
            """Chat client that raises LLMError for every request."""

            async def simple_chat(
                self,
                user_prompt: str,  # noqa: ARG002
                system_prompt: str = "",  # noqa: ARG002
                model: str | None = None,  # noqa: ARG002
                temperature: float = 0.0,  # noqa: ARG002
            ) -> str:
                raise LLMError("LLM unavailable")

        agent = JudgeAgent(llm_client=FailingClient())
        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        for score in evaluation.scores.values():
            assert score.score == 3
            assert score.explanation == "LLM evaluation failed; default score used."

    @pytest.mark.asyncio
    async def test_pydantic_ai_path_success(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should use Pydantic AI agent when enabled and configured."""
        mock_output = JudgeMetricOutput(score=5, explanation="test")
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_output)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ) as mock_factory:
            agent = JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            result = await agent.evaluate(sample_assessment, sample_transcript)

        mock_factory.assert_called_once()
        assert mock_agent.run.call_count == 4
        assert len(result.scores) == 4
        assert result.scores[EvaluationMetric.COHERENCE].score == 5
