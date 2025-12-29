"""Tests for judge agent and feedback loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.output_models import JudgeMetricOutput
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.domain.enums import EvaluationMetric
from tests.fixtures.mock_llm import MockLLMClient


class TestJudgeAgent:
    """Tests for JudgeAgent."""

    @pytest.fixture
    def mock_judge_output_high(self) -> JudgeMetricOutput:
        """Create high-score output for mocking."""
        return JudgeMetricOutput(score=5, explanation="The assessment is highly specific.")

    @pytest.fixture
    def mock_judge_output_low(self) -> JudgeMetricOutput:
        """Create low-score output for mocking."""
        return JudgeMetricOutput(score=2, explanation="The assessment is too vague.")

    @pytest.fixture
    def mock_agent_factory(
        self, mock_judge_output_high: JudgeMetricOutput
    ) -> Generator[AsyncMock, None, None]:
        """Patch create_judge_metric_agent to return a mock agent."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_judge_output_high)

        patcher = patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        )
        mock = patcher.start()
        yield mock
        patcher.stop()

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
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_evaluate_all_metrics(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should evaluate all 4 metrics."""
        agent = JudgeAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

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
        """Should extract correct numeric scores from Pydantic AI output."""
        # Create outputs with different scores for each metric
        outputs = {
            EvaluationMetric.COHERENCE: JudgeMetricOutput(score=5, explanation="Good coherence"),
            EvaluationMetric.COMPLETENESS: JudgeMetricOutput(
                score=2, explanation="Bad completeness"
            ),
            EvaluationMetric.SPECIFICITY: JudgeMetricOutput(
                score=5, explanation="Good specificity"
            ),
            EvaluationMetric.ACCURACY: JudgeMetricOutput(score=2, explanation="Bad accuracy"),
        }

        call_count = 0
        metrics_order = list(EvaluationMetric.all_metrics())

        async def mock_run(prompt: str, **kwargs: Any) -> AsyncMock:
            nonlocal call_count
            metric = metrics_order[call_count]
            call_count += 1
            return AsyncMock(output=outputs[metric])

        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run = mock_run

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ):
            agent = JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            evaluation = await agent.evaluate(sample_assessment, sample_transcript)

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
        # Create outputs with one low score
        outputs = {
            EvaluationMetric.COHERENCE: JudgeMetricOutput(score=5, explanation="Good"),
            EvaluationMetric.COMPLETENESS: JudgeMetricOutput(score=2, explanation="Bad"),
            EvaluationMetric.SPECIFICITY: JudgeMetricOutput(score=5, explanation="Good"),
            EvaluationMetric.ACCURACY: JudgeMetricOutput(score=5, explanation="Good"),
        }

        call_count = 0
        metrics_order = list(EvaluationMetric.all_metrics())

        async def mock_run(prompt: str, **kwargs: Any) -> AsyncMock:
            nonlocal call_count
            metric = metrics_order[call_count]
            call_count += 1
            return AsyncMock(output=outputs[metric])

        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run = mock_run

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ):
            agent = JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        feedback = agent.get_feedback_for_low_scores(evaluation)

        assert len(feedback) == 1
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
        outputs = {
            EvaluationMetric.COHERENCE: JudgeMetricOutput(score=4, explanation="OK"),
            EvaluationMetric.COMPLETENESS: JudgeMetricOutput(score=3, explanation="Low"),
            EvaluationMetric.SPECIFICITY: JudgeMetricOutput(score=2, explanation="Low"),
            EvaluationMetric.ACCURACY: JudgeMetricOutput(score=5, explanation="Good"),
        }

        call_count = 0
        metrics_order = list(EvaluationMetric.all_metrics())

        async def mock_run(prompt: str, **kwargs: Any) -> AsyncMock:
            nonlocal call_count
            metric = metrics_order[call_count]
            call_count += 1
            return AsyncMock(output=outputs[metric])

        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run = mock_run

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ):
            agent = JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        feedback = agent.get_feedback_for_low_scores(evaluation, threshold=2)

        assert len(feedback) == 1
        low_metric = evaluation.low_scores_for_threshold(2)[0]
        assert low_metric == EvaluationMetric.SPECIFICITY
        assert low_metric.value in feedback

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
                pydantic_ai_settings=PydanticAISettings(enabled=True, timeout_seconds=123.0),
                ollama_base_url="http://localhost:11434",
            )
            result = await agent.evaluate(sample_assessment, sample_transcript)

        mock_factory.assert_called_once()
        assert mock_agent.run.call_count == 4
        for call in mock_agent.run.call_args_list:
            assert call.kwargs["model_settings"]["timeout"] == 123.0
        assert len(result.scores) == 4
        assert result.scores[EvaluationMetric.COHERENCE].score == 5

    @pytest.mark.asyncio
    async def test_pydantic_ai_failure_raises(
        self,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should raise ValueError when Pydantic AI call fails."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.side_effect = RuntimeError("LLM timeout")

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
            return_value=mock_agent,
        ):
            agent = JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            with pytest.raises(ValueError, match="Pydantic AI evaluation failed"):
                await agent.evaluate(sample_assessment, sample_transcript)

    def test_init_without_ollama_url_raises(self) -> None:
        """Should raise ValueError when Pydantic AI enabled but no ollama_base_url."""
        with pytest.raises(ValueError, match="ollama_base_url"):
            JudgeAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url=None,
            )

    @pytest.mark.asyncio
    async def test_evaluate_without_agent_raises(self) -> None:
        """Should raise ValueError when agent not initialized."""
        agent = JudgeAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=False),
        )
        assessment = QualitativeAssessment(
            overall="Test",
            phq8_symptoms="Test",
            social_factors="Test",
            biological_factors="Test",
            risk_factors="Test",
            participant_id=1,
        )
        transcript = Transcript(participant_id=1, text="Test")

        with pytest.raises(ValueError, match="Pydantic AI metric agent not initialized"):
            await agent.evaluate(assessment, transcript)
