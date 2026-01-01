"""Tests for feedback loop service."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.config import FeedbackLoopSettings
from ai_psychiatrist.domain.entities import (
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import EvaluationMetric
from ai_psychiatrist.domain.value_objects import EvaluationScore
from ai_psychiatrist.services.feedback_loop import FeedbackLoopResult, FeedbackLoopService

pytestmark = pytest.mark.unit


class TestFeedbackLoopService:
    """Tests for FeedbackLoopService."""

    @pytest.fixture
    def mock_qualitative_agent(self) -> AsyncMock:
        """Mock qualitative agent."""
        agent = AsyncMock(spec_set=QualitativeAssessmentAgent)
        agent.assess.return_value = QualitativeAssessment(
            overall="Initial",
            phq8_symptoms="Initial",
            social_factors="Initial",
            biological_factors="Initial",
            risk_factors="Initial",
            participant_id=123,
        )
        agent.refine.return_value = QualitativeAssessment(
            overall="Refined",
            phq8_symptoms="Refined",
            social_factors="Refined",
            biological_factors="Refined",
            risk_factors="Refined",
            participant_id=123,
        )
        return agent

    @pytest.fixture
    def mock_judge_agent(self) -> AsyncMock:
        """Mock judge agent."""
        agent = AsyncMock(spec_set=JudgeAgent)
        # Default behavior: Acceptable scores
        agent.evaluate.return_value = QualitativeEvaluation(
            scores={
                m: EvaluationScore(metric=m, score=5, explanation="Good") for m in EvaluationMetric
            },
            assessment_id=Mock(),
            iteration=0,
        )
        agent.get_feedback_for_low_scores = Mock(return_value={})
        return agent

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Sample transcript."""
        return Transcript(participant_id=123, text="Hello")

    @pytest.fixture
    def default_settings(self) -> FeedbackLoopSettings:
        """Default settings."""
        return FeedbackLoopSettings(
            enabled=True,
            max_iterations=3,
            score_threshold=3,  # Scores <= 3 trigger refinement
        )

    @pytest.mark.asyncio
    async def test_initial_pass(
        self,
        mock_qualitative_agent: AsyncMock,
        mock_judge_agent: AsyncMock,
        sample_transcript: Transcript,
        default_settings: FeedbackLoopSettings,
    ) -> None:
        """Should return result immediately if initial assessment is good."""
        service = FeedbackLoopService(mock_qualitative_agent, mock_judge_agent, default_settings)

        result = await service.run(sample_transcript)

        assert result.iterations_used == 0
        assert not result.improved
        assert len(result.history) == 1
        mock_qualitative_agent.assess.assert_called_once()
        mock_qualitative_agent.refine.assert_not_called()

    @pytest.mark.asyncio
    async def test_refinement_loop(
        self,
        mock_qualitative_agent: AsyncMock,
        mock_judge_agent: AsyncMock,
        sample_transcript: Transcript,
        default_settings: FeedbackLoopSettings,
    ) -> None:
        """Should loop until scores improve."""
        # 1. Initial: Low score
        # 2. Refined: Good score

        bad_eval = QualitativeEvaluation(
            scores={
                m: EvaluationScore(metric=m, score=2, explanation="Bad") for m in EvaluationMetric
            },
            assessment_id=Mock(),
            iteration=0,
        )

        good_eval = QualitativeEvaluation(
            scores={
                m: EvaluationScore(metric=m, score=5, explanation="Good") for m in EvaluationMetric
            },
            assessment_id=Mock(),
            iteration=1,
        )

        mock_judge_agent.evaluate.side_effect = [bad_eval, good_eval]
        mock_judge_agent.get_feedback_for_low_scores.return_value = {"metric": "feedback"}

        service = FeedbackLoopService(mock_qualitative_agent, mock_judge_agent, default_settings)

        result = await service.run(sample_transcript)

        assert result.iterations_used == 1
        assert result.improved
        assert len(result.history) == 2
        mock_qualitative_agent.assess.assert_called_once()
        mock_qualitative_agent.refine.assert_called_once()
        assert mock_judge_agent.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations(
        self,
        mock_qualitative_agent: AsyncMock,
        mock_judge_agent: AsyncMock,
        sample_transcript: Transcript,
        default_settings: FeedbackLoopSettings,
    ) -> None:
        """Should stop after max iterations."""
        bad_eval = QualitativeEvaluation(
            scores={
                m: EvaluationScore(metric=m, score=2, explanation="Bad") for m in EvaluationMetric
            },
            assessment_id=Mock(),
            iteration=0,
        )

        # Always return bad eval
        mock_judge_agent.evaluate.return_value = bad_eval
        mock_judge_agent.get_feedback_for_low_scores.return_value = {"metric": "feedback"}

        service = FeedbackLoopService(mock_qualitative_agent, mock_judge_agent, default_settings)

        result = await service.run(sample_transcript)

        assert result.iterations_used == default_settings.max_iterations
        assert not result.improved  # Initial and final are both bad (same score)
        assert len(result.history) == default_settings.max_iterations + 1
        assert mock_qualitative_agent.refine.call_count == default_settings.max_iterations

    @pytest.mark.asyncio
    async def test_disabled(
        self,
        mock_qualitative_agent: AsyncMock,
        mock_judge_agent: AsyncMock,
        sample_transcript: Transcript,
    ) -> None:
        """Should skip loop if disabled."""
        settings = FeedbackLoopSettings(enabled=False)

        # Even if eval is bad
        bad_eval = QualitativeEvaluation(
            scores={
                m: EvaluationScore(metric=m, score=2, explanation="Bad") for m in EvaluationMetric
            },
            assessment_id=Mock(),
            iteration=0,
        )
        mock_judge_agent.evaluate.return_value = bad_eval

        service = FeedbackLoopService(mock_qualitative_agent, mock_judge_agent, settings)

        result = await service.run(sample_transcript)

        assert result.iterations_used == 0
        mock_qualitative_agent.refine.assert_not_called()

    def test_result_improved_empty_history(self) -> None:
        """Should return False if history is empty."""
        result = FeedbackLoopResult(
            final_assessment=Mock(),
            final_evaluation=Mock(),
            iterations_used=0,
            history=[],
        )
        assert not result.improved
