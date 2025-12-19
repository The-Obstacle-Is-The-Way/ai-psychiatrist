"""Tests for judge agent and feedback loop."""

from __future__ import annotations

import pytest

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.domain.enums import EvaluationMetric
from ai_psychiatrist.domain.exceptions import LLMError
from tests.fixtures.mock_llm import MockLLMClient


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
        # Note: Order of iteration over enum might vary, but usually consistent.
        # But MockLLMClient consumes responses FIFO.
        # Let's mock all to return specific scores by inspecting call args if needed.
        # Simpler: just ensure we get the values we expect if we feed different ones.
        # Since we can't easily control which response goes to which metric with simple FIFO list,
        # we will use the same mock response structure but different scores.

        responses = [
            "Explanation: Good\nScore: 5",
            "Explanation: Bad\nScore: 2",
            "Explanation: Good\nScore: 5",
            "Explanation: Bad\nScore: 2",
        ]

        mock_client = MockLLMClient(chat_responses=responses)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        # We don't know which metric got which score without peeking iteration order.
        # But we know we should have two 5s and two 2s.
        scores = [s.score for s in evaluation.scores.values()]
        assert scores.count(5) == 2
        assert scores.count(2) == 2

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
        responses = [
            "Explanation: Good\nScore: 5",  # Coherence
            "Explanation: Bad\nScore: 2",  # Completeness
            "Explanation: Good\nScore: 5",  # Specificity
            "Explanation: Good\nScore: 5",  # Accuracy
        ]
        # NOTE: This relies on Enum order. If Enum order changes, this test might be flaky.
        # But logically, one will be low.

        mock_client = MockLLMClient(chat_responses=responses)
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        feedback = agent.get_feedback_for_low_scores(evaluation)

        assert len(feedback) == 1
        # The low score one
        low_metric = evaluation.low_scores[0]
        assert low_metric.value in feedback
        assert "Scored 2/5" in feedback[low_metric.value]
        assert "Bad" in feedback[low_metric.value]

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
                temperature: float = 0.2,  # noqa: ARG002
                top_k: int = 20,  # noqa: ARG002
                top_p: float = 0.8,  # noqa: ARG002
            ) -> str:
                raise LLMError("LLM unavailable")

        agent = JudgeAgent(llm_client=FailingClient())
        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        for score in evaluation.scores.values():
            assert score.score == 3
            assert score.explanation == "LLM evaluation failed; default score used."
