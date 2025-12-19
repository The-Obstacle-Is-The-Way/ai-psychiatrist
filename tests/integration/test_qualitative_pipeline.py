"""Integration tests for the full qualitative assessment pipeline."""

from __future__ import annotations

import pytest

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.config import FeedbackLoopSettings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService
from tests.fixtures.mock_llm import MockLLMClient


class TestQualitativePipeline:
    """Integration tests for qualitative pipeline."""

    @pytest.fixture
    def mock_assessment_response(self) -> str:
        return """
<assessment>
Patient seems depressed.
</assessment>
<PHQ8_symptoms>
- Depressed mood: Frequent
</PHQ8_symptoms>
<social_factors>
Isolated.
</social_factors>
<biological_factors>
None.
</biological_factors>
<risk_factors>
None.
</risk_factors>
"""

    @pytest.fixture
    def mock_refined_assessment_response(self) -> str:
        return """
<assessment>
Patient shows clear signs of major depression with specific symptoms.
</assessment>
<PHQ8_symptoms>
- Depressed mood: Nearly every day (Score 3)
- Anhedonia: Several days (Score 1)
</PHQ8_symptoms>
<social_factors>
Social isolation due to recent divorce.
</social_factors>
<biological_factors>
Family history of depression (Mother).
</biological_factors>
<risk_factors>
No current suicidal ideation.
</risk_factors>
"""

    @pytest.fixture
    def mock_low_score_response(self) -> str:
        return "Explanation: Vague.\nScore: 2"

    @pytest.fixture
    def mock_high_score_response(self) -> str:
        return "Explanation: Excellent.\nScore: 5"

    @pytest.mark.asyncio
    async def test_feedback_loop_refinement(
        self,
        mock_assessment_response: str,
        mock_refined_assessment_response: str,
        mock_low_score_response: str,
        mock_high_score_response: str,
    ) -> None:
        """Verify full feedback loop flow: Assess -> Judge -> Refine -> Judge."""

        # Sequence of LLM calls:
        # 1. Initial Assessment
        # 2. Judge (4 metrics) -> all low
        # 3. Refinement
        # 4. Judge (4 metrics) -> all high

        responses = [
            mock_assessment_response,          # 1. Assess
            mock_low_score_response,           # 2. Judge Metric 1
            mock_low_score_response,           #    Judge Metric 2
            mock_low_score_response,           #    Judge Metric 3
            mock_low_score_response,           #    Judge Metric 4
            mock_refined_assessment_response,  # 3. Refine
            mock_high_score_response,          # 4. Judge Metric 1
            mock_high_score_response,          #    Judge Metric 2
            mock_high_score_response,          #    Judge Metric 3
            mock_high_score_response,          #    Judge Metric 4
        ]

        client = MockLLMClient(chat_responses=responses)

        qual_agent = QualitativeAssessmentAgent(client)
        judge_agent = JudgeAgent(client)

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
        assert len(result.history) == 2 # Initial + 1 refinement

        # Verify initial was bad
        initial_eval = result.history[0][1]
        assert initial_eval.average_score == 2.0

        # Verify final was good
        final_eval = result.final_evaluation
        assert final_eval.all_acceptable
