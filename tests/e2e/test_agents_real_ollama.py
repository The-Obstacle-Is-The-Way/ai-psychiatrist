from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_psychiatrist.agents import (
    JudgeAgent,
    QualitativeAssessmentAgent,
    QuantitativeAssessmentAgent,
)
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, EvaluationMetric
from ai_psychiatrist.infrastructure.llm.responses import extract_score_from_text

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings
    from ai_psychiatrist.infrastructure.llm import OllamaClient


@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.slow
class TestAgentsRealOllama:
    async def test_qualitative_agent_assess_real_ollama(
        self,
        ollama_client: OllamaClient,
        app_settings: Settings,
        sample_transcript: str,
    ) -> None:
        agent = QualitativeAssessmentAgent(
            llm_client=ollama_client,
            model_settings=app_settings.model,
        )

        transcript = Transcript(participant_id=999_999, text=sample_transcript)
        assessment = await agent.assess(transcript)

        assert assessment.participant_id == transcript.participant_id
        assert assessment.overall.strip()
        assert assessment.phq8_symptoms.strip()
        assert assessment.social_factors.strip()
        assert assessment.biological_factors.strip()
        assert assessment.risk_factors.strip()

    async def test_judge_agent_evaluate_real_ollama_scores_parseable(
        self,
        ollama_client: OllamaClient,
        app_settings: Settings,
        sample_transcript: str,
    ) -> None:
        qual_agent = QualitativeAssessmentAgent(
            llm_client=ollama_client,
            model_settings=app_settings.model,
        )
        judge = JudgeAgent(
            llm_client=ollama_client,
            model_settings=app_settings.model,
        )

        transcript = Transcript(participant_id=999_999, text=sample_transcript)
        assessment = await qual_agent.assess(transcript)
        evaluation = await judge.evaluate(assessment, transcript)

        assert set(evaluation.scores.keys()) == set(EvaluationMetric.all_metrics())
        for score in evaluation.scores.values():
            assert 1 <= score.score <= 5
            assert extract_score_from_text(score.explanation) is not None

    async def test_quantitative_agent_assess_real_ollama_has_some_numeric_scores(
        self,
        ollama_client: OllamaClient,
        app_settings: Settings,
        sample_transcript: str,
    ) -> None:
        agent = QuantitativeAssessmentAgent(
            llm_client=ollama_client,
            embedding_service=None,
            mode=AssessmentMode.ZERO_SHOT,
            model_settings=app_settings.model,
        )

        transcript = Transcript(participant_id=999_999, text=sample_transcript)
        result = await agent.assess(transcript)

        assert result.participant_id == transcript.participant_id
        assert result.mode == AssessmentMode.ZERO_SHOT
        assert len(result.items) == 8

        numeric_scores = [item.score for item in result.items.values() if item.score is not None]
        assert len(numeric_scores) >= 1

        # Validate score ranges for non-N/A items (PHQ-8 valid range: 0-3)
        for item in result.items.values():
            if item.score is not None:
                assert 0 <= item.score <= 3, f"Score {item.score} out of valid PHQ-8 range [0,3]"
