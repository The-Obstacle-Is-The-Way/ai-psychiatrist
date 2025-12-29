"""Tests for meta-review agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.meta_review import MetaReviewAgent
from ai_psychiatrist.agents.output_models import MetaReviewOutput
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import (
    PHQ8Assessment,
    QualitativeAssessment,
    Transcript,
)
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item, SeverityLevel
from ai_psychiatrist.domain.value_objects import ItemAssessment
from tests.fixtures.mock_llm import MockLLMClient


class TestMetaReviewAgent:
    """Tests for MetaReviewAgent."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text="Ellie: How are you feeling?\nParticipant: Not great, been feeling down.",
        )

    @pytest.fixture
    def sample_qualitative(self) -> QualitativeAssessment:
        """Create sample qualitative assessment."""
        return QualitativeAssessment(
            overall="Patient shows moderate depression symptoms.",
            phq8_symptoms="Reports feeling down, low energy, and sleep issues.",
            social_factors="Financial stress and relationship difficulties.",
            biological_factors="History of depression in family.",
            risk_factors="Previous episodes of depression.",
            participant_id=123,
            supporting_quotes=["I've been feeling down lately."],
        )

    @pytest.fixture
    def sample_quantitative(self) -> PHQ8Assessment:
        """Create sample quantitative assessment."""
        items = {}
        scores = [2, 2, 1, 1, 1, 0, 1, 0]  # Total = 8, Mild severity
        for i, item in enumerate(PHQ8Item.all_items()):
            items[item] = ItemAssessment(
                item=item,
                evidence="Test evidence",
                reason="Test reason",
                score=scores[i],
            )
        return PHQ8Assessment(
            items=items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=123,
        )

    @pytest.fixture
    def mock_meta_review_output(self) -> MetaReviewOutput:
        """Create valid MetaReviewOutput for mocking."""
        return MetaReviewOutput(
            severity=2,
            explanation="The participant shows moderate depressive symptoms.",
        )

    @pytest.fixture
    def mock_agent_factory(
        self, mock_meta_review_output: MetaReviewOutput
    ) -> Generator[AsyncMock, None, None]:
        """Patch create_meta_review_agent to return a mock agent."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_meta_review_output)

        patcher = patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        )
        mock = patcher.start()
        yield mock
        patcher.stop()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_review_returns_meta_review(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should return MetaReview entity with correct fields."""
        agent = MetaReviewAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.severity == SeverityLevel.MODERATE
        assert meta_review.participant_id == 123
        assert "moderate" in meta_review.explanation.lower()
        assert meta_review.quantitative_assessment_id == sample_quantitative.id
        assert meta_review.qualitative_assessment_id == sample_qualitative.id

    @pytest.mark.asyncio
    async def test_review_parses_severity_correctly(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should parse severity values 0-4 correctly."""
        test_cases = [
            (0, SeverityLevel.MINIMAL),
            (1, SeverityLevel.MILD),
            (2, SeverityLevel.MODERATE),
            (3, SeverityLevel.MOD_SEVERE),
            (4, SeverityLevel.SEVERE),
        ]

        for severity_value, expected_severity in test_cases:
            mock_output = MetaReviewOutput(severity=severity_value, explanation="Test")
            mock_agent = AsyncMock(spec_set=Agent)
            mock_agent.run.return_value = AsyncMock(output=mock_output)

            with patch(
                "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
                return_value=mock_agent,
            ):
                agent = MetaReviewAgent(
                    llm_client=MockLLMClient(),
                    pydantic_ai_settings=PydanticAISettings(enabled=True),
                    ollama_base_url="http://mock",
                )

                meta_review = await agent.review(
                    transcript=sample_transcript,
                    qualitative=sample_qualitative,
                    quantitative=sample_quantitative,
                )

                assert meta_review.severity == expected_severity

    @pytest.mark.asyncio
    async def test_review_extracts_explanation(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should extract explanation from Pydantic AI output."""
        explanation_text = "This is the detailed clinical explanation."
        mock_output = MetaReviewOutput(severity=2, explanation=explanation_text)
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_output)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        ):
            agent = MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )

            meta_review = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

            assert meta_review.explanation == explanation_text

    @pytest.mark.asyncio
    async def test_review_is_mdd_property(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """is_mdd should reflect severity >= MODERATE."""
        # Test MDD case (severity >= 2)
        mock_output_mdd = MetaReviewOutput(severity=2, explanation="Test")
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_output_mdd)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        ):
            agent = MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )

            meta_review = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

            assert meta_review.is_mdd is True

        # Test non-MDD case (severity < 2)
        mock_output_no_mdd = MetaReviewOutput(severity=1, explanation="Test")
        mock_agent.run.return_value = AsyncMock(output=mock_output_no_mdd)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        ):
            agent = MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )

            meta_review = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

            assert meta_review.is_mdd is False

    @pytest.mark.asyncio
    async def test_pydantic_ai_path_success(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should use Pydantic AI agent when enabled and configured."""
        mock_output = MetaReviewOutput(
            severity=3,
            explanation="Pydantic AI Explanation",
        )
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_output)

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        ) as mock_factory:
            agent = MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True, timeout_seconds=123.0),
                ollama_base_url="http://localhost:11434",
            )
            result = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

        mock_factory.assert_called_once()
        mock_agent.run.assert_called_once()
        assert mock_agent.run.call_args.kwargs["model_settings"]["timeout"] == 123.0
        assert result.severity == SeverityLevel.MOD_SEVERE
        assert result.explanation == "Pydantic AI Explanation"

    @pytest.mark.asyncio
    async def test_pydantic_ai_failure_raises(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should raise ValueError when Pydantic AI call fails."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.side_effect = RuntimeError("LLM timeout")

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_meta_review_agent",
            return_value=mock_agent,
        ):
            agent = MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            with pytest.raises(ValueError, match="Pydantic AI meta-review failed"):
                await agent.review(
                    transcript=sample_transcript,
                    qualitative=sample_qualitative,
                    quantitative=sample_quantitative,
                )

    def test_init_without_ollama_url_raises(self) -> None:
        """Should raise ValueError when Pydantic AI enabled but no ollama_base_url."""
        with pytest.raises(ValueError, match="ollama_base_url"):
            MetaReviewAgent(
                llm_client=MockLLMClient(),
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url=None,
            )

    def test_review_without_agent_raises(self) -> None:
        """Should raise ValueError when agent not initialized."""
        agent = MetaReviewAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=False),
        )
        transcript = Transcript(participant_id=1, text="Test")
        qualitative = QualitativeAssessment(
            overall="Test",
            phq8_symptoms="Test",
            social_factors="Test",
            biological_factors="Test",
            risk_factors="Test",
            participant_id=1,
        )
        items = {}
        for item in PHQ8Item.all_items():
            items[item] = ItemAssessment(
                item=item,
                evidence="Test",
                reason="Test",
                score=0,
            )
        quantitative = PHQ8Assessment(
            items=items,
            mode=AssessmentMode.ZERO_SHOT,
            participant_id=1,
        )

        with pytest.raises(ValueError, match="Pydantic AI review agent not initialized"):
            asyncio.get_event_loop().run_until_complete(
                agent.review(transcript, qualitative, quantitative)
            )
