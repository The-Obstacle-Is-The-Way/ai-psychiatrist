"""Tests for meta-review agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

from ai_psychiatrist.agents.meta_review import MetaReviewAgent
from ai_psychiatrist.agents.output_models import MetaReviewOutput
from ai_psychiatrist.config import ModelSettings, PydanticAISettings
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
    def mock_severity_response(self) -> str:
        """Mock response with severity and explanation."""
        return """Based on the assessments provided, I have analyzed the participant's condition.

<severity>2</severity>
<explanation>The participant shows moderate depressive symptoms. The PHQ-8 scores indicate \
clinically significant depression with symptoms present more than half the days. The \
qualitative assessment reveals social stressors and biological predisposition.</explanation>
"""

    @pytest.mark.asyncio
    async def test_review_returns_meta_review(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
        mock_severity_response: str,
    ) -> None:
        """Should return MetaReview entity with correct fields."""
        mock_client = MockLLMClient(chat_responses=[mock_severity_response])
        agent = MetaReviewAgent(llm_client=mock_client)

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
            ("<severity>0</severity>", SeverityLevel.MINIMAL),
            ("<severity>1</severity>", SeverityLevel.MILD),
            ("<severity>2</severity>", SeverityLevel.MODERATE),
            ("<severity>3</severity>", SeverityLevel.MOD_SEVERE),
            ("<severity>4</severity>", SeverityLevel.SEVERE),
        ]

        for response, expected_severity in test_cases:
            mock_client = MockLLMClient(
                chat_responses=[f"{response}<explanation>Test</explanation>"]
            )
            agent = MetaReviewAgent(llm_client=mock_client)

            meta_review = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

            assert meta_review.severity == expected_severity

    @pytest.mark.asyncio
    async def test_review_clamps_out_of_range_severity(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should clamp severity values outside 0-4 range."""
        # Test value above 4
        mock_client = MockLLMClient(
            chat_responses=["<severity>5</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.severity == SeverityLevel.SEVERE  # Clamped to 4

        # Test negative value
        mock_client = MockLLMClient(
            chat_responses=["<severity>-1</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.severity == SeverityLevel.MINIMAL  # Clamped to 0

    @pytest.mark.asyncio
    async def test_review_falls_back_on_parse_failure(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should fall back to quantitative severity on parse failure."""
        # Response without proper severity tag
        mock_client = MockLLMClient(
            chat_responses=["The patient seems fine.<explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        # Should use quantitative severity (8 = MILD)
        assert meta_review.severity == sample_quantitative.severity

    @pytest.mark.asyncio
    async def test_review_extracts_explanation(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should extract explanation from response."""
        explanation_text = "This is the detailed clinical explanation."
        mock_client = MockLLMClient(
            chat_responses=[f"<severity>2</severity><explanation>{explanation_text}</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.explanation == explanation_text

    @pytest.mark.asyncio
    async def test_review_uses_raw_response_if_no_explanation_tag(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should use raw response if explanation tag is missing."""
        raw_response = "<severity>1</severity>The patient has mild symptoms."
        mock_client = MockLLMClient(chat_responses=[raw_response])
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.explanation == raw_response.strip()

    @pytest.mark.asyncio
    async def test_review_formats_quantitative_scores(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should format quantitative scores in prompt."""
        mock_client = MockLLMClient(
            chat_responses=["<severity>1</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        # Verify prompt contains formatted scores
        assert mock_client.chat_call_count == 1
        request = mock_client.chat_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")

        # Should contain score tags
        assert "<nointerest_score>" in user_message.content.lower()
        assert "</nointerest_score>" in user_message.content.lower()

    @pytest.mark.asyncio
    async def test_review_includes_transcript_in_prompt(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should include transcript in prompt."""
        mock_client = MockLLMClient(
            chat_responses=["<severity>1</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        request = mock_client.chat_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")

        assert sample_transcript.text in user_message.content

    @pytest.mark.asyncio
    async def test_review_uses_meta_review_model_settings(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """Should use ModelSettings meta-review parameters in LLM call."""
        model_settings = ModelSettings(
            meta_review_model="meta-review-model",
            temperature=0.7,
        )
        mock_client = MockLLMClient(
            chat_responses=["<severity>1</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client, model_settings=model_settings)

        await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert mock_client.chat_call_count == 1
        request = mock_client.chat_requests[0]
        assert request.model == "meta-review-model"
        assert request.temperature == 0.7

    @pytest.mark.asyncio
    async def test_review_is_mdd_property(
        self,
        sample_transcript: Transcript,
        sample_qualitative: QualitativeAssessment,
        sample_quantitative: PHQ8Assessment,
    ) -> None:
        """is_mdd should reflect severity >= MODERATE."""
        # Test MDD case (severity >= 2)
        mock_client = MockLLMClient(
            chat_responses=["<severity>2</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

        meta_review = await agent.review(
            transcript=sample_transcript,
            qualitative=sample_qualitative,
            quantitative=sample_quantitative,
        )

        assert meta_review.is_mdd is True

        # Test non-MDD case (severity < 2)
        mock_client = MockLLMClient(
            chat_responses=["<severity>1</severity><explanation>Test</explanation>"]
        )
        agent = MetaReviewAgent(llm_client=mock_client)

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
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://localhost:11434",
            )
            result = await agent.review(
                transcript=sample_transcript,
                qualitative=sample_qualitative,
                quantitative=sample_quantitative,
            )

        mock_factory.assert_called_once()
        mock_agent.run.assert_called_once()
        assert result.severity == SeverityLevel.MOD_SEVERE
        assert result.explanation == "Pydantic AI Explanation"
