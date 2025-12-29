"""Integration tests for dual-path assessment pipeline.

Spec 09.5: Post-Quantitative Path Integration Checkpoint

This test verifies that both qualitative and quantitative assessment
paths work correctly and can integrate for meta-review processing.

Test Flow:
    Transcript CSV
        │
        ├──→ TranscriptService ──→ QualitativeAgent ──→ JudgeAgent ──→ Qualitative Assessment
        │                                                                      │
        └──→ EmbeddingService ──→ QuantitativeAgent ───────────────────────────┤
                                                                               ▼
                                                                  [Ready for Meta-Review]
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.output_models import (
    EvidenceOutput,
    JudgeMetricOutput,
    QualitativeOutput,
    QuantitativeOutput,
)
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import FeedbackLoopSettings, PydanticAISettings
from ai_psychiatrist.domain.entities import (
    PHQ8Assessment,
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item, SeverityLevel
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService
from tests.fixtures.mock_llm import MockLLMClient

if TYPE_CHECKING:
    from collections.abc import Generator

# Sample transcript representing a DAIC-WOZ interview
SAMPLE_DAIC_TRANSCRIPT = """Ellie: Hello, my name is Ellie. I'm here to learn about you. I'll ask a few questions to get us started.

Ellie: How are you doing today?
Participant: I'm okay, I guess. Not great.

Ellie: What's been on your mind lately?
Participant: I don't really enjoy things anymore. Activities that used to be fun just feel like a chore now.

Ellie: Can you tell me more about that?
Participant: I'm always tired. Even when I sleep, I wake up exhausted. I can't fall asleep most nights.

Ellie: How about your appetite?
Participant: I've lost some weight. Don't really feel like eating most days.

Ellie: How do you feel about yourself?
Participant: I feel like a failure sometimes. Like I'm letting everyone down.

Ellie: What about your ability to concentrate?
Participant: My memory is shot. I forgot what I was doing multiple times today. Can't focus on anything.

Ellie: How would you describe your mood overall?
Participant: Feeling pretty down lately. Things seem pretty hopeless sometimes.

Ellie: Is there anything else you'd like to share?
Participant: No, that's about it. Just tired of feeling this way."""


# Sample Pydantic AI outputs
def make_qualitative_output() -> QualitativeOutput:
    """Create sample QualitativeOutput for mocking."""
    return QualitativeOutput(
        assessment="The participant presents with multiple symptoms consistent with moderate depression.",
        phq8_symptoms=(
            "Anhedonia: clear loss of interest. "
            "Sleep issues: trouble falling asleep. "
            "Fatigue: constant tiredness reported. "
        ),
        social_factors="No specific social factors mentioned.",
        biological_factors="Sleep disturbance patterns suggest possible circadian rhythm disruption.",
        risk_factors="Expression of hopelessness noted. No active suicidal ideation.",
        exact_quotes=[
            "I don't really enjoy things anymore",
            "I'm always tired",
            "I can't fall asleep most nights",
        ],
    )


def make_judge_output_high() -> JudgeMetricOutput:
    """Create high-score JudgeMetricOutput."""
    return JudgeMetricOutput(score=5, explanation="The assessment is thorough and well-supported.")


def make_quantitative_output() -> QuantitativeOutput:
    """Create sample QuantitativeOutput for mocking."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(
            evidence="I don't really enjoy things anymore", reason="Clear anhedonia", score=2
        ),
        PHQ8_Depressed=EvidenceOutput(
            evidence="Feeling pretty down lately", reason="Depressed mood", score=2
        ),
        PHQ8_Sleep=EvidenceOutput(
            evidence="I can't fall asleep most nights", reason="Sleep disturbance", score=3
        ),
        PHQ8_Tired=EvidenceOutput(evidence="I'm always tired", reason="Fatigue", score=3),
        PHQ8_Appetite=EvidenceOutput(
            evidence="I've lost some weight", reason="Appetite changes", score=2
        ),
        PHQ8_Failure=EvidenceOutput(
            evidence="I feel like a failure", reason="Negative self-perception", score=1
        ),
        PHQ8_Concentrating=EvidenceOutput(
            evidence="My memory is shot", reason="Concentration difficulties", score=2
        ),
        PHQ8_Moving=EvidenceOutput(evidence="None", reason="Not discussed", score=None),
    )


def make_minimal_quantitative_output() -> QuantitativeOutput:
    """Create minimal score QuantitativeOutput."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=0),
    )


def make_mild_quantitative_output() -> QuantitativeOutput:
    """Create mild severity QuantitativeOutput (total=7)."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=0),
    )


def make_moderate_quantitative_output() -> QuantitativeOutput:
    """Create moderate severity QuantitativeOutput (total=10)."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=2),
        PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=2),
        PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=2),
        PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=2),
        PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=1),
        PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=0),
        PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=0),
    )


def make_severe_quantitative_output() -> QuantitativeOutput:
    """Create severe QuantitativeOutput (total=24)."""
    return QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=3),
        PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=3),
    )


@pytest.fixture
def mock_qualitative_agent() -> Generator[AsyncMock, None, None]:
    """Patch create_qualitative_agent to return a mock."""
    mock_agent = AsyncMock(spec_set=Agent)
    mock_agent.run.return_value = AsyncMock(output=make_qualitative_output())
    with patch(
        "ai_psychiatrist.agents.pydantic_agents.create_qualitative_agent",
        return_value=mock_agent,
    ):
        yield mock_agent


@pytest.fixture
def mock_judge_agent() -> Generator[AsyncMock, None, None]:
    """Patch create_judge_metric_agent to return a mock."""
    mock_agent = AsyncMock(spec_set=Agent)
    mock_agent.run.return_value = AsyncMock(output=make_judge_output_high())
    with patch(
        "ai_psychiatrist.agents.pydantic_agents.create_judge_metric_agent",
        return_value=mock_agent,
    ):
        yield mock_agent


@pytest.fixture
def mock_quantitative_agent() -> Generator[AsyncMock, None, None]:
    """Patch create_quantitative_agent to return a mock."""
    mock_agent = AsyncMock(spec_set=Agent)
    mock_agent.run.return_value = AsyncMock(output=make_quantitative_output())
    with patch(
        "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
        return_value=mock_agent,
    ):
        yield mock_agent


@pytest.mark.usefixtures("mock_qualitative_agent", "mock_judge_agent", "mock_quantitative_agent")
class TestDualPathPipeline:
    """Integration tests for the dual-path assessment pipeline."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample DAIC-WOZ style transcript."""
        return Transcript(
            participant_id=300,
            text=SAMPLE_DAIC_TRANSCRIPT,
        )

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_qualitative_path_produces_valid_assessment(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Qualitative path should produce complete assessment."""
        qual_agent = QualitativeAssessmentAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        judge_agent = JudgeAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        settings = FeedbackLoopSettings(
            enabled=True,
            max_iterations=1,
            score_threshold=3,
        )

        service = FeedbackLoopService(qual_agent, judge_agent, settings)
        result = await service.run(sample_transcript)

        # Verify qualitative assessment
        assert isinstance(result.final_assessment, QualitativeAssessment)
        assert result.final_assessment.participant_id == 300
        assert result.final_assessment.overall
        assert result.final_assessment.phq8_symptoms
        assert len(result.final_assessment.supporting_quotes) > 0

        # Verify evaluation
        assert isinstance(result.final_evaluation, QualitativeEvaluation)
        assert result.final_evaluation.all_acceptable

    @pytest.mark.asyncio
    async def test_quantitative_path_produces_valid_assessment(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should produce PHQ-8 scores."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        assessment = await quant_agent.assess(sample_transcript)

        # Verify PHQ-8 assessment
        assert isinstance(assessment, PHQ8Assessment)
        assert assessment.participant_id == 300
        assert len(assessment.items) == 8

        # Verify all items present
        for item in PHQ8Item.all_items():
            assert item in assessment.items

        # Verify scores are valid
        for _item, item_assessment in assessment.items.items():
            if item_assessment.score is not None:
                assert 0 <= item_assessment.score <= 3

    @pytest.mark.asyncio
    async def test_dual_path_runs_independently(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Both paths should run independently and produce valid outputs."""
        # Qualitative path
        qual_agent = QualitativeAssessmentAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        judge_agent = JudgeAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        settings = FeedbackLoopSettings(enabled=True, max_iterations=1, score_threshold=3)
        qual_service = FeedbackLoopService(qual_agent, judge_agent, settings)

        # Quantitative path
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        # Run both paths
        qual_result = await qual_service.run(sample_transcript)
        quant_result = await quant_agent.assess(sample_transcript)

        # Verify both succeeded
        assert qual_result.final_assessment is not None
        assert quant_result is not None

        # Verify both use same participant_id
        assert qual_result.final_assessment.participant_id == quant_result.participant_id

    @pytest.mark.asyncio
    async def test_quantitative_total_score_calculation(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should calculate correct total score."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        assessment = await quant_agent.assess(sample_transcript)

        # Total score: 2+2+3+3+2+1+2+0(N/A) = 15
        assert assessment.total_score == 15

        # 15 = MOD_SEVERE
        assert assessment.severity == SeverityLevel.MOD_SEVERE

    @pytest.mark.asyncio
    async def test_quantitative_na_count(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should track N/A counts correctly."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        assessment = await quant_agent.assess(sample_transcript)

        # Only Moving is N/A
        assert assessment.na_count == 1
        assert assessment.available_count == 7

    @pytest.mark.asyncio
    async def test_outputs_ready_for_meta_review(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Both path outputs should have data needed for meta-review."""
        # Qualitative path
        qual_agent = QualitativeAssessmentAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        judge_agent = JudgeAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        settings = FeedbackLoopSettings(enabled=True, max_iterations=1, score_threshold=3)
        qual_service = FeedbackLoopService(qual_agent, judge_agent, settings)

        # Quantitative path
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        # Run both
        qual_result = await qual_service.run(sample_transcript)
        quant_result = await quant_agent.assess(sample_transcript)

        # Verify qualitative has required fields for meta-review
        qual_assessment = qual_result.final_assessment
        assert qual_assessment.id is not None  # UUID for meta-review reference
        assert qual_assessment.overall  # Summary text
        assert qual_assessment.phq8_symptoms  # Symptom analysis

        # Verify quantitative has required fields for meta-review
        assert quant_result.id is not None  # UUID for meta-review reference
        assert quant_result.total_score >= 0  # Numeric score
        assert quant_result.severity is not None  # Severity classification
        assert quant_result.mode is not None  # Assessment mode

        # Verify item-level data available
        for _item, item_assessment in quant_result.items.items():
            assert item_assessment.evidence is not None
            assert item_assessment.reason is not None


@pytest.mark.usefixtures("mock_qualitative_agent", "mock_judge_agent", "mock_quantitative_agent")
class TestCrossPathConsistency:
    """Tests for cross-path consistency checks from Spec 09.5."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=300,
            text=SAMPLE_DAIC_TRANSCRIPT,
        )

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_both_paths_cover_all_symptoms(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Both paths should cover all 8 PHQ-8 symptoms."""
        # Qualitative path covers symptoms via PHQ8_symptoms field
        qual_agent = QualitativeAssessmentAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        qual_result = await qual_agent.assess(sample_transcript)

        # Quantitative path covers all 8 explicitly
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        quant_result = await quant_agent.assess(sample_transcript)

        # Verify qualitative mentions symptoms
        assert (
            "anhedonia" in qual_result.phq8_symptoms.lower()
            or "interest" in qual_result.phq8_symptoms.lower()
        )
        assert "sleep" in qual_result.phq8_symptoms.lower()
        assert (
            "tired" in qual_result.phq8_symptoms.lower()
            or "fatigue" in qual_result.phq8_symptoms.lower()
        )

        # Verify quantitative covers all 8
        assert len(quant_result.items) == 8

    @pytest.mark.asyncio
    async def test_no_shared_state_between_paths(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
        mock_qualitative_agent: AsyncMock,
        mock_quantitative_agent: AsyncMock,
    ) -> None:
        """Paths should not share state (independent agents)."""
        qual_agent = QualitativeAssessmentAgent(
            mock_client,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock",
        )

        # Run both
        await qual_agent.assess(sample_transcript)
        await quant_agent.assess(sample_transcript)

        # Verify agents were called independently
        mock_qualitative_agent.run.assert_called()
        mock_quantitative_agent.run.assert_called()


class TestSeverityMapping:
    """Tests for severity level mapping per paper."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(participant_id=1, text="Test transcript content")

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_minimal_severity(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Total 0-4 should map to MINIMAL."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=make_minimal_quantitative_output())

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = QuantitativeAssessmentAgent(
                llm_client=mock_client,
                mode=AssessmentMode.ZERO_SHOT,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            result = await agent.assess(sample_transcript)

        assert result.total_score == 0
        assert result.severity == SeverityLevel.MINIMAL

    @pytest.mark.asyncio
    async def test_mild_severity(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Total 5-9 should map to MILD."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=make_mild_quantitative_output())

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = QuantitativeAssessmentAgent(
                llm_client=mock_client,
                mode=AssessmentMode.ZERO_SHOT,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            result = await agent.assess(sample_transcript)

        assert result.total_score == 7
        assert result.severity == SeverityLevel.MILD

    @pytest.mark.asyncio
    async def test_moderate_severity_mdd_threshold(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Total 10-14 should map to MODERATE (MDD threshold)."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=make_moderate_quantitative_output())

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = QuantitativeAssessmentAgent(
                llm_client=mock_client,
                mode=AssessmentMode.ZERO_SHOT,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            result = await agent.assess(sample_transcript)

        assert result.total_score == 10
        assert result.severity == SeverityLevel.MODERATE
        assert result.severity.is_mdd  # MDD threshold check

    @pytest.mark.asyncio
    async def test_severe_severity(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Total 20-24 should map to SEVERE."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=make_severe_quantitative_output())

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = QuantitativeAssessmentAgent(
                llm_client=mock_client,
                mode=AssessmentMode.ZERO_SHOT,
                pydantic_ai_settings=PydanticAISettings(enabled=True),
                ollama_base_url="http://mock",
            )
            result = await agent.assess(sample_transcript)

        assert result.total_score == 24
        assert result.severity == SeverityLevel.SEVERE
