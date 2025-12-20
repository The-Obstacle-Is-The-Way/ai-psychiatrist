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

import json

import pytest

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import FeedbackLoopSettings
from ai_psychiatrist.domain.entities import (
    PHQ8Assessment,
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item, SeverityLevel
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService
from tests.fixtures.mock_llm import MockLLMClient

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


# Sample qualitative assessment response
SAMPLE_QUALITATIVE_RESPONSE = """
<assessment>
The participant presents with multiple symptoms consistent with moderate depression.
Key concerns include anhedonia, sleep disturbances, fatigue, decreased appetite,
negative self-perception, concentration difficulties, and depressed mood.
</assessment>

<PHQ8_symptoms>
- Anhedonia: "I don't really enjoy things anymore" - Several days
- Sleep issues: "I can't fall asleep most nights" - Nearly every day
- Fatigue: "I'm always tired" - Nearly every day
- Appetite changes: "I've lost some weight" - More than half days
- Negative self-view: "I feel like a failure" - Several days
- Concentration: "My memory is shot" - More than half days
- Depressed mood: "Feeling pretty down lately" - More than half days
</PHQ8_symptoms>

<social_factors>
No specific social factors mentioned in this transcript.
</social_factors>

<biological_factors>
Sleep disturbance patterns suggest possible circadian rhythm disruption.
</biological_factors>

<risk_factors>
Expression of hopelessness noted: "Things seem pretty hopeless sometimes"
No active suicidal ideation reported.
</risk_factors>

<exact_quotes>
- "I don't really enjoy things anymore"
- "I'm always tired"
- "I can't fall asleep most nights"
- "I've lost some weight"
- "I feel like a failure sometimes"
- "My memory is shot"
- "Feeling pretty down lately"
</exact_quotes>
"""

# Sample judge high score response
SAMPLE_JUDGE_HIGH_RESPONSE = """Explanation: The assessment is thorough, specific, and well-supported by evidence.
Score: 5"""

# Sample quantitative evidence response
SAMPLE_QUANT_EVIDENCE_RESPONSE = json.dumps(
    {
        "PHQ8_NoInterest": [
            "I don't really enjoy things anymore.",
            "Activities that used to be fun just feel like a chore now.",
        ],
        "PHQ8_Depressed": ["Feeling pretty down lately.", "Things seem pretty hopeless sometimes."],
        "PHQ8_Sleep": [
            "I can't fall asleep most nights.",
            "Even when I sleep, I wake up exhausted.",
        ],
        "PHQ8_Tired": ["I'm always tired."],
        "PHQ8_Appetite": ["I've lost some weight.", "Don't really feel like eating most days."],
        "PHQ8_Failure": ["I feel like a failure sometimes.", "Like I'm letting everyone down."],
        "PHQ8_Concentrating": [
            "My memory is shot.",
            "I forgot what I was doing multiple times today.",
            "Can't focus on anything.",
        ],
        "PHQ8_Moving": [],
    }
)

# Sample quantitative scoring response
SAMPLE_QUANT_SCORING_RESPONSE = """<thinking>
Analyzing each PHQ-8 item based on transcript evidence...
</thinking>

<answer>
{
    "PHQ8_NoInterest": {"evidence": "I don't really enjoy things anymore", "reason": "Clear statement of anhedonia", "score": 2},
    "PHQ8_Depressed": {"evidence": "Feeling pretty down lately", "reason": "Reports depressed mood and hopelessness", "score": 2},
    "PHQ8_Sleep": {"evidence": "I can't fall asleep most nights", "reason": "Significant sleep disturbance", "score": 3},
    "PHQ8_Tired": {"evidence": "I'm always tired", "reason": "Constant fatigue reported", "score": 3},
    "PHQ8_Appetite": {"evidence": "I've lost some weight", "reason": "Appetite changes with weight loss", "score": 2},
    "PHQ8_Failure": {"evidence": "I feel like a failure sometimes", "reason": "Negative self-perception", "score": 1},
    "PHQ8_Concentrating": {"evidence": "My memory is shot", "reason": "Clear concentration difficulties", "score": 2},
    "PHQ8_Moving": {"evidence": "No relevant evidence found", "reason": "Psychomotor changes not discussed", "score": "N/A"}
}
</answer>"""


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
    def qualitative_mock_client(self) -> MockLLMClient:
        """Create mock client for qualitative path."""
        return MockLLMClient(
            chat_responses=[
                SAMPLE_QUALITATIVE_RESPONSE,  # Assess
                SAMPLE_JUDGE_HIGH_RESPONSE,  # Judge Metric 1
                SAMPLE_JUDGE_HIGH_RESPONSE,  # Judge Metric 2
                SAMPLE_JUDGE_HIGH_RESPONSE,  # Judge Metric 3
                SAMPLE_JUDGE_HIGH_RESPONSE,  # Judge Metric 4
            ]
        )

    @pytest.fixture
    def quantitative_mock_client(self) -> MockLLMClient:
        """Create mock client for quantitative path."""
        return MockLLMClient(
            chat_responses=[
                SAMPLE_QUANT_EVIDENCE_RESPONSE,  # Evidence extraction
                SAMPLE_QUANT_SCORING_RESPONSE,  # PHQ-8 scoring
            ]
        )

    @pytest.mark.asyncio
    async def test_qualitative_path_produces_valid_assessment(
        self,
        qualitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Qualitative path should produce complete assessment."""
        qual_agent = QualitativeAssessmentAgent(qualitative_mock_client)
        judge_agent = JudgeAgent(qualitative_mock_client)

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
        quantitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should produce PHQ-8 scores."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quantitative_mock_client,
            mode=AssessmentMode.ZERO_SHOT,
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
        qualitative_mock_client: MockLLMClient,
        quantitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Both paths should run independently and produce valid outputs."""
        # Qualitative path
        qual_agent = QualitativeAssessmentAgent(qualitative_mock_client)
        judge_agent = JudgeAgent(qualitative_mock_client)
        settings = FeedbackLoopSettings(enabled=True, max_iterations=1, score_threshold=3)
        qual_service = FeedbackLoopService(qual_agent, judge_agent, settings)

        # Quantitative path
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quantitative_mock_client,
            mode=AssessmentMode.ZERO_SHOT,
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
        quantitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should calculate correct total score."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quantitative_mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )

        assessment = await quant_agent.assess(sample_transcript)

        # Total score: 2+2+3+3+2+1+2+0(N/A) = 15
        assert assessment.total_score == 15

        # 15 = MOD_SEVERE
        assert assessment.severity == SeverityLevel.MOD_SEVERE

    @pytest.mark.asyncio
    async def test_quantitative_na_count(
        self,
        quantitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Quantitative path should track N/A counts correctly."""
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quantitative_mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )

        assessment = await quant_agent.assess(sample_transcript)

        # Only Moving is N/A
        assert assessment.na_count == 1
        assert assessment.available_count == 7

    @pytest.mark.asyncio
    async def test_outputs_ready_for_meta_review(
        self,
        qualitative_mock_client: MockLLMClient,
        quantitative_mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Both path outputs should have data needed for meta-review."""
        # Qualitative path
        qual_agent = QualitativeAssessmentAgent(qualitative_mock_client)
        judge_agent = JudgeAgent(qualitative_mock_client)
        settings = FeedbackLoopSettings(enabled=True, max_iterations=1, score_threshold=3)
        qual_service = FeedbackLoopService(qual_agent, judge_agent, settings)

        # Quantitative path
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quantitative_mock_client,
            mode=AssessmentMode.ZERO_SHOT,
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


class TestCrossPathConsistency:
    """Tests for cross-path consistency checks from Spec 09.5."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=300,
            text=SAMPLE_DAIC_TRANSCRIPT,
        )

    @pytest.mark.asyncio
    async def test_both_paths_cover_all_symptoms(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Both paths should cover all 8 PHQ-8 symptoms."""
        # Qualitative path covers symptoms via PHQ8_symptoms field
        qual_client = MockLLMClient(
            chat_responses=[
                SAMPLE_QUALITATIVE_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
            ]
        )
        qual_agent = QualitativeAssessmentAgent(qual_client)
        qual_result = await qual_agent.assess(sample_transcript)

        # Quantitative path covers all 8 explicitly
        quant_client = MockLLMClient(
            chat_responses=[SAMPLE_QUANT_EVIDENCE_RESPONSE, SAMPLE_QUANT_SCORING_RESPONSE]
        )
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quant_client, mode=AssessmentMode.ZERO_SHOT
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
        sample_transcript: Transcript,
    ) -> None:
        """Paths should not share state (independent LLM clients)."""
        qual_client = MockLLMClient(
            chat_responses=[
                SAMPLE_QUALITATIVE_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
                SAMPLE_JUDGE_HIGH_RESPONSE,
            ]
        )
        quant_client = MockLLMClient(
            chat_responses=[SAMPLE_QUANT_EVIDENCE_RESPONSE, SAMPLE_QUANT_SCORING_RESPONSE]
        )

        qual_agent = QualitativeAssessmentAgent(qual_client)
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=quant_client, mode=AssessmentMode.ZERO_SHOT
        )

        # Run both
        await qual_agent.assess(sample_transcript)
        await quant_agent.assess(sample_transcript)

        # Verify call counts are independent
        # Qualitative makes 1 call for assessment
        assert qual_client.chat_call_count == 1

        # Quantitative makes 2 calls (evidence + scoring)
        assert quant_client.chat_call_count == 2


class TestSeverityMapping:
    """Tests for severity level mapping per paper."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(participant_id=1, text="Test transcript content")

    @pytest.mark.asyncio
    async def test_minimal_severity(self, sample_transcript: Transcript) -> None:
        """Total 0-4 should map to MINIMAL."""
        scoring_response = json.dumps(
            {
                f"PHQ8_{item.value}": {"evidence": "test", "reason": "test", "score": 0}
                for item in PHQ8Item.all_items()
            }
        )
        client = MockLLMClient(chat_responses=["{}", scoring_response])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.total_score == 0
        assert result.severity == SeverityLevel.MINIMAL

    @pytest.mark.asyncio
    async def test_mild_severity(self, sample_transcript: Transcript) -> None:
        """Total 5-9 should map to MILD."""
        # Create scores that sum to 7
        scores = [1, 1, 1, 1, 1, 1, 1, 0]  # Sum = 7
        scoring_data = {}
        for i, item in enumerate(PHQ8Item.all_items()):
            scoring_data[f"PHQ8_{item.value}"] = {
                "evidence": "test",
                "reason": "test",
                "score": scores[i],
            }
        scoring_response = json.dumps(scoring_data)
        client = MockLLMClient(chat_responses=["{}", scoring_response])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.total_score == 7
        assert result.severity == SeverityLevel.MILD

    @pytest.mark.asyncio
    async def test_moderate_severity_mdd_threshold(self, sample_transcript: Transcript) -> None:
        """Total 10-14 should map to MODERATE (MDD threshold)."""
        # Create scores that sum to 10
        scores = [2, 2, 2, 2, 1, 1, 0, 0]  # Sum = 10
        scoring_data = {}
        for i, item in enumerate(PHQ8Item.all_items()):
            scoring_data[f"PHQ8_{item.value}"] = {
                "evidence": "test",
                "reason": "test",
                "score": scores[i],
            }
        scoring_response = json.dumps(scoring_data)
        client = MockLLMClient(chat_responses=["{}", scoring_response])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.total_score == 10
        assert result.severity == SeverityLevel.MODERATE
        assert result.severity.is_mdd  # MDD threshold check

    @pytest.mark.asyncio
    async def test_severe_severity(self, sample_transcript: Transcript) -> None:
        """Total 20-24 should map to SEVERE."""
        scoring_response = json.dumps(
            {
                f"PHQ8_{item.value}": {"evidence": "test", "reason": "test", "score": 3}
                for item in PHQ8Item.all_items()
            }
        )
        client = MockLLMClient(chat_responses=["{}", scoring_response])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.total_score == 24
        assert result.severity == SeverityLevel.SEVERE
