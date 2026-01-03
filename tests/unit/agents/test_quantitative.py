"""Tests for quantitative assessment agent.

Tests verify the agent correctly:
- Predicts PHQ-8 scores (0-3) for all 8 items
- Supports zero-shot and few-shot modes
- Parses JSON responses with multi-level repair
- Handles N/A scores for insufficient evidence
- Calculates total score bounds and severity bounds correctly
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import Agent

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_psychiatrist.agents.output_models import EvidenceOutput, QuantitativeOutput
from ai_psychiatrist.agents.prompts.quantitative import (
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import PydanticAISettings
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item, SeverityLevel
from ai_psychiatrist.domain.value_objects import ItemAssessment
from tests.fixtures.mock_llm import MockLLMClient

pytestmark = pytest.mark.unit


def _to_smart_quotes(text: str) -> str:
    """Convert ASCII quotes to smart quotes for parsing tests."""
    left_double = True
    out: list[str] = []
    for char in text:
        if char == '"':
            out.append("\u201c" if left_double else "\u201d")
            left_double = not left_double
        elif char == "'":
            out.append("\u2019")
        else:
            out.append(char)
    return "".join(out)


# Sample evidence extraction response
SAMPLE_EVIDENCE_RESPONSE = json.dumps(
    {
        "PHQ8_NoInterest": ["I don't really enjoy things anymore."],
        "PHQ8_Depressed": ["Feeling pretty down lately."],
        "PHQ8_Sleep": ["I can't sleep most nights.", "Wake up at 3am every night."],
        "PHQ8_Tired": ["I'm always tired."],
        "PHQ8_Appetite": [],
        "PHQ8_Failure": ["I feel like a failure."],
        "PHQ8_Concentrating": ["Can't focus on anything."],
        "PHQ8_Moving": [],
    }
)

# Sample scoring response with <answer> tags
SAMPLE_SCORING_RESPONSE = """<thinking>
Let me analyze each PHQ-8 item based on the transcript evidence.

For NoInterest, the participant explicitly states they don't enjoy things anymore.
This suggests at least several days a week (score 1-2).
</thinking>

<answer>
{
    "PHQ8_NoInterest": {"evidence": "I don't really enjoy things anymore", "reason": "Direct statement of anhedonia", "score": 2},
    "PHQ8_Depressed": {"evidence": "Feeling pretty down lately", "reason": "Reports depressed mood", "score": 1},
    "PHQ8_Sleep": {"evidence": "I can't sleep most nights, wake up at 3am", "reason": "Clear sleep disturbance", "score": 2},
    "PHQ8_Tired": {"evidence": "I'm always tired", "reason": "Reports constant fatigue", "score": 2},
    "PHQ8_Appetite": {"evidence": "No relevant evidence found", "reason": "Appetite not discussed", "score": "N/A"},
    "PHQ8_Failure": {"evidence": "I feel like a failure", "reason": "Direct negative self-perception", "score": 1},
    "PHQ8_Concentrating": {"evidence": "Can't focus on anything", "reason": "Reports concentration issues", "score": 2},
    "PHQ8_Moving": {"evidence": "No relevant evidence found", "reason": "Psychomotor changes not mentioned", "score": "N/A"}
}
</answer>"""

SAMPLE_TRANSCRIPT_TEXT = """Ellie: How are you feeling today?
Participant: Not great, honestly. I don't really enjoy things anymore.
Ellie: Can you tell me more about that?
Participant: I'm always tired. I can't sleep most nights. Wake up at 3am every night.
Ellie: How about your mood overall?
Participant: Feeling pretty down lately. I feel like a failure sometimes.
Ellie: What about your ability to concentrate?
Participant: Can't focus on anything at work. My memory is shot."""


class TestQuantitativeAssessmentAgent:
    """Tests for QuantitativeAssessmentAgent."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=300,
            text=SAMPLE_TRANSCRIPT_TEXT,
        )

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client with sample responses for EVIDENCE extraction only."""
        return MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE])

    @pytest.fixture
    def mock_quantitative_output(self) -> QuantitativeOutput:
        """Create valid QuantitativeOutput object."""
        return QuantitativeOutput(
            PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=1, confidence=4),
            PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=None),  # N/A
            PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=None),  # N/A
        )

    @pytest.fixture
    def mock_agent_factory(
        self, mock_quantitative_output: QuantitativeOutput
    ) -> Generator[AsyncMock, None, None]:
        """Patch create_quantitative_agent to return a mock agent."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_quantitative_output)

        patcher = patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        )
        mock = patcher.start()
        yield mock
        patcher.stop()

    def create_agent(
        self, client: MockLLMClient, mode: AssessmentMode = AssessmentMode.ZERO_SHOT
    ) -> QuantitativeAssessmentAgent:
        """Helper to create agent with Pydantic AI enabled."""
        return QuantitativeAssessmentAgent(
            llm_client=client,
            mode=mode,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock-ollama:11434",
        )

    @pytest.mark.asyncio
    async def test_pydantic_agent_run_error_not_masked(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Exceptions from Pydantic AI should not be converted to ValueError (Spec 39)."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.side_effect = RuntimeError("boom")

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = self.create_agent(mock_client)
            with pytest.raises(RuntimeError, match="boom"):
                await agent.assess(sample_transcript)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_returns_all_items(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should include all 8 PHQ-8 items."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        assert isinstance(result, PHQ8Assessment)
        assert len(result.items) == 8
        for item in PHQ8Item.all_items():
            assert item in result.items

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_returns_valid_scores(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should have valid scores (0-3 or None)."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        for item, assessment in result.items.items():
            assert isinstance(assessment, ItemAssessment)
            assert assessment.item == item
            if assessment.score is not None:
                assert 0 <= assessment.score <= 3

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_sets_correct_participant_id(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should have correct participant_id."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        assert result.participant_id == 300

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_sets_correct_mode(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should record the mode used."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        assert result.mode == AssessmentMode.ZERO_SHOT

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_propagates_verbalized_confidence(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Verbalized confidence should be preserved per item (Spec 048)."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        assert result.items[PHQ8Item.NO_INTEREST].verbalized_confidence == 4

    @pytest.mark.asyncio
    async def test_assess_propagates_token_signals_when_available(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
        mock_quantitative_output: QuantitativeOutput,
    ) -> None:
        """Token-level confidence signals should be attached when logprobs exist (Spec 051)."""
        logprobs = [
            {
                "logprob": math.log(0.9),
                "top_logprobs": [{"logprob": math.log(0.9)}, {"logprob": math.log(0.1)}],
            }
        ]

        fake_result = AsyncMock()
        fake_result.output = mock_quantitative_output
        fake_result.response = AsyncMock()
        fake_result.response.provider_details = {"logprobs": logprobs}

        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = fake_result

        with patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        ):
            agent = self.create_agent(mock_client)
            result = await agent.assess(sample_transcript)

        item = result.items[PHQ8Item.NO_INTEREST]
        assert item.token_msp == pytest.approx(0.9)

        expected_entropy = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        assert item.token_pe == pytest.approx(expected_entropy)
        assert item.token_energy == pytest.approx(0.0)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_calculates_total_score(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should calculate total score correctly."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        # 6 items score 1, 2 items score None. Total = 6.
        assert result.total_score == 6

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_determines_severity(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Partial assessments should not emit a single severity label."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        # Scores: 6 items are scored, 2 are N/A (unknown). The severity is bounded, not determinate.
        assert result.severity is None
        assert result.severity_lower_bound == SeverityLevel.MILD

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_with_consistency_sets_modal_score_and_signals(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Consistency mode should attach modal agreement signals per item (Spec 050)."""
        agent = self.create_agent(mock_client)

        def make_items(score_no_interest: int | None) -> dict[PHQ8Item, ItemAssessment]:
            items: dict[PHQ8Item, ItemAssessment] = {}
            for phq_item in PHQ8Item.all_items():
                score = score_no_interest if phq_item == PHQ8Item.NO_INTEREST else 1
                items[phq_item] = ItemAssessment(
                    item=phq_item,
                    evidence="test",
                    reason="test",
                    score=score,
                )
            return items

        # Samples: [2, 2, 2, 1, 1] => modal=2, count=3, conf=0.6
        scores = [2, 2, 2, 1, 1]
        agent._score_items = AsyncMock(side_effect=[make_items(s) for s in scores])  # type: ignore[method-assign]

        result = await agent.assess_with_consistency(
            sample_transcript, n_samples=5, temperature=0.3
        )
        item = result.items[PHQ8Item.NO_INTEREST]
        assert item.score == 2
        assert item.consistency_modal_score == 2
        assert item.consistency_modal_count == 3
        assert item.consistency_modal_confidence == pytest.approx(0.6)
        assert item.consistency_na_rate == pytest.approx(0.0)
        assert item.consistency_samples == tuple(scores)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_with_consistency_recovers_from_sample_failure(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Consistency mode should tolerate occasional sample failures and keep running."""
        agent = self.create_agent(mock_client)

        def make_items(score_no_interest: int | None) -> dict[PHQ8Item, ItemAssessment]:
            items: dict[PHQ8Item, ItemAssessment] = {}
            for phq_item in PHQ8Item.all_items():
                score = score_no_interest if phq_item == PHQ8Item.NO_INTEREST else 1
                items[phq_item] = ItemAssessment(
                    item=phq_item,
                    evidence="test",
                    reason="test",
                    score=score,
                )
            return items

        scores = [2, 2, 2, 1, 1]
        side_effect: list[object] = [RuntimeError("boom"), *[make_items(s) for s in scores]]
        agent._score_items = AsyncMock(side_effect=side_effect)  # type: ignore[method-assign]

        result = await agent.assess_with_consistency(
            sample_transcript, n_samples=5, temperature=0.3
        )
        item = result.items[PHQ8Item.NO_INTEREST]
        assert item.score == 2
        assert item.consistency_samples == tuple(scores)
        assert agent._score_items.call_count == 6

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_counts_na_items(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment should count N/A items correctly."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        # 2 items have N/A (Appetite, Moving)
        assert result.na_count == 2
        assert result.available_count == 6

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_assess_includes_evidence(
        self, mock_client: MockLLMClient, sample_transcript: Transcript
    ) -> None:
        """Assessment items should include evidence text."""
        agent = self.create_agent(mock_client)
        result = await agent.assess(sample_transcript)

        for assessment in result.items.values():
            assert assessment.evidence == "test"
            assert isinstance(assessment.evidence, str)

    @pytest.mark.asyncio
    async def test_score_items_propagates_cancel(
        self,
        sample_transcript: Transcript,
        mock_agent_factory: AsyncMock,
    ) -> None:
        """Should propagate CancelledError without fallback."""
        mock_agent = mock_agent_factory.return_value
        mock_agent.run.side_effect = asyncio.CancelledError

        agent = self.create_agent(MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE]))

        with pytest.raises(asyncio.CancelledError):
            await agent.assess(sample_transcript)

    @pytest.mark.asyncio
    async def test_pydantic_ai_failure_preserves_exception_type(
        self,
        sample_transcript: Transcript,
        mock_agent_factory: AsyncMock,
    ) -> None:
        """Should preserve the original exception type if Pydantic AI fails (Spec 39)."""
        mock_agent = mock_agent_factory.return_value
        mock_agent.run.side_effect = RuntimeError("Something went wrong")

        agent = self.create_agent(MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE]))

        with pytest.raises(RuntimeError, match="Something went wrong"):
            await agent.assess(sample_transcript)


class TestQuantitativePrompts:
    """Tests for prompt template functions."""

    def test_make_scoring_prompt_includes_transcript(self) -> None:
        """Scoring prompt should include transcript text."""
        transcript = "Ellie: Hello\nParticipant: Hi there"
        prompt = make_scoring_prompt(transcript, "")

        assert transcript in prompt
        assert "<transcript>" in prompt
        assert "</transcript>" in prompt

    def test_make_scoring_prompt_includes_references_when_provided(self) -> None:
        """Scoring prompt should include reference bundle."""
        references = "[NoInterest]\n<Reference Examples>\nExample text\n</Reference Examples>"
        prompt = make_scoring_prompt("test transcript", references)

        assert references in prompt

    def test_make_scoring_prompt_omits_references_when_empty(self) -> None:
        """Scoring prompt should not have empty reference section."""
        prompt = make_scoring_prompt("test transcript", "")

        # Should have answer format instructions but no extra blank lines
        assert "<answer>" in prompt

    def test_make_scoring_prompt_specifies_json_format(self) -> None:
        """Scoring prompt should specify expected JSON structure."""
        prompt = make_scoring_prompt("test", "")

        assert "PHQ8_NoInterest" in prompt
        assert "evidence" in prompt
        assert "reason" in prompt
        assert "score" in prompt
        assert "N/A" in prompt

    def test_make_evidence_prompt_includes_transcript(self) -> None:
        """Evidence prompt should include transcript text."""
        transcript = "Interview content here"
        prompt = make_evidence_prompt(transcript)

        assert transcript in prompt

    def test_make_evidence_prompt_lists_all_domains(self) -> None:
        """Evidence prompt should list all PHQ-8 domains."""
        prompt = make_evidence_prompt("test")

        assert "nointerest" in prompt.lower()
        assert "depressed" in prompt.lower()
        assert "sleep" in prompt.lower()
        assert "tired" in prompt.lower()
        assert "appetite" in prompt.lower()
        assert "failure" in prompt.lower()
        assert "concentrating" in prompt.lower()
        assert "moving" in prompt.lower()

    def test_system_prompt_includes_scoring_scale(self) -> None:
        """System prompt should explain 0-3 scoring scale."""
        assert "0 = Not at all" in QUANTITATIVE_SYSTEM_PROMPT
        assert "1 = Several days" in QUANTITATIVE_SYSTEM_PROMPT
        assert "2 = More than half" in QUANTITATIVE_SYSTEM_PROMPT
        assert "3 = Nearly every day" in QUANTITATIVE_SYSTEM_PROMPT

    def test_system_prompt_mentions_na_handling(self) -> None:
        """System prompt should explain N/A handling."""
        assert "N/A" in QUANTITATIVE_SYSTEM_PROMPT
        assert "no relevant evidence" in QUANTITATIVE_SYSTEM_PROMPT.lower()


class TestFewShotMode:
    """Tests for few-shot mode functionality."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(participant_id=789, text=SAMPLE_TRANSCRIPT_TEXT)

    @pytest.fixture
    def mock_quantitative_output(self) -> QuantitativeOutput:
        """Create valid QuantitativeOutput object."""
        return QuantitativeOutput(
            PHQ8_NoInterest=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Depressed=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Sleep=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Tired=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Appetite=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Failure=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Concentrating=EvidenceOutput(evidence="test", reason="test", score=1),
            PHQ8_Moving=EvidenceOutput(evidence="test", reason="test", score=1),
        )

    @pytest.fixture
    def mock_agent_factory(
        self, mock_quantitative_output: QuantitativeOutput
    ) -> Generator[AsyncMock, None, None]:
        """Patch create_quantitative_agent to return a mock agent."""
        mock_agent = AsyncMock(spec_set=Agent)
        mock_agent.run.return_value = AsyncMock(output=mock_quantitative_output)

        patcher = patch(
            "ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent",
            return_value=mock_agent,
        )
        mock = patcher.start()
        yield mock
        patcher.stop()

    def create_agent(
        self, client: MockLLMClient, mode: AssessmentMode, embedding_service: None = None
    ) -> QuantitativeAssessmentAgent:
        return QuantitativeAssessmentAgent(
            llm_client=client,
            embedding_service=embedding_service,
            mode=mode,
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://mock-ollama:11434",
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_few_shot_without_embedding_service_works(
        self, sample_transcript: Transcript
    ) -> None:
        """Few-shot mode without embedding service should still work (no references)."""
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE])
        agent = self.create_agent(client, AssessmentMode.FEW_SHOT, embedding_service=None)

        result = await agent.assess(sample_transcript)

        # Should complete but without reference examples
        assert result.mode == AssessmentMode.FEW_SHOT
        assert len(result.items) == 8

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_agent_factory")
    async def test_zero_shot_does_not_require_embedding_service(
        self, sample_transcript: Transcript
    ) -> None:
        """Zero-shot mode should work without embedding service."""
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE])
        agent = self.create_agent(client, AssessmentMode.ZERO_SHOT, embedding_service=None)

        result = await agent.assess(sample_transcript)

        assert result.mode == AssessmentMode.ZERO_SHOT
        assert len(result.items) == 8
