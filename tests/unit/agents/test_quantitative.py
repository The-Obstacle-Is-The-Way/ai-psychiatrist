"""Tests for quantitative assessment agent.

Tests verify the agent correctly:
- Predicts PHQ-8 scores (0-3) for all 8 items
- Supports zero-shot and few-shot modes
- Extracts evidence with keyword backfill
- Parses JSON responses with multi-level repair
- Handles N/A scores for insufficient evidence
- Calculates total score and severity correctly
"""

from __future__ import annotations

import json

import pytest

from ai_psychiatrist.agents.prompts.quantitative import (
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
from ai_psychiatrist.config.domain_constants import DOMAIN_KEYWORDS
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item, SeverityLevel
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient
from tests.fixtures.mock_llm import MockLLMClient


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

# Sample scoring response (pure JSON now)
SAMPLE_SCORING_RESPONSE = json.dumps(
    {
        "PHQ8_NoInterest": {
            "evidence": "I don't really enjoy things anymore",
            "reason": "Direct statement of anhedonia",
            "score": 2,
        },
        "PHQ8_Depressed": {
            "evidence": "Feeling pretty down lately",
            "reason": "Reports depressed mood",
            "score": 1,
        },
        "PHQ8_Sleep": {
            "evidence": "I can't sleep most nights, wake up at 3am",
            "reason": "Clear sleep disturbance",
            "score": 2,
        },
        "PHQ8_Tired": {
            "evidence": "I'm always tired",
            "reason": "Reports constant fatigue",
            "score": 2,
        },
        "PHQ8_Appetite": {
            "evidence": "No relevant evidence found",
            "reason": "Appetite not discussed",
            "score": "N/A",
        },
        "PHQ8_Failure": {
            "evidence": "I feel like a failure",
            "reason": "Direct negative self-perception",
            "score": 1,
        },
        "PHQ8_Concentrating": {
            "evidence": "Can't focus on anything",
            "reason": "Reports concentration issues",
            "score": 2,
        },
        "PHQ8_Moving": {
            "evidence": "No relevant evidence found",
            "reason": "Psychomotor changes not mentioned",
            "score": "N/A",
        },
    }
)

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
        """Create mock LLM client with sample responses."""
        return MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, SAMPLE_SCORING_RESPONSE])

    @pytest.mark.asyncio
    async def test_assess_returns_all_items(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should include all 8 PHQ-8 items."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        assert isinstance(result, PHQ8Assessment)
        assert len(result.items) == 8
        for item in PHQ8Item.all_items():
            assert item in result.items

    @pytest.mark.asyncio
    async def test_assess_returns_valid_scores(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should have valid scores (0-3 or None)."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        for item, assessment in result.items.items():
            assert isinstance(assessment, ItemAssessment)
            assert assessment.item == item
            if assessment.score is not None:
                assert 0 <= assessment.score <= 3

    @pytest.mark.asyncio
    async def test_assess_sets_correct_participant_id(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should have correct participant_id."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        assert result.participant_id == 300

    @pytest.mark.asyncio
    async def test_assess_sets_correct_mode(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should record the mode used."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        assert result.mode == AssessmentMode.ZERO_SHOT

    @pytest.mark.asyncio
    async def test_assess_calculates_total_score(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should calculate total score correctly."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        # N/A scores contribute 0 to total (sum = 10)
        # Note: In new implementation, we rely on robust JSON parsing.
        # If any item failed to parse (e.g. smart quotes issue in mock response),
        # it might be None. We assert total_score matches expectation from SAMPLE_SCORING_RESPONSE.
        # SAMPLE_SCORING_RESPONSE: 2 + 1 + 2 + 2 + N/A + 1 + 2 + N/A = 10
        assert result.total_score == 10

    @pytest.mark.asyncio
    async def test_assess_determines_severity(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should determine correct severity level."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        # Total score 10 = MODERATE severity
        assert result.severity == SeverityLevel.MODERATE

    @pytest.mark.asyncio
    async def test_assess_counts_na_items(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should count N/A items correctly."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        # 2 items have N/A (Appetite, Moving)
        assert result.na_count == 2
        assert result.available_count == 6

    @pytest.mark.asyncio
    async def test_assess_includes_evidence(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment items should include evidence text."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        for assessment in result.items.values():
            assert assessment.evidence
            assert isinstance(assessment.evidence, str)

    @pytest.mark.asyncio
    async def test_assess_includes_reasoning(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment items should include reasoning."""
        agent = QuantitativeAssessmentAgent(
            llm_client=mock_client,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        for assessment in result.items.values():
            assert assessment.reason
            assert isinstance(assessment.reason, str)


class TestQuantitativeAgentParsing:
    """Tests for JSON response parsing."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(participant_id=123, text=SAMPLE_TRANSCRIPT_TEXT)

    @pytest.mark.asyncio
    async def test_parses_json_without_answer_tags(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should parse bare JSON without <answer> tags."""
        bare_json = json.dumps(
            {
                "PHQ8_NoInterest": {"evidence": "test", "reason": "test", "score": 1},
                "PHQ8_Depressed": {"evidence": "test", "reason": "test", "score": 0},
                "PHQ8_Sleep": {"evidence": "test", "reason": "test", "score": "N/A"},
                "PHQ8_Tired": {"evidence": "test", "reason": "test", "score": 2},
                "PHQ8_Appetite": {"evidence": "test", "reason": "test", "score": 3},
                "PHQ8_Failure": {"evidence": "test", "reason": "test", "score": "N/A"},
                "PHQ8_Concentrating": {"evidence": "test", "reason": "test", "score": 1},
                "PHQ8_Moving": {"evidence": "test", "reason": "test", "score": 0},
            }
        )
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, bare_json])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.items[PHQ8Item.NO_INTEREST].score == 1
        assert result.items[PHQ8Item.SLEEP].score is None

    @pytest.mark.asyncio
    async def test_parses_json_with_markdown_code_block(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should parse JSON wrapped in markdown code block."""
        markdown_json = """Some text here...
```json
{
    "PHQ8_NoInterest": {"evidence": "test", "reason": "test", "score": 2},
    "PHQ8_Depressed": {"evidence": "test", "reason": "test", "score": 1},
    "PHQ8_Sleep": {"evidence": "test", "reason": "test", "score": 0},
    "PHQ8_Tired": {"evidence": "test", "reason": "test", "score": 3},
    "PHQ8_Appetite": {"evidence": "test", "reason": "test", "score": "N/A"},
    "PHQ8_Failure": {"evidence": "test", "reason": "test", "score": 1},
    "PHQ8_Concentrating": {"evidence": "test", "reason": "test", "score": 2},
    "PHQ8_Moving": {"evidence": "test", "reason": "test", "score": "N/A"}
}
```"""
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, markdown_json])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.items[PHQ8Item.NO_INTEREST].score == 2

    @pytest.mark.skip(reason="Legacy heuristic parsing removed in favor of robust JSON mode (BUG-002)")
    @pytest.mark.asyncio
    async def test_handles_trailing_commas(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should handle JSON with trailing commas."""
        # NOTE: This test is skipped because we removed the manual regex fixups
        # in favor of relying on the LLM provider's JSON mode which should guarantee
        # valid JSON. If we want to support this in unit tests, we'd need to mock
        # json.loads to accept it or re-add the fixup logic if we don't trust the provider.
        pass

    @pytest.mark.skip(reason="Legacy heuristic parsing removed in favor of robust JSON mode (BUG-002)")
    @pytest.mark.asyncio
    async def test_handles_smart_quotes(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should handle JSON with smart quotes."""
        # NOTE: Skipped for same reason as trailing commas. Robust JSON mode
        # implies the LLM outputs valid JSON quotes.
        pass

    @pytest.mark.asyncio
    async def test_handles_missing_items_with_defaults(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should fill missing items with N/A."""
        partial_json = json.dumps(
            {
                "PHQ8_NoInterest": {"evidence": "test", "reason": "test", "score": 2},
                # Missing all other items
            }
        )
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, partial_json])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        assert result.items[PHQ8Item.NO_INTEREST].score == 2
        # All missing items should have default values
        assert result.items[PHQ8Item.DEPRESSED].score is None
        assert result.items[PHQ8Item.DEPRESSED].evidence == "No relevant evidence found"

    @pytest.mark.asyncio
    async def test_falls_back_to_empty_on_malformed_json(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should return empty skeleton when JSON is completely broken."""
        # Mock client that returns garbage for both evidence and scoring,
        # plus the repair attempt
        client = MockLLMClient(
            chat_responses=[
                "garbage evidence",
                "this is not json at all {{{",
                "repair also failed ]][[",
            ]
        )
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(sample_transcript)

        # Should have all 8 items with N/A scores
        assert len(result.items) == 8
        for item in result.items.values():
            assert item.score is None


class TestKeywordBackfill:
    """Tests for keyword backfill functionality."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create transcript with keyword-rich text."""
        return Transcript(
            participant_id=456,
            text="""Ellie: How are you sleeping?
Participant: I can't fall asleep at night. I'm exhausted all the time.
Ellie: And your appetite?
Participant: I've lost weight recently. Don't bother eating much.
Ellie: How is your concentration?
Participant: I forgot what I was doing multiple times today.""",
        )

    @pytest.mark.asyncio
    async def test_backfill_adds_missed_evidence(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Keyword backfill should add evidence when LLM misses it."""
        # LLM returns empty evidence
        empty_evidence = json.dumps({k: [] for k in DOMAIN_KEYWORDS})
        scoring_response = json.dumps(
            {k: {"evidence": "test", "reason": "test", "score": 0} for k in DOMAIN_KEYWORDS}
        )
        client = MockLLMClient(chat_responses=[empty_evidence, scoring_response])
        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)

        # Access internal method to test backfill
        evidence = await agent._extract_evidence(sample_transcript.text)

        # Should have found evidence via keywords
        assert len(evidence["PHQ8_Sleep"]) > 0
        assert len(evidence["PHQ8_Tired"]) > 0
        assert len(evidence["PHQ8_Appetite"]) > 0
        assert len(evidence["PHQ8_Concentrating"]) > 0


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
        # In new version we don't use <answer> tags, but pure JSON
        assert "Return ONLY a valid JSON object" in prompt

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


class TestDomainKeywords:
    """Tests for domain keyword definitions."""

    def test_all_phq8_domains_have_keywords(self) -> None:
        """All 8 PHQ-8 domains should have keyword lists."""
        expected_domains = [
            "PHQ8_NoInterest",
            "PHQ8_Depressed",
            "PHQ8_Sleep",
            "PHQ8_Tired",
            "PHQ8_Appetite",
            "PHQ8_Failure",
            "PHQ8_Concentrating",
            "PHQ8_Moving",
        ]
        for domain in expected_domains:
            assert domain in DOMAIN_KEYWORDS
            assert isinstance(DOMAIN_KEYWORDS[domain], list)
            assert len(DOMAIN_KEYWORDS[domain]) > 0

    def test_keywords_are_lowercase(self) -> None:
        """Keywords should be lowercase for case-insensitive matching."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {domain} not lowercase"


class TestAgentProtocol:
    """Tests for SimpleChatClient protocol compatibility."""

    def test_mock_client_implements_protocol(self) -> None:
        """MockLLMClient should implement SimpleChatClient protocol."""
        client = MockLLMClient()
        assert isinstance(client, SimpleChatClient)

    @pytest.mark.asyncio
    async def test_agent_works_with_protocol(self) -> None:
        """Agent should work with any SimpleChatClient implementation."""

        class CustomClient:
            """Custom chat client for testing."""

            async def simple_chat(
                self,
                user_prompt: str,  # noqa: ARG002
                system_prompt: str = "",  # noqa: ARG002
                model: str | None = None,  # noqa: ARG002
                temperature: float = 0.2,  # noqa: ARG002
                top_k: int = 20,  # noqa: ARG002
                top_p: float = 0.8,  # noqa: ARG002
                response_format: str | None = None,  # noqa: ARG002
            ) -> str:
                return json.dumps(
                    {k: {"evidence": "test", "reason": "test", "score": 1} for k in DOMAIN_KEYWORDS}
                )

        client = CustomClient()
        assert isinstance(client, SimpleChatClient)

        agent = QuantitativeAssessmentAgent(llm_client=client, mode=AssessmentMode.ZERO_SHOT)
        transcript = Transcript(participant_id=1, text="Test content")
        result = await agent.assess(transcript)

        assert result.total_score == 8  # All 8 items scored 1


class TestFewShotMode:
    """Tests for few-shot mode functionality."""

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(participant_id=789, text=SAMPLE_TRANSCRIPT_TEXT)

    @pytest.mark.asyncio
    async def test_few_shot_without_embedding_service_works(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Few-shot mode without embedding service should still work (no references)."""
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, SAMPLE_SCORING_RESPONSE])
        agent = QuantitativeAssessmentAgent(
            llm_client=client,
            embedding_service=None,
            mode=AssessmentMode.FEW_SHOT,
        )
        result = await agent.assess(sample_transcript)

        # Should complete but without reference examples
        assert result.mode == AssessmentMode.FEW_SHOT
        assert len(result.items) == 8

    @pytest.mark.asyncio
    async def test_zero_shot_does_not_require_embedding_service(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Zero-shot mode should work without embedding service."""
        client = MockLLMClient(chat_responses=[SAMPLE_EVIDENCE_RESPONSE, SAMPLE_SCORING_RESPONSE])
        agent = QuantitativeAssessmentAgent(
            llm_client=client,
            embedding_service=None,
            mode=AssessmentMode.ZERO_SHOT,
        )
        result = await agent.assess(sample_transcript)

        assert result.mode == AssessmentMode.ZERO_SHOT
        assert len(result.items) == 8
