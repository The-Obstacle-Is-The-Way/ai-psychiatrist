"""Tests for qualitative assessment agent.

Tests verify the agent correctly:
- Generates assessments across all 4 clinical domains
- Extracts supporting quotes from transcripts
- Parses XML responses from LLM
- Handles malformed responses gracefully
- Supports feedback-based refinement
"""

from __future__ import annotations

import pytest

from ai_psychiatrist.agents.prompts.qualitative import (
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient
from tests.fixtures.mock_llm import MockLLMClient

# Sample LLM response for fixtures
SAMPLE_LLM_RESPONSE = """
<assessment>
The participant shows signs of moderate depression with evident sleep disturbances
and low energy. They expressed feelings of hopelessness related to recent job loss.
</assessment>

<PHQ8_symptoms>
<little_interest_or_pleasure>
Participant expresses decreased interest in activities.
Frequency: Several days a week
</little_interest_or_pleasure>
<feeling_tired_little_energy>
Reports significant fatigue.
Frequency: Nearly every day
</feeling_tired_little_energy>
</PHQ8_symptoms>

<social_factors>
Lives with partner and two children. Reports strained relationship.
Currently unemployed, previously worked in tech.
</social_factors>

<biological_factors>
Family history of depression. No current medications reported.
Reports history of anxiety in college.
</biological_factors>

<risk_factors>
Expresses passive suicidal ideation. No active plan reported.
Recent job loss appears to be primary stressor.
</risk_factors>

<exact_quotes>
- "I don't really enjoy things anymore."
- "I'm always tired, can't seem to get enough sleep."
- "Things have been tense at home."
- "My mother had depression."
- "Sometimes I think it would be easier if I just didn't wake up."
</exact_quotes>
"""


SAMPLE_TRANSCRIPT_TEXT = """Ellie: How are you feeling today?
Participant: Not great, honestly. I don't really enjoy things anymore.
Ellie: Can you tell me more about that?
Participant: I'm always tired, can't seem to get enough sleep.
Ellie: How is your family life?
Participant: Things have been tense at home."""


class TestQualitativeAssessmentAgent:
    """Tests for QualitativeAssessmentAgent."""

    @pytest.fixture
    def sample_llm_response(self) -> str:
        """Sample LLM response with all XML tags."""
        return SAMPLE_LLM_RESPONSE

    @pytest.fixture
    def mock_client(self, sample_llm_response: str) -> MockLLMClient:
        """Create mock LLM client with sample response."""
        return MockLLMClient(chat_responses=[sample_llm_response])

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text=SAMPLE_TRANSCRIPT_TEXT,
        )

    @pytest.mark.asyncio
    async def test_assess_returns_all_domains(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should include all required domains."""
        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        result = await agent.assess(sample_transcript)

        assert isinstance(result, QualitativeAssessment)
        assert result.overall
        assert result.phq8_symptoms
        assert result.social_factors
        assert result.biological_factors
        assert result.risk_factors
        assert result.participant_id == 123

    @pytest.mark.asyncio
    async def test_assess_extracts_quotes(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should extract supporting quotes."""
        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        result = await agent.assess(sample_transcript)

        assert len(result.supporting_quotes) > 0
        assert any("tired" in q.lower() for q in result.supporting_quotes)

    @pytest.mark.asyncio
    async def test_assess_extracts_inline_quotes_when_no_exact_quotes(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should extract quotes embedded in domain text."""
        response = """
<assessment>
Participant reports feeling "empty and hopeless".
</assessment>
<PHQ8_symptoms>
Sleep issues: "I can't sleep most nights."
</PHQ8_symptoms>
<social_factors>
No major changes reported.
</social_factors>
<biological_factors>
No family history mentioned.
</biological_factors>
<risk_factors>
None noted.
</risk_factors>
"""
        client = MockLLMClient(chat_responses=[response])
        agent = QualitativeAssessmentAgent(llm_client=client)
        result = await agent.assess(sample_transcript)

        assert "empty and hopeless" in result.supporting_quotes
        assert any("can't sleep" in q for q in result.supporting_quotes)

    @pytest.mark.asyncio
    async def test_assess_sends_correct_prompt(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should send transcript in prompt."""
        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        await agent.assess(sample_transcript)

        assert mock_client.chat_call_count == 1
        request = mock_client.chat_requests[0]
        # User message should contain transcript
        user_msg = next(m for m in request.messages if m.role == "user")
        assert sample_transcript.text in user_msg.content

    @pytest.mark.asyncio
    async def test_assess_includes_system_prompt(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment should include system prompt."""
        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        await agent.assess(sample_transcript)

        request = mock_client.chat_requests[0]
        system_msgs = [m for m in request.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "psychiatrist" in system_msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_assess_handles_partial_response(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should handle response with missing tags."""
        partial_response = """
<assessment>
Brief assessment only.
</assessment>
<risk_factors>
Some risk noted.
</risk_factors>
"""
        client = MockLLMClient(chat_responses=[partial_response])
        agent = QualitativeAssessmentAgent(llm_client=client)
        result = await agent.assess(sample_transcript)

        assert result.overall == "Brief assessment only."
        assert result.risk_factors == "Some risk noted."
        # Missing tags should have defaults
        assert result.phq8_symptoms == "Not assessed"
        assert result.social_factors == "Not assessed"
        assert result.biological_factors == "Not assessed"

    @pytest.mark.asyncio
    async def test_assess_handles_empty_response(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should handle empty LLM response."""
        client = MockLLMClient(chat_responses=[""])
        agent = QualitativeAssessmentAgent(llm_client=client)
        result = await agent.assess(sample_transcript)

        assert result.overall == "Assessment not generated"
        assert result.participant_id == 123

    @pytest.mark.asyncio
    async def test_assess_handles_malformed_xml(
        self,
        sample_transcript: Transcript,
    ) -> None:
        """Should handle malformed XML gracefully."""
        # Malformed response with missing closing tags
        malformed = """
Some preamble text...
<assessment>
Valid assessment here.
</assessment>
<PHQ8_symptoms>
Symptoms without proper closing
<social_factors>
Nested unclosed tags
"""
        client = MockLLMClient(chat_responses=[malformed])
        agent = QualitativeAssessmentAgent(llm_client=client)
        result = await agent.assess(sample_transcript)

        # Extract what we can, defaults for what we can't
        assert result.overall == "Valid assessment here."
        # PHQ8_symptoms not closed, so empty
        assert result.phq8_symptoms == "Not assessed"
        assert result.social_factors == "Not assessed"
        assert result.participant_id == 123

    @pytest.mark.asyncio
    async def test_refine_sends_feedback_in_prompt(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Refinement should include feedback in prompt."""
        # Need two responses: one for assess, one for refine
        mock_client.add_chat_response(SAMPLE_LLM_RESPONSE)

        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        initial = await agent.assess(sample_transcript)

        feedback = {
            "completeness": "Missing analysis of sleep symptoms",
            "evidence": "Need more quotes to support claims",
        }
        await agent.refine(initial, feedback, sample_transcript)

        assert mock_client.chat_call_count == 2
        refine_request = mock_client.chat_requests[1]
        user_msg = next(m for m in refine_request.messages if m.role == "user")

        assert "Missing analysis of sleep symptoms" in user_msg.content
        assert "Need more quotes" in user_msg.content
        assert "COMPLETENESS" in user_msg.content
        assert "EVIDENCE" in user_msg.content

    @pytest.mark.asyncio
    async def test_refine_includes_original_assessment(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Refinement should include original assessment."""
        mock_client.add_chat_response(SAMPLE_LLM_RESPONSE)

        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        initial = await agent.assess(sample_transcript)

        await agent.refine(initial, {"test": "feedback"}, sample_transcript)

        refine_request = mock_client.chat_requests[1]
        user_msg = next(m for m in refine_request.messages if m.role == "user")
        assert initial.overall in user_msg.content

    @pytest.mark.asyncio
    async def test_full_text_property(
        self,
        mock_client: MockLLMClient,
        sample_transcript: Transcript,
    ) -> None:
        """Assessment full_text should contain all sections."""
        agent = QualitativeAssessmentAgent(llm_client=mock_client)
        result = await agent.assess(sample_transcript)

        full_text = result.full_text
        assert "Overall Assessment:" in full_text
        assert "PHQ-8 Symptoms:" in full_text
        assert "Social Factors:" in full_text
        assert "Biological Factors:" in full_text
        assert "Risk Factors:" in full_text


class TestQualitativePrompts:
    """Tests for prompt template functions."""

    def test_make_qualitative_prompt_includes_transcript(self) -> None:
        """Prompt should include transcript text."""
        transcript = "Ellie: Hello\nParticipant: Hi there"
        prompt = make_qualitative_prompt(transcript)

        assert transcript in prompt
        assert "Ellie" in prompt
        assert "Participant" in prompt

    def test_make_qualitative_prompt_includes_domains(self) -> None:
        """Prompt should mention all assessment domains."""
        prompt = make_qualitative_prompt("test transcript")

        assert "PHQ-8" in prompt
        assert "social" in prompt.lower()
        assert "biological" in prompt.lower()
        assert "risk" in prompt.lower()

    def test_make_qualitative_prompt_includes_xml_format(self) -> None:
        """Prompt should specify expected XML format."""
        prompt = make_qualitative_prompt("test")

        assert "<assessment>" in prompt
        assert "<PHQ8_symptoms>" in prompt
        assert "<social_factors>" in prompt
        assert "<biological_factors>" in prompt
        assert "<risk_factors>" in prompt
        assert "<exact_quotes>" in prompt

    def test_make_qualitative_prompt_includes_examples(self) -> None:
        """Prompt should include example guidance."""
        prompt = make_qualitative_prompt("test")

        assert "Examples" in prompt

    def test_make_feedback_prompt_includes_feedback(self) -> None:
        """Feedback prompt should include all feedback metrics."""
        feedback = {
            "completeness": "Missing details",
            "accuracy": "Some inaccuracies",
        }
        prompt = make_feedback_prompt("original", feedback, "transcript")

        assert "COMPLETENESS" in prompt
        assert "Missing details" in prompt
        assert "ACCURACY" in prompt
        assert "Some inaccuracies" in prompt

    def test_make_feedback_prompt_includes_original(self) -> None:
        """Feedback prompt should include original assessment."""
        original = "This is the original assessment"
        prompt = make_feedback_prompt(original, {}, "transcript")

        assert original in prompt
        assert "ORIGINAL ASSESSMENT" in prompt

    def test_make_feedback_prompt_includes_transcript(self) -> None:
        """Feedback prompt should include transcript."""
        transcript = "Interview transcript here"
        prompt = make_feedback_prompt("original", {}, transcript)

        assert transcript in prompt
        assert "TRANSCRIPT" in prompt


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
                temperature: float = 0.0,  # noqa: ARG002
            ) -> str:
                return "<assessment>Custom response</assessment>"

        client = CustomClient()
        assert isinstance(client, SimpleChatClient)

        agent = QualitativeAssessmentAgent(llm_client=client)
        transcript = Transcript(participant_id=1, text="Test")
        result = await agent.assess(transcript)

        assert result.overall == "Custom response"
