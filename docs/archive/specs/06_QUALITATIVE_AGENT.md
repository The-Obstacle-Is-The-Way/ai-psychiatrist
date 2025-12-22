# Spec 06: Qualitative Assessment Agent

## Objective

Implement the qualitative assessment agent that analyzes interview transcripts to identify PHQ-8 symptoms, risk factors, and generates clinical summaries.

## Paper Reference

- **Section 2.3.1**: Qualitative Assessment Agent
- **Appendix B**: Four assessment domains (PHQ-8 symptoms, biological, social, risk factors)

## As-Is Implementation (Repo)

- Demo qualitative agent used by `server.py`: `agents/qualitative_assessor_f.py:QualitativeAssessor`
- Transport: `POST /api/generate` (streaming) with a **single prompt string** (no system/user separation)
- Default model (demo): `"llama3"`
- Return type (demo): raw model text; no XML parsing/validation
- Known issue: the embedded “XML template” in the prompt contains malformed tags (documented below verbatim for parity)

### Research Script (Cluster)

- File: `qualitative_assessment/qual_assessment.py`
- Uses `POST /api/chat` with a separate `system_prompt` and multiple user prompts (overall, per-symptom, social, biological, risk)
- Typical model: `"gemma3:27b"`
- Options: `{"temperature": 0, "top_k": 20, "top_p": 0.9}`
- Output tags (as requested by the script): `<assessment>...</assessment>` and `<quotes>...</quotes>`

## Deliverables

1. `src/ai_psychiatrist/agents/qualitative.py` - Qualitative assessment agent
2. `src/ai_psychiatrist/agents/prompts/qualitative.py` - Prompt templates
3. `tests/unit/agents/test_qualitative.py` - Comprehensive tests

## Implementation

### 1. Prompt Templates (prompts/qualitative.py)

#### As-Is Prompt (Verbatim from `agents/qualitative_assessor_f.py`)

```python
prompt = f"""
            You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities. Use exact quotes from the transcript to support your assessment for each tag.
            
            The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
            {interview_text}
            
               Please:
              1. Provide an overall qualitative assessment of the subject's mental health.
              2. Summarize PHQ-8 symptoms if available (if not, state 'not available'), as well as frequency and severity if available.
              3. Summarize social aspects that may influence the subject's mental health. (for example, familial relationships, frienship dynamics, work environment, etc. that are relevant to the subjects mental health)
              4. Summarize biological aspects that may influence the subject's mental health. (for example, famillial history of mental health issues, previous or pre-existing mental health issues, stress levels, etc. that are relevant to the subjects mental health)
              5. Identify potential risk factors the subject may be experiencing.
              6. Use exact quotes from the transcript to support your assessment for each tag.
            
              Output should be formatted as bullet points with headings for each section using stars. Example: **Tiredness** <explanation of tiredness>. Do not include any additional text outside the bullet points
              Please answer in this XML format with each tag on a new line, properly indented. Use straight quotes instead of curly quotes, and do not include any additional text outside the XML tags:
            
              <assessment>
                <!-- Summary of participant's overall mental health -->
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </assessment>
            
              <PHQ8_symptoms>
                <!-- Summary of PHQ-8 symptoms mentioned in the trancript:
                - Little interest or pleasure in doing things
                - Feeling down, depressed, or hopeless
                - Trouble falling or staying asleep, or sleeping too much
                - Feeling tired or having little energy
                - Poor appetite or overeating
                - Feeling bad about yourself — or that you are a failure or have let yourself or your family down
                - Trouble concentrating on things, such as reading the newspaper or watching television
                - Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual
            
                 For each symptom present, note:
                - Frequency (not at all, several days, more than half the days, nearly every day)
                - Duration (how long experienced)
                - Severity/impact on functioning
            
               If symptoms are not discussed, state "not assessed in interview" -->
            
               <little_interest_or_pleasure>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
               </little_interest or pleasure>
            
                <feeling_down_depressed_hopeless>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_down_depressed_hopeless>
            
                <trouble_sleeping>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </trouble_sleeping>
            
                <feeling_tired_little_energy>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_tired_little_energy>
            
                <poor_appetite_overeating>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </poor_appetite_overeating>
            
                <feeling_bad_about_self>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </feeling_bad_about_self>
            
                <trouble_concentrating>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </trouble_concentrating>
            
                <moving_speaking_slowly_or_fidgety>
                <!-- Details on this symptom -->
                <!-- Frequency, duration, severity if available -->
                </moving_speaking_slowly_or_fidgety>
            
            
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </PHQ8_symptoms>
            
              <social_factors>
                <!-- Summary of social influences on patient's health -->
                <exact_quotes>
              </social_factors>
            
              <biological_factors>
                <!-- Summary of biological influences on patient's health -->
               <exact_quotes>
                <!-- Quotes from the transcript that support the assessment -->
                </exact_quotes>
              </biological_factors>
            
              <risk_factors>
                <!-- Summary of potential risk factors -->
                 <exact_quotes>
               <!-- Quotes from the transcript that support the assessment -->
               </exact_quotes>
              </risk_factors>
            """
```

#### Target Prompt (Paper-Aligned Schema)

```python
"""Prompt templates for qualitative assessment."""

from __future__ import annotations


QUALITATIVE_SYSTEM_PROMPT = """You are a psychiatrist. Your job is to assess and diagnose patients for depression. Be as objective, concise, and accurate as possible. Be specific and avoid vague generalities. Use exact quotes from the transcript to support your assessment for each tag."""


def make_qualitative_prompt(transcript: str) -> str:
    """Generate qualitative assessment prompt.

    Args:
        transcript: Interview transcript text.

    Returns:
        Formatted user prompt.
    """
return f"""The following is an interview transcript from a psychiatric assessment of a subject who is being screened for depression. Please note that 'Ellie' is the interviewer, and 'Participant' is the subject being assessed.
{transcript}

Please produce a qualitative assessment across these domains:
1) Overall assessment of the subject's mental health
2) PHQ-8 symptoms discussed (include frequency/duration/severity when available)
3) Social factors that may influence mental health
4) Biological factors that may influence mental health
5) Risk factors the subject may be experiencing

Examples (for format only, do NOT reuse content):
- PHQ-8 symptoms: "I don't enjoy anything anymore" (frequency: nearly every day)
- Social factors: "Things have been tense at home"
- Biological factors: "My mother had depression"
- Risk factors: "I feel isolated since losing my job"

Requirements:
- Be objective, concise, and clinically grounded (avoid vague generalities).
- Use exact quotes from the transcript as evidence within each domain.
- Collect all quoted evidence again in <exact_quotes> as bullet points.
- If a domain is not covered in the interview, write \"not assessed in interview\".

Return ONLY this XML (each tag on its own line; no additional text outside the tags):

<assessment>...</assessment>
<PHQ8_symptoms>...</PHQ8_symptoms>
<social_factors>...</social_factors>
<biological_factors>...</biological_factors>
<risk_factors>...</risk_factors>
<exact_quotes>...</exact_quotes>
"""


def make_feedback_prompt(
    original_assessment: str,
    feedback: dict[str, str],
    transcript: str,
) -> str:
    """Generate prompt for assessment refinement based on feedback.

    Args:
        original_assessment: Previous assessment output.
        feedback: Dictionary of metric -> feedback text.
        transcript: Original transcript.

    Returns:
        Formatted refinement prompt.
    """
    feedback_text = "\n".join(
        f"- **{metric.upper()}**: {text}" for metric, text in feedback.items()
    )

    return f"""The following qualitative assessment has been evaluated and needs improvement.

EVALUATION FEEDBACK:
{feedback_text}

ORIGINAL ASSESSMENT:
{original_assessment}

TRANSCRIPT:
{transcript}

Please provide an improved assessment that addresses the feedback above. Use the same XML format:

<assessment>...</assessment>
<PHQ8_symptoms>...</PHQ8_symptoms>
<social_factors>...</social_factors>
<biological_factors>...</biological_factors>
<risk_factors>...</risk_factors>
<exact_quotes>...</exact_quotes>

Ensure:
1. More specific evidence with exact quotes
2. Complete coverage of all PHQ-8 symptoms
3. Logical consistency throughout
4. Accurate alignment with clinical criteria

Return only the XML (no additional text outside the tags)."""
```

### 2. Qualitative Agent (agents/qualitative.py)

```python
"""Qualitative assessment agent implementation."""

from __future__ import annotations

import re
from typing import ClassVar, Protocol, runtime_checkable

from ai_psychiatrist.agents.prompts.qualitative import (
    QUALITATIVE_SYSTEM_PROMPT,
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.llm.responses import extract_xml_tags
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM clients with simple_chat method."""

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> str:
        """Send a simple chat prompt and return response."""
        ...


class QualitativeAssessmentAgent:
    """Agent for generating qualitative assessments from interview transcripts.

    This agent implements the qualitative assessment described in Section 2.3.1
    of the paper. It analyzes transcripts to identify:
    - PHQ-8 symptoms with supporting evidence
    - Social factors affecting mental health
    - Biological factors and history
    - Risk factors and warning signs
    """

    # XML tags to extract from LLM response
    ASSESSMENT_TAGS: ClassVar[list[str]] = [
        "assessment",
        "PHQ8_symptoms",
        "social_factors",
        "biological_factors",
        "risk_factors",
    ]

    def __init__(self, llm_client: ChatClient) -> None:
        """Initialize qualitative assessment agent.

        Args:
            llm_client: LLM client for chat completions.
        """
        self._llm_client = llm_client

    async def assess(self, transcript: Transcript) -> QualitativeAssessment:
        """Generate qualitative assessment for a transcript.

        Args:
            transcript: Interview transcript to assess.

        Returns:
            Qualitative assessment with all domains.
        """
        logger.info(
            "Starting qualitative assessment",
            participant_id=transcript.participant_id,
            word_count=transcript.word_count,
        )

        # Generate assessment prompt
        user_prompt = make_qualitative_prompt(transcript.text)

        # Call LLM
        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
        )

        # Parse response
        assessment = self._parse_response(raw_response, transcript.participant_id)

        logger.info(
            "Qualitative assessment complete",
            participant_id=transcript.participant_id,
            overall_length=len(assessment.overall),
        )

        return assessment

    async def refine(
        self,
        original_assessment: QualitativeAssessment,
        feedback: dict[str, str],
        transcript: Transcript,
    ) -> QualitativeAssessment:
        """Refine assessment based on evaluation feedback.

        Args:
            original_assessment: Previous assessment to improve.
            feedback: Dictionary of metric -> feedback text.
            transcript: Original transcript.

        Returns:
            Improved qualitative assessment.
        """
        logger.info(
            "Refining qualitative assessment",
            participant_id=transcript.participant_id,
            feedback_metrics=list(feedback.keys()),
        )

        user_prompt = make_feedback_prompt(
            original_assessment=original_assessment.full_text,
            feedback=feedback,
            transcript=transcript.text,
        )

        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
        )

        assessment = self._parse_response(raw_response, transcript.participant_id)

        logger.info(
            "Assessment refinement complete",
            participant_id=transcript.participant_id,
        )

        return assessment

    def _parse_response(
        self,
        raw_response: str,
        participant_id: int,
    ) -> QualitativeAssessment:
        """Parse LLM response into QualitativeAssessment.

        Args:
            raw_response: Raw LLM output with XML tags.
            participant_id: Participant identifier.

        Returns:
            Parsed QualitativeAssessment entity.
        """
        # Extract XML tags
        extracted = extract_xml_tags(raw_response, self.ASSESSMENT_TAGS)

        # Extract quotes if present or embedded inline
        quotes = self._extract_quotes(raw_response, extracted)

        return QualitativeAssessment(
            overall=extracted.get("assessment") or "Assessment not generated",
            phq8_symptoms=extracted.get("PHQ8_symptoms") or "Not assessed",
            social_factors=extracted.get("social_factors") or "Not assessed",
            biological_factors=extracted.get("biological_factors") or "Not assessed",
            risk_factors=extracted.get("risk_factors") or "Not assessed",
            supporting_quotes=quotes,
            participant_id=participant_id,
        )

    def _extract_quotes(
        self,
        raw_response: str,
        extracted: dict[str, str],
    ) -> list[str]:
        """Extract supporting quotes from response."""
        quotes: list[str] = []

        if "exact_quotes" in raw_response.lower():
            quotes_section = extract_xml_tags(raw_response, ["exact_quotes"])
            if quotes_section.get("exact_quotes"):
                quotes = [
                    self._clean_quote_line(line)
                    for line in quotes_section["exact_quotes"].split("\n")
                ]
                quotes = [q for q in quotes if q]

        if not quotes:
            combined = "\n".join(value for value in extracted.values() if value)
            quotes = self._extract_inline_quotes(combined)

        return quotes

    @staticmethod
    def _clean_quote_line(line: str) -> str:
        """Normalize a quote line from an exact_quotes block."""
        cleaned = line.strip()
        if not cleaned or cleaned == "-":
            return ""
        if cleaned[0] in {"-", "*", "•"}:
            cleaned = cleaned[1:].strip()
        return cleaned

    @staticmethod
    def _extract_inline_quotes(text: str) -> list[str]:
        """Extract quoted substrings from assessment sections."""
        matches: list[str] = []
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text):
            value = match.group(1) or match.group(2) or ""
            cleaned = value.strip()
            if cleaned:
                matches.append(cleaned)

        deduped: list[str] = []
        seen: set[str] = set()
        for quote in matches:
            if quote not in seen:
                seen.add(quote)
                deduped.append(quote)

        return deduped
```

### 3. Tests (test_qualitative.py)

```python
"""Tests for qualitative assessment agent."""

from __future__ import annotations

import pytest

from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.domain.entities import Transcript
from tests.fixtures.mock_llm import MockLLMClient


class TestQualitativeAssessmentAgent:
    """Tests for QualitativeAssessmentAgent."""

    @pytest.fixture
    def sample_llm_response(self) -> str:
        """Sample LLM response with all XML tags."""
        return """
<assessment>
The participant shows signs of moderate depression.
</assessment>

<PHQ8_symptoms>
<little_interest_or_pleasure>
Participant expresses lack of interest.
</little_interest_or_pleasure>
</PHQ8_symptoms>

<social_factors>
Lives with children.
</social_factors>

<biological_factors>
History of suicide attempt.
</biological_factors>

<risk_factors>
Current thoughts of "not waking up".
</risk_factors>
"""

    @pytest.fixture
    def mock_client(self, sample_llm_response: str) -> MockLLMClient:
        """Create mock LLM client with sample response."""
        return MockLLMClient(chat_responses=[sample_llm_response])

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text="Ellie: How are you?\nParticipant: Not great, feeling down.",
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

        assert result.overall
        assert result.phq8_symptoms
        assert result.social_factors
        assert result.biological_factors
        assert result.risk_factors
        assert result.participant_id == 123
```

## Acceptance Criteria

- [ ] Generates assessment covering all 4 domains (PHQ-8, social, biological, risk)
- [ ] Extracts and includes supporting quotes from transcript
- [ ] Uses XML format matching paper description
- [ ] Includes domain examples in the prompt (paper Section 2.3.1)
- [ ] Supports refinement based on feedback
- [ ] Handles malformed LLM responses gracefully
- [ ] Logs assessment progress and metrics
- [ ] Full test coverage with mocked LLM

## Dependencies

- **Spec 02**: Domain entities (QualitativeAssessment, Transcript)
- **Spec 04**: LLM infrastructure (OllamaClient)

## Specs That Depend on This

- **Spec 07**: Judge Agent (evaluates qualitative output)
- **Spec 10**: Meta-Review Agent
- **Spec 11**: Full Pipeline
