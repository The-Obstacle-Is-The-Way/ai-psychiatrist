"""Qualitative assessment agent implementation.

Paper Reference:
    - Section 2.3.1: Qualitative Assessment Agent
    - Appendix B: Four assessment domains (PHQ-8 symptoms, biological, social, risk factors)

This agent analyzes interview transcripts to generate structured qualitative
assessments across clinical domains, supporting the feedback loop for
iterative refinement.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from ai_psychiatrist.agents.prompts.qualitative import (
    QUALITATIVE_SYSTEM_PROMPT,
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.llm.responses import (
    SimpleChatClient,
    extract_xml_tags,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import ModelSettings

logger = get_logger(__name__)


class QualitativeAssessmentAgent:
    """Agent for generating qualitative assessments from interview transcripts.

    This agent implements the qualitative assessment described in Section 2.3.1
    of the paper. It analyzes transcripts to identify:
    - PHQ-8 symptoms with supporting evidence
    - Social factors affecting mental health
    - Biological factors and history
    - Risk factors and warning signs

    Example:
        >>> from tests.fixtures.mock_llm import MockLLMClient
        >>> client = MockLLMClient(chat_responses=["<assessment>...</assessment>..."])
        >>> agent = QualitativeAssessmentAgent(llm_client=client)
        >>> transcript = Transcript(participant_id=123, text="...")
        >>> assessment = await agent.assess(transcript)
    """

    # XML tags to extract from LLM response
    ASSESSMENT_TAGS: ClassVar[list[str]] = [
        "assessment",
        "PHQ8_symptoms",
        "social_factors",
        "biological_factors",
        "risk_factors",
    ]

    def __init__(
        self,
        llm_client: SimpleChatClient,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize qualitative assessment agent.

        Args:
            llm_client: LLM client for chat completions.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm_client = llm_client
        self._model_settings = model_settings

    def _get_llm_params(self) -> tuple[str | None, float, int, float]:
        """Get LLM parameters from model settings or defaults.

        Returns:
            Tuple of (model, temperature, top_k, top_p).
        """
        if self._model_settings:
            return (
                self._model_settings.qualitative_model,
                self._model_settings.temperature,
                self._model_settings.top_k,
                self._model_settings.top_p,
            )
        return None, 0.2, 20, 0.8

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

        # Get LLM params (Paper Section 2.2: Gemma 3 27B)
        model, temperature, top_k, top_p = self._get_llm_params()

        # Call LLM
        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
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

        # Get LLM params
        model, temperature, top_k, top_p = self._get_llm_params()

        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
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

        # Extract quotes if present
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
        """Extract supporting quotes from response.

        Tries to extract from an explicit <exact_quotes> tag first,
        then falls back to inline quoted strings from assessment sections.

        Args:
            raw_response: Raw LLM output.
            extracted: Previously extracted tag content for fallback.

        Returns:
            List of extracted quotes.
        """
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
