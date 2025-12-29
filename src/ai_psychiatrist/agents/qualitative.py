"""Qualitative assessment agent implementation.

Paper Reference:
    - Section 2.3.1: Qualitative Assessment Agent
    - Appendix B: Four assessment domains (PHQ-8 symptoms, biological, social, risk factors)

This agent analyzes interview transcripts to generate structured qualitative
assessments across clinical domains, supporting the feedback loop for
iterative refinement.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, ClassVar

from ai_psychiatrist.agents.prompts.qualitative import (
    QUALITATIVE_SYSTEM_PROMPT,
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.config import PydanticAISettings, get_model_name
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.llm.responses import (
    SimpleChatClient,
    extract_xml_tags,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ai_psychiatrist.agents.output_models import QualitativeOutput
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
        pydantic_ai_settings: PydanticAISettings | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        """Initialize qualitative assessment agent.

        Args:
            llm_client: LLM client for chat completions.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
            pydantic_ai_settings: Pydantic AI configuration. If None, uses defaults.
            ollama_base_url: Ollama base URL for Pydantic AI agent. Required when
                pydantic_ai_settings.enabled is True.
        """
        self._llm_client = llm_client
        self._model_settings = model_settings
        self._pydantic_ai = pydantic_ai_settings or PydanticAISettings()
        self._ollama_base_url = ollama_base_url
        self._agent: Agent[None, QualitativeOutput] | None = None

        if self._pydantic_ai.enabled:
            if not self._ollama_base_url:
                logger.warning(
                    "Pydantic AI enabled but no ollama_base_url provided; falling back to legacy",
                )
            else:
                from ai_psychiatrist.agents.pydantic_agents import (  # noqa: PLC0415
                    create_qualitative_agent,
                )

                self._agent = create_qualitative_agent(
                    model_name=get_model_name(model_settings, "qualitative"),
                    base_url=self._ollama_base_url,
                    retries=self._pydantic_ai.retries,
                    system_prompt=QUALITATIVE_SYSTEM_PROMPT,
                )

    def _get_llm_params(self) -> tuple[str | None, float]:
        """Get LLM parameters from model settings or defaults.

        Returns:
            Tuple of (model, temperature).
        """
        model = get_model_name(self._model_settings, "qualitative")
        temperature = self._model_settings.temperature if self._model_settings else 0.0
        return model, temperature

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
        model, temperature = self._get_llm_params()

        # Try Pydantic AI path first if enabled
        if self._agent is not None:
            try:
                timeout = self._pydantic_ai.timeout_seconds
                result = await self._agent.run(
                    user_prompt,
                    model_settings={
                        "temperature": temperature,
                        **({"timeout": timeout} if timeout is not None else {}),
                    },
                )
                assessment = self._from_qualitative_output(result.output, transcript.participant_id)
                logger.info(
                    "Qualitative assessment complete (Pydantic AI)",
                    participant_id=transcript.participant_id,
                    overall_length=len(assessment.overall),
                )
                return assessment
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Intentionally broad: Fallback for any Pydantic AI error
                # (see docs/specs/21-broad-exception-handling.md)
                logger.error(
                    "Pydantic AI call failed during assessment; falling back to legacy",
                    error=str(e),
                )

        # Call LLM (Legacy Path)
        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
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
        model, temperature = self._get_llm_params()

        # Try Pydantic AI path first if enabled
        if self._agent is not None:
            try:
                timeout = self._pydantic_ai.timeout_seconds
                result = await self._agent.run(
                    user_prompt,
                    model_settings={
                        "temperature": temperature,
                        **({"timeout": timeout} if timeout is not None else {}),
                    },
                )
                assessment = self._from_qualitative_output(result.output, transcript.participant_id)
                logger.info(
                    "Assessment refinement complete (Pydantic AI)",
                    participant_id=transcript.participant_id,
                )
                return assessment
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Intentionally broad: Fallback for any Pydantic AI error
                # (see docs/specs/21-broad-exception-handling.md)
                logger.error(
                    "Pydantic AI call failed during refinement; falling back to legacy",
                    error=str(e),
                )

        raw_response = await self._llm_client.simple_chat(
            user_prompt=user_prompt,
            system_prompt=QUALITATIVE_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
        )

        assessment = self._parse_response(raw_response, transcript.participant_id)

        logger.info(
            "Assessment refinement complete",
            participant_id=transcript.participant_id,
        )

        return assessment

    def _from_qualitative_output(
        self,
        output: QualitativeOutput,
        participant_id: int,
    ) -> QualitativeAssessment:
        """Convert validated QualitativeOutput into domain entity."""
        return QualitativeAssessment(
            overall=output.assessment,
            phq8_symptoms=output.phq8_symptoms,
            social_factors=output.social_factors,
            biological_factors=output.biological_factors,
            risk_factors=output.risk_factors,
            supporting_quotes=output.exact_quotes,
            participant_id=participant_id,
        )

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
