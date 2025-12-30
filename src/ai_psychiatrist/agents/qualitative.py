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
from typing import TYPE_CHECKING

from ai_psychiatrist.agents.prompts.qualitative import (
    QUALITATIVE_SYSTEM_PROMPT,
    make_feedback_prompt,
    make_qualitative_prompt,
)
from ai_psychiatrist.config import PydanticAISettings, get_model_name
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ai_psychiatrist.agents.output_models import QualitativeOutput
    from ai_psychiatrist.config import ModelSettings
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

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
        >>> client = MockLLMClient()
        >>> agent = QualitativeAssessmentAgent(
        ...     llm_client=client,
        ...     ollama_base_url="http://localhost:11434",
        ... )
        >>> transcript = Transcript(participant_id=123, text="...")
        >>> assessment = await agent.assess(transcript)
    """

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
                raise ValueError(
                    "Pydantic AI enabled but no ollama_base_url provided. "
                    "Legacy fallback is disabled."
                )
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

        if self._agent is None:
            raise ValueError("Pydantic AI agent not initialized")

        # Generate assessment prompt
        user_prompt = make_qualitative_prompt(transcript.text)

        # Get LLM params (Paper Section 2.2: Gemma 3 27B)
        _, temperature = self._get_llm_params()

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
                "Qualitative assessment complete",
                participant_id=transcript.participant_id,
                overall_length=len(assessment.overall),
            )
            return assessment
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail fast - no legacy fallback
            logger.error(
                "Pydantic AI call failed during assessment",
                error=str(e),
                error_type=type(e).__name__,
                participant_id=transcript.participant_id,
                prompt_chars=len(user_prompt),
                temperature=temperature,
            )
            raise

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

        if self._agent is None:
            raise ValueError("Pydantic AI agent not initialized")

        user_prompt = make_feedback_prompt(
            original_assessment=original_assessment.full_text,
            feedback=feedback,
            transcript=transcript.text,
        )

        # Get LLM params
        _, temperature = self._get_llm_params()

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
                "Assessment refinement complete",
                participant_id=transcript.participant_id,
            )
            return assessment
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail fast - no legacy fallback
            logger.error(
                "Pydantic AI call failed during refinement",
                error=str(e),
                error_type=type(e).__name__,
                participant_id=transcript.participant_id,
                prompt_chars=len(user_prompt),
                temperature=temperature,
            )
            raise

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
