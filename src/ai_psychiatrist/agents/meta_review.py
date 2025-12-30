"""Meta-review agent for integrating assessments.

Paper Reference:
    - Section 2.3.3: Meta Review
    - Section 3.3: Meta Review Results (78% accuracy, comparable to human expert)

The meta-review agent integrates qualitative and quantitative assessments
to predict overall depression severity (0-4 scale).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ai_psychiatrist.agents.prompts.meta_review import (
    META_REVIEW_SYSTEM_PROMPT,
    make_meta_review_prompt,
)
from ai_psychiatrist.config import PydanticAISettings, get_model_name
from ai_psychiatrist.domain.entities import (
    MetaReview,
    PHQ8Assessment,
    QualitativeAssessment,
    Transcript,
)
from ai_psychiatrist.domain.enums import PHQ8Item, SeverityLevel
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ai_psychiatrist.agents.output_models import MetaReviewOutput
    from ai_psychiatrist.config import ModelSettings
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

logger = get_logger(__name__)


class MetaReviewAgent:
    """Agent for integrating assessments into final severity prediction.

    Takes the qualitative assessment, quantitative PHQ-8 scores, and original
    transcript to generate a final integrated severity prediction with explanation.

    Paper Reference:
        - Section 2.3.3: Meta Review
        - Section 3.3: 78% severity prediction accuracy
    """

    def __init__(
        self,
        llm_client: SimpleChatClient,
        model_settings: ModelSettings | None = None,
        pydantic_ai_settings: PydanticAISettings | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        """Initialize meta-review agent.

        Args:
            llm_client: LLM client for generating reviews.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
            pydantic_ai_settings: Pydantic AI configuration. If None, uses defaults.
            ollama_base_url: Ollama base URL for Pydantic AI agent. Required when
                pydantic_ai_settings.enabled is True.
        """
        self._llm = llm_client
        self._model_settings = model_settings
        self._pydantic_ai = pydantic_ai_settings or PydanticAISettings()
        self._ollama_base_url = ollama_base_url
        self._review_agent: Agent[None, MetaReviewOutput] | None = None

        if self._pydantic_ai.enabled:
            if not self._ollama_base_url:
                raise ValueError(
                    "Pydantic AI enabled but no ollama_base_url provided. "
                    "Legacy fallback is disabled."
                )
            from ai_psychiatrist.agents.pydantic_agents import (  # noqa: PLC0415
                create_meta_review_agent,
            )

            self._review_agent = create_meta_review_agent(
                model_name=get_model_name(model_settings, "meta_review"),
                base_url=self._ollama_base_url,
                retries=self._pydantic_ai.retries,
                system_prompt=META_REVIEW_SYSTEM_PROMPT,
            )

    async def review(
        self,
        transcript: Transcript,
        qualitative: QualitativeAssessment,
        quantitative: PHQ8Assessment,
    ) -> MetaReview:
        """Generate meta-review integrating all assessments.

        Args:
            transcript: Original interview transcript.
            qualitative: Qualitative assessment output.
            quantitative: Quantitative PHQ-8 scores.

        Returns:
            MetaReview with severity prediction and explanation.
        """
        logger.info(
            "Starting meta-review",
            participant_id=transcript.participant_id,
            phq8_total=quantitative.total_score,
            phq8_na_count=quantitative.na_count,
        )

        if self._review_agent is None:
            raise ValueError("Pydantic AI review agent not initialized")

        # Format quantitative scores for prompt
        quant_text = self._format_quantitative(quantitative)

        # Format qualitative assessment
        qual_text = qualitative.full_text

        prompt = make_meta_review_prompt(
            transcript=transcript.text,
            qualitative=qual_text,
            quantitative=quant_text,
        )

        # Get model and sampling params from settings
        temperature = self._model_settings.temperature if self._model_settings else 0.0

        try:
            timeout = self._pydantic_ai.timeout_seconds
            result = await self._review_agent.run(
                prompt,
                model_settings={
                    "temperature": temperature,
                    **({"timeout": timeout} if timeout is not None else {}),
                },
            )
            output = result.output
            severity = SeverityLevel(output.severity)
            explanation = output.explanation.strip()
            logger.info(
                "Meta-review complete",
                participant_id=transcript.participant_id,
                severity=severity.name,
                is_mdd=severity.is_mdd,
            )
            return MetaReview(
                severity=severity,
                explanation=explanation,
                quantitative_assessment_id=quantitative.id,
                qualitative_assessment_id=qualitative.id,
                participant_id=transcript.participant_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail fast - no legacy fallback
            logger.error(
                "Pydantic AI call failed during meta-review",
                participant_id=transcript.participant_id,
                error=str(e),
                error_type=type(e).__name__,
                prompt_chars=len(prompt),
                temperature=temperature,
            )
            raise

    def _format_quantitative(self, assessment: PHQ8Assessment) -> str:
        """Format PHQ-8 scores for prompt.

        Formats scores in XML-like tags matching the original implementation's format.

        Args:
            assessment: PHQ-8 assessment to format.

        Returns:
            Formatted string with score tags.
        """
        lines: list[str] = []
        for item in PHQ8Item.all_items():
            item_assessment = assessment.get_item(item)
            score = item_assessment.score if item_assessment.is_available else "N/A"
            reason = item_assessment.reason

            key_lower = item.value.lower()
            if score != "N/A":
                lines.append(f"<{key_lower}_score>{score}</{key_lower}_score>")
                lines.append(f"<{key_lower}_explanation>{reason}</{key_lower}_explanation>")

        return "\n".join(lines)
