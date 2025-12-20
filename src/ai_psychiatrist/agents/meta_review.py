"""Meta-review agent for integrating assessments.

Paper Reference:
    - Section 2.3.3: Meta Review
    - Section 3.3: Meta Review Results (78% accuracy, comparable to human expert)

The meta-review agent integrates qualitative and quantitative assessments
to predict overall depression severity (0-4 scale).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_psychiatrist.agents.prompts.meta_review import (
    META_REVIEW_SYSTEM_PROMPT,
    make_meta_review_prompt,
)
from ai_psychiatrist.domain.entities import (
    MetaReview,
    PHQ8Assessment,
    QualitativeAssessment,
    Transcript,
)
from ai_psychiatrist.domain.enums import PHQ8Item, SeverityLevel
from ai_psychiatrist.infrastructure.llm.responses import (
    SimpleChatClient,
    extract_xml_tags,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import ModelSettings

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
    ) -> None:
        """Initialize meta-review agent.

        Args:
            llm_client: LLM client for generating reviews.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm = llm_client
        self._model_settings = model_settings

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
        model = self._model_settings.meta_review_model if self._model_settings else None
        temperature = self._model_settings.temperature if self._model_settings else 0.2
        top_k = self._model_settings.top_k if self._model_settings else 20
        top_p = self._model_settings.top_p if self._model_settings else 0.8

        response = await self._llm.simple_chat(
            user_prompt=prompt,
            system_prompt=META_REVIEW_SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        severity, explanation = self._parse_response(response, quantitative)

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

    def _parse_response(
        self,
        raw: str,
        quantitative: PHQ8Assessment,
    ) -> tuple[SeverityLevel, str]:
        """Parse severity and explanation from LLM response.

        Args:
            raw: Raw LLM response text.
            quantitative: Quantitative assessment for fallback.

        Returns:
            Tuple of (severity level, explanation text).
        """
        tags = extract_xml_tags(raw, ["severity", "explanation"])

        # Parse severity with fallback
        severity_str = tags.get("severity", "").strip()
        try:
            severity_int = int(severity_str)
            # Clamp to valid range 0-4
            severity = SeverityLevel(max(0, min(4, severity_int)))
        except (ValueError, TypeError):
            # Fall back to quantitative-derived severity
            logger.warning(
                "Failed to parse severity from response, using quantitative fallback",
                raw_severity=severity_str[:50] if severity_str else "empty",
            )
            severity = quantitative.severity

        # Get explanation, fallback to raw response if not tagged
        explanation = tags.get("explanation", "").strip()
        if not explanation:
            # Try to extract meaningful content from raw response
            explanation = raw.strip()

        return severity, explanation
