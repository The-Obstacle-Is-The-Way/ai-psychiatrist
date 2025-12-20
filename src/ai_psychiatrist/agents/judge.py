"""Judge agent for evaluating qualitative assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai_psychiatrist.agents.prompts.judge import make_evaluation_prompt
from ai_psychiatrist.domain.entities import (
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import EvaluationMetric
from ai_psychiatrist.domain.exceptions import LLMError
from ai_psychiatrist.domain.value_objects import EvaluationScore
from ai_psychiatrist.infrastructure.llm.responses import extract_score_from_text
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import ModelSettings
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

logger = get_logger(__name__)


class JudgeAgent:
    """Agent for evaluating qualitative assessments.

    Implements the LLM-as-a-judge approach described in Section 2.3.1.
    Evaluates assessments on 4 metrics using a 5-point Likert scale.
    """

    def __init__(
        self,
        llm_client: SimpleChatClient,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Initialize judge agent.

        Args:
            llm_client: LLM client for evaluations.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm_client = llm_client
        self._model_settings = model_settings

    async def evaluate(
        self,
        assessment: QualitativeAssessment,
        transcript: Transcript,
        iteration: int = 0,
    ) -> QualitativeEvaluation:
        """Evaluate a qualitative assessment on all metrics.

        Args:
            assessment: Qualitative assessment to evaluate.
            transcript: Original transcript for reference.
            iteration: Current feedback loop iteration (0 = initial).

        Returns:
            QualitativeEvaluation with scores for all metrics.
        """
        logger.info(
            "Starting qualitative evaluation",
            participant_id=transcript.participant_id,
            iteration=iteration,
        )

        scores: dict[EvaluationMetric, EvaluationScore] = {}

        for metric in EvaluationMetric.all_metrics():
            score = await self._evaluate_metric(
                metric=metric,
                transcript=transcript.text,
                assessment=assessment.full_text,
            )
            scores[metric] = score

            logger.debug(
                "Metric evaluated",
                metric=metric.value,
                score=score.score,
                participant_id=transcript.participant_id,
            )

        evaluation = QualitativeEvaluation(
            scores=scores,
            assessment_id=assessment.id,
            iteration=iteration,
        )

        logger.info(
            "Evaluation complete",
            participant_id=transcript.participant_id,
            average_score=f"{evaluation.average_score:.2f}",
            low_scores=[m.value for m in evaluation.low_scores],
        )

        return evaluation

    async def _evaluate_metric(
        self,
        metric: EvaluationMetric,
        transcript: str,
        assessment: str,
    ) -> EvaluationScore:
        """Evaluate a single metric.

        Args:
            metric: Metric to evaluate.
            transcript: Original transcript text.
            assessment: Assessment text to evaluate.

        Returns:
            EvaluationScore for the metric.
        """
        prompt = make_evaluation_prompt(metric, transcript, assessment)

        # Use model settings if provided (Spec 07 mandates temperature=0.0 for Judge)
        model = self._model_settings.judge_model if self._model_settings else None
        # Judge always uses temperature=0.0 for deterministic evaluation
        temperature = self._model_settings.temperature_judge if self._model_settings else 0.0
        top_k = self._model_settings.top_k if self._model_settings else 20
        top_p = self._model_settings.top_p if self._model_settings else 0.8

        try:
            response = await self._llm_client.simple_chat(
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        except LLMError as e:
            logger.error(
                "LLM call failed during metric evaluation",
                metric=metric.value,
                error=str(e),
            )
            # Return default score on LLM failure (triggers refinement as fail-safe)
            return EvaluationScore(
                metric=metric,
                score=3,
                explanation="LLM evaluation failed; default score used.",
            )

        # Extract score from response
        score = extract_score_from_text(response)

        # Default to 3 if extraction fails
        if score is None:
            logger.warning(
                "Could not extract score, defaulting to 3",
                metric=metric.value,
                response_preview=response[:200],
            )
            score = 3

        return EvaluationScore(
            metric=metric,
            score=score,
            explanation=response.strip(),
        )

    def get_feedback_for_low_scores(
        self,
        evaluation: QualitativeEvaluation,
        threshold: int | None = None,
    ) -> dict[str, str]:
        """Extract feedback text for low-scoring metrics.

        Args:
            evaluation: Evaluation with scores.
            threshold: Optional score threshold override.

        Returns:
            Dictionary of metric name -> feedback explanation.
        """
        feedback = {}
        low_scores = (
            evaluation.low_scores
            if threshold is None
            else evaluation.low_scores_for_threshold(threshold)
        )
        for metric in low_scores:
            score = evaluation.get_score(metric)
            feedback[metric.value] = f"Scored {score.score}/5. {score.explanation}"
        return feedback
