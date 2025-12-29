"""Judge agent for evaluating qualitative assessments."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ai_psychiatrist.agents.prompts.judge import make_evaluation_prompt
from ai_psychiatrist.config import PydanticAISettings, get_model_name
from ai_psychiatrist.domain.entities import (
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.enums import EvaluationMetric
from ai_psychiatrist.domain.value_objects import EvaluationScore
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from ai_psychiatrist.agents.output_models import JudgeMetricOutput
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
        pydantic_ai_settings: PydanticAISettings | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        """Initialize judge agent.

        Args:
            llm_client: LLM client for evaluations.
            model_settings: Model configuration. If None, uses OllamaClient defaults.
        """
        self._llm_client = llm_client
        self._model_settings = model_settings
        self._pydantic_ai = pydantic_ai_settings or PydanticAISettings()
        self._ollama_base_url = ollama_base_url
        self._metric_agent: Agent[None, JudgeMetricOutput] | None = None

        if self._pydantic_ai.enabled:
            if not self._ollama_base_url:
                raise ValueError(
                    "Pydantic AI enabled but no ollama_base_url provided. "
                    "Legacy fallback is disabled."
                )
            from ai_psychiatrist.agents.pydantic_agents import (  # noqa: PLC0415
                create_judge_metric_agent,
            )

            self._metric_agent = create_judge_metric_agent(
                model_name=get_model_name(model_settings, "judge"),
                base_url=self._ollama_base_url,
                retries=self._pydantic_ai.retries,
                system_prompt="",
            )

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
        if self._metric_agent is None:
            raise ValueError("Pydantic AI metric agent not initialized")

        prompt = make_evaluation_prompt(metric, transcript, assessment)

        # Use model settings if provided (GAP-001: temp=0.0 for clinical reproducibility)
        temperature = self._model_settings.temperature if self._model_settings else 0.0

        try:
            timeout = self._pydantic_ai.timeout_seconds
            result = await self._metric_agent.run(
                prompt,
                model_settings={
                    "temperature": temperature,
                    **({"timeout": timeout} if timeout is not None else {}),
                },
            )
            output = result.output
            return EvaluationScore(
                metric=metric,
                score=output.score,
                explanation=output.explanation.strip(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Fail fast - no legacy fallback
            logger.error(
                "Pydantic AI call failed during metric evaluation",
                metric=metric.value,
                error=str(e),
            )
            raise ValueError(f"Pydantic AI evaluation failed for {metric.value}: {e}") from e

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
