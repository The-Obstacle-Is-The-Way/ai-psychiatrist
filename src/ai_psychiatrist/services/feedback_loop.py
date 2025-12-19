"""Iterative self-refinement feedback loop service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.agents.judge import JudgeAgent
    from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
    from ai_psychiatrist.config import FeedbackLoopSettings
    from ai_psychiatrist.domain.entities import (
        QualitativeAssessment,
        QualitativeEvaluation,
        Transcript,
    )

logger = get_logger(__name__)


@dataclass
class FeedbackLoopResult:
    """Result of the feedback loop refinement process."""

    final_assessment: QualitativeAssessment
    final_evaluation: QualitativeEvaluation
    iterations_used: int
    history: list[tuple[QualitativeAssessment, QualitativeEvaluation]] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Check if assessment improved from initial."""
        if len(self.history) < 1:
            return False
        initial_avg = self.history[0][1].average_score
        final_avg = self.final_evaluation.average_score
        return final_avg > initial_avg


class FeedbackLoopService:
    """Service for iterative assessment refinement.

    Implements the feedback loop described in Section 2.3.1:
    1. Generate initial qualitative assessment
    2. Evaluate with judge agent
    3. If any score <= threshold, provide feedback and regenerate
    4. Repeat until all scores acceptable or max iterations reached
    """

    def __init__(
        self,
        qualitative_agent: QualitativeAssessmentAgent,
        judge_agent: JudgeAgent,
        settings: FeedbackLoopSettings,
    ) -> None:
        """Initialize feedback loop service.

        Args:
            qualitative_agent: Agent for generating assessments.
            judge_agent: Agent for evaluating assessments.
            settings: Feedback loop configuration.
        """
        self._qualitative_agent = qualitative_agent
        self._judge_agent = judge_agent
        self._max_iterations = settings.max_iterations
        self._score_threshold = settings.score_threshold
        self._enabled = settings.enabled

    async def run(self, transcript: Transcript) -> FeedbackLoopResult:
        """Run the complete feedback loop for a transcript.

        Args:
            transcript: Transcript to assess.

        Returns:
            FeedbackLoopResult with final assessment and history.
        """
        logger.info(
            "Starting feedback loop",
            participant_id=transcript.participant_id,
            max_iterations=self._max_iterations,
            enabled=self._enabled,
        )

        # Initial assessment
        assessment = await self._qualitative_agent.assess(transcript)
        evaluation = await self._judge_agent.evaluate(assessment, transcript, iteration=0)

        history: list[tuple[QualitativeAssessment, QualitativeEvaluation]] = [
            (assessment, evaluation)
        ]

        # Skip refinement if disabled
        if not self._enabled:
            logger.info("Feedback loop disabled, returning initial assessment")
            return FeedbackLoopResult(
                final_assessment=assessment,
                final_evaluation=evaluation,
                iterations_used=0,
                history=history,
            )

        iteration = 0

        # Refinement loop
        while self._needs_improvement(evaluation) and iteration < self._max_iterations:
            iteration += 1

            logger.info(
                "Refinement iteration",
                iteration=iteration,
                low_scores=[m.value for m in evaluation.low_scores],
                participant_id=transcript.participant_id,
            )

            # Get feedback for low-scoring metrics
            feedback = await self._judge_agent.get_feedback_for_low_scores(evaluation)

            # Refine assessment
            assessment = await self._qualitative_agent.refine(
                original_assessment=assessment,
                feedback=feedback,
                transcript=transcript,
            )

            # Re-evaluate
            evaluation = await self._judge_agent.evaluate(
                assessment, transcript, iteration=iteration
            )

            history.append((assessment, evaluation))

            logger.info(
                "Refinement complete",
                iteration=iteration,
                average_score=f"{evaluation.average_score:.2f}",
                needs_improvement=self._needs_improvement(evaluation),
            )

        # Log final result
        if self._needs_improvement(evaluation):
            logger.warning(
                "Max iterations reached without full improvement",
                participant_id=transcript.participant_id,
                iterations=iteration,
                remaining_low=[m.value for m in evaluation.low_scores],
            )
        else:
            logger.info(
                "Feedback loop successful",
                participant_id=transcript.participant_id,
                iterations=iteration,
                final_average=f"{evaluation.average_score:.2f}",
            )

        return FeedbackLoopResult(
            final_assessment=assessment,
            final_evaluation=evaluation,
            iterations_used=iteration,
            history=history,
        )

    def _needs_improvement(self, evaluation: QualitativeEvaluation) -> bool:
        """Check if evaluation needs improvement based on configured threshold."""
        return any(score.score <= self._score_threshold for score in evaluation.scores.values())
