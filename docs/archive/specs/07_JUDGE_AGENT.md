# Spec 07: Judge Agent & Feedback Loop

## Objective

Implement the judge agent that evaluates qualitative assessments and the iterative self-refinement feedback loop described in the paper.

## Paper Reference

- **Section 2.3.1**: Judge agent evaluation with 4 metrics
- **Appendix B**: Metric definitions (specificity, completeness, coherence, accuracy)
- **Figure 2**: Evaluation scores before/after feedback loop

## As-Is Implementation (Repo)

### Demo Judge (Used by `server.py`)

- File: `agents/qualitive_evaluator.py`
- Class: `QualitativeEvaluatorAgent`
- Endpoint: `POST /api/chat` (non-stream)
- Default model: `"llama3"`
- Options: `{"temperature": 0, "top_k": 20, "top_p": 0.9}`
- Behavior: evaluates metrics **once** and returns a dict of `{metric: score}`; no iterative loop in `server.py`.

### Research Feedback Loop Script (Not Wired Into `server.py`)

- File: `qualitative_assessment/feedback_loop.py`
- Trigger condition: any metric score `<= 2` (not paper’s `< 4`)
- Max iterations: `10`
- Output format expectations differ (it requests `Score:` then `Explanation:` and enforces integer-only scores)

### Paper vs Repo Note

The paper’s feedback loop trigger is “original evaluation score **below four**” with a max of **10** iterations.
The as-is demo server does not implement this loop; only the standalone cluster script implements a loop (with a different threshold).

## Deliverables

1. `src/ai_psychiatrist/agents/judge.py` - Judge evaluation agent
2. `src/ai_psychiatrist/services/feedback_loop.py` - Iterative refinement service
3. `tests/unit/agents/test_judge.py` - Comprehensive tests
4. `tests/integration/test_feedback_loop.py` - Integration tests

## Implementation

### As-Is Metric Prompts (Verbatim)

From `agents/qualitive_evaluator.py`:

```python
coherence_prompt = f"""Evaluate the following qualitative assessment output for COHERENCE only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

            Coherence (1–5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions within the assessment.

            Score from 1-5 solely based on the following criteria:

             - score of 5 : 0 mistakes
             - score of 4 : 1-2 mistakes
             - score of 3 : 3-4 mistakes
             - score of 2 : 5-6 mistakes
             - score of 1 : 7 or more mistakes

            Format your response as:
            Explanation: [Your rating, as text]
            Score: [your score, as a number between 1 and 5]

            ---
            Here is the transcript:
            {transcript}

            Here is the assessment based on the transcript:
            {qualitative_output}
            ---"""

completeness_prompt = f"""Evaluate the following qualitative assessment output for COMPLETENESS only. Compare qualitative_assessment to the provided transcript. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Completeness (1–5): Does the assessment cover all relevant symptoms, severities, duration/frequency? Mistakes are missed PHQ-8 symptoms, or duration/frequency details.

        Score from 1-5 solely based on the following criteria:
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        Here is the transcript:
        {transcript}

        Here is the assessment based on the transcript:
        {qualitative_output}
        ---"""

specificity_prompt = f"""Evaluate the following qualitative assessment output for SPECIFICITY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Specificity (1–5): Is the assessment specific? Mistakes include using vague/generic statements like 'the patient seems depressed'.

        Score from 1-5 solely based on the following criteria:
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        ---
        Here is the transcript:
        {transcript}

        Here is the assessment based on the transcript:
        {qualitative_output}
        ---"""

accuracy_prompt = f"""Evaluate the following qualitative assessment output for ACCURACY only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

        Accuracy (1–5): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Mistakes are incorrect symptoms or incorrect duration/frequency.

        Score from 1-5 solely based on the following criteria:
         - score of 5 : 0 mistakes
         - score of 4 : 1-2 mistakes
         - score of 3 : 3-4 mistakes
         - score of 2 : 5-6 mistakes
         - score of 1 : 7 or more mistakes

        Format your response as:
        Explanation: [Your rating, as text]
        Score: [your score, as a number between 1 and 5]

        ---
        Here is the transcript:
        {transcript}

        Here is the assessment based on the transcript:
        {qualitative_output}
        ---"""
```

### 1. Judge Agent (agents/judge.py)

```python
"""Judge agent for evaluating qualitative assessments."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient

logger = get_logger(__name__)


# Metric definitions from Appendix B and agents/qualitive_evaluator.py
METRIC_DEFINITIONS = {
    EvaluationMetric.SPECIFICITY: """
Specificity (1-5): Is the assessment specific? Mistakes include using vague/generic statements like 'the patient seems depressed'.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",
    EvaluationMetric.COMPLETENESS: """
Completeness (1-5): Does the assessment cover all relevant symptoms, severities, duration/frequency? Mistakes are missed PHQ-8 symptoms, or duration/frequency details.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",
    EvaluationMetric.COHERENCE: """
Coherence (1-5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions within the assessment.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",
    EvaluationMetric.ACCURACY: """
Accuracy (1-5): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Mistakes are incorrect symptoms or incorrect duration/frequency.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",
}


def make_evaluation_prompt(
    metric: EvaluationMetric,
    transcript: str,
    assessment: str,
) -> str:
    """Generate evaluation prompt for a specific metric.

    Args:
        metric: Evaluation metric to assess.
        transcript: Original interview transcript.
        assessment: Qualitative assessment to evaluate.

    Returns:
        Formatted evaluation prompt.
    """
    definition = METRIC_DEFINITIONS[metric]

    return f"""Evaluate the following qualitative assessment output for {metric.value.upper()} only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.

{definition}

Format your response as:
Explanation: [Your rating, as text]
Score: [your score, as a number between 1 and 5]

---
Here is the transcript:
{transcript}

Here is the assessment based on the transcript:
{assessment}
---"""


class JudgeAgent:
    """Agent for evaluating qualitative assessments.

    Implements the LLM-as-a-judge approach described in Section 2.3.1.
    Evaluates assessments on 4 metrics using a 5-point Likert scale.
    """

    def __init__(self, llm_client: SimpleChatClient) -> None:
        """Initialize judge agent.

        Args:
            llm_client: LLM client for evaluations.
        """
        self._llm_client = llm_client

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

        # Note: The original implementation used temperature=0, top_k=20, top_p=0.9
        try:
            response = await self._llm_client.simple_chat(
                user_prompt=prompt,
                temperature=0.0,
            )
        except LLMError as e:
            logger.error(
                "LLM call failed during metric evaluation",
                metric=metric.value,
                error=str(e),
            )
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
    ) -> dict[str, str]:
        """Extract feedback text for low-scoring metrics.

        Args:
            evaluation: Evaluation with scores.

        Returns:
            Dictionary of metric name -> feedback explanation.
        """
        feedback = {}
        for metric in evaluation.low_scores:
            score = evaluation.get_score(metric)
            feedback[metric.value] = (
                f"Scored {score.score}/5. {score.explanation}"
            )
        return feedback
```

### 2. Feedback Loop Service (services/feedback_loop.py)

```python
"""Iterative self-refinement feedback loop service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_psychiatrist.domain.entities import (
    QualitativeAssessment,
    QualitativeEvaluation,
    Transcript,
)
from ai_psychiatrist.domain.exceptions import MaxIterationsError
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.agents.judge import JudgeAgent
    from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
    from ai_psychiatrist.config import FeedbackLoopSettings

logger = get_logger(__name__)


@dataclass
class FeedbackLoopResult:
    """Result of the feedback loop refinement process."""

    final_assessment: QualitativeAssessment
    final_evaluation: QualitativeEvaluation
    iterations_used: int
    history: list[tuple[QualitativeAssessment, QualitativeEvaluation]] = field(
        default_factory=list
    )

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

        Raises:
            MaxIterationsError: If max iterations reached without acceptable scores.
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
        while evaluation.needs_improvement and iteration < self._max_iterations:
            iteration += 1

            logger.info(
                "Refinement iteration",
                iteration=iteration,
                low_scores=[m.value for m in evaluation.low_scores],
                participant_id=transcript.participant_id,
            )

            # Get feedback for low-scoring metrics
            feedback = self._judge_agent.get_feedback_for_low_scores(evaluation)

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
                needs_improvement=evaluation.needs_improvement,
            )

        # Log final result
        if evaluation.needs_improvement:
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
```

### 3. Tests (test_judge.py)

```python
"""Tests for judge agent and feedback loop."""

from __future__ import annotations

import pytest
from uuid import uuid4

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.domain.entities import QualitativeAssessment, Transcript
from ai_psychiatrist.domain.enums import EvaluationMetric
from tests.fixtures.mock_llm import MockLLMClient


class TestJudgeAgent:
    """Tests for JudgeAgent."""

    @pytest.fixture
    def mock_high_score_response(self) -> str:
        """Response indicating high score."""
        return """
Explanation: The assessment is highly specific.
Score: 5
"""

    @pytest.fixture
    def mock_low_score_response(self) -> str:
        """Response indicating low score."""
        return """
Explanation: The assessment is too vague.
Score: 2
"""

    @pytest.fixture
    def sample_assessment(self) -> QualitativeAssessment:
        """Create sample assessment."""
        return QualitativeAssessment(
            overall="Patient shows moderate depression symptoms.",
            phq8_symptoms="Multiple symptoms present.",
            social_factors="Financial stress mentioned.",
            biological_factors="History of depression.",
            risk_factors="Previous suicide attempt.",
            participant_id=123,
        )

    @pytest.fixture
    def sample_transcript(self) -> Transcript:
        """Create sample transcript."""
        return Transcript(
            participant_id=123,
            text="Ellie: How are you?\nParticipant: Not well.",
        )

    @pytest.mark.asyncio
    async def test_evaluate_all_metrics(
        self,
        mock_high_score_response: str,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should evaluate all 4 metrics."""
        # 4 responses for 4 metrics
        mock_client = MockLLMClient(
            chat_responses=[mock_high_score_response] * 4
        )
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        assert len(evaluation.scores) == 4
        assert EvaluationMetric.COHERENCE in evaluation.scores
        assert EvaluationMetric.COMPLETENESS in evaluation.scores
        assert EvaluationMetric.SPECIFICITY in evaluation.scores
        assert EvaluationMetric.ACCURACY in evaluation.scores

    @pytest.mark.asyncio
    async def test_extracts_scores_correctly(
        self,
        mock_high_score_response: str,
        mock_low_score_response: str,
        sample_assessment: QualitativeAssessment,
        sample_transcript: Transcript,
    ) -> None:
        """Should extract correct numeric scores."""
        # Mix of high and low scores
        mock_client = MockLLMClient(
            chat_responses=[
                mock_high_score_response,  # coherence: 5
                mock_low_score_response,   # completeness: 2
                mock_high_score_response,  # specificity: 5
                mock_low_score_response,   # accuracy: 2
            ]
        )
        agent = JudgeAgent(llm_client=mock_client)

        evaluation = await agent.evaluate(sample_assessment, sample_transcript)

        assert evaluation.scores[EvaluationMetric.COHERENCE].score == 5
        assert evaluation.scores[EvaluationMetric.COMPLETENESS].score == 2
        assert evaluation.needs_improvement
        assert EvaluationMetric.COMPLETENESS in evaluation.low_scores
```

## Acceptance Criteria

- [ ] Evaluates all 4 metrics (coherence, completeness, specificity, accuracy)
- [ ] Scores extracted correctly from LLM responses (1-5 scale)
- [ ] Feedback loop triggered by configurable threshold:
  - **As-is code**: `score <= 2` triggers (default for parity)
  - **Paper behavior**: `score < 4` triggers (set `FEEDBACK_SCORE_THRESHOLD=3`)
- [ ] Feedback loop respects max iterations (default: 10, per paper)
- [ ] Assessment improves through iterations
- [ ] Can be disabled via configuration (`FEEDBACK_ENABLED=false`)
- [ ] History preserved for analysis
- [ ] Comprehensive logging throughout

## Dependencies

- **Spec 02**: Domain entities (QualitativeEvaluation, EvaluationScore)
- **Spec 04**: LLM infrastructure
- **Spec 06**: Qualitative Agent

## Specs That Depend on This

- **Spec 11**: Full Pipeline (uses feedback loop)
