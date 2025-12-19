"""Prompts for Judge Agent.

Contains the evaluation prompts for the 4 metrics (Specificity, Completeness,
Coherence, Accuracy) as defined in the paper's Appendix B and Spec 07.
"""

from __future__ import annotations

from ai_psychiatrist.domain.enums import EvaluationMetric

# Metric definitions from Appendix B
METRIC_DEFINITIONS = {
    EvaluationMetric.SPECIFICITY: """
Specificity (1-5): Is the assessment specific? Mistakes include using vague/generic statements like 'the patient seems depressed'.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",  # noqa: E501
    EvaluationMetric.COMPLETENESS: """
Completeness (1-5): Does the assessment cover all relevant symptoms, severities, duration/frequency? Mistakes are missed PHQ-8 symptoms, or duration/frequency details.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",  # noqa: E501
    EvaluationMetric.COHERENCE: """
Coherence (1-5): Is the response logically consistent? Mistakes are logically inconsistent statements or contradictions within the assessment.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",  # noqa: E501
    EvaluationMetric.ACCURACY: """
Accuracy (1-5): Are the signs/symptoms aligned with DSM-5 or PHQ-8? Mistakes are incorrect symptoms or incorrect duration/frequecy.

Score from 1-5 solely based on the following criteria:
 - score of 5 : 0 mistakes
 - score of 4 : 1-2 mistakes
 - score of 3 : 3-4 mistakes
 - score of 2 : 5-6 mistakes
 - score of 1 : 7 or more mistakes
""",  # noqa: E501
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

    return f"""Evaluate the following qualitative assessment output for {metric.value.upper()} only. Consider the qualitative assessment (qualitative_assessment) of the transcript (transcript) and compare qualitative_assessment to the provided transcript.  # noqa: E501

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