"""Meta-review agent prompts.

Paper Reference:
    - Section 2.3.3: Meta Review
    - Section 3.3: Meta Review Results (78% accuracy)

The meta-review agent integrates qualitative and quantitative assessments
to predict overall depression severity.
"""

from __future__ import annotations

META_REVIEW_SYSTEM_PROMPT = """You are an AI psychiatrist assistant specializing in depression. \
Your task is to analyze the interview transcript and related qualitative and quantitative \
assessments, and then predict the diagnosis label and severity level."""


def make_meta_review_prompt(
    transcript: str,
    qualitative: str,
    quantitative: str,
) -> str:
    """Create meta-review prompt from assessments.

    Args:
        transcript: Full interview transcript text.
        qualitative: Formatted qualitative assessment.
        quantitative: Formatted quantitative PHQ-8 scores.

    Returns:
        Formatted prompt for the meta-review agent.
    """
    return f"""Please review a participant's interview transcript, qualitative assessment, \
and quantitative assessment below.

Here is the interview transcript in <transcript> tags:
<transcript>
{transcript}
</transcript>

Here are the overall assessment, social and biological risk factors that may influence \
the participant's mental health in <qualitative_assessment> tags:
<qualitative_assessment>
{qualitative}
</qualitative_assessment>

Here are the predicted PHQ-8 scores and explanations based on the available information \
in the interview transcript in <quantitative_assessment> tags:
<quantitative_assessment>
{quantitative}
</quantitative_assessment>

According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), \
a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; \
a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; \
15 to 19, moderately severe; and 20 to 24, severe.

Note that the interview may not cover all eight PHQ-8 items. Therefore, directly \
summing the available scores and deriving a severity level could underestimate the \
participant's condition. Please use the available information to infer and predict \
the participant's condition as accurately as possible.

Please predict their severity level in <severity> tags, where 0 means no significant \
depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means \
moderately severe symptoms, and 4 means severe symptoms.

Please explain your predictions in <explanation> tags. Please provide answers in the \
XML format with each tag on a new line."""
