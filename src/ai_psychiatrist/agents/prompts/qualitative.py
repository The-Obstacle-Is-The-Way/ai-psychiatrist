"""Prompt templates for qualitative assessment.

Paper Reference:
    - Section 2.3.1: Qualitative Assessment Agent
    - Appendix B: Four assessment domains (PHQ-8 symptoms, biological, social, risk factors)
"""

from __future__ import annotations

QUALITATIVE_SYSTEM_PROMPT = (
    "You are a psychiatrist. Your job is to assess and diagnose patients for "
    "depression. Be as objective, concise, and accurate as possible. Be specific "
    "and avoid vague generalities. Use exact quotes from the transcript to "
    "support your assessment for each tag."
)


def make_qualitative_prompt(transcript: str) -> str:
    """Generate qualitative assessment prompt.

    Args:
        transcript: Interview transcript text.

    Returns:
        Formatted user prompt.
    """
    intro = (
        "The following is an interview transcript from a psychiatric assessment "
        "of a subject who is being screened for depression. Please note that "
        "'Ellie' is the interviewer, and 'Participant' is the subject being assessed."
    )
    return f"""{intro}
{transcript}

Please produce a qualitative assessment across these domains:
1) Overall assessment of the subject's mental health
2) PHQ-8 symptoms discussed (include frequency/duration/severity when available)
3) Social factors that may influence mental health
4) Biological factors that may influence mental health
5) Risk factors the subject may be experiencing

Examples (for format only, do NOT reuse content):
- PHQ-8 symptoms: "I don't enjoy anything anymore" (frequency: nearly every day)
- Social factors: "Things have been tense at home"
- Biological factors: "My mother had depression"
- Risk factors: "I feel isolated since losing my job"

Requirements:
- Be objective, concise, and clinically grounded (avoid vague generalities).
- Use exact quotes from the transcript as evidence within each domain.
- Collect all quoted evidence again in <exact_quotes> as bullet points.
- If a domain is not covered in the interview, write "not assessed in interview".

Return ONLY this XML (each tag on its own line; no additional text outside the tags):

<assessment>...</assessment>
<PHQ8_symptoms>...</PHQ8_symptoms>
<social_factors>...</social_factors>
<biological_factors>...</biological_factors>
<risk_factors>...</risk_factors>
<exact_quotes>...</exact_quotes>
"""


def make_feedback_prompt(
    original_assessment: str,
    feedback: dict[str, str],
    transcript: str,
) -> str:
    """Generate prompt for assessment refinement based on feedback.

    Args:
        original_assessment: Previous assessment output.
        feedback: Dictionary of metric -> feedback text.
        transcript: Original transcript.

    Returns:
        Formatted refinement prompt.
    """
    feedback_text = "\n".join(
        f"- **{metric.upper()}**: {text}" for metric, text in feedback.items()
    )

    return f"""The following qualitative assessment has been evaluated and needs improvement.

EVALUATION FEEDBACK:
{feedback_text}

ORIGINAL ASSESSMENT:
{original_assessment}

TRANSCRIPT:
{transcript}

Please provide an improved assessment that addresses the feedback above. Use the same XML format:

<assessment>...</assessment>
<PHQ8_symptoms>...</PHQ8_symptoms>
<social_factors>...</social_factors>
<biological_factors>...</biological_factors>
<risk_factors>...</risk_factors>
<exact_quotes>...</exact_quotes>

Ensure:
1. More specific evidence with exact quotes
2. Complete coverage of all PHQ-8 symptoms
3. Logical consistency throughout
4. Accurate alignment with clinical criteria

Return only the XML (no additional text outside the tags)."""
