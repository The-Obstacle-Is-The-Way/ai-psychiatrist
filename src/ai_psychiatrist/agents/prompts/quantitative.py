"""Quantitative assessment prompt templates.

Paper Reference:
    - Section 2.3.2: Quantitative Assessment
    - Section 2.4.2: Few-shot prompting workflow
    - Appendix D: Hyperparameter optimization

These prompts implement the embedding-based few-shot approach for
predicting PHQ-8 scores from interview transcripts.
"""

from __future__ import annotations

from typing import Literal

# The PHQ-8 item keys used throughout the prompts and parsers.
PHQ8_DOMAIN_KEYS: tuple[str, ...] = (
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
)

QUANTITATIVE_SYSTEM_PROMPT = """You are an AI psychiatrist assessment tool specialized in analyzing \
interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a \
validated depression screening instrument that assesses eight of the nine DSM-5 criteria for \
depressive disorders over the past 2 weeks.

PHQ-8 Scoring System:
- 0 = Not at all (0-1 days in the past 2 weeks)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

Key Principles:
1. Base all assessments strictly on evidence from the transcript
2. When no relevant evidence exists for a symptom, output "N/A" rather than assuming absence (score of 0)
3. Consider both explicit statements and behavioral indicators in the interview
4. Be conservative - only assign scores when there is clear evidence of frequency
5. Remember that absence of mention does not necessarily mean absence of symptom
6. When reference examples are provided, use them for cross-analysis to inform scoring decisions
7. Weight reference examples based on their similarity to the current evidence - highly similar references should be considered more heavily
8. Follow the format specified exactly"""

EVIDENCE_EXTRACT_PROMPT = """Analyze the following therapy transcript and extract specific text chunks \
that provide evidence for each PHQ-8 domain. Based on the evidence found, determine the appropriate \
PHQ-8 score for each domain.

PHQ-8 scoring reference:
- 0 = not at all (0-1 days)
- 1 = several days (2-6 days)
- 2 = more than half the days (7-11 days)
- 3 = nearly every day (12-14 days)

For each domain below, extract direct quotes from the transcript that indicate the presence, absence, \
frequency, or severity of symptoms. Then determine the most appropriate score based on the evidence.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain. Each quote should be \
a direct excerpt from the transcript that supports your scoring decision. Include as many relevant \
quotes as you find - this could range from 0 quotes (if no evidence) to multiple quotes per domain.

Therapy transcript:
{transcript}

Respond with valid JSON matching this structure:
{{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2"],
    "PHQ8_Tired": ["evidence_1"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1"]
}}

Important: Extract UNIQUE quotes only and do not reformat them."""


def make_scoring_prompt(
    transcript: str,
    reference_bundle: str,
    *,
    severity_inference_mode: Literal["strict", "infer"] = "strict",
) -> str:
    """Create the main scoring prompt with optional references.

    Args:
        transcript: Interview transcript text.
        reference_bundle: Formatted reference examples (empty string for zero-shot).
        severity_inference_mode: Prompt policy for scoring frequency/severity.

    Returns:
        Complete user prompt for PHQ-8 scoring.
    """
    reference_section = f"\n{reference_bundle}\n" if reference_bundle else ""

    if severity_inference_mode == "strict":
        return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>
{reference_section}
Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence
4. "confidence": integer 1-5 indicating how confident you are in the score (omit if score is "N/A"):
   - 1: Very uncertain - guessing based on minimal evidence
   - 2: Somewhat uncertain - evidence is weak or ambiguous
   - 3: Moderately confident - some supporting evidence
   - 4: Fairly confident - clear supporting evidence
   - 5: Very confident - strong, unambiguous evidence

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Depressed": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Sleep": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Tired": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Appetite": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Failure": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Concentrating": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}
- "PHQ8_Moving": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ...}}"""

    if severity_inference_mode == "infer":
        return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>
{reference_section}
FREQUENCY INFERENCE GUIDE:

When explicit day-counts are not stated, infer approximate frequency:

| Language Pattern | Inferred Frequency | Score |
|------------------|-------------------|-------|
| "every day", "constantly", "all the time", "always" | 12-14 days | 3 |
| "most days", "usually", "often", "frequently" | 7-11 days | 2 |
| "sometimes", "occasionally", "lately", "recently" | 2-6 days | 1 |
| "once", "rarely", "not really", "never" | 0-1 days | 0 |

For symptom mentions without temporal markers:
- If impact is severe (can't function) → Score 2-3
- If impact is mentioned but manageable → Score 1
- If mentioned casually without distress → Score 0

IMPORTANT: Document your inference reasoning in the 'reason' field.

Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes or behavioral evidence related to each PHQ-8 symptom
2. When reference examples are provided, compare the current evidence with similar reference cases
3. Evaluate the frequency/severity based on available evidence and reference comparisons
4. Consider how similar the reference examples are to the current evidence - if highly similar, give more weight to the reference scores; if less similar, rely more on direct analysis
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Assign numeric scores (0-3) based on evidence and the FREQUENCY INFERENCE GUIDE above.
7. Only output N/A if there is truly no mention of the symptom.

After your analysis, provide your final assessment in <answer> tags as a JSON object.

For each symptom, provide:
1. "evidence": exact quotes from transcript (use "No relevant evidence found" if not discussed)
2. "reason": explanation of scoring decision, including cross-reference analysis when applicable and why N/A if applicable
3. "score": integer 0-3 based on evidence, or "N/A" if no relevant evidence
4. "confidence": integer 1-5 indicating how confident you are in the score (omit if score is "N/A"):
   - 1: Very uncertain - guessing based on minimal evidence
   - 2: Somewhat uncertain - evidence is weak or ambiguous
   - 3: Moderately confident - some supporting evidence
   - 4: Fairly confident - clear supporting evidence
   - 5: Very confident - strong, unambiguous evidence
5. "inference_used": boolean (true if you inferred frequency/severity; false if explicit day-count evidence)
6. "inference_type": string or null ("temporal_marker", "intensity_marker", "impact_statement")
7. "inference_marker": string or null (the word/phrase triggering inference, e.g., "always")

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Depressed": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Sleep": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Tired": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Appetite": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Failure": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Concentrating": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}
- "PHQ8_Moving": {{"evidence": "...", "reason": "...", "score": ..., "confidence": ..., "inference_used": ..., "inference_type": ..., "inference_marker": ...}}"""

    raise ValueError(f"Unknown severity_inference_mode: {severity_inference_mode!r}")


def make_evidence_prompt(transcript: str) -> str:
    """Create the evidence extraction prompt.

    Args:
        transcript: Interview transcript text.

    Returns:
        Complete user prompt for evidence extraction.
    """
    return EVIDENCE_EXTRACT_PROMPT.format(transcript=transcript)
