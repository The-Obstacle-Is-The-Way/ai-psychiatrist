"""Quantitative assessment prompt templates.

Paper Reference:
    - Section 2.3.2: Quantitative Assessment
    - Section 2.4.2: Few-shot prompting workflow
    - Appendix D: Hyperparameter optimization

These prompts implement the embedding-based few-shot approach for
predicting PHQ-8 scores from interview transcripts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _get_keywords_path() -> Path:
    """Get path to PHQ-8 keywords YAML file.

    Returns:
        Path to data/keywords/phq8_keywords.yaml relative to project root.
    """
    # Navigate from this file to project root
    # This file: src/ai_psychiatrist/agents/prompts/quantitative.py
    # Project root: 4 levels up
    project_root = Path(__file__).parent.parent.parent.parent.parent
    return project_root / "data" / "keywords" / "phq8_keywords.yaml"


@lru_cache(maxsize=1)
def _load_domain_keywords() -> dict[str, list[str]]:
    """Load PHQ-8 domain keywords from YAML file.

    Keywords are cached after first load for performance.

    Returns:
        Dictionary mapping PHQ-8 item keys to keyword lists.

    Raises:
        FileNotFoundError: If keywords file is missing.
        yaml.YAMLError: If YAML is malformed.
    """
    keywords_path = _get_keywords_path()
    with keywords_path.open("r") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    # Validate structure
    result: dict[str, list[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            result[key] = [str(v) for v in value]
    return result


# Domain keywords for keyword backfill
# Loaded from data/keywords/phq8_keywords.yaml for easier clinical review
# Used to catch evidence when LLM extraction misses relevant sentences
DOMAIN_KEYWORDS: dict[str, list[str]] = _load_domain_keywords()

QUANTITATIVE_SYSTEM_PROMPT = """You are an AI psychiatrist assessment tool specialized in analyzing \
interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a \
validated depression screening instrument that assesses eight of the nine DSM-IV criteria for \
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


def make_scoring_prompt(transcript: str, reference_bundle: str) -> str:
    """Create the main scoring prompt with optional references.

    Args:
        transcript: Interview transcript text.
        reference_bundle: Formatted reference examples (empty string for zero-shot).

    Returns:
        Complete user prompt for PHQ-8 scoring.
    """
    reference_section = f"\n{reference_bundle}\n" if reference_bundle else ""

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

Return ONLY a JSON object in <answer> tags with these exact keys:
- "PHQ8_NoInterest": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Depressed": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Sleep": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Tired": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Appetite": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Failure": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Concentrating": {{"evidence": "...", "reason": "...", "score": ...}}
- "PHQ8_Moving": {{"evidence": "...", "reason": "...", "score": ...}}"""


def make_evidence_prompt(transcript: str) -> str:
    """Create the evidence extraction prompt.

    Args:
        transcript: Interview transcript text.

    Returns:
        Complete user prompt for evidence extraction.
    """
    return EVIDENCE_EXTRACT_PROMPT.format(transcript=transcript)
