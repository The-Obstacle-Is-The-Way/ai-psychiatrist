# Spec 10: Meta-Review Agent

## Objective

Implement the meta-review agent that integrates qualitative and quantitative assessments to predict overall depression severity.

## Paper Reference

- **Section 2.3.3**: Meta Review
- **Section 3.3**: Meta Review Results (78% accuracy, comparable to human expert)
- **Table 1**: Performance metrics

## As-Is Implementation (Repo)

### Demo Meta Reviewer (Used by `server.py`)

- File: `agents/meta_reviewer.py`
- Client: `ollama.Client(host=\"http://localhost:11434\")`
- Model: hardcoded `"llama3"`
- Prompt: single user message (no separate system message)
- Expected output tags: `<severity>` (0–4) and `<explanation>`
- Note: prompt text asks for a “diagnosis label” but the required output tags do not include a diagnosis field.

### Research Meta Review Script + Visualization

- `meta_review/meta_review.py` runs meta-review over a dataset, typically with `gemma3-optimized:27b` and deterministic options.
- `visualization/meta_review_heatmap.ipynb` contains the ground-truth→severity/diagnosis mapping and computes accuracy, balanced accuracy, precision, recall, and F1 (Table 1).

## Deliverables

1. `src/ai_psychiatrist/agents/meta_review.py` - Meta-review agent
2. `tests/unit/agents/test_meta_review.py` - Tests

## Implementation

### As-Is Prompt (Verbatim from `agents/meta_reviewer.py`)

```python
prompt = f'''You are an AI psychiatrist assistant specializing in depression. Your task is to analyze the interview transcript and related qualitative and quantitative assessments, and then predict the diagnosis label and severity level.
        
                Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.  

                Here is the interview transcript in <transcript> tags:
                <transcript>
                {interview}
                </transcript>

                Here are the overall assessment, social and biological risk factors that may influence the participant's mental health in <qualitative_assessment> tags:
                <qualitative_assessment>
                {qualitative}
                </qualitative_assessment>

                Here are the predicted PHQ-8 scores and explanations based on the available information in the interview transcript in <quantitative_assessment> tags:
                <quantitative_assessment>\n{quantitative}</quantitative_assessment>

                According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; 15 to 19, moderately severe; and 20 to 24, severe. \
                Note that the interview may not cover all eight PHQ-8 items. Therefore, directly summing the available scores and deriving a severity level could underestimate the participant's condition. Please use the available information to infer and predict the participant's condition as accurately as possible. \
                Please predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms. \
                Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
                '''
```

```python
"""Meta-review agent for integrating assessments."""

from __future__ import annotations

from ai_psychiatrist.domain.entities import (
    MetaReview,
    PHQ8Assessment,
    QualitativeAssessment,
    Transcript,
)
from ai_psychiatrist.domain.enums import SeverityLevel


META_REVIEW_SYSTEM_PROMPT = """You are an AI psychiatrist assistant specializing in depression. Your task is to analyze the interview transcript and related qualitative and quantitative assessments, and then predict the diagnosis label and severity level."""

def make_meta_review_prompt(transcript: str, qualitative: str, quantitative: str) -> str:
    return f"""Please review a participant's interview transcript, qualitative assessment, and quantitative assessment below.  

        Here is the interview transcript in <transcript> tags:
        <transcript>
        {transcript}
        </transcript>

        Here are the overall assessment, social and biological risk factors that may influence the participant's mental health in <qualitative_assessment> tags:
        <qualitative_assessment>
        {qualitative}
        </qualitative_assessment>

        Here are the predicted PHQ-8 scores and explanations based on the available information in the interview transcript in <quantitative_assessment> tags:
        <quantitative_assessment>\n{quantitative}</quantitative_assessment>

        According to the Patient Health Questionnaire eight-item depression scale (PHQ-8), a total PHQ-8 score of 0 to 4 represents no significant depressive symptoms; a total score of 5 to 9 represents mild depressive symptoms; 10 to 14, moderate; 15 to 19, moderately severe; and 20 to 24, severe.
        Note that the interview may not cover all eight PHQ-8 items. Therefore, directly summing the available scores and deriving a severity level could underestimate the participant's condition. Please use the available information to infer and predict the participant's condition as accurately as possible.
        Please predict their severity level in <severity> tags, where 0 means no significant depressive symptoms, 1 means mild symptoms, 2 means moderate symptoms, 3 means moderately severe symptoms, and 4 means severe symptoms.
        Please explain your predictions in <explanation> tags. Please provide answers in the XML format with each tag on a new line.
        """


class MetaReviewAgent:
    """Agent for integrating assessments into final severity prediction."""

    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm = llm_client

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
        # Format quantitative scores
        quant_text = self._format_quantitative(quantitative)
        # Format qualitative assessment (stripping outer tags if present to avoid nesting issues)
        qual_text = qualitative.full_text

        prompt = make_meta_review_prompt(
            transcript=transcript.text,
            qualitative=qual_text,
            quantitative=quant_text,
        )

        response = await self._llm.simple_chat(
            user_prompt=prompt,
            system_prompt=META_REVIEW_SYSTEM_PROMPT
        )

        severity, explanation = self._parse_response(response, quantitative)

        return MetaReview(
            severity=severity,
            explanation=explanation,
            quantitative_assessment_id=quantitative.id,
            qualitative_assessment_id=qualitative.id,
            participant_id=transcript.participant_id,
        )

    def _format_quantitative(self, assessment: PHQ8Assessment) -> str:
        """Format PHQ-8 scores for prompt to match original implementation's format."""
        # The original implementation constructed XML-like tags for each score
        lines = []
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
        self, raw: str, quantitative: PHQ8Assessment
    ) -> tuple[SeverityLevel, str]:
        """Parse severity and explanation from response."""
        from ai_psychiatrist.infrastructure.llm.responses import extract_xml_tags

        tags = extract_xml_tags(raw, ["severity", "explanation"])

        # Parse severity
        try:
            severity_int = int(tags.get("severity", "0").strip())
            severity = SeverityLevel(max(0, min(4, severity_int)))
        except (ValueError, TypeError):
            # Fall back to quantitative-derived severity
            severity = quantitative.severity

        explanation = tags.get("explanation", raw.strip())

        return severity, explanation

## Acceptance Criteria

- [ ] Integrates transcript + qualitative + quantitative inputs
- [ ] Predicts severity level (0-4)
- [ ] Handles N/A PHQ-8 items appropriately
- [ ] Provides clinical reasoning explanation
- [ ] Falls back to quantitative severity on parse failure
- [ ] Paper accuracy reproducible (~78%)

## Dependencies

- **Spec 02**: Domain entities (MetaReview, SeverityLevel)
- **Spec 04**: LLM infrastructure
- **Spec 06**: Qualitative assessment
- **Spec 09**: Quantitative assessment

## Specs That Depend on This

- **Spec 11**: Full Pipeline

```
