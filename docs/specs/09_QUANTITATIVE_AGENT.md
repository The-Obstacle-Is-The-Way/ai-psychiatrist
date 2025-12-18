# Spec 09: Quantitative Assessment Agent

## Objective

Implement the quantitative assessment agent that predicts PHQ-8 scores using embedding-based few-shot prompting.

## Paper Reference

- **Section 2.3.2**: Quantitative Assessment
- **Section 2.4.2**: Few-shot prompting workflow
- **Figure 4-5**: Prediction performance comparison
- **Appendix F**: MedGemma achieves 18% better MAE

## Target Configuration (Paper-Optimal)

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| Chat model | MedGemma 27B (example Ollama tag: `alibayram/medgemma:27b`) | Appendix F (MAE 0.505; fewer predictions) |
| Mode | Few-shot | Section 2.4.2 |
| top_k references | 2 per item | Appendix D |
| Temperature | 0.2 | As-is code |

**Expected Performance (Paper):**
- Few-shot MAE: 0.505 (MedGemma) / 0.619 (Gemma 3)
- Zero-shot MAE: 0.796

## As-Is Implementation (Repo)

### Demo Agents (Used by `server.py`)

- Few-shot: `agents/quantitative_assessor_f.py:QuantitativeAssessor`
  - Chat endpoint: `POST /api/chat` with `options={"temperature": 0.2, "top_k": 20, "top_p": 0.8}`
  - Embedding endpoint: `POST /api/embeddings` (L2-normalized; optional truncation via `dim`)
  - Default models: `chat_model="llama3"`, `emb_model="dengcao/Qwen3-Embedding-8B:Q4_K_M"`
  - Default `top_k` references: `3` (paper optimal: `2`)
  - Output schema: dict keyed by `PHQ8_*` with `{"evidence","reason","score"}` plus `_total` and `_severity`
- Zero-shot: `agents/quantitative_assessor_z.py:QuantitativeAssessorZ`
  - Generate endpoint: `POST /api/generate` (streaming)
  - Output schema: raw model text (expects `<answer>{...}</answer>` JSON, but not validated in code)

### Research Implementations (Scripts/Notebooks)

- `quantitative_assessment/quantitative_analysis.py` and `quantitative_assessment/basic_quantitative_analysis.ipynb` implement zero-shot scoring + evaluation metrics and write `results.csv` + `results_detailed.jsonl`.
- `quantitative_assessment/embedding_batch_script.py` and `quantitative_assessment/embedding_quantitative_analysis.ipynb` implement few-shot retrieval runs, hyperparameter sweeps, and t-SNE/retrieval diagnostics.

## Deliverables

1. `src/ai_psychiatrist/agents/quantitative.py` - Quantitative agent
2. `src/ai_psychiatrist/agents/prompts/quantitative.py` - Prompt templates
3. `tests/unit/agents/test_quantitative.py` - Comprehensive tests

## Implementation

### Key Components

```python
"""Quantitative assessment agent for PHQ-8 scoring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ai_psychiatrist.domain.entities import PHQ8Assessment, Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.services.embedding import EmbeddingService
    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient

logger = get_logger(__name__)

# Constants from original implementation
DOMAIN_KEYWORDS = {
    "PHQ8_NoInterest": ["can't be bothered", "no interest", "nothing really", "not enjoy", "no pleasure", "what's the point", "can’t be bothered", "cant be bothered"],
    "PHQ8_Depressed": ["fed up", "miserable", "depressed", "very black", "hopeless", "low"],
    "PHQ8_Sleep": ["sleep", "fall asleep", "wake up", "insomnia", "clock", "tired in the morning"],
    "PHQ8_Tired": ["exhausted", "tired", "little energy", "fatigue", "no energy"],
    "PHQ8_Appetite": ["appetite", "weight", "lost weight", "eat", "eating", "don’t bother", "dont bother", "looser"],
    "PHQ8_Failure": ["useless", "failure", "bad about myself", "burden"],
    "PHQ8_Concentrating": ["concentrat", "memory", "forgot", "thinking of something else", "focus"],
    "PHQ8_Moving": ["moving slowly", "restless", "fidget", "speaking slowly", "psychomotor"]
}


class QuantitativeAssessmentAgent:
    """Agent for predicting PHQ-8 scores from transcripts.

    Supports two modes:
    - Zero-shot: Direct prediction without reference examples
    - Few-shot: Uses embedding-based reference retrieval (paper optimal)
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        embedding_service: EmbeddingService | None = None,
        mode: AssessmentMode = AssessmentMode.FEW_SHOT,
    ) -> None:
        """Initialize quantitative agent."""
        self._llm = llm_client
        self._embedding = embedding_service
        self._mode = mode

    async def assess(self, transcript: Transcript) -> PHQ8Assessment:
        """Generate PHQ-8 assessment for transcript.

        Pipeline:
        1. Extract evidence for each PHQ-8 item (with keyword backfill)
        2. (Few-shot) Build reference bundle via embeddings
        3. Score with LLM using evidence + references
        4. Parse and validate scores with repair
        """
        logger.info("Starting quantitative assessment", mode=self._mode)

        # Step 1: Extract evidence
        evidence = await self._extract_evidence(transcript.text)

        # Step 2: Build references (few-shot only)
        references = ""
        if self._mode == AssessmentMode.FEW_SHOT and self._embedding:
            bundle = await self._embedding.build_reference_bundle(evidence)
            references = bundle.format_for_prompt()

        # Step 3: Score with LLM
        prompt = make_scoring_user_prompt(transcript.text, references)
        raw_response = await self._llm.simple_chat(
            user_prompt=prompt,
            system_prompt=QUANTITATIVE_SYSTEM_PROMPT,
            temperature=0.2, # Original code uses 0.2
        )

        # Step 4: Parse response
        items = await self._parse_response(raw_response, prompt)

        return PHQ8Assessment(
            items=items,
            mode=self._mode,
            participant_id=transcript.participant_id,
        )

    async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
        """Extract evidence quotes for each PHQ-8 item."""
        user_prompt = EVIDENCE_EXTRACT_PROMPT.replace("{transcript}", transcript_text)
        
        raw = await self._llm.simple_chat(user_prompt)
        
        try:
            # Basic JSON parsing
            obj = json.loads(self._strip_json_block(raw))
        except Exception:
            obj = {}
            
        # Clean up extraction
        evidence_dict = {}
        for k in DOMAIN_KEYWORDS.keys():
            arr = obj.get(k, []) if isinstance(obj, dict) else []
            if not isinstance(arr, list):
                arr = []
            evidence_dict[k] = list(set(str(q).strip() for q in arr if str(q).strip()))

        # Keyword Backfill (from original code)
        enriched = self._keyword_backfill(transcript_text, evidence_dict)
        return enriched

    def _keyword_backfill(self, transcript: str, current: dict[str, list[str]], cap: int = 3) -> dict[str, list[str]]:
        """Add keyword-matched sentences when LLM misses evidence."""
        import re
        parts = re.split(r'(?<=[\.?!])\s+|\n+', transcript.strip())
        sents = [p.strip() for p in parts if p and len(p.strip()) > 0]
        
        out = {k: list(v) for k, v in current.items()}
        
        for key, kws in DOMAIN_KEYWORDS.items():
            need = max(0, cap - len(out.get(key, [])))
            if need == 0:
                continue
                
            hits = []
            for s in sents:
                s_lower = s.lower()
                if any(kw in s_lower for kw in kws):
                    hits.append(s)
                if len(hits) >= need:
                    break
            
            if hits:
                existing = set(out.get(key, []))
                merged = out.get(key, []) + [h for h in hits if h not in existing]
                out[key] = merged[:cap]
                
        return out

    async def _parse_response(self, raw: str, original_prompt: str) -> dict[PHQ8Item, ItemAssessment]:
        """Parse JSON response with multi-level repair."""
        # Strategy 1: Clean and Parse
        try:
            clean = self._strip_json_block(raw)
            clean = self._tolerant_fixups(clean)
            data = json.loads(clean)
            return self._validate_and_normalize(data)
        except Exception:
            pass
            
        # Strategy 2: Extract <answer> block
        try:
            import re
            m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw, flags=re.S)
            if m:
                block = m.group(1)
                clean = self._tolerant_fixups(block)
                data = json.loads(clean)
                return self._validate_and_normalize(data)
        except Exception:
            pass

        # Strategy 3: LLM Repair
        try:
            repaired_json = await self._llm_repair(raw)
            if repaired_json:
                return self._validate_and_normalize(repaired_json)
        except Exception:
            pass
            
        # Fallback: Return empty skeleton
        logger.error("Failed to parse quantitative response after all attempts")
        return self._validate_and_normalize({})

    async def _llm_repair(self, malformed: str) -> dict | None:
        """Ask LLM to fix broken JSON."""
        repair_user = (
            "You will be given malformed JSON for a PHQ-8 result. "
            "Output ONLY a valid JSON object with these EXACT keys:\n"
            f"{', '.join(DOMAIN_KEYWORDS.keys())}\n"
            'Each value must be an object: {"evidence": <string>, "reason": <string>, "score": <int 0-3 or \"N/A\">}.\n'
            "If something is missing or unclear, fill with "
            '{"evidence":"No relevant evidence found","reason":"Auto-repaired","score":"N/A"}.\n\n'
            "Malformed JSON:\n"
            f"{malformed}\n\n"
            "Return only the fixed JSON. No prose, no markdown, no tags."
        )
        try:
            fixed = await self._llm.simple_chat(user_prompt=repair_user)
            clean = self._strip_json_block(fixed)
            clean = self._tolerant_fixups(clean)
            return json.loads(clean)
        except Exception:
            return None

    def _strip_json_block(self, s: str) -> str:
        """Strip markdown code blocks and tags."""
        t = s.strip()
        if "<answer>" in t and "</answer>" in t:
            t = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        if t.startswith("```json"):
            t = t[len("```json"):
].strip()
        if t.startswith("```"):
            t = t[len("```"):
].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
        return t

    def _tolerant_fixups(self, s: str) -> str:
        """Fix common JSON syntax errors."""
        import re
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        s = re.sub(r",\s*([\}\]])", r"\1", s) # Remove trailing commas
        return s

    def _validate_and_normalize(self, data: dict) -> dict[PHQ8Item, ItemAssessment]:
        """Convert raw dict to typed ItemAssessment objects."""
        # Implementation to map keys (PHQ8_NoInterest) to enums and ensure types
        # ...
        pass
```

### Prompt Templates (prompts/quantitative.py)

```python
QUANTITATIVE_SYSTEM_PROMPT = """You are a AI psychiatrist assessment tool specialized in analyzing interview transcripts to evaluate PHQ-8 (Patient Health Questionnaire-8) scores. The PHQ-8 is a validated depression screening instrument that assesses eight of the nine DSM-IV criteria for depressive disorders over the past 2 weeks.

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
8. Follow the format specified exactly
"""

EVIDENCE_EXTRACT_PROMPT = """Analyze the following therapy transcript and extract specific text chunks that provide evidence for each PHQ-8 domain. Based on the evidence found, determine the appropriate PHQ-8 score for each domain.

PHQ-8 scoring reference:
- 0 = not at all (0-1 days)
- 1 = several days (2-6 days)
- 2 = more than half the days (7-11 days)
- 3 = nearly every day (12-14 days)

For each domain below, extract direct quotes from the transcript that indicate the presence, absence, frequency, or severity of symptoms. Then determine the most appropriate score based on the evidence.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain. Each quote should be a direct excerpt from the transcript that supports your scoring decision. Include as many relevant quotes as you find - this could range from 0 quotes (if no evidence) to multiple quotes per domain.

Therapy transcript:
{transcript}

Respond with valid JSON matching this structure:
{
    "PHQ8_NoInterest": ["evidence_1", "evidence_2"],
    "PHQ8_Depressed": ["evidence_1"],
    "PHQ8_Sleep": ["evidence_1", "evidence_2"],
    "PHQ8_Tired": ["evidence_1"],
    "PHQ8_Appetite": [],
    "PHQ8_Failure": ["evidence_1"],
    "PHQ8_Concentrating": ["evidence_1"],
    "PHQ8_Moving": ["evidence_1"]
}

Important: Extract UNIQUE quotes only and do not reformat them.
"""

def make_scoring_user_prompt(transcript: str, reference_bundle: str) -> str:
    return f"""Analyze the following interview transcript and predict PHQ-8 scores for each symptom domain.

Interview transcript:
<transcript>
{transcript}
</transcript>

{reference_bundle}

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
- "PHQ8_NoInterest": {{evidence, reason, score}}
- "PHQ8_Depressed": {{evidence, reason, score}}
- "PHQ8_Sleep": {{evidence, reason, score}}
- "PHQ8_Tired": {{evidence, reason, score}}
- "PHQ8_Appetite": {{evidence, reason, score}}
- "PHQ8_Failure": {{evidence, reason, score}}
- "PHQ8_Concentrating": {{evidence, reason, score}}
- "PHQ8_Moving": {{evidence, reason, score}}"""
```

## Acceptance Criteria

- [ ] Supports zero-shot and few-shot modes
- [ ] Extracts evidence with keyword backfill (using DOMAIN_KEYWORDS)
- [ ] Builds reference bundle for few-shot using EmbeddingService
- [ ] Implements multi-level JSON repair (fixups -> regex -> LLM repair)
- [ ] Calculates total score and severity
- [ ] Handles N/A scores correctly
- [ ] Paper metrics reproducible (MAE 0.619 few-shot vs 0.796 zero-shot)

## Dependencies

- **Spec 02**: Domain entities (PHQ8Assessment)
- **Spec 04**: LLM infrastructure
- **Spec 08**: Embedding service (few-shot mode)

## Specs That Depend on This

- **Spec 10**: Meta-Review Agent
- **Spec 11**: Full Pipeline
