# Spec 13: Structured Output Migration

> **STATUS: DEFERRED - POST-REPLICATION ENHANCEMENT**
>
> This spec is deferred until paper replication is fully validated with real
> E2E testing. The current XML-based parsing pipeline is complete and working.
>
> This spec covers two related enhancements:
> 1. **Pydantic AI** for validated structured outputs (GitHub #28)
> 2. **Ollama JSON mode** as a simpler alternative (GitHub #29)
>
> **Tracked by**:
> - [GitHub Issue #28](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/28) - Pydantic AI
> - [GitHub Issue #29](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/29) - Ollama JSON mode
>
> **Last Updated**: 2025-12-25

---

## Objective

Improve LLM output parsing reliability by migrating from manual XML/JSON repair
cascades to validated structured outputs. Two approaches are evaluated:

1. **Pydantic AI**: Full schema validation with type safety
2. **Ollama JSON mode**: Native JSON generation without repair

Both approaches preserve the paper-aligned XML path for replication fidelity.

---

## Current State

### What Works

The current quantitative agent uses a multi-level repair cascade:

```python
# Current parsing pipeline in src/ai_psychiatrist/agents/quantitative.py

def _parse_response(self, raw: str) -> dict:
    """Parse LLM response with repair cascade."""
    # Level 1: Strip XML tags and code fences
    cleaned = self._strip_json_block(raw)

    # Level 2: Tolerant syntax fixups
    fixed = self._tolerant_fixups(cleaned)

    # Level 3: Try JSON parse
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        # Level 4: LLM repair (recursive call with error context)
        return self._llm_repair(raw)
```

**Statistics from recent runs**:
- ~15% of responses require Level 2 fixups
- ~3% require Level 4 LLM repair
- <1% fail completely (return fallback skeleton)

### What Could Be Better

| Concern | Current State | With Structured Outputs |
|---------|--------------|------------------------|
| Type safety | Runtime checks only | Compile-time validation |
| Schema validation | Manual field checks | Automatic via Pydantic |
| Error messages | Generic JSON errors | Field-level validation errors |
| Maintenance | Repair code is complex | Declarative schemas |

---

## Approach 1: Pydantic AI (Comprehensive)

### Overview

Use [Pydantic AI](https://ai.pydantic.dev/) to obtain validated structured outputs
from LLMs with automatic schema enforcement.

### Deliverables

1. **Output Schemas** (`src/ai_psychiatrist/agents/output_models.py`):

```python
"""Pydantic output models for LLM responses."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class EvidenceOutput(BaseModel):
    """Evidence for a single PHQ-8 item."""

    evidence: str = Field(description="Direct quote from transcript")
    reason: str = Field(description="Reasoning for score assignment")
    score: int | Literal["N/A"] = Field(
        description="PHQ-8 score (0-3) or N/A if insufficient evidence"
    )


class QuantitativeOutput(BaseModel):
    """Complete quantitative assessment output."""

    PHQ8_NoInterest: EvidenceOutput
    PHQ8_Depressed: EvidenceOutput
    PHQ8_Sleep: EvidenceOutput
    PHQ8_Tired: EvidenceOutput
    PHQ8_Appetite: EvidenceOutput
    PHQ8_Failure: EvidenceOutput
    PHQ8_Concentrating: EvidenceOutput
    PHQ8_Moving: EvidenceOutput


class JudgeFeedbackOutput(BaseModel):
    """Judge agent evaluation output."""

    specificity: int = Field(ge=1, le=5)
    specificity_reasoning: str
    completeness: int = Field(ge=1, le=5)
    completeness_reasoning: str
    coherence: int = Field(ge=1, le=5)
    coherence_reasoning: str
    accuracy: int = Field(ge=1, le=5)
    accuracy_reasoning: str
    overall_feedback: str


class QualitativeOutput(BaseModel):
    """Qualitative assessment output."""

    phq8_symptoms: str
    biological_factors: str
    social_factors: str
    risk_factors: str


class MetaReviewOutput(BaseModel):
    """Meta-review agent output."""

    severity: Literal[
        "minimal", "mild", "moderate", "moderately_severe", "severe"
    ]
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
```

2. **Pydantic AI Adapter** (`src/ai_psychiatrist/infrastructure/llm/pydantic_ai_client.py`):

```python
"""Pydantic AI client adapter."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from ai_psychiatrist.config import get_settings

T = TypeVar("T", bound=BaseModel)


class PydanticAIClient:
    """Adapter for Pydantic AI structured outputs."""

    def __init__(self):
        settings = get_settings()
        # Ollama exposes OpenAI-compatible API at /v1
        self.model = OpenAIModel(
            model_name=settings.model.quantitative_model,
            base_url=f"{settings.ollama.base_url}/v1",
        )

    async def generate_structured(
        self,
        prompt: str,
        output_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate validated structured output."""
        agent = Agent(
            model=self.model,
            result_type=output_model,
            system_prompt=system_prompt or "",
        )
        result = await agent.run(prompt)
        return result.data
```

3. **Configuration** (additions to `src/ai_psychiatrist/config.py`):

```python
class LLMOutputSettings(BaseSettings):
    """LLM output mode configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_OUTPUT_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    mode: Literal["xml", "structured", "json"] = Field(
        default="xml",
        description="Output mode: xml (paper), structured (Pydantic AI), json (Ollama)",
    )
```

4. **Agent Updates** (example for quantitative agent):

```python
class QuantitativeAssessmentAgent:
    """Quantitative agent with configurable output mode."""

    def __init__(self, output_mode: str = "xml"):
        self.output_mode = output_mode
        if output_mode == "structured":
            self.pydantic_client = PydanticAIClient()

    async def assess(self, transcript: str, ...) -> PHQ8Assessment:
        if self.output_mode == "structured":
            return await self._assess_structured(transcript, ...)
        else:
            return await self._assess_xml(transcript, ...)

    async def _assess_structured(self, transcript: str, ...) -> PHQ8Assessment:
        """Use Pydantic AI for validated output."""
        output = await self.pydantic_client.generate_structured(
            prompt=self._build_prompt(transcript),
            output_model=QuantitativeOutput,
            system_prompt=QUANTITATIVE_SYSTEM_PROMPT,
        )
        return self._to_domain(output)

    def _to_domain(self, output: QuantitativeOutput) -> PHQ8Assessment:
        """Map Pydantic output to domain entity."""
        # ... mapping logic
```

---

## Approach 2: Ollama JSON Mode (Simpler)

### Overview

Use Ollama's native `format: json` parameter to get cleaner JSON outputs without
the repair cascade. This is simpler than Pydantic AI but provides less validation.

### Implementation

**File**: `src/ai_psychiatrist/infrastructure/llm/ollama.py`

```python
async def chat_json(
    self,
    messages: list[dict],
    model: str | None = None,
) -> dict:
    """Chat with JSON format enforcement."""
    response = await self._post(
        self.settings.chat_url,
        json={
            "model": model or self.settings.model.quantitative_model,
            "messages": messages,
            "format": "json",  # <-- Ollama JSON mode
            "stream": False,
        },
    )
    return json.loads(response["message"]["content"])
```

### Trade-offs

| Aspect | Pydantic AI | Ollama JSON Mode |
|--------|-------------|------------------|
| Validation | Full schema + types | JSON syntax only |
| Dependencies | pydantic-ai package | None (native Ollama) |
| Complexity | Medium | Low |
| Error handling | Field-level | Parse-level |
| Schema evolution | Easy (Pydantic models) | Manual |

---

## Evaluation Criteria

Before choosing an approach, evaluate:

1. **Reliability**: Does JSON mode eliminate repair needs?
2. **Performance**: Any latency impact from structured generation?
3. **Accuracy**: Does enforced schema affect model reasoning?
4. **Maintenance**: Is the complexity reduction worth the migration?

### Evaluation Script

```python
"""Evaluate structured output approaches."""

import asyncio
from pathlib import Path

async def evaluate_approaches():
    """Compare XML, JSON mode, and Pydantic AI."""
    test_transcripts = list(Path("data/transcripts").glob("*/"))[:10]

    results = {
        "xml": {"success": 0, "repair_needed": 0, "failed": 0, "latency": []},
        "json": {"success": 0, "repair_needed": 0, "failed": 0, "latency": []},
        "pydantic": {"success": 0, "repair_needed": 0, "failed": 0, "latency": []},
    }

    for transcript_dir in test_transcripts:
        # Test each approach and collect metrics
        ...

    # Report comparison
    print("=== Structured Output Evaluation ===")
    for approach, metrics in results.items():
        print(f"\n{approach.upper()}:")
        print(f"  Success rate: {metrics['success'] / len(test_transcripts) * 100:.1f}%")
        print(f"  Avg latency: {sum(metrics['latency']) / len(metrics['latency']):.2f}s")
```

---

## Implementation Plan

### Phase 1: Evaluate Ollama JSON Mode (Quick Win)

1. Add `chat_json()` method to OllamaClient
2. Test on 10 transcripts
3. Compare repair rate vs current cascade
4. **Decision point**: If JSON mode eliminates repairs, use it

### Phase 2: Pydantic AI Integration (If Needed)

1. Install pydantic-ai dependency
2. Create output schema models
3. Create PydanticAIClient adapter
4. Add feature flag
5. Test on validation set

### Phase 3: Migration (If Approved)

1. Update agents to use structured output
2. Keep XML path as fallback
3. Update tests
4. Document migration

---

## Acceptance Criteria

- [ ] XML path unchanged and still the default
- [ ] Feature flag controls output mode: `LLM_OUTPUT_MODE=xml|json|structured`
- [ ] Structured path produces domain entities equivalent to XML path
- [ ] Pydantic AI usage is fully optional and gated by config
- [ ] No domain code depends on Pydantic
- [ ] Tests cover all output modes
- [ ] Evaluation results documented

---

## Testing

### Unit Tests

```python
def test_quantitative_output_schema():
    """Test output schema validation."""
    valid = QuantitativeOutput(
        PHQ8_NoInterest=EvidenceOutput(
            evidence="I don't enjoy anything",
            reason="Direct statement of anhedonia",
            score=2,
        ),
        # ... other items
    )
    assert valid.PHQ8_NoInterest.score == 2


def test_invalid_score_rejected():
    """Invalid scores should be rejected."""
    with pytest.raises(ValidationError):
        EvidenceOutput(evidence="test", reason="test", score=5)  # >3


def test_xml_path_unchanged():
    """XML parsing should be unaffected by new code."""
    # Regression test for paper fidelity
```

### Integration Tests

```python
@pytest.mark.integration
async def test_structured_vs_xml_equivalence():
    """Structured output should match XML output."""
    transcript = load_test_transcript()

    xml_result = await agent.assess(transcript, mode="xml")
    structured_result = await agent.assess(transcript, mode="structured")

    assert xml_result.total_score == structured_result.total_score
```

---

## Dependencies

- `pydantic-ai>=0.1.0` (optional, only for structured mode)
- Ollama with `/v1` OpenAI-compatible endpoint (for Pydantic AI)

---

## Rollout Notes

1. Keep structured output path **off by default**
2. Evaluate on validation set before enabling
3. Only enable after paper replication is complete
4. Monitor repair rate metrics to verify improvement

---

## References

- GitHub Issue #28: Pydantic AI migration request
- GitHub Issue #29: Ollama JSON mode evaluation
- Pydantic AI docs: https://ai.pydantic.dev/
- Ollama API: `format: json` parameter
- Current repair cascade: `src/ai_psychiatrist/agents/quantitative.py`
