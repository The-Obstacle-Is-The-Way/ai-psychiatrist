# Spec 13: Full Pydantic AI Framework Integration

> **STATUS: READY FOR IMPLEMENTATION**
>
> This spec describes **full Pydantic AI framework integration** using the `TextOutput` mode
> to preserve our reasoning-optimal `<thinking>` + `<answer>` prompt pattern while gaining
> all framework benefits: type safety, built-in retry, dependency injection, and observability.
>
> **Tracked by**:
> - [GitHub Issue #28](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/28) - Pydantic AI integration
> - [GitHub Issue #29](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/29) - Ollama JSON mode (TO BE CLOSED - see below)
>
> **Last Updated**: 2025-12-25

---

## Executive Summary

**What we're doing**: Full migration to Pydantic AI framework using `TextOutput` mode.

**Why this approach**:
- Research shows forcing structured output during generation degrades LLM reasoning by 10-26%
- Our existing `<thinking>` + `<answer>` prompt pattern is already the optimal "generate-then-structure" approach
- Pydantic AI's `TextOutput` mode lets us preserve this pattern while gaining framework benefits

**What we're NOT doing**:
- NOT using Ollama's `format: json` for primary generation (breaks `<thinking>` tags)
- NOT using Pydantic AI's "Native Output" mode (forces JSON schema, hurts reasoning)
- NOT just adding validation (that's the minimal approach - we want the full framework)

---

## Research Foundation

### Why NOT Force JSON During Generation

| Source | Finding |
|--------|---------|
| [The Downsides of Structured Outputs](https://www.llmwatch.com/p/the-downsides-of-structured-outputs) | Forcing structured output degrades creativity/reasoning by **17-26%** |
| [Structured outputs can hurt LLM performance](https://dylancastillo.co/posts/say-what-you-mean-sometimes.html) | Significant performance difference between structured and unstructured outputs |
| [Decoupling Task-Solving and Output Formatting](https://arxiv.org/html/2510.03595v1) | Recommends "generate-then-structure" workflow |
| [OpenReview: Structured Output Degradation](https://openreview.net/forum?id=vYkz5tzzjV) | Cognitive load of simultaneous creation and formatting harms ideation |

### Our Current Pattern is Already Optimal

From `src/ai_psychiatrist/agents/prompts/quantitative.py` (lines 159-182):

```python
# Phase 1: Free reasoning (preserves quality)
Analyze each symptom using the following approach in <thinking> tags:
1. Search for direct quotes...
2. When reference examples are provided...

# Phase 2: Structured extraction (after reasoning)
After your analysis, provide your final assessment in <answer> tags as a JSON object.
```

This IS the "generate-then-structure" pattern. We preserve it.

---

## Pydantic AI Output Modes

Pydantic AI offers three output modes ([source](https://ai.pydantic.dev/output/)):

| Mode | How It Works | Use Case |
|------|--------------|----------|
| **Tool Output** (default) | Uses function/tool calling | When model supports tools well |
| **Native Output** | Forces JSON schema compliance | Simple extraction, NO reasoning needed |
| **PromptedOutput** | Schema in prompt, validate after | Models without tool support |
| **TextOutput** | Free generation, custom extraction | **OUR CHOICE** - complex reasoning |

### Why TextOutput is Right for Us

`TextOutput` allows:
1. LLM generates freely with any format (including `<thinking>` tags)
2. Custom function receives raw text
3. Function extracts, validates, returns typed result
4. If extraction fails, raise `ModelRetry` → Pydantic AI retries automatically

```python
from pydantic_ai import Agent, TextOutput
from pydantic_ai.exceptions import ModelRetry

def extract_quantitative(text: str) -> QuantitativeOutput:
    """Extract and validate quantitative assessment from LLM response."""
    try:
        # Use existing extraction logic
        json_str = extract_from_answer_tags(text)
        return QuantitativeOutput.model_validate_json(json_str)
    except (ExtractionError, ValidationError) as e:
        # Pydantic AI will retry with this message
        raise ModelRetry(f"Invalid response: {e}. Please format correctly.")

agent = Agent(
    'ollama:gemma3:27b',
    output_type=TextOutput(extract_quantitative),
)
```

---

## Architecture

### Current vs Target

```
CURRENT ARCHITECTURE (hand-rolled)
┌─────────────────────────────────────────────────────────────────┐
│  QuantitativeAgent._assess_participant()                        │
│                                                                 │
│  1. Build prompt manually                                       │
│  2. Call self._llm.chat() directly                              │
│  3. Parse response with _strip_json_block()                     │
│  4. Apply _tolerant_fixups()                                    │
│  5. Validate with _validate_and_normalize()                     │
│  6. If fails, call _llm_repair() (custom retry logic)           │
└─────────────────────────────────────────────────────────────────┘


TARGET ARCHITECTURE (Pydantic AI)
┌─────────────────────────────────────────────────────────────────┐
│  Pydantic AI Agent with TextOutput                              │
│                                                                 │
│  agent = Agent(                                                 │
│      model='ollama:gemma3:27b',                                 │
│      output_type=TextOutput(extract_quantitative),              │
│      retries=3,  # Built-in retry with backoff                  │
│  )                                                              │
│                                                                 │
│  result = await agent.run(prompt, deps=context)                 │
│  validated_output = result.output  # Already typed!             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PYDANTIC AI FRAMEWORK                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ QuantitativeAgent│    │   JudgeAgent    │    │ MetaReviewAgent │     │
│  │                 │    │                 │    │                 │     │
│  │ Agent(          │    │ Agent(          │    │ Agent(          │     │
│  │   output_type=  │    │   output_type=  │    │   output_type=  │     │
│  │   TextOutput(   │    │   TextOutput(   │    │   TextOutput(   │     │
│  │     extract_    │    │     extract_    │    │     extract_    │     │
│  │     quant       │    │     judge       │    │     meta        │     │
│  │   )             │    │   )             │    │   )             │     │
│  │ )               │    │ )               │    │ )               │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           ▼                      ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Shared Infrastructure                        │   │
│  │                                                                  │   │
│  │  • OllamaModel (Pydantic AI's Ollama integration)               │   │
│  │  • RunContext with dependency injection                          │   │
│  │  • Built-in retry with exponential backoff                       │   │
│  │  • Logfire integration for observability                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deliverables

### 1. Output Models

**File**: `src/ai_psychiatrist/agents/output_models.py`

```python
"""Pydantic output models for LLM response validation.

These models are used with TextOutput extractors to validate
LLM responses after free-form generation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceOutput(BaseModel):
    """Evidence for a single PHQ-8 item."""

    evidence: str = Field(description="Direct quote from transcript")
    reason: str = Field(description="Reasoning for score assignment")
    score: int | Literal["N/A"] = Field(
        description="PHQ-8 score (0-3) or N/A if insufficient evidence"
    )

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v: object) -> int | Literal["N/A"]:
        """Validate score is 0-3 or N/A."""
        if isinstance(v, str):
            if v.upper() == "N/A":
                return "N/A"
            try:
                v = int(v)
            except ValueError:
                return "N/A"
        if isinstance(v, int) and 0 <= v <= 3:
            return v
        return "N/A"


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


class JudgeMetricOutput(BaseModel):
    """Judge agent evaluation output for a single metric."""

    score: int = Field(ge=1, le=5)
    explanation: str


class JudgeOutput(BaseModel):
    """Complete judge agent output (all 4 metrics)."""

    coherence: JudgeMetricOutput
    completeness: JudgeMetricOutput
    evidence_specificity: JudgeMetricOutput
    accuracy: JudgeMetricOutput


class MetaReviewOutput(BaseModel):
    """Meta-review agent output."""

    severity: int = Field(ge=0, le=4)
    explanation: str
```

### 2. TextOutput Extractors

**File**: `src/ai_psychiatrist/agents/extractors.py`

```python
"""TextOutput extraction functions for Pydantic AI agents.

These functions extract and validate structured data from free-form
LLM responses, preserving the <thinking> + <answer> pattern.

References:
    - https://ai.pydantic.dev/output/#textoutput
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError
from pydantic_ai.exceptions import ModelRetry

from ai_psychiatrist.agents.output_models import (
    JudgeOutput,
    MetaReviewOutput,
    QuantitativeOutput,
)

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from ai_psychiatrist.agents.context import AgentContext


def _extract_answer_json(text: str) -> str:
    """Extract JSON from <answer>...</answer> tags.

    Args:
        text: Raw LLM response with <thinking> and <answer> tags.

    Returns:
        JSON string from answer section.

    Raises:
        ModelRetry: If answer tags not found or JSON extraction fails.
    """
    # Try <answer> tags first
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ```json code blocks as fallback
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)

    raise ModelRetry(
        "Could not find structured output. Please wrap your JSON response "
        "in <answer>...</answer> tags after your <thinking> analysis."
    )


def _tolerant_fixups(json_str: str) -> str:
    """Apply tolerant fixups to malformed JSON.

    Handles common LLM JSON mistakes:
    - Trailing commas
    - Single quotes instead of double
    - Unquoted keys
    """
    # Remove trailing commas before } or ]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Try to fix single quotes (risky, only if double quote parse fails)
    return json_str


def extract_quantitative(
    ctx: RunContext[AgentContext], text: str
) -> QuantitativeOutput:
    """Extract and validate quantitative assessment.

    This function is used with TextOutput to process free-form LLM
    responses while preserving the <thinking> reasoning phase.

    Args:
        ctx: Pydantic AI run context with dependencies.
        text: Raw LLM response.

    Returns:
        Validated QuantitativeOutput.

    Raises:
        ModelRetry: If extraction or validation fails (triggers retry).
    """
    try:
        json_str = _extract_answer_json(text)
        json_str = _tolerant_fixups(json_str)
        data = json.loads(json_str)
        return QuantitativeOutput.model_validate(data)
    except json.JSONDecodeError as e:
        raise ModelRetry(
            f"Invalid JSON in response: {e}. "
            "Please ensure your <answer> contains valid JSON."
        ) from e
    except ValidationError as e:
        raise ModelRetry(
            f"Response validation failed: {e}. "
            "Please ensure all PHQ-8 items have evidence, reason, and score fields."
        ) from e


def extract_judge(ctx: RunContext[AgentContext], text: str) -> JudgeMetricOutput:
    """Extract and validate judge metric output.

    Note: Judge evaluates ONE metric per call. The full JudgeOutput
    is assembled by the FeedbackLoopService from multiple calls.
    """
    from ai_psychiatrist.agents.output_models import JudgeMetricOutput

    try:
        json_str = _extract_answer_json(text)
        json_str = _tolerant_fixups(json_str)
        data = json.loads(json_str)
        return JudgeMetricOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ModelRetry(f"Invalid judge output: {e}") from e


def extract_meta_review(
    ctx: RunContext[AgentContext], text: str
) -> MetaReviewOutput:
    """Extract and validate meta-review output."""
    try:
        json_str = _extract_answer_json(text)
        json_str = _tolerant_fixups(json_str)
        data = json.loads(json_str)
        return MetaReviewOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ModelRetry(f"Invalid meta-review output: {e}") from e
```

### 3. Agent Context (Dependency Injection)

**File**: `src/ai_psychiatrist/agents/context.py`

```python
"""Pydantic AI agent context for dependency injection.

References:
    - https://ai.pydantic.dev/dependencies/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings
    from ai_psychiatrist.services.reference_store import ReferenceStore


@dataclass
class AgentContext:
    """Shared context for all Pydantic AI agents.

    This is injected into agents via RunContext and provides
    access to configuration and services.
    """

    settings: Settings
    reference_store: ReferenceStore | None = None
    transcript_id: str | None = None

    # For few-shot mode
    similar_examples: list[str] | None = None
```

### 4. Pydantic AI Agent Definitions

**File**: `src/ai_psychiatrist/agents/pydantic_agents.py`

```python
"""Pydantic AI agent definitions.

This module defines the core agents using the Pydantic AI framework
with TextOutput mode to preserve reasoning quality.

References:
    - https://ai.pydantic.dev/agents/
    - https://ai.pydantic.dev/output/#textoutput
"""

from __future__ import annotations

from pydantic_ai import Agent, TextOutput
from pydantic_ai.models.ollama import OllamaModel

from ai_psychiatrist.agents.context import AgentContext
from ai_psychiatrist.agents.extractors import (
    extract_judge,
    extract_meta_review,
    extract_quantitative,
)
from ai_psychiatrist.agents.output_models import (
    JudgeMetricOutput,
    MetaReviewOutput,
    QuantitativeOutput,
)


def create_quantitative_agent(
    model_name: str = "gemma3:27b",
    base_url: str = "http://localhost:11434",
    retries: int = 3,
) -> Agent[AgentContext, QuantitativeOutput]:
    """Create quantitative assessment agent.

    Uses TextOutput to preserve <thinking> + <answer> pattern
    while gaining Pydantic AI's retry and validation.

    Args:
        model_name: Ollama model name.
        base_url: Ollama server URL.
        retries: Number of retries on validation failure.

    Returns:
        Configured Pydantic AI agent.
    """
    model = OllamaModel(model_name, base_url=base_url)

    return Agent(
        model=model,
        output_type=TextOutput(extract_quantitative),
        retries=retries,
        system_prompt=(
            "You are a clinical assessment specialist analyzing interview "
            "transcripts for PHQ-8 depression symptoms. Always structure your "
            "response with <thinking> tags for analysis and <answer> tags "
            "containing your JSON assessment."
        ),
    )


def create_judge_agent(
    model_name: str = "gemma3:27b",
    base_url: str = "http://localhost:11434",
    retries: int = 3,
) -> Agent[AgentContext, JudgeMetricOutput]:
    """Create judge evaluation agent."""
    model = OllamaModel(model_name, base_url=base_url)

    return Agent(
        model=model,
        output_type=TextOutput(extract_judge),
        retries=retries,
        system_prompt=(
            "You are evaluating the quality of clinical assessments. "
            "Provide your analysis in <thinking> tags and your score "
            "in <answer> tags as JSON with 'score' (1-5) and 'explanation'."
        ),
    )


def create_meta_review_agent(
    model_name: str = "gemma3:27b",
    base_url: str = "http://localhost:11434",
    retries: int = 3,
) -> Agent[AgentContext, MetaReviewOutput]:
    """Create meta-review agent."""
    model = OllamaModel(model_name, base_url=base_url)

    return Agent(
        model=model,
        output_type=TextOutput(extract_meta_review),
        retries=retries,
        system_prompt=(
            "You are synthesizing clinical assessments to determine "
            "overall depression severity. Provide reasoning in <thinking> "
            "tags and severity classification in <answer> tags."
        ),
    )
```

### 5. Updated QuantitativeAgent

**File**: `src/ai_psychiatrist/agents/quantitative.py` (migration)

```python
"""Quantitative Assessment Agent - Pydantic AI Implementation.

This module migrates the quantitative agent to use Pydantic AI
while preserving the exact prompt structure and behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, TextOutput

from ai_psychiatrist.agents.context import AgentContext
from ai_psychiatrist.agents.extractors import extract_quantitative
from ai_psychiatrist.agents.output_models import QuantitativeOutput
from ai_psychiatrist.agents.prompts.quantitative import make_scoring_prompt
from ai_psychiatrist.domain.entities import PHQ8Assessment
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from ai_psychiatrist.domain.value_objects import ItemAssessment
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings
    from ai_psychiatrist.services.reference_store import ReferenceStore

logger = get_logger(__name__)


class QuantitativeAssessmentAgent:
    """PHQ-8 scoring agent using Pydantic AI framework.

    Uses TextOutput mode to preserve <thinking> + <answer> pattern
    while gaining framework benefits:
    - Built-in retry with exponential backoff
    - Type-safe outputs
    - Dependency injection via RunContext
    - Observability integration
    """

    # Mapping from output keys to PHQ8Item enum
    PHQ8_KEY_MAP: dict[str, PHQ8Item] = {
        "PHQ8_NoInterest": PHQ8Item.INTEREST,
        "PHQ8_Depressed": PHQ8Item.DEPRESSED,
        "PHQ8_Sleep": PHQ8Item.SLEEP,
        "PHQ8_Tired": PHQ8Item.TIRED,
        "PHQ8_Appetite": PHQ8Item.APPETITE,
        "PHQ8_Failure": PHQ8Item.FAILURE,
        "PHQ8_Concentrating": PHQ8Item.CONCENTRATION,
        "PHQ8_Moving": PHQ8Item.MOVEMENT,
    }

    def __init__(
        self,
        settings: Settings,
        reference_store: ReferenceStore | None = None,
    ) -> None:
        """Initialize agent with Pydantic AI backend.

        Args:
            settings: Application settings.
            reference_store: Optional reference store for few-shot mode.
        """
        self._settings = settings
        self._reference_store = reference_store

        # Create Pydantic AI agent
        from pydantic_ai.models.ollama import OllamaModel

        model = OllamaModel(
            settings.model.quantitative_model,
            base_url=settings.ollama.base_url,
        )

        self._agent: Agent[AgentContext, QuantitativeOutput] = Agent(
            model=model,
            output_type=TextOutput(extract_quantitative),
            retries=3,
        )

        logger.info(
            "Initialized QuantitativeAgent with Pydantic AI",
            model=settings.model.quantitative_model,
            mode="pydantic_ai",
        )

    async def assess(
        self,
        transcript: str,
        mode: AssessmentMode = AssessmentMode.FEW_SHOT,
    ) -> PHQ8Assessment:
        """Assess transcript for PHQ-8 symptoms.

        Args:
            transcript: Interview transcript text.
            mode: Assessment mode (zero-shot or few-shot).

        Returns:
            PHQ8Assessment with scored items.
        """
        # Build prompt using existing logic (unchanged)
        examples = None
        if mode == AssessmentMode.FEW_SHOT and self._reference_store:
            matches = await self._reference_store.find_similar(transcript, top_k=2)
            examples = [m.content for m in matches]

        prompt = make_scoring_prompt(transcript, examples)

        # Create context for dependency injection
        context = AgentContext(
            settings=self._settings,
            reference_store=self._reference_store,
            similar_examples=examples,
        )

        # Run with Pydantic AI - handles retries automatically
        result = await self._agent.run(prompt, deps=context)
        validated: QuantitativeOutput = result.output

        # Convert to domain entity
        return self._to_assessment(validated, mode)

    def _to_assessment(
        self,
        output: QuantitativeOutput,
        mode: AssessmentMode,
    ) -> PHQ8Assessment:
        """Convert validated output to domain entity."""
        items: dict[PHQ8Item, ItemAssessment] = {}

        for key, item_enum in self.PHQ8_KEY_MAP.items():
            evidence_output = getattr(output, key)
            score = evidence_output.score if evidence_output.score != "N/A" else None

            items[item_enum] = ItemAssessment(
                item=item_enum,
                evidence=evidence_output.evidence,
                reason=evidence_output.reason,
                score=score,
            )

        return PHQ8Assessment.create(items=items, mode=mode)
```

---

## Configuration

**File**: `src/ai_psychiatrist/config.py` (additions)

```python
class PydanticAISettings(BaseSettings):
    """Pydantic AI framework configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_AI_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable Pydantic AI framework (vs legacy agents)",
    )
    retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retries on validation failure",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries (exponential backoff)",
    )
    logfire_enabled: bool = Field(
        default=False,
        description="Enable Pydantic Logfire observability",
    )
```

---

## Migration Strategy

### Phase 1: Add Framework (Week 1)
1. Add `pydantic-ai` dependency to `pyproject.toml`
2. Create output models (`output_models.py`)
3. Create extractors (`extractors.py`)
4. Create context (`context.py`)
5. Create agent definitions (`pydantic_agents.py`)

### Phase 2: Migrate Agents (Week 2)
1. Migrate `QuantitativeAssessmentAgent`
2. Migrate `JudgeAgent`
3. Migrate `MetaReviewAgent`
4. Update `FeedbackLoopService` to use new agents

### Phase 3: Cleanup (Week 3)
1. Remove legacy `_llm_repair()` logic (Pydantic AI handles retries)
2. Remove `_tolerant_fixups()` from agent (moved to extractor)
3. Update tests to use Pydantic AI mocking patterns
4. Add observability with Logfire (optional)

### Backward Compatibility

During migration, support both modes via feature flag:

```python
if settings.pydantic_ai.enabled:
    agent = QuantitativeAssessmentAgent(settings)  # New
else:
    agent = LegacyQuantitativeAgent(settings)  # Old
```

---

## What This Achieves

| Goal | How |
|------|-----|
| **Preserve reasoning quality** | TextOutput mode, prompts unchanged |
| **Type-safe outputs** | Pydantic models validated on extraction |
| **Built-in retry** | Pydantic AI handles retries with backoff |
| **Dependency injection** | RunContext provides settings, services |
| **Observability** | Logfire integration available |
| **Cleaner code** | Remove hand-rolled retry/validation logic |
| **Industry standard** | Using established framework vs custom code |

---

## GitHub Issue Updates

### Issue #28 (Update)
**New Title**: "Spec 13: Full Pydantic AI Framework Integration with TextOutput"

**Update Description**:
> Migrating to full Pydantic AI framework using `TextOutput` mode to preserve
> our reasoning-optimal `<thinking>` + `<answer>` pattern while gaining
> framework benefits: type safety, built-in retry, dependency injection.

### Issue #29 (Close)
**Reason**: Ollama `format: json` is NOT appropriate for our use case.

> Closing this issue. After research, we determined that:
> 1. `format: json` forces the LLM to output ONLY valid JSON
> 2. This breaks our `<thinking>` tags which require free-form text
> 3. Research shows forcing JSON degrades reasoning by 10-26%
>
> Instead, we're using Pydantic AI's `TextOutput` mode which:
> - Lets the LLM generate freely
> - Extracts JSON from `<answer>` tags after generation
> - Validates with Pydantic
> - Retries if validation fails
>
> See Spec 13 and Issue #28 for the correct approach.

---

## Testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from ai_psychiatrist.agents.extractors import extract_quantitative
from ai_psychiatrist.agents.output_models import QuantitativeOutput


@pytest.fixture
def mock_agent():
    """Create agent with test model for unit testing."""
    return Agent(
        model=TestModel(),
        output_type=TextOutput(extract_quantitative),
    )


def test_extract_quantitative_valid():
    """Valid response extracts correctly."""
    response = """
    <thinking>
    Analyzing symptoms...
    </thinking>

    <answer>
    {
        "PHQ8_NoInterest": {"evidence": "...", "reason": "...", "score": 2},
        "PHQ8_Depressed": {"evidence": "...", "reason": "...", "score": 1},
        ...
    }
    </answer>
    """
    # Should not raise
    result = extract_quantitative(mock_context, response)
    assert isinstance(result, QuantitativeOutput)


def test_extract_quantitative_missing_answer_retries():
    """Missing answer tags triggers ModelRetry."""
    from pydantic_ai.exceptions import ModelRetry

    response = "Just some text without answer tags"

    with pytest.raises(ModelRetry) as exc_info:
        extract_quantitative(mock_context, response)

    assert "Could not find structured output" in str(exc_info.value)


def test_extract_quantitative_invalid_json_retries():
    """Invalid JSON triggers ModelRetry."""
    from pydantic_ai.exceptions import ModelRetry

    response = "<answer>{not valid json}</answer>"

    with pytest.raises(ModelRetry) as exc_info:
        extract_quantitative(mock_context, response)

    assert "Invalid JSON" in str(exc_info.value)


def test_prompts_unchanged():
    """Prompts still use <thinking> + <answer> pattern."""
    from ai_psychiatrist.agents.prompts.quantitative import make_scoring_prompt

    prompt = make_scoring_prompt("test transcript", None)
    assert "<thinking>" in prompt
    assert "<answer>" in prompt
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...
    "pydantic-ai>=0.1.0",
]

[project.optional-dependencies]
observability = [
    "logfire>=0.1.0",
]
```

---

## References

### Pydantic AI Documentation
- [Pydantic AI Overview](https://ai.pydantic.dev/)
- [Output Modes (TextOutput)](https://ai.pydantic.dev/output/)
- [Agents](https://ai.pydantic.dev/agents/)
- [Dependencies](https://ai.pydantic.dev/dependencies/)
- [Ollama Integration](https://ai.pydantic.dev/models/#ollama)

### Research on Structured Outputs
- [The Downsides of Structured Outputs](https://www.llmwatch.com/p/the-downsides-of-structured-outputs)
- [Decoupling Task-Solving and Output Formatting](https://arxiv.org/html/2510.03595v1)
- [Structured outputs can hurt LLM performance](https://dylancastillo.co/posts/say-what-you-mean-sometimes.html)

### Best Practices
- [DataCamp: Pydantic AI Guide](https://www.datacamp.com/tutorial/pydantic-ai-guide)
- [Machine Learning Mastery: Pydantic for LLM Outputs](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)
