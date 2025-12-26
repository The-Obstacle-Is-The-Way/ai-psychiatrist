# Spec 13: Full Pydantic AI Framework Integration

> **STATUS: IMPLEMENTED (Quantitative scoring path; `PYDANTIC_AI_ENABLED` opt-in)**
>
> This spec describes Pydantic AI framework integration using the `TextOutput` mode
> to preserve our reasoning-optimal `<thinking>` + `<answer>` prompt pattern while gaining
> framework benefits: type safety and built-in retry loops.
>
> **Tracked by**:
> - [GitHub Issue #28](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/28) - Pydantic AI integration
> - [GitHub Issue #29](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/29) - Ollama JSON mode (TO BE CLOSED - see below)
>
> **Last Updated**: 2025-12-26

---

## Executive Summary

**What we're doing**: Use Pydantic AI `TextOutput` for the quantitative scoring step (opt-in).

**Why this approach**:
- Strict format constraints can reduce performance on some tasks/models; we keep our existing “reason → then structure” pattern and validate after generation
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
| [Decoupling Task-Solving and Output Formatting](https://arxiv.org/html/2510.03595v1) | Shows that entangling task-solving instructions with strict format constraints can hurt performance; proposes decoupling format from reasoning |
| [Structured outputs can hurt LLM performance](https://dylancastillo.co/posts/say-what-you-mean-sometimes.html) | Survey + experiments showing outcomes depend on model, task, and prompt design (structured is not “always better”) |
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

Pydantic AI offers four output modes ([source](https://ai.pydantic.dev/output/)):

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
from pydantic_ai import Agent, ModelRetry, TextOutput

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
│      retries=3,  # Built-in retry loop on validation errors     │
│  )                                                              │
│                                                                 │
│  result = await agent.run(prompt)                               │
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
│  │  • OpenAIChatModel + OllamaProvider (Ollama /v1 OpenAI-compatible) │   │
│  │  • Built-in retry loop on validation errors                       │   │
│  │  • TextOutput extractors (parse/validate after generation)        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deliverables

### 1. Output Models

**File**: `src/ai_psychiatrist/agents/output_models.py`

Authoritative implementation: `src/ai_psychiatrist/agents/output_models.py`.

Schema (implemented):
- `EvidenceOutput`: `evidence: str`, `reason: str`, `score: int | None` (`None` means N/A)
- `QuantitativeOutput`: 8 PHQ-8 keys, each an `EvidenceOutput`:
  `PHQ8_NoInterest`, `PHQ8_Depressed`, `PHQ8_Sleep`, `PHQ8_Tired`, `PHQ8_Appetite`,
  `PHQ8_Failure`, `PHQ8_Concentrating`, `PHQ8_Moving`
- `JudgeMetricOutput`: `score: int` (1–5), `explanation: str`
- `JudgeOutput`: `coherence`, `completeness`, `specificity`, `accuracy` (each a `JudgeMetricOutput`)
- `MetaReviewOutput`: `severity: int` (0–4), `explanation: str`

### 2. TextOutput Extractors

**File**: `src/ai_psychiatrist/agents/extractors.py`

These are small, single-purpose extractors that:
- Prefer `<answer>...</answer>` JSON (fallbacks: code fences, then first `{...}` block)
- Apply tolerant fixups (smart quotes, trailing commas)
- Parse + validate to Pydantic models
- Raise `ModelRetry(...)` with actionable guidance to trigger framework retries

Entry points (implemented):
- `extract_quantitative(text: str) -> QuantitativeOutput`
- `extract_judge_metric(text: str) -> JudgeMetricOutput`
- `extract_meta_review(text: str) -> MetaReviewOutput`

Implementation is authoritative in `src/ai_psychiatrist/agents/extractors.py`.

### 3. Pydantic AI Agent Definitions

**File**: `src/ai_psychiatrist/agents/pydantic_agents.py`

Implemented factories:
- `create_quantitative_agent(...)` (wired into `QuantitativeAssessmentAgent` when enabled)
- `create_judge_metric_agent(...)` and `create_meta_review_agent(...)` (not yet wired into the pipeline)

Pydantic AI talks to Ollama via the OpenAI-compatible `/v1` endpoint using
`OpenAIChatModel` + `OllamaProvider` (see `src/ai_psychiatrist/agents/pydantic_agents.py`).

### 4. Updated `QuantitativeAssessmentAgent` (Scoring Step Migration)

**File**: `src/ai_psychiatrist/agents/quantitative.py`

We keep the **public API** and the full pipeline intact:
- Evidence extraction (LLM → JSON)
- Optional keyword backfill
- Optional embedding-based reference retrieval (few-shot)
- **Scoring** (LLM → `<thinking>` + `<answer>` JSON) ← **this is where we integrate Pydantic AI**

The only behavioral change is the scoring call + parsing:
- **Legacy path**: one LLM call + hand-rolled parsing/repair.
- **Pydantic AI path** (when `PYDANTIC_AI_ENABLED=true`): Pydantic AI performs the LLM call and retries automatically until `extract_quantitative(...)` returns a validated `QuantitativeOutput`.

Notes:
- Pydantic AI uses Ollama’s **OpenAI-compatible `/v1` endpoint** (see `create_quantitative_agent`).
- `extract_quantitative(...)` raises `ModelRetry(...)` when it cannot parse/validate the `<answer>` JSON.
- We keep the legacy path for rollback and to avoid forcing network calls in unit tests.

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
        default=False,
        description="Enable Pydantic AI scoring path (legacy remains default until validated)",
    )
    retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retries on validation failure",
    )
```

---

## Migration Strategy

### Phase 1: Add Framework (Week 1)
1. Add `pydantic-ai` dependency to `pyproject.toml`
2. Create output models (`output_models.py`)
3. Create extractors (`extractors.py`)
4. Create agent definitions (`pydantic_agents.py`)

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
agent = QuantitativeAssessmentAgent(
    llm_client=llm_client,  # existing SimpleChatClient (e.g., OllamaClient)
    embedding_service=embedding_service,
    mode=mode,
    model_settings=settings.model,
    quantitative_settings=settings.quantitative,
    pydantic_ai_settings=settings.pydantic_ai,
    ollama_base_url=settings.ollama.base_url,
)
```

---

## What This Achieves

| Goal | How |
|------|-----|
| **Preserve reasoning quality** | TextOutput mode, prompts unchanged |
| **Type-safe outputs** | Pydantic models validated on extraction |
| **Built-in retry** | Pydantic AI handles retry loops on validation errors |
| **Cleaner scoring path (opt-in)** | Pydantic AI replaces parsing/repair for the scoring step when enabled; legacy remains for rollback |
| **Industry standard** | Using established framework vs custom code |

---

## GitHub Issue Updates

### Issue #28 (Update)
**New Title**: "Spec 13: Full Pydantic AI Framework Integration with TextOutput"

**Update Description**:
> Migrating to full Pydantic AI framework using `TextOutput` mode to preserve
> our reasoning-optimal `<thinking>` + `<answer>` pattern while gaining
> framework benefits: type safety and built-in retry loops.

### Issue #29 (Close)
**Reason**: Ollama `format: json` is NOT appropriate for our use case.

> Closing this issue. After research, we determined that:
> 1. `format: json` forces the LLM to output ONLY valid JSON
> 2. This breaks our `<thinking>` tags which require free-form text
> 3. Evidence suggests strict format constraints can hurt performance on some tasks/models; we avoid mixing “solve” and “format” in the same constrained channel
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

Authoritative unit tests:
- `tests/unit/agents/test_pydantic_ai_extractors.py` (extractors raise `ModelRetry` and validate schemas)
- `tests/unit/agents/test_quantitative.py` (prompt shape and legacy behavior remain stable)

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...
    "pydantic-ai>=1.39.0",
]
```

---

## References

### Pydantic AI Documentation
- [Pydantic AI Overview](https://ai.pydantic.dev/)
- [Output Modes (TextOutput)](https://ai.pydantic.dev/output/)
- [Agents](https://ai.pydantic.dev/agents/)
- [Dependencies](https://ai.pydantic.dev/dependencies/)
- [Ollama Integration](https://ai.pydantic.dev/models/openai/#ollama)

### Research on Structured Outputs
- [Decoupling Task-Solving and Output Formatting](https://arxiv.org/html/2510.03595v1)
- [Structured outputs can hurt LLM performance](https://dylancastillo.co/posts/say-what-you-mean-sometimes.html)

### Best Practices
- [DataCamp: Pydantic AI Guide](https://www.datacamp.com/tutorial/pydantic-ai-guide)
- [Machine Learning Mastery: Pydantic for LLM Outputs](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)
