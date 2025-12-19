# Should We Integrate Pydantic AI?

**Date**: 2025-12-19
**Status**: BRAINSTORMING
**Reference**: https://ai.pydantic.dev/

---

## What Is It?

Pydantic AI is a Python agent framework built by the Pydantic team. It brings "the FastAPI feeling" to GenAI development with:

- **Type-safe agents**: Full IDE support, validation at write-time
- **Model-agnostic**: OpenAI, Anthropic, Gemini, Ollama (OpenAI-compatible), and many more
- **Structured outputs**: Native Pydantic model validation for LLM responses
- **Durable execution**: Checkpointing across failures
- **Graph-based workflows**: Type-hinted control flow
- **MCP/A2A protocols**: Model Context Protocol and Agent-to-Agent interop

---

## What We Currently Have

We're already using **Pydantic 2.12** for:

| Current Usage | Location |
|--------------|----------|
| Settings validation | `config.py` (pydantic-settings) |
| Domain entities | Dataclasses (not Pydantic models) |
| API validation | FastAPI (uses Pydantic internally) |
| LLM responses | Manual XML parsing → domain entities |

**Key observation**: We use Pydantic for config/API but NOT for domain models or LLM response parsing.

---

## Potential Benefits

### 1. Structured LLM Output Validation (HIGH VALUE)

Currently:
```python
# Our approach: XML parsing with fallbacks
extracted = extract_xml_tags(raw_response, self.ASSESSMENT_TAGS)
return QualitativeAssessment(
    overall=extracted.get("assessment") or "Assessment not generated",
    ...
)
```

With Pydantic AI:
```python
# Pydantic AI approach: Direct structured output
from pydantic import BaseModel
from pydantic_ai import Agent

class QualitativeAssessment(BaseModel):
    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    supporting_quotes: list[str]

agent = Agent('ollama:gemma3:27b', output_type=QualitativeAssessment)
result = await agent.run(prompt)
assessment = result.output  # Validated QualitativeAssessment
```

**Benefits:**
- No XML parsing needed
- Validation happens automatically
- Type safety throughout
- LLM retry on validation failure

### 2. Native Ollama Support (MEDIUM VALUE)
```python
from pydantic_ai import Agent

agent = Agent('ollama:gemma3:27b', output_type=QualitativeAssessment)
```
**Note**: Pydantic AI uses Ollama via the OpenAI-compatible `/v1` endpoint
(`OLLAMA_BASE_URL=http://localhost:11434/v1`).

### 3. Tool/Function Definitions (MEDIUM VALUE)
```python
@agent.tool
def get_phq8_criteria() -> dict:
    """Return PHQ-8 symptom criteria for reference."""
    return PHQ8_CRITERIA
```

### 4. Streaming with Validation (FUTURE VALUE)
- Stream partial results with real-time validation
- Useful for long assessments

---

## Potential Costs

### 1. Domain Model Migration

We use **dataclasses** for domain entities:
```python
@dataclass
class QualitativeAssessment:
    overall: str
    phq8_symptoms: str
    ...
```

Pydantic AI requires **Pydantic models for structured LLM outputs**:
```python
class QualitativeAssessment(BaseModel):
    overall: str
    phq8_symptoms: str
    ...
```

**Migration scope (minimum):**
- Add Pydantic models for LLM output schemas
- Map Pydantic outputs → existing dataclass domain entities
- Keep domain dataclasses unchanged (optional domain migration later)

### 2. Architecture Philosophy

Our current design follows **Clean Architecture**:
```
Domain (dataclasses) ← Infrastructure (Pydantic for external)
```

**Option A (preferred for paper replication):**
```
Domain (dataclasses) ← Infrastructure (Pydantic AI output models)
```

**Option B (post-replication refactor):**
```
Domain (Pydantic models) ← Infrastructure (also Pydantic)
```

Option B is a philosophical shift - domain becomes coupled to Pydantic.

### 3. Paper Alignment

The paper doesn't specify structured output parsing. It uses XML-style extraction:
```
<assessment>...</assessment>
<PHQ8_symptoms>...</PHQ8_symptoms>
```

Switching to JSON structured outputs changes the prompt/response format.

### 4. Ollama API Compatibility

Pydantic AI uses Ollama via the OpenAI-compatible `/v1` endpoint. Our current
`OllamaClient` uses `/api/chat` and `/api/embed`. Adopting Pydantic AI would
require either:
- Switching to `/v1` (set `OLLAMA_BASE_URL=http://localhost:11434/v1`), or
- Building an adapter layer to keep our existing `/api/*` usage.

### 5. Dependency Chain
```
pydantic-ai
├── pydantic (already have)
├── httpx (already have)
├── opentelemetry-* (new)
└── various model SDKs
```

---

## Comparison: Pydantic AI vs MS Agent Framework

| Aspect | Pydantic AI | MS Agent Framework |
|--------|------------|-------------------|
| **Focus** | Type-safe structured outputs | Multi-agent orchestration |
| **Maturity** | More mature (Pydantic team) | Preview (`--pre`) |
| **Ollama** | Native support | Native support |
| **Our need** | Better response parsing | Better agent orchestration |
| **Migration** | Output models + mapping (domain rewrite optional) | LLM client rewrite |
| **Philosophy** | Pydantic-friendly outputs | Framework-agnostic core |

---

## Decision Matrix

| Factor | Now (During Paper Replication) | Later (Post-Replication) |
|--------|-------------------------------|--------------------------|
| **Value** | Medium - cleaner parsing | High - production validation |
| **Risk** | Medium - new output path + parity validation | Lower - can plan migration |
| **Effort** | Medium - output models + mapping | Medium - phased approach |
| **Alignment** | Diverges from paper's XML format | Acceptable for production |

---

## Recommendation

### Wait, But Consider Earlier Than MS Agent Framework

**Rationale:**

1. **Smaller Scope**: Pydantic AI is about response parsing, not orchestration. It's a more targeted change.

2. **Already Using Pydantic**: We have Pydantic in our stack. The philosophical leap is smaller.

3. **Type Safety ROI**: Structured outputs would catch LLM response errors at parse-time, not later in the pipeline.

**However:**

4. **Paper Fidelity**: The paper uses XML. For replication, we should match the paper.

5. **Domain Purity**: Dataclasses are simpler. Pydantic models have more magic.

### When To Reconsider

Integrate Pydantic AI when:
- [ ] Paper replication is complete
- [ ] We want to move to JSON structured outputs
- [ ] We need validation-with-retry for production reliability
- [ ] We're ready to adopt Pydantic output models (domain migration optional)

**Proposed spec placement**: Post-replication, after **Spec 12.5** (Final Cleanup),
as an optional **Spec 13** focused on structured output migration.

---

## If We Integrate: Migration Path

```
Phase 1: Parallel Models
  - Keep dataclasses for domain
  - Create Pydantic models for LLM responses
  - Map between them

Phase 2: Response Migration
  - Replace XML parsing with Pydantic AI structured output
  - Keep domain dataclasses
  - Verify response quality matches

Phase 3: Optional Domain Migration
  - If Phase 2 works well, consider migrating domain to Pydantic
  - Or keep separation (LLM layer uses Pydantic, domain uses dataclasses)
```

---

## Alternative: Structured Outputs Without Full Framework

We could use Pydantic for response validation without the full Pydantic AI framework:

```python
# Current: XML parsing
extracted = extract_xml_tags(raw_response, tags)

# Alternative: JSON mode + Pydantic validation
class AssessmentResponse(BaseModel):
    assessment: str
    PHQ8_symptoms: str
    ...

# In prompt: "Return JSON matching this schema: {schema}"
# Then: AssessmentResponse.model_validate_json(response)
```

This gives us validation without framework lock-in.

---

## References

- [Pydantic AI Docs](https://ai.pydantic.dev/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)
- [Agents Documentation](https://ai.pydantic.dev/agents/)
