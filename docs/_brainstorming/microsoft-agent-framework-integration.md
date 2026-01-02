# Should We Integrate Microsoft Agent Framework?

**Date**: 2025-12-19
**Status**: BRAINSTORMING
**Reference**: https://github.com/microsoft/agent-framework

---

## What Is It?

Microsoft Agent Framework is a comprehensive multi-language framework for building, orchestrating, and deploying AI agents. It provides:

- **ChatAgent abstraction**: Wraps LLM clients with instructions, tools, and orchestration
- **Multi-agent orchestration**: GroupChat, Sequential, Concurrent, and Handoff patterns
- **Workflow engine**: Graph-based workflows with streaming, checkpointing, and human-in-the-loop
- **Ollama support**: Native `OllamaChatClient` (agent-framework-ollama, uses Ollama Python SDK) against the same Ollama server
- **Observability**: Built-in OpenTelemetry integration
- **Tool/function calling**: Schema-inferred tools via `@ai_function` (type hints + optional Pydantic `Field`)

---

## What We Currently Have

Our codebase already implements:

| Component | Our Implementation | MS Agent Framework Equivalent |
|-----------|-------------------|------------------------------|
| LLM Client | `OllamaClient` (httpx-based) | `OllamaChatClient` (ollama SDK) |
| Agent abstraction | `QualitativeAssessmentAgent` | `ChatAgent` |
| Prompt templates | `src/ai_psychiatrist/agents/prompts/qualitative.py` | `instructions` parameter |
| Response parsing | `extract_xml_tags()` | Response middleware |
| Protocols | `ChatClient` protocol | `BaseChatClient` ABC |

---

## Potential Benefits

### 1. Multi-Agent Orchestration (HIGH VALUE)
Our paper describes a multi-agent pipeline:
```text
Transcript → Qualitative Agent → Judge Agent → [Feedback Loop] → Quantitative Agent → Meta-Review
```

MS Agent Framework provides:
- **Sequential orchestration**: Exactly what we need for the pipeline
- **GroupChat**: Could enable parallel qualitative + quantitative paths
- **Handoff patterns**: Judge agent could hand back to Qualitative agent for refinement

### 2. Workflow Checkpointing (MEDIUM VALUE)
- Long-running assessments could checkpoint between agents
- Useful for production where API failures happen
- Time-travel debugging for research

### 3. Human-in-the-Loop (FUTURE VALUE)
- Clinician review step before final assessment
- Tool approval for sensitive operations

### 4. Observability (MEDIUM VALUE)
- Built-in OpenTelemetry tracing
- We'd need this for production anyway

---

## Potential Costs

### 1. Abstraction Mismatch
Our agents have **domain-specific** semantics:
- `QualitativeAssessment` entity with specific fields
- XML extraction for structured output
- Paper-specified prompt templates

MS Agent Framework is **generic**:
- Returns `ChatResponse` with `TextContent`
- Would need adapters to convert to our domain entities

### 2. Dependency Weight
```text
agent-framework-core
agent-framework-ollama
+ transitive deps (ollama SDK, pydantic, opentelemetry, etc.)
```
vs our current:
```text
httpx (already have for Ollama)
```

### 3. Migration Effort
- Rewrite `OllamaClient` → adapt to `OllamaChatClient`
- Rewrite agents to extend `ChatAgent`
- Rewrite prompts to use `instructions` pattern
- Update all tests

### 4. Lock-in Risk
- Microsoft could change API
- Framework is still in preview (`--pre`)

---

## Decision Matrix

| Factor | Now (During Paper Replication) | Later (Post-Replication) |
|--------|-------------------------------|--------------------------|
| **Value** | Low - adds complexity without paper benefit | High - orchestration for production |
| **Risk** | High - framework is in preview | Lower - more stable |
| **Effort** | High - major refactor mid-implementation | Medium - can plan properly |
| **Alignment** | Paper doesn't mention frameworks | Production needs differ from paper |

---

## Recommendation

### Wait Until After Paper Replication

**Rationale:**

1. **Paper Fidelity First**: The paper describes a specific architecture. Adding a framework layer makes it harder to verify we match the paper exactly.

2. **Framework Stability**: MS Agent Framework is in `--pre` (preview). By the time we finish replication, it may be more stable.

3. **Clear Refactor Point**: After replication, we'll know exactly what orchestration patterns we need. The refactor will be targeted, not speculative.

4. **Minimal Viable Now**: Our current `ChatClient` protocol + domain-specific agents work. They're simpler and paper-aligned.

### When To Reconsider

Integrate MS Agent Framework when:
- [ ] Paper replication is complete and validated
- [ ] We need production orchestration (retries, checkpoints, parallel agents)
- [ ] We want to add human-in-the-loop review
- [ ] Framework reaches 1.0 stable release

---

## If We Integrate Later: Migration Path

```text
Phase 1: Adapter Layer
  - Create OllamaChatClientAdapter that wraps MS client
  - Keep our domain entities and parsing
  - Verify behavior parity

Phase 2: Orchestration
  - Replace manual agent chaining with Sequential orchestration
  - Add checkpointing for long assessments

Phase 3: Full Migration
  - Migrate agents to extend ChatAgent
  - Use workflow graphs for full pipeline
  - Add observability hooks
```

---

## References

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [MS Learn Documentation](https://learn.microsoft.com/en-us/agent-framework/)
- [Ollama Integration](https://github.com/microsoft/agent-framework/tree/main/python/packages/ollama)
