# Future Architecture: Agent Orchestration Options

This document captures our research on evolving AI Psychiatrist's orchestration layer from pure Python to a modern agent framework.

**Last Updated:** December 2025

---

## Current State

### How Orchestration Works Today

The `server.py` file handles both **API exposure** (FastAPI) and **agent orchestration** (calling agents in sequence):

```python
# Current: Manual orchestration in server.py
loop_result = await feedback_loop.run(transcript)        # Qualitative + Judge loop
quant_result = await quant_agent.assess(transcript)      # Quantitative
meta_review = await meta_review_agent.review(...)        # Meta-Review
```

This works, but:
- No workflow-level observability/tracing (beyond existing structured logging)
- Manual workflow-level retry/error handling (agent-level validation/retries are handled separately)
- State management is implicit
- The feedback loop is a while-loop, not explicit graph structure

### Current Pipeline as a Graph

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Qualitative │────>│    Judge    │────>│ Quantitative│
└─────────────┘     └──────┬──────┘     └──────┬──────┘
       ▲                   │                   │
       │            score <= 3?                │
       │                   │                   ▼
       └───── YES ─────────┘            ┌─────────────┐
                                        │ Meta-Review │
                                        └─────────────┘
```

This is a **cyclic graph** (has a loop), not a DAG.

---

## Graph Architecture Explained

### What "Graph" Means (Not Neo4j)

In agent orchestration, a "graph" is a **workflow representation**:

| Term | Graph Database (Neo4j) | Agent Orchestration (LangGraph) |
|------|------------------------|----------------------------------|
| **Node** | A data record | A function or agent that does work |
| **Edge** | A relationship between records | Control flow (what runs next) |
| **Purpose** | Store and query connected data | Execute a workflow |

### Why Use Graph-Based Orchestration?

| Aspect | Linear Code (Current) | Graph-Based |
|--------|----------------------|-------------|
| **Visualization** | Hidden in if/while statements | Explicit, visual structure |
| **Conditional Logic** | `if score <= 3: loop` | Conditional edges |
| **State Management** | Manual variables | Built-in state machine |
| **Debugging** | Print statements, logs | Node-by-node inspection |
| **Observability** | DIY | Built-in tracing |

---

## Framework Comparison

### Overview

Data privacy depends primarily on the **model backend** (local models via Ollama/HuggingFace vs.
hosted APIs), not the orchestration framework.

| Framework | License | Data Privacy | Best For | Maturity Note |
|-----------|---------|--------------|----------|---------------|
| **Pydantic AI** | MIT | Backend-dependent | Type-safe agent definitions | Stable (verify upstream) |
| **LangGraph** | MIT | Backend-dependent | Graph-based orchestration | Active (verify upstream) |
| **Microsoft Agent Framework** | MIT | Backend-dependent | Enterprise workflows | Check upstream (often preview) |
| **CrewAI** | MIT | Backend-dependent | Role-based teams | Active (verify upstream) |

### Recommended: Pydantic AI + LangGraph

These frameworks complement each other:

| Layer | Framework | Responsibility |
|-------|-----------|----------------|
| **Agent Definition** | Pydantic AI | What each agent does (type-safe, validated) |
| **Orchestration** | LangGraph | When/how agents run (graph, state, control flow) |

**Why this combination:**

1. **Pydantic AI** matches our existing stack (Pydantic everywhere, mypy strict)
2. **LangGraph** makes the feedback loop explicit as a graph cycle
3. Both are **MIT-licensed, fully open source, self-hosted**
4. No data leaves your infrastructure
5. [Well-documented integration pattern](https://dotzlaw.com/ai-2/combining-the-power-of-langgraph-with-pydantic-ai-agents/)

---

## Pydantic AI

### What It Is

[Pydantic AI](https://ai.pydantic.dev/) is a Python agent framework from the Pydantic team, designed to bring "that FastAPI feeling to GenAI development."

### Key Features

- **Type-safe agents** with Pydantic validation
- **Dependency injection** (matches our current pattern)
- **Multi-model support** (OpenAI, Anthropic, Ollama, etc.)
- **Structured outputs** guaranteed by schema
- **Native async/await**

### How It Would Look

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class QualitativeOutput(BaseModel):
    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str

qualitative_agent = Agent(
    model="ollama:gemma3:27b",
    result_type=QualitativeOutput,
    system_prompt="You are a clinical psychologist...",
)

# Type-safe, validated output
result = await qualitative_agent.run(transcript_text)
# result.data is QualitativeOutput, guaranteed
```

### Migration Path

Our current agents already follow this pattern conceptually:
- `QualitativeAssessmentAgent` → Pydantic AI `Agent` with `QualitativeAssessment` output
- `JudgeAgent` → Pydantic AI `Agent` with `QualitativeEvaluation` output
- etc.

---

## LangGraph

### What It Is

[LangGraph](https://www.langchain.com/langgraph) is a graph-based orchestration framework for building stateful, multi-agent workflows.

### Key Features

- **Graph-based workflows** with explicit nodes and edges
- **Built-in state management** (checkpoints, persistence)
- **Conditional edges** for branching logic
- **Cycle support** (required for our feedback loop)
- **Human-in-the-loop** capabilities

### License & Privacy

> "LangGraph is an MIT-licensed open-source library and is free to use."

- Core library: MIT, fully open source
- No data sent externally
- Self-host on your infrastructure
- LangSmith (observability) is optional, not required

### How Our Pipeline Would Look

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class PipelineState(TypedDict):
    transcript: str
    qualitative: QualitativeAssessment | None
    evaluation: QualitativeEvaluation | None
    quantitative: PHQ8Assessment | None
    meta_review: MetaReview | None
    iteration: int

# Define the graph
graph = StateGraph(PipelineState)

# Add nodes (our agents)
graph.add_node("qualitative", run_qualitative_agent)
graph.add_node("judge", run_judge_agent)
graph.add_node("quantitative", run_quantitative_agent)
graph.add_node("meta_review", run_meta_review_agent)

# Add edges (control flow)
graph.add_edge("qualitative", "judge")
graph.add_conditional_edges(
    "judge",
    should_refine,  # Returns "refine" or "proceed"
    {
        "refine": "qualitative",  # Loop back
        "proceed": "quantitative",
    }
)
graph.add_edge("quantitative", "meta_review")
graph.add_edge("meta_review", END)

# Set entry point
graph.set_entry_point("qualitative")

# Compile
pipeline = graph.compile()

# Run
result = await pipeline.ainvoke({"transcript": transcript_text, "iteration": 0})
```

### Benefits for This Codebase

1. **Feedback loop is explicit**: The `qualitative → judge → (maybe qualitative)` cycle is a graph edge, not a while-loop
2. **State is managed**: `PipelineState` TypedDict tracks everything
3. **Debuggable**: Can inspect state at each node
4. **Extensible**: Adding new agents = adding nodes

---

## Microsoft Agent Framework

### What It Is

[Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview) is a Microsoft-maintained framework for orchestrating agents and multi-agent workflows.

### Status

| Aspect | Status |
|--------|--------|
| Release | Check upstream documentation (status can change over time) |
| Languages | Python + .NET |
| License | MIT |

### Why Not Now

- Often in **preview** (treat as an adoption risk until you verify current status)
- Azure-native (good for Azure shops, not required though)
- Recommendation: Revisit once maturity/stability is confirmed (and only if Azure-centric workflows are needed)

---

## Recommended Evolution Path

### Phase 1: Document & Prepare (Now)

- [x] Document current architecture
- [x] Document future options (this file)
- [x] Add to `docs/index.md` navigation

### Phase 2: Pydantic AI Integration

**Goal:** Type-safe agent definitions with validated outputs

**Changes:**
1. Refactor `src/ai_psychiatrist/agents/*.py` to use Pydantic AI `Agent` class
2. Define dedicated Pydantic output schemas (separate from domain dataclasses)
3. Map validated outputs into domain dataclasses (domain remains SSOT)
4. Keep orchestration in `server.py` initially

**Example:**
```python
# Before: Custom agent class
class QualitativeAssessmentAgent:
    async def assess(self, transcript: Transcript) -> QualitativeAssessment:
        response = await self._llm.simple_chat(...)
        return self._parse_response(response)

# After: Pydantic AI agent
class QualitativeAssessmentOutput(BaseModel):
    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str

qualitative_agent = Agent(
    model="ollama:gemma3:27b",
    result_type=QualitativeAssessmentOutput,
    system_prompt=QUALITATIVE_SYSTEM_PROMPT,
)
```

### Phase 3: LangGraph Orchestration (Optional)

**Goal:** Explicit graph-based workflow with built-in state

**Changes:**
1. Create `src/ai_psychiatrist/orchestration/pipeline.py` (proposed; does not exist yet)
2. Define `StateGraph` with nodes for each agent
3. Replace `FeedbackLoopService` with conditional edges
4. `server.py` becomes thin API layer

**New Structure:**
```
src/ai_psychiatrist/
├── agents/              # Pydantic AI agent definitions
├── orchestration/       # NEW: LangGraph pipeline
│   ├── __init__.py
│   ├── pipeline.py      # StateGraph definition
│   └── state.py         # PipelineState TypedDict
├── domain/              # Unchanged
├── services/            # Reduced (embedding, transcript only)
└── infrastructure/      # Unchanged
```
Note: The `orchestration/` package is a future-state proposal. The current repo performs orchestration in `server.py`.

### Phase 4: Production Hardening (Future)

- Add observability (LangSmith optional, or OpenTelemetry)
- Add persistence (checkpoints for long-running assessments)
- Add human-in-the-loop (review before final assessment)

---

## Code Examples

### Current vs. Future Comparison

#### Feedback Loop: Current

```python
# src/ai_psychiatrist/services/feedback_loop.py
class FeedbackLoopService:
    async def run(self, transcript: Transcript) -> FeedbackLoopResult:
        assessment = await self._qualitative_agent.assess(transcript)
        evaluation = await self._judge_agent.evaluate(assessment, transcript)

        iteration = 0
        while self._needs_improvement(evaluation) and iteration < self._max_iterations:
            iteration += 1
            feedback = self._judge_agent.get_feedback_for_low_scores(evaluation)
            assessment = await self._qualitative_agent.refine(assessment, feedback, transcript)
            evaluation = await self._judge_agent.evaluate(assessment, transcript, iteration)

        return FeedbackLoopResult(...)
```

#### Feedback Loop: LangGraph

```python
# Proposed: src/ai_psychiatrist/orchestration/pipeline.py (not implemented yet)
def should_refine(state: PipelineState) -> str:
    """Conditional edge: decide whether to refine or proceed."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "proceed"
    if state["evaluation"].needs_improvement:
        return "refine"
    return "proceed"

graph = StateGraph(PipelineState)
graph.add_node("qualitative", qualitative_node)
graph.add_node("judge", judge_node)
graph.add_conditional_edges("judge", should_refine, {
    "refine": "qualitative",
    "proceed": "quantitative",
})
```

The loop is now an **explicit edge in the graph**, not hidden in a while-loop.

---

## Installation

### Pydantic AI

```bash
uv add pydantic-ai
```

### LangGraph

```bash
uv add langgraph
```

### Both (Recommended)

```bash
uv add pydantic-ai langgraph
```

---

## References

### Official Documentation

- [Pydantic AI](https://ai.pydantic.dev/)
- [LangGraph](https://www.langchain.com/langgraph)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)

### Articles & Comparisons

- [Combining LangGraph with Pydantic AI](https://dotzlaw.com/ai-2/combining-the-power-of-langgraph-with-pydantic-ai-agents/)
- [Pydantic AI vs LangGraph Comparison](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)
- [Best AI Agent Frameworks 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
- [LangGraph Architecture Explained](https://www.ibm.com/think/topics/langgraph)
- [LangGraph Pricing (Self-Host is Free)](https://www.zenml.io/blog/langgraph-pricing)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Dec 2025 | Document options before implementing | Avoid premature optimization |
| Dec 2025 | Recommend Pydantic AI + LangGraph | MIT license, matches stack, proven integration |
| Dec 2025 | Defer MS Agent Framework | Maturity/status can change; revisit once stability is confirmed and value is clear |
| Dec 2025 | Keep server.py for API | Still need HTTP endpoints regardless of orchestration |

---

## See Also

- [Architecture](./architecture.md) - Current system design
- [Pipeline](./pipeline.md) - Current 4-agent pipeline
- [Configuration](../configs/configuration.md) - Settings reference
