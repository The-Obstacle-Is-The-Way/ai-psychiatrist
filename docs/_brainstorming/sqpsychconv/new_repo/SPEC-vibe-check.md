# SPEC: vibe-check

**Repository**: `vibe-check`
**Version**: 1.0-draft
**Date**: 2026-01-02
**Status**: DRAFT - Awaiting Senior Review

---

## 1. Executive Summary

**vibe-check** is a production-grade multi-agent LLM system that scores synthetic therapy conversations with PHQ-8 depression severity labels using frontier model consensus.

### The Problem

The `ai-psychiatrist` pipeline achieves strong PHQ-8 prediction but cannot be deployed because:

- DAIC-WOZ has restrictive academic licensing
- Embeddings derived from DAIC-WOZ cannot be redistributed
- Few-shot retrieval requires reference examples with ground truth scores

### The Solution

Score SQPsychConv (2,090 synthetic therapy dialogues) with PHQ-8 labels using frontier LLM consensus, creating a freely redistributable retrieval corpus validated against DAIC-WOZ ground truth.

### Why "vibe-check"?

The system literally checks the "vibe" of therapy conversations to assess mental health severity. Also: it's memorable and the domain `vibe-check.ai` might be available.

---

## 2. January 2026 Frontier Models

### 2.1 Model Selection (Verified Jan 2026)

| Role | Model | Model ID | Provider | Price (in/out per 1M) |
|------|-------|----------|----------|----------------------|
| **Juror A** | GPT-5.2 Thinking | `gpt-5.2` | OpenAI | $1.75 / $14.00 |
| **Juror B** | Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Anthropic | $3.00 / $15.00 |
| **Juror C** | Gemini 3 Flash | `gemini-3-flash-preview` | Google | $0.50 / $3.00 |
| **Judge** | Claude Opus 4.5 | `claude-opus-4-5-20251101` | Anthropic | $15.00 / $75.00 |

### 2.2 Why These Models?

**GPT-5.2 Thinking** (Dec 2025):

- First model to perform at or above human expert level on GDPval (70.9% beat/tie rate)
- 55.6% on SWE-Bench Pro (new SOTA for software engineering)
- Uses adaptive reasoning - allocates more compute to harder problems
- Knowledge cutoff: August 2025

**Claude Sonnet 4.5** (Sep 2025):

- 77.2% on SWE-bench Verified (82.0% with high compute)
- 61.4% on OSWorld (computer use benchmark) - up from 42.2% four months prior
- Maintains focus for 30+ hours on multi-step tasks
- 0% error rate on internal code editing benchmarks (down from 9%)
- Knowledge cutoff: July 2025

**Gemini 3 Flash** (Dec 2025):

- 90.4% on GPQA Diamond (PhD-level reasoning)
- 78% on SWE-bench Verified (outperforms even Gemini 3 Pro for coding)
- Thinking level parameter (minimal/low/medium/high) for cost control
- 1M token context window
- **Cheapest frontier model** - $0.50/$3 per M tokens
- Knowledge cutoff: January 2025

**Claude Opus 4.5** (Judge):

- Most capable Anthropic model for complex arbitration
- Different family from majority jurors (avoids correlated errors)
- Used sparingly - only for disagreements (~20% of items)

### 2.3 Cost Estimate (Updated)

| Component | Calculation | Cost |
|-----------|-------------|------|
| GPT-5.2 (2 runs × 2,090) | 4,180 calls × 2.5K tokens | ~$26 |
| Sonnet 4.5 (2 runs × 2,090) | 4,180 calls × 2.5K tokens | ~$38 |
| Gemini 3 Flash (2 runs × 2,090) | 4,180 calls × 2.5K tokens | ~$6 |
| Opus 4.5 Judge (~20% arbitration) | ~830 calls × 3K tokens | ~$45 |
| **Total (one pass)** | | **~$115** |
| **With batch discounts (~50%)** | | **~$60** |

---

## 3. Definitive Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Scoring Metric** | PHQ-8 + self-harm boolean tag | DAIC-WOZ alignment; avoids PHQ-9 safety refusals |
| **Framework** | LangGraph 1.0 | Production checkpointing, graph-based state, fault tolerance |
| **Aggregation** | Distributional posterior | Principled uncertainty; better than mean/mode |
| **Disagreement Threshold** | Range ≥ 2 per item | Simple, interpretable; catches real disagreements |
| **Runs per Model** | 2 runs × 3 models = 6 passes | Balances cost vs stability |
| **Preprocessing** | Client/Participant utterances only | Avoids therapist prompt bias |
| **Checkpoint Storage** | SQLite (dev) / PostgreSQL (prod) | LangGraph native persistence |
| **Structured Output** | Pydantic models inside LangGraph | Type-safe, validated responses |

---

## 4. Deep Dive: LangGraph Architecture

### 4.1 Why LangGraph (Not PydanticAI Alone)

You're familiar with PydanticAI - here's why we need LangGraph on top:

| Concern | PydanticAI Alone | LangGraph |
|---------|------------------|-----------|
| **Checkpointing** | Manual (you write the code) | Native (`SqliteSaver`, `PostgresSaver`) |
| **Resume after crash** | Custom state management | Automatic - exact position recovery |
| **Parallel fan-out** | `asyncio.gather()` | `Send()` API with state isolation |
| **Conditional branching** | if/else in code | Declarative conditional edges |
| **Observability** | Custom logging | LangSmith integration |
| **Human-in-the-loop** | Manual implementation | Native `interrupt()` API |
| **Memory across sessions** | Manual persistence | Built-in memory stores |

**Key insight**: PydanticAI excels at defining *what* each agent does (structured outputs, type safety). LangGraph excels at *orchestrating* multiple agents (state, flow, persistence). **Use both together.**

### 4.2 The Hybrid Pattern: LangGraph + Pydantic

```python
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from pydantic_ai import Agent

# 1. Define Pydantic schemas (what the agent outputs)
class PHQ8ItemScore(BaseModel):
    score: Literal[0, 1, 2, 3]
    confidence: float
    evidence: list[str]

class PHQ8Report(BaseModel):
    items: dict[str, PHQ8ItemScore]
    total_score: int

# 2. Define PydanticAI agents (how each model scores)
gpt_agent = Agent(
    "openai:gpt-5.2",
    output_type=PHQ8Report,
    system_prompt="You are scoring PHQ-8...",
)

claude_agent = Agent(
    "anthropic:claude-sonnet-4-5-20250929",
    output_type=PHQ8Report,
    system_prompt="You are scoring PHQ-8...",
)

gemini_agent = Agent(
    "google:gemini-3-flash-preview",
    output_type=PHQ8Report,
    system_prompt="You are scoring PHQ-8...",
)

# 3. LangGraph orchestrates them
class ScoringState(TypedDict):
    file_id: str
    client_text: str
    jury_results: list[PHQ8Report]
    needs_arbitration: bool
    final_output: AggregatedPHQ8

async def jury_node(state: ScoringState) -> dict:
    """Run all three jurors in parallel using PydanticAI agents."""
    results = await asyncio.gather(
        gpt_agent.run(state["client_text"]),
        claude_agent.run(state["client_text"]),
        gemini_agent.run(state["client_text"]),
    )
    return {"jury_results": [r.output for r in results]}
```

### 4.3 LangGraph Concepts Explained

#### State: The Shared Memory

```python
from typing import TypedDict, Annotated
import operator

class ScoringState(TypedDict):
    # Identity
    file_id: str
    client_text: str

    # Accumulated results (operator.add allows multiple nodes to append)
    jury_results: Annotated[list[PHQ8Report], operator.add]

    # Control flow
    needs_arbitration: bool

    # Final output
    final_output: AggregatedPHQ8 | None
```

**Why `Annotated[list, operator.add]`?** When multiple parallel nodes return results, LangGraph needs to know how to combine them. `operator.add` says "concatenate the lists."

#### Nodes: The Processing Steps

```python
async def preprocess_node(state: ScoringState) -> dict:
    """Extract client-only text from dialogue."""
    client_text = extract_client_utterances(state["dialogue"])
    return {"client_text": client_text}

async def jury_node(state: ScoringState) -> dict:
    """Run 3 models × 2 runs = 6 scoring passes."""
    # ... parallel scoring ...
    return {"jury_results": results}

def aggregate_node(state: ScoringState) -> dict:
    """Compute distributional posterior and check disagreement."""
    agg = compute_aggregation(state["jury_results"])
    needs_arb = check_disagreement(agg)
    return {"final_output": agg, "needs_arbitration": needs_arb}

async def arbitrate_node(state: ScoringState) -> dict:
    """Judge resolves contested items."""
    resolution = await judge_agent.run(
        contested_items=find_contested_items(state),
        jury_results=state["jury_results"],
    )
    return {"final_output": resolution.output}
```

#### Edges: The Control Flow

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(ScoringState)

# Add nodes
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("jury", jury_node)
workflow.add_node("aggregate", aggregate_node)
workflow.add_node("arbitrate", arbitrate_node)

# Linear edges
workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "jury")
workflow.add_edge("jury", "aggregate")

# Conditional edge: branch based on state
def route_after_aggregate(state: ScoringState) -> str:
    if state["needs_arbitration"]:
        return "arbitrate"
    return END

workflow.add_conditional_edges("aggregate", route_after_aggregate)
workflow.add_edge("arbitrate", END)
```

#### Checkpointing: The Magic

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpointer
checkpointer = SqliteSaver.from_conn_string("sqlite:///vibe_check.db")

# Compile graph with persistence
app = workflow.compile(checkpointer=checkpointer)

# Run with thread_id for state isolation
config = {"configurable": {"thread_id": "file_id_active436"}}
result = await app.ainvoke(initial_state, config=config)

# If it crashes at "aggregate" node, next run resumes from there!
```

**What gets saved?**

- Every node's input and output
- The current position in the graph
- All accumulated state

**When does it save?**

- After every node completes
- Configurable: can batch writes for performance

### 4.4 Map-Reduce: Processing 2,090 Dialogues

```python
from langgraph.constants import Send

class BatchState(TypedDict):
    dialogues: list[dict]  # [{file_id, dialogue}, ...]
    completed: Annotated[list[AggregatedPHQ8], operator.add]

def orchestrator_node(state: BatchState) -> list[Send]:
    """Fan out to process each dialogue independently."""
    return [
        Send(
            "score_single",  # Target node (the subgraph)
            {"file_id": d["file_id"], "dialogue": d["dialogue"]}
        )
        for d in state["dialogues"]
    ]

def collector_node(state: BatchState) -> dict:
    """Fan in - all results arrive here."""
    return {"completed": state["completed"]}

# Main graph
batch_graph = StateGraph(BatchState)
batch_graph.add_node("orchestrate", orchestrator_node)
batch_graph.add_node("score_single", scoring_subgraph)  # The per-dialogue graph
batch_graph.add_node("collect", collector_node)

batch_graph.set_entry_point("orchestrate")
batch_graph.add_edge("orchestrate", "score_single")
batch_graph.add_edge("score_single", "collect")
```

**Why `Send()` instead of `asyncio.gather()`?**

1. **State isolation**: Each dialogue gets its own state, no cross-contamination
2. **Independent checkpointing**: If dialogue 1,500 fails, 1,499 are already saved
3. **Rate limiting**: LangGraph can throttle concurrent `Send()` calls
4. **Retry granularity**: Retry just the failed dialogue, not the whole batch

### 4.5 Error Handling and Retries

```python
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential

# Option 1: Tenacity decorators on individual functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
async def call_openai(prompt: str) -> PHQ8Report:
    ...

# Option 2: LangGraph's built-in retry (per-node)
workflow.add_node(
    "jury",
    jury_node,
    retry_policy={"max_retries": 3, "backoff_factor": 2.0}
)

# Option 3: Explicit error node
def error_handler_node(state: ScoringState) -> dict:
    """Handle failures gracefully."""
    return {
        "final_output": AggregatedPHQ8(
            file_id=state["file_id"],
            error="Scoring failed after retries",
            partial_results=state.get("jury_results", []),
        )
    }

workflow.add_conditional_edges(
    "jury",
    lambda s: "error" if s.get("error") else "aggregate",
    {"error": "error_handler", "aggregate": "aggregate"}
)
```

---

## 5. Complete System Architecture

### 5.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           vibe-check Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────────────────────┐ │
│  │ HuggingFace  │      │ Preprocess   │      │      CONSENSUS ENGINE      │ │
│  │ SQPsychConv  │─────▶│ Extract      │─────▶│                            │ │
│  │ 2,090 dlgs   │      │ Client Text  │      │  ┌─────┐ ┌─────┐ ┌─────┐   │ │
│  └──────────────┘      └──────────────┘      │  │GPT  │ │Clau │ │Gemi │   │ │
│                                              │  │5.2  │ │4.5  │ │3.0  │   │ │
│                                              │  └──┬──┘ └──┬──┘ └──┬──┘   │ │
│                                              │     │       │       │      │ │
│                                              │     └───────┼───────┘      │ │
│                                              │             │              │ │
│                                              │     ┌───────▼───────┐      │ │
│                                              │     │  Aggregate    │      │ │
│                                              │     │  Posterior    │      │ │
│                                              │     └───────┬───────┘      │ │
│                                              │             │              │ │
│                                              │      ┌──────┴──────┐       │ │
│                                              │      │             │       │ │
│                                              │      ▼             ▼       │ │
│                                              │  ┌──────┐     ┌──────┐     │ │
│                                              │  │Accept│     │Judge │     │ │
│                                              │  │(≤1)  │     │Opus  │     │ │
│                                              │  └──────┘     └──────┘     │ │
│                                              │                            │ │
│                                              └────────────────────────────┘ │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT ARTIFACTS                             │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │  • scored_sqpsychconv.jsonl     (full structured output)             │   │
│  │  • scored_sqpsychconv.csv       (flat for pandas)                    │   │
│  │  • embeddings/*.npz             (vector store for retrieval)         │   │
│  │  • validation_report.json       (inter-model agreement stats)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 LangGraph State Machine

```
                    ┌─────────────────┐
                    │     START       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   preprocess    │
                    │ Extract client  │
                    │ text only       │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │            jury             │
              │  ┌───────┬───────┬───────┐  │
              │  │GPT×2  │CLU×2  │GEM×2  │  │
              │  │runs   │runs   │runs   │  │
              │  └───────┴───────┴───────┘  │
              │     (6 parallel calls)      │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   aggregate     │
                    │ Compute post-   │
                    │ erior, entropy  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            range < 2           range ≥ 2
                    │                 │
                    ▼                 ▼
           ┌──────────────┐  ┌──────────────┐
           │   ACCEPT     │  │  arbitrate   │
           │  Return agg  │  │  Judge Opus  │
           └──────┬───────┘  │  resolves    │
                  │          └──────┬───────┘
                  │                 │
                  └────────┬────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │      END        │
                    │  Save to DB     │
                    └─────────────────┘
```

---

## 6. Scoring Metric: PHQ-8 + Self-Harm Tag

### 6.1 Why PHQ-8 (NOT PHQ-9)

| Factor | PHQ-8 | PHQ-9 |
|--------|-------|-------|
| DAIC-WOZ alignment | Direct match (0-24 scale) | Scale mismatch (0-27) |
| LLM safety refusals | None | 10-74% refusal rate on suicide items |
| SQPsychConv coverage | N/A | Only 0.3% explicit mentions |
| PHQ-9 correlation | r = 0.996 | Full scale |
| Liability | Severity only | Suicide detection (high-stakes) |

### 6.2 PHQ-8 Items (0-3 Scale)

| # | Item | CSV Column | Description |
|---|------|------------|-------------|
| 1 | Anhedonia | `PHQ8_NoInterest` | Little interest or pleasure |
| 2 | Depressed Mood | `PHQ8_Depressed` | Feeling down, hopeless |
| 3 | Sleep | `PHQ8_Sleep` | Sleep disturbance |
| 4 | Fatigue | `PHQ8_Tired` | Low energy |
| 5 | Appetite | `PHQ8_Appetite` | Appetite changes |
| 6 | Guilt | `PHQ8_Failure` | Feeling like a failure |
| 7 | Concentration | `PHQ8_Concentrating` | Trouble focusing |
| 8 | Psychomotor | `PHQ8_Moving` | Restlessness or slowness |

**Score Anchors**:

- `0` = Not at all
- `1` = Several days
- `2` = More than half the days
- `3` = Nearly every day

### 6.3 Self-Harm Boolean Tag

Instead of PHQ-9 Item 9 (0-3), store a binary flag:

```json
{
  "mentions_self_harm_or_death": true,
  "self_harm_evidence": [
    "wondering if anyone'd notice if I just vanished"
  ],
  "disclaimer": "NOT validated for suicide risk assessment"
}
```

### 6.4 Severity Buckets

| PHQ-8 Total | Severity | Clinical Interpretation |
|-------------|----------|------------------------|
| 0-4 | Minimal | No significant symptoms |
| 5-9 | Mild | Watchful waiting |
| 10-14 | Moderate | Treatment consideration |
| 15-19 | Moderately Severe | Active treatment indicated |
| 20-24 | Severe | Immediate intervention |

---

## 7. Consensus Architecture

### 7.1 Distributional Aggregation (Not Simple Mean)

For each PHQ-8 item, given 6 votes `v₁, v₂, ..., v₆`:

**Step 1: Compute counts**

```python
counts = {0: 0, 1: 0, 2: 0, 3: 0}
for vote in votes:
    counts[vote] += 1
# Example: {0: 1, 1: 0, 2: 3, 3: 2}
```

**Step 2: Apply Dirichlet smoothing**

```python
alpha = 0.5  # Weak prior
total = sum(counts.values()) + 4 * alpha
posterior = {
    score: (count + alpha) / total
    for score, count in counts.items()
}
# Example: {0: 0.19, 1: 0.06, 2: 0.44, 3: 0.31}
```

**Step 3: Extract statistics**

```python
# Mode (most voted)
item_mode = max(posterior, key=posterior.get)  # 2

# Expected value
item_expected = sum(s * p for s, p in posterior.items())  # 1.87

# Entropy (uncertainty)
item_entropy = -sum(p * log(p) for p in posterior.values() if p > 0)  # 1.21
```

**Why distributional?**

- Simple mean of `[0, 2, 2, 2, 3, 3]` = 2.0
- But `[1, 1, 1, 2, 2, 3]` also = 1.67 ≈ 2

The distributions are different! The first has clear consensus on 2-3, the second is dispersed. Entropy captures this.

### 7.2 Disagreement Threshold

Trigger Meta-Judge arbitration if ANY item has:

1. **Range ≥ 2**: e.g., votes `{0, 2, 2}` have range 2 → arbitrate
2. **Insufficient evidence** flagged by ≥2 jurors
3. **Total score std ≥ 2.0** across all juror reports

**Why range ≥ 2?**

- Range 1 (e.g., `{2, 3, 2}`) is natural variance between adjacent scores
- Range 2 (e.g., `{1, 3, 2}`) indicates real disagreement about symptom presence
- Range 3 (e.g., `{0, 3, 1}`) is extreme disagreement - definitely needs arbitration

### 7.3 Meta-Judge Prompt

```text
You are a senior clinical psychologist arbitrating a disagreement between three AI scorers.

## Contested Item: {item_name}
- PHQ-8 definition: {item_definition}

## Scorer Outputs:
- GPT-5.2: Score {gpt_score}, Evidence: "{gpt_evidence}"
- Claude Sonnet 4.5: Score {claude_score}, Evidence: "{claude_evidence}"
- Gemini 3 Flash: Score {gemini_score}, Evidence: "{gemini_evidence}"

## Client Transcript (relevant excerpt):
{transcript_excerpt}

## Your Task:
1. Analyze which scorer's interpretation best matches the PHQ-8 scoring criteria
2. Consider the frequency ("nearly every day" vs "several days") implied in the text
3. Provide a final score (0-3) with justification

Respond in JSON:
{
  "final_score": <0-3>,
  "rationale": "<Why this score is most appropriate>",
  "confidence": <0.0-1.0>
}
```

---

## 8. Data Schemas (Pydantic)

### 8.1 Input Schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class SQPsychConvDialogue(BaseModel):
    """Raw input from HuggingFace."""
    file_id: str
    condition: Literal["mdd", "control"]
    client_model: str
    therapist_model: str
    dialogue: str
```

### 8.2 Per-Item Score

```python
class PHQ8ItemScore(BaseModel):
    """Single item assessment from one model run."""
    score: Literal[0, 1, 2, 3]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, max_length=3)
    insuff_evidence: bool = False

class PHQ8Report(BaseModel):
    """Full PHQ-8 report from one model run."""
    model_id: str
    run_number: int

    # 8 items
    anhedonia: PHQ8ItemScore
    depressed_mood: PHQ8ItemScore
    sleep: PHQ8ItemScore
    fatigue: PHQ8ItemScore
    appetite: PHQ8ItemScore
    guilt: PHQ8ItemScore
    concentration: PHQ8ItemScore
    psychomotor: PHQ8ItemScore

    # Derived
    total_score: int = Field(ge=0, le=24)

    # Safety tag
    mentions_self_harm: bool = False
    self_harm_evidence: list[str] = Field(default_factory=list)
```

### 8.3 Aggregated Output

```python
class ItemAggregation(BaseModel):
    """Aggregated statistics for one item."""
    vote_counts: dict[str, int]  # {"0": 1, "1": 0, "2": 3, "3": 2}
    posterior: dict[str, float]  # {"0": 0.19, "1": 0.06, ...}
    mode: int
    expected: float
    entropy: float
    range: int  # max - min of votes

class AggregatedPHQ8(BaseModel):
    """Final output for one dialogue."""
    # Identity
    file_id: str
    condition: Literal["mdd", "control"]

    # Per-item aggregations
    items: dict[str, ItemAggregation]

    # Total scores
    total_mode: int = Field(ge=0, le=24)
    total_expected: float
    total_std: float
    severity_bucket: Literal["0-4", "5-9", "10-14", "15-19", "20-24"]

    # Consensus metadata
    triggered_arbitration: bool
    arbitration_items: list[str] = Field(default_factory=list)

    # Safety
    mentions_self_harm: bool = False
    self_harm_evidence: list[str] = Field(default_factory=list)

    # Audit trail
    juror_reports: list[PHQ8Report]
    judge_resolution: dict | None = None

    # Provenance
    prompt_version: str
    scored_at: datetime
```

---

## 9. Repository Structure

```
vibe-check/
├── README.md
├── pyproject.toml
├── .env.example
├── CLAUDE.md
│
├── src/
│   └── vibe_check/
│       ├── __init__.py
│       ├── config.py                 # Pydantic Settings
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── input.py              # SQPsychConvDialogue
│       │   ├── scoring.py            # PHQ8ItemScore, PHQ8Report
│       │   └── output.py             # AggregatedPHQ8
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py               # PydanticAI agent factory
│       │   ├── jurors.py             # GPT, Claude, Gemini agents
│       │   └── judge.py              # Opus arbitration agent
│       │
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── state.py              # ScoringState, BatchState
│       │   ├── nodes.py              # preprocess, jury, aggregate, arbitrate
│       │   ├── edges.py              # Conditional routing logic
│       │   ├── workflow.py           # Single-dialogue graph
│       │   └── batch.py              # Map-reduce batch graph
│       │
│       ├── aggregation/
│       │   ├── __init__.py
│       │   ├── posterior.py          # Dirichlet smoothing
│       │   ├── disagreement.py       # Threshold checks
│       │   └── metrics.py            # Entropy, ICC, Krippendorff
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── client_extractor.py   # Extract client utterances
│       │   └── cleaner.py            # CJK removal, normalization
│       │
│       └── export/
│           ├── __init__.py
│           ├── jsonl.py
│           ├── csv.py
│           └── embeddings.py
│
├── scripts/
│   ├── score_corpus.py               # Main CLI entry point
│   ├── validate_on_daic_woz.py       # Phase 0: scorer competence
│   ├── generate_embeddings.py        # Create retrieval corpus
│   └── evaluate_transfer.py          # Sim-to-real metrics
│
├── tests/
│   ├── unit/
│   │   ├── test_aggregation.py
│   │   ├── test_preprocessing.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_single_dialogue.py
│   │   └── test_graph_checkpointing.py
│   └── e2e/
│       └── test_full_pipeline.py
│
├── data/
│   ├── raw/                          # HuggingFace downloads
│   ├── checkpoints/                  # SQLite checkpoint DBs
│   └── outputs/                      # Final scored datasets
│
└── docs/
    ├── architecture.md
    ├── langgraph-tutorial.md
    └── validation-protocol.md
```

---

## 10. Dependencies

```toml
[project]
name = "vibe-check"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Orchestration
    "langgraph>=1.0.0",

    # Agents (PydanticAI for structured outputs)
    "pydantic-ai>=1.0.0",

    # Provider clients
    "langchain-openai>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-google-genai>=2.1.0",

    # Data
    "pydantic>=2.10.0",
    "datasets>=3.0.0",
    "pandas>=2.2.0",
    "numpy>=2.0.0",

    # Metrics
    "scikit-learn>=1.5.0",
    "scipy>=1.14.0",

    # Utilities
    "tenacity>=9.0.0",
    "aiolimiter>=1.2.0",
    "structlog>=25.0.0",
    "rich>=14.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.9.0",
    "mypy>=1.14.0",
    "pre-commit>=4.0.0",
]
```

---

## 11. Configuration

### 11.1 Environment Variables

```env
# ─────────────────────────────────────────────────────────────
# Provider API Keys
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# ─────────────────────────────────────────────────────────────
# Model Selection (January 2026 Frontier)
# ─────────────────────────────────────────────────────────────
JUROR_GPT_MODEL=gpt-5.2
JUROR_CLAUDE_MODEL=claude-sonnet-4-5-20250929
JUROR_GEMINI_MODEL=gemini-3-flash-preview
JUDGE_MODEL=claude-opus-4-5-20251101

# ─────────────────────────────────────────────────────────────
# Scoring Configuration
# ─────────────────────────────────────────────────────────────
RUNS_PER_MODEL=2
DISAGREEMENT_RANGE_THRESHOLD=2
ARBITRATION_TOTAL_STD_THRESHOLD=2.0
DIRICHLET_ALPHA=0.5

# ─────────────────────────────────────────────────────────────
# Rate Limiting (requests per minute)
# ─────────────────────────────────────────────────────────────
OPENAI_RPM=100
ANTHROPIC_RPM=60
GOOGLE_RPM=100

# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────
CHECKPOINT_DB=sqlite:///data/checkpoints/vibe_check.db
# For production: postgresql://user:pass@host:5432/vibe_check

# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR=./data/outputs
PROMPT_VERSION=v1.0.0
```

### 11.2 Pydantic Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API Keys
    openai_api_key: str
    anthropic_api_key: str
    google_api_key: str

    # Models
    juror_gpt_model: str = "gpt-5.2"
    juror_claude_model: str = "claude-sonnet-4-5-20250929"
    juror_gemini_model: str = "gemini-3-flash-preview"
    judge_model: str = "claude-opus-4-5-20251101"

    # Scoring
    runs_per_model: int = 2
    disagreement_range_threshold: int = 2
    arbitration_total_std_threshold: float = 2.0
    dirichlet_alpha: float = 0.5

    # Rate limiting
    openai_rpm: int = 100
    anthropic_rpm: int = 60
    google_rpm: int = 100

    # Checkpointing
    checkpoint_db: str = "sqlite:///data/checkpoints/vibe_check.db"

    # Output
    output_dir: str = "./data/outputs"
    prompt_version: str = "v1.0.0"
```

---

## 12. Validation Protocol

### 12.1 Phase 0: Scorer Competence on DAIC-WOZ

Before scoring SQPsychConv, prove the ensemble works on real data:

```bash
uv run python scripts/validate_on_daic_woz.py \
    --split paper-dev \
    --output data/outputs/scorer_validation.json
```

**Success Criteria**:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Total MAE | ≤ 3.0 | Transcript-only has limits |
| Binary AUC | > 0.80 | Strong discrimination |
| ICC(2,k) | ≥ 0.80 | High inter-model reliability |
| Krippendorff's α | ≥ 0.70 | Ordinal agreement |

### 12.2 Phase 1: Score SQPsychConv

```bash
uv run python scripts/score_corpus.py \
    --input AIMH/SQPsychConv_qwq \
    --output data/outputs/scored_sqpsychconv.jsonl
```

**Internal Sanity Checks**:

- MDD dialogues should have higher mean PHQ-8 than control
- Cronbach's α > 0.7 (internal consistency)
- Arbitration rate < 30% (most items reach consensus)

### 12.3 Phase 2: Generate Embeddings

```bash
uv run python scripts/generate_embeddings.py \
    --input data/outputs/scored_sqpsychconv.jsonl \
    --output data/outputs/embeddings/
```

### 12.4 Phase 3: Sim-to-Real Evaluation

```bash
uv run python scripts/evaluate_transfer.py \
    --synthetic data/outputs/embeddings/ \
    --real-test paper-test \
    --k 5 \
    --output data/outputs/transfer_eval.json
```

**Ablations**:

| Configuration | Description |
|---------------|-------------|
| k ∈ {3, 5, 10, 20} | Number of retrieved exemplars |
| participant-only vs full | Embedding strategy |
| total_mode vs total_expected | Label to use |
| low-entropy filter | Only retrieve confident exemplars |

**Success Criteria**:

| Metric | Target | Rationale |
|--------|--------|-----------|
| DAIC-WOZ MAE | < 6.0 | SOTA range is 5.0-6.0 |
| Binary AUC | > 0.75 | Useful discrimination |
| Δ vs zero-shot | > 5% relative | Proves transfer value |

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **API rate limits** | `aiolimiter` + exponential backoff with jitter |
| **Batch job crash** | LangGraph checkpointing; resume from exact node |
| **Invalid JSON** | Pydantic validation + PydanticAI auto-retry |
| **Model refuses** | "Clinical research assistant" role framing |
| **Synthetic circularity** | Cross-vendor scorers; validate on DAIC-WOZ |
| **Cost overrun** | Gemini 3 Flash is cheapest; batch API discounts |

---

## 14. Success Criteria Summary

| Criterion | Target |
|-----------|--------|
| Scorer ICC | ≥ 0.80 |
| Krippendorff's α | ≥ 0.70 |
| DAIC-WOZ transfer MAE | < 6.0 |
| DAIC-WOZ binary AUC | > 0.75 |
| Improvement over zero-shot | > 5% |
| Arbitration rate | < 30% |
| Processing completion | 100% (2,090 dialogues) |

---

## 15. Next Steps

### Immediate (Before Implementation)

1. [ ] Senior review of this spec
2. [ ] Approve model selection and costs
3. [ ] Set up API keys and verify access
4. [ ] Finalize prompt templates

### Phase 1: Scaffold (3-5 days)

1. [ ] Initialize repo with `uv init`
2. [ ] Implement Pydantic schemas
3. [ ] Implement preprocessing (client text extraction)
4. [ ] Write unit tests for schemas

### Phase 2: Agents (3-5 days)

1. [ ] Implement PydanticAI juror agents
2. [ ] Implement aggregation logic
3. [ ] Implement judge agent
4. [ ] Write integration tests

### Phase 3: LangGraph (5-7 days)

1. [ ] Build single-dialogue workflow
2. [ ] Add checkpointing
3. [ ] Build batch map-reduce workflow
4. [ ] Test crash recovery

### Phase 4: Validation (3-5 days)

1. [ ] Run Phase 0 on DAIC-WOZ
2. [ ] Score full SQPsychConv
3. [ ] Generate embeddings
4. [ ] Run sim-to-real evaluation

---

## 16. References

### Research Papers

- [Traub et al. (2024): AUGRC for selective classification](https://arxiv.org/abs/2407.01032)
- [Amazon CollabEval: Multi-agent LLM-as-Judge](https://www.amazon.science/publications/enhancing-llm-as-a-judge-via-multi-agent-collaboration)
- [PHQ-8 vs PHQ-9 equivalency](https://stacks.cdc.gov/view/cdc/84248)
- [SQPsychConv dataset](https://arxiv.org/abs/2510.25384)

### Model Announcements

- [GPT-5.2 Introduction (OpenAI, Dec 2025)](https://openai.com/index/introducing-gpt-5-2/)
- [Claude Sonnet 4.5 (Anthropic, Sep 2025)](https://www.anthropic.com/news/claude-sonnet-4-5)
- [Gemini 3 Flash (Google, Dec 2025)](https://blog.google/products/gemini/gemini-3-flash/)

### Technical Documentation

- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph Send API](https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd)
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [OpenAI GPT-5.2 API](https://platform.openai.com/docs/models/gpt-5.2)

### Framework Comparisons

- [LangGraph vs PydanticAI (ZenML)](https://www.zenml.io/blog/pydantic-ai-vs-langgraph)
- [Best AI Agent Frameworks 2025 (LangWatch)](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)

---

*This specification synthesizes research from Gemini Deep Research and GPT Deep Research, updated with January 2026 frontier models and production-grade LangGraph architecture.*
