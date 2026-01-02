# SPEC: vibe-check

**Repository**: `vibe-check`
**Version**: 1.0-draft
**Date**: 2026-01-02
**Status**: DRAFT - Awaiting Senior Review
**Senior review prompt**: `docs/_brainstorming/sqpsychconv/new_repo/SENIOR-REVIEW-REQUEST-vibe-check.md`

---

## 1. Executive Summary

**vibe-check** is a production-grade multi-agent LLM system that scores synthetic therapy conversations with PHQ-8 depression severity labels using frontier model consensus.

### The Problem

The `ai-psychiatrist` pipeline achieves strong PHQ-8 prediction but cannot be deployed because:

- DAIC-WOZ has restrictive academic licensing
- Embeddings derived from DAIC-WOZ cannot be redistributed
- Few-shot retrieval requires reference examples with ground truth scores

### The Solution

Score SQPsychConv (2,090 synthetic therapy dialogues) with PHQ-8 labels using frontier LLM consensus, intended to create a redistributable retrieval corpus validated against DAIC-WOZ ground truth (contingent on SQPsychConv licensing confirmation; see Section 3.1).

### Why "vibe-check"?

The system literally checks the "vibe" of therapy conversations to assess mental health severity. Also: it's memorable and the domain `vibe-check.ai` might be available.

### Known Sharp Edges (Read First)

- **SQPsychConv “train/test” splits may be identical** (observed in local exports for some variants). Treat HF splits as untrusted and implement a deterministic resplit based on `file_id`.
- **DAIC-WOZ is evaluation-only**: it must not be redistributed, including derived artifacts (raw text, embeddings, etc.).
- **No DAIC-WOZ egress**: do not send DAIC-WOZ transcripts to third-party APIs (OpenAI/Anthropic/Google). Keep DAIC-WOZ evaluation local-only (see Section 3.2).
- **LLM JSON failures can be deterministic at temperature=0**: do not rely on “retry until it parses” as the only mitigation.
- **External facts drift** (pricing/model IDs/benchmarks): any time-sensitive values must be re-verified against provider SSOTs before implementation.

---

## 2. January 2026 Frontier Models

### 2.1 Model Selection (Needs Re-Verification)

| Role | Model | Model ID | Provider | Price (in/out per 1M) |
|------|-------|----------|----------|----------------------|
| **Juror A** | GPT-5.2 Thinking | `gpt-5.2` | OpenAI | $1.25 / $10.00† |
| **Juror B** | Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Anthropic | $3.00 / $15.00 |
| **Juror C** | Gemini 3 Flash | `gemini-3-flash-preview` | Google | $0.50 / $3.00 |
| **Judge** | Claude Opus 4.5 | `claude-opus-4-5-20251101` | Anthropic | $5.00 / $25.00 |

**†Hidden Reasoning Tokens Warning**: GPT-5.2 Thinking uses adaptive reasoning that generates hidden chain-of-thought tokens. These tokens are **billed as output tokens but not visible via the API**. Clinical assessments may consume 2,000+ hidden tokens per call. Budget accordingly (see Section 2.3).

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

### 2.3 Cost Estimate (Updated - Hidden Token Aware)

**Important**: GPT-5.2 Thinking generates hidden reasoning tokens billed as output. The estimates below include a **3x multiplier** for GPT-5.2 output tokens to account for this.

| Component | Calculation | Visible Tokens | Hidden Tokens Est. | Cost |
|-----------|-------------|----------------|-------------------|------|
| GPT-5.2 (2 runs × 2,090) | 4,180 calls | 2.5K in / 1K out | +2K hidden out | ~$105† |
| Sonnet 4.5 (2 runs × 2,090) | 4,180 calls | 2.5K in / 1K out | N/A | ~$38 |
| Gemini 3 Flash (2 runs × 2,090) | 4,180 calls | 2.5K in / 1K out | N/A | ~$6 |
| Opus 4.5 Judge (~20% arbitration) | ~830 calls | 3K in / 1.5K out | N/A | ~$15 |
| **Total (one pass)** | | | | **~$165** |
| **With batch discounts (~50%)** | | | | **~$85** |

†GPT-5.2 hidden token estimate: 2,000 reasoning tokens × $10/1M × 4,180 calls = ~$84 additional. Actual costs vary by task complexity—clinical assessments are reasoning-heavy.

**Cost Savings Options**:
- Prompt caching: up to 90% savings on repeated context
- Batch API: 50% savings (OpenAI/Anthropic)

---

## 3. Data Governance & Licensing Gates

**This section is a hard gate for redistribution and for any processing of restricted datasets (e.g., DAIC-WOZ) via third-party APIs.** You can still implement the pipeline and run it on non-restricted data, but do not claim or attempt public release of derived artifacts until licensing is confirmed, and do not send restricted transcripts to external vendors without institutional approval.

### 3.1 SQPsychConv License

**Status**: UNKNOWN - Requires author confirmation before redistribution

| What We Know | Source |
|--------------|--------|
| Paper is CC BY 4.0 | arXiv license metadata |
| Project website is CC BY-SA 4.0 | sqpsych.github.io footer |
| HuggingFace dataset card | **Empty** - no license displayed |
| Paper text | Does not explicitly license the dataset |

**Critical Distinction**: The arXiv paper being CC BY 4.0 does **not** mean the dataset is CC BY 4.0. The paper license covers the paper text, not the data artifacts.

**Action Required** (Hard Gate):

1. Check HuggingFace dataset card for explicit license before production run
2. If license is absent: contact AIMH authors directly to confirm redistribution rights
3. Until confirmed: treat as "research use only, no redistribution of derived artifacts"

**Fallback Position**: If license remains unclear, the vibe-check corpus can still be used for internal validation but embeddings/labels cannot be publicly redistributed until licensing is resolved.

### 3.2 DAIC-WOZ EULA Restrictions

DAIC-WOZ is restricted to academic/non-profit use, and the safest interpretation is to treat it as **non-exportable**.

**Policy (non-negotiable for this spec)**:

- **Do not send DAIC-WOZ transcripts to third-party APIs** (OpenAI/Anthropic/Google).
- The `vibe-check` **core labeling pipeline must not require DAIC-WOZ**. DAIC-WOZ data and derived artifacts must never be checked into the repo.
- Any DAIC-WOZ evaluation happens **locally** (e.g., in `ai-psychiatrist` with Ollama/vLLM) and must avoid cloud logging/telemetry that could capture text.

If you later obtain explicit institutional/legal approval to process DAIC-WOZ with vendor APIs, that is a separate decision and requires a spec revision.

### 3.3 Logging & Observability Policy

To avoid accidental transcript leakage:

| Artifact | Allowed Content | Prohibited Content |
|----------|-----------------|-------------------|
| Checkpoint DB | File IDs, scores, entropy, status | Raw transcript text |
| Exception traces | Stack traces, error codes | Transcript snippets |
| LangSmith traces | Node timing, token counts | Prompt/response content |
| Run manifests | Counts, aggregate stats | Individual utterances |
| Job ledger | Status, attempts, error codes | Transcript excerpts |

**Implementation**: Use a `SensitiveString` wrapper type that refuses to serialize to logs/JSON.

### 3.4 Corpus Integrity (Trust No Split)

**Problem**: SQPsychConv HuggingFace "train/test" splits may have 100% overlap (observed in local exports).

**Solution**: Implement deterministic resplit based on `file_id`:

```python
import hashlib

def compute_split(file_id: str) -> str:
    """Deterministic split based on file_id hash."""
    hash_val = int(hashlib.sha256(file_id.encode()).hexdigest(), 16)
    bucket = hash_val % 10
    if bucket < 8:
        return "train"  # 80%
    elif bucket < 9:
        return "dev"    # 10%
    else:
        return "test"   # 10%
```

**Corpus Integrity Checks** (run before scoring):
- `file_id` uniqueness: 0 duplicates allowed
- Dialogue deduplication: SHA256 hash of `dialogue_clean`
- Split leakage check: train ∩ test = ∅
- Near-duplicate detection: MinHash/LSH for high-similarity pairs

**Output**: `corpus_integrity_manifest.json` with counts + warnings (no transcript text).

---

## 4. Definitive Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Scoring Metric** | PHQ-8 + self-harm boolean tag | DAIC-WOZ alignment; avoids PHQ-9 safety refusals |
| **Framework** | LangGraph 1.0 + PydanticAI | Production checkpointing, structured outputs, fault tolerance |
| **Aggregation** | Posterior convolution (Section 7.2) | Principled uncertainty; credible intervals on total score |
| **Disagreement Threshold** | Posterior-based + range fallback (Section 7.3) | Entropy/max-prob primary; range ≥ 2 as safety net |
| **Runs per Model** | 2 runs × 3 models = 6 passes | Balances cost vs stability |
| **Embedding View** | `client_qa` (Section 5.3.1) | Avoids semantic void; acceptable therapist question context |
| **Scoring View** | `client_qa` | Minimal Q/A context for short-answer interpretation |
| **Checkpoint Storage** | SQLite (dev) / PostgreSQL (prod) | LangGraph native persistence + psycopg driver |
| **Structured Output** | Provider-specific modes (Section 13) | JSON schema mode per provider; repair fallback chain |
| **Concurrency** | Global semaphore (MAX=50) | Prevents file descriptor exhaustion |
| **Job Tracking** | Job ledger (Section 14) | Idempotent restarts; error categorization |

---

## 5. Deep Dive: LangGraph Architecture

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
    dialogue: str
    scoring_text: str
    jury_results: list[PHQ8Report]
    needs_arbitration: bool
    final_output: AggregatedPHQ8 | None

async def jury_node(state: ScoringState) -> dict:
    """Run all three jurors in parallel using PydanticAI agents."""
    results = await asyncio.gather(
        gpt_agent.run(state["scoring_text"]),
        claude_agent.run(state["scoring_text"]),
        gemini_agent.run(state["scoring_text"]),
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
    dialogue: str
    scoring_text: str

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
    """Build bias-aware text views from the raw dialogue."""
    views = build_dialogue_views(state["dialogue"])
    return {"scoring_text": views.client_qa_text}

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

**Critical Infrastructure Note**: Even with `aiolimiter` for API rate limiting, Python's `asyncio` loop may attempt to open thousands of sockets/files simultaneously, hitting OS `ulimit` (often 1024) and causing `OSError: Too many open files`. Use a **global semaphore** to limit concurrent dialogues.

```python
from asyncio import Semaphore
from langgraph.constants import Send

# Global concurrency limit
MAX_CONCURRENT_DIALOGUES = 50  # Tune based on OS limits and API quotas

class BatchState(TypedDict):
    dialogues: list[dict]  # [{file_id, dialogue}, ...]
    completed: Annotated[list[AggregatedPHQ8], operator.add]

# Global semaphore for resource protection
workflow_sem = Semaphore(MAX_CONCURRENT_DIALOGUES)

async def score_single_wrapper(state: ScoringState) -> dict:
    """Wrapper that enforces concurrency limit."""
    async with workflow_sem:
        return await score_single(state)

def orchestrator_node(state: BatchState) -> list[Send]:
    """Fan out to process each dialogue independently."""
    return [
        Send(
            "score_single",  # Target node (the subgraph with semaphore)
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
batch_graph.add_node("score_single", score_single_wrapper)  # Uses semaphore
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
5. **Resource protection**: Combined with semaphore, prevents file descriptor exhaustion

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

## 6. Complete System Architecture

### 5.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           vibe-check Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌────────────────────────────┐ │
│  │ HuggingFace  │      │ Preprocess   │      │      CONSENSUS ENGINE      │ │
│  │ SQPsychConv  │─────▶│ Extract      │─────▶│                            │ │
│  │ 2,090 dlgs   │      │ Text Views   │      │  ┌─────┐ ┌─────┐ ┌─────┐   │ │
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
                    │ Build dialogue  │
                    │ text views      │
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

### 5.3 Preprocessing: Bias-Aware Dialogue Views

**Problem**: In clinical interviews (and synthetic therapy dialogues), the therapist/interviewer contributes a strong *protocol prior*. For DAIC-WOZ in particular, research shows models can exploit interviewer prompts as a shortcut signal rather than learning patient-language indicators of depression severity.

**Key insight**: Even if an LLM can distinguish speakers in the final prompt, **embeddings and retrieval cannot reliably disambiguate “therapist asked about X” from “client reported X.”** Speaker leakage at the embedding layer can corrupt few-shot selection before the scorer ever sees the examples.

This spec therefore standardizes **multiple deterministic dialogue views** and uses them *intentionally* by stage.

#### 5.3.1 Dialogue Views (Single Source of Truth)

For every input dialogue, preprocessing produces:

- `dialogue_clean`: normalized speaker labels + whitespace (no semantic rewriting)
- `client_only_text`: client/participant utterances only (**WARNING: semantic void risk for embeddings**)
- `client_qa_text`: client utterances plus the **single most recent** therapist prompt for each contiguous client block
- `client_contextualized`: client utterances rewritten with question context embedded (see 5.3.1.1)

**Defaults (Updated Based on Senior Review)**:

- **Embeddings/retrieval**: `client_qa_text` (recommended) OR `client_contextualized` (best quality, higher cost)
- **Scoring/jurors**: `client_qa_text`

**Rationale Change**: The original default of `client_only_text` for embeddings creates a **semantic void problem**:

> *Therapist:* "How is your sleep?" → *Client:* "Terrible."
> *Therapist:* "How is your relationship?" → *Client:* "Terrible."

If you embed just "Terrible", the vector has no semantic connection to "Sleep" or "Relationship". Your retrieval system will fetch random complaints, not symptom-specific matches.

**Tradeoff Analysis**:

| View | Embedding Quality | Protocol Leakage Risk | Cost |
|------|------------------|----------------------|------|
| `client_only_text` | Poor (semantic void) | None | Free |
| `client_qa_text` | Good | Low (question only) | Free |
| `client_contextualized` | Best | None | ~$5 preprocessing |

**Recommendation**: Use `client_qa_text` for embeddings. The therapist question context is necessary for semantic grounding, and the "protocol leakage" concern is less severe than semantic void. Reserve `client_only_text` for scoring prompts where you want to minimize prompt injection risk.

#### 5.3.1.1 Contextualized Rewriting (Optional, Best Quality)

For highest embedding quality, use a cheap model (Gemini Flash) to rewrite client responses with context:

```python
async def contextualize_utterance(
    therapist_question: str,
    client_response: str,
    model: str = "gemini-3-flash-preview"
) -> str:
    """Rewrite client response with embedded context.

    Example:
        Input:  Q="How is your sleep?" A="Terrible"
        Output: "The participant reports terrible sleep quality."
    """
    prompt = f"""Rewrite the client's response to include the question context.
    Do NOT change the meaning. Do NOT add information not present.
    Output ONLY the rewritten sentence.

    Therapist: {therapist_question}
    Client: {client_response}
    Rewritten:"""

    return await call_model(prompt, model)
```

**Cost**: ~$5 for 2,090 dialogues at ~1K tokens each with Gemini Flash ($0.50/1M in).

Deterministic rule for `client_qa_text`:

- Keep every client utterance.
- For each contiguous block of client utterances, include the immediately preceding therapist line **once** (do not repeat it before every client line).

#### 5.3.2 First-Principles Example (Why Views Matter)

Full dialogue:

```
Therapist: On a scale of 1-10, how hopeless have you felt?
Client: About an 8.
```

- `client_only_text` → `About an 8.` (ambiguous; high hallucination risk)
- `client_qa_text` → includes the therapist question, enabling correct interpretation.

Juror prompts must still enforce: **score PHQ-8 based only on client statements**; therapist text is context, not evidence.

#### 5.3.3 DAIC-WOZ Phase 0: Deterministic Transcript Hygiene

Phase 0 validation must apply the same *mechanical* transcript corrections that the DAIC-WOZ preprocessing literature and reference tooling document, to avoid silently biasing evaluation.

Minimum deterministic rules (parity with the Bailey/Plumbley DAIC-WOZ preprocessing tool and this repo’s implementation patterns):

- Validate required schema (`start_time`, `stop_time`, `speaker`, `value`).
- Normalize speakers (`ellie` → `Ellie`, `participant` → `Participant`).
- Remove pre-interview preamble (rows before first Ellie utterance when Ellie exists).
- Remove sync markers: `<sync>`, `<synch>`, `[sync]`, `[synch]`, `[syncing]`, `[synching]` (case/whitespace tolerant).
- Remove known interruption windows:
  - 373: `[395, 428]` seconds
  - 444: `[286, 387]` seconds
- Handle known “no Ellie transcript” sessions without failing (451/458/480).
- Preserve nonverbal annotations (e.g., `<laughter>`) by default for LLM scoring; treat removal as an explicit ablation.

Operational constraint:

- Never overwrite raw transcripts; write processed variants to a new directory and record a **counts-only** manifest (no transcript text).

#### 5.3.4 Required Ablations (Preprocessing Is Not “Set and Forget”)

To avoid accidental benchmark gaming and to quantify the prompt-leakage tradeoff, Phase 0 and Phase 3 must include:

- scoring view: `client_qa_text` vs `client_only_text`
- embedding view: `client_only_text` vs `dialogue_clean` (expected to be worse; included as a sanity check)

References:

- Burdisso et al. (2024): interviewer prompt shortcut behavior in DAIC-WOZ.

---

## 7. Scoring Metric: PHQ-8 + Self-Harm Tag

### 6.1 Why PHQ-8 (NOT PHQ-9)

| Factor | PHQ-8 | PHQ-9 |
|--------|-------|-------|
| DAIC-WOZ alignment | Direct match (0-24 scale) | Scale mismatch (0-27) |
| LLM safety refusals | Lower risk | Higher refusal risk for suicide-related content (varies by provider/policy) |
| SQPsychConv coverage | N/A | Only 0.3% explicit mentions |
| PHQ-8/PHQ-9 total correlation | r = 0.996 | Total scores are near-equivalent |
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

## 8. Consensus Architecture

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

### 7.2 Total-Score Posterior via Convolution

**Problem**: Computing per-item posteriors then summarizing totals via mode/expected/std of juror totals can produce inconsistent results—item modes may sum to a total that isn't the most probable total.

**Solution**: Convolve item posteriors to get a proper total-score distribution.

```python
import numpy as np
from scipy.signal import convolve

def compute_total_posterior(item_posteriors: list[np.ndarray]) -> np.ndarray:
    """Convolve 8 item posteriors to get total-score distribution.

    Args:
        item_posteriors: List of 8 arrays, each shape (4,) for scores 0-3

    Returns:
        Array of shape (25,) for total scores 0-24
    """
    # Start with first item
    total_dist = item_posteriors[0]

    # Convolve remaining items
    for item_post in item_posteriors[1:]:
        total_dist = convolve(total_dist, item_post)

    return total_dist  # Shape (25,) for scores 0-24

def compute_severity_bucket_probs(total_posterior: np.ndarray) -> dict[str, float]:
    """Compute probability of each severity bucket."""
    return {
        "0-4 (minimal)": float(total_posterior[0:5].sum()),
        "5-9 (mild)": float(total_posterior[5:10].sum()),
        "10-14 (moderate)": float(total_posterior[10:15].sum()),
        "15-19 (mod_severe)": float(total_posterior[15:20].sum()),
        "20-24 (severe)": float(total_posterior[20:25].sum()),
    }

def compute_credible_interval(total_posterior: np.ndarray, alpha: float = 0.10) -> tuple[int, int]:
    """Compute (1-alpha) credible interval for total score."""
    cdf = np.cumsum(total_posterior)
    lower = int(np.searchsorted(cdf, alpha / 2))
    upper = int(np.searchsorted(cdf, 1 - alpha / 2))
    return (lower, upper)
```

**Output Extensions**:

- `total_posterior: dict[int, float]` — P(total=k) for k in 0..24
- `total_ci_90: tuple[int, int]` — 90% credible interval
- `severity_bucket_probs: dict[str, float]` — P(severity=bucket)

### 7.3 Disagreement Threshold (Posterior-Based)

**Primary Trigger** (posterior uncertainty):

1. **Low max posterior**: `max(posterior[item]) < 0.60` → item is uncertain
2. **High entropy**: `entropy(posterior[item]) > 1.2` → votes are dispersed
3. **Clinical threshold ambiguity**: `P(item ∈ {2,3}) ∈ [0.4, 0.6]` → borderline clinical significance

**Safety Net Trigger** (keep for interpretability):

4. **Range ≥ 2**: e.g., votes `{0, 2, 2}` have range 2 → arbitrate
5. **Insufficient evidence** flagged by ≥2 jurors
6. **Total score std ≥ 2.0** across all juror reports

**Trigger Logic**:

```python
def should_arbitrate(item_posterior: np.ndarray, votes: list[int]) -> bool:
    """Determine if item needs arbitration."""
    max_prob = item_posterior.max()
    entropy_val = -np.sum(item_posterior * np.log(item_posterior + 1e-10))
    clinical_prob = item_posterior[2] + item_posterior[3]  # P(score >= 2)

    # Posterior-based (primary)
    if max_prob < 0.60:
        return True
    if entropy_val > 1.2:
        return True
    if 0.4 <= clinical_prob <= 0.6:
        return True

    # Range-based (safety net)
    if max(votes) - min(votes) >= 2:
        return True

    return False
```

**Why posterior-based arbitration?**

- Range ≥ 2 is a blunt instrument—votes `[1, 2, 2, 2, 2, 3]` have range 2 but high consensus on 2
- Max posterior < 0.60 catches cases where votes are dispersed even within adjacent scores
- Entropy captures overall uncertainty, not just extremes
- Clinical threshold check ensures we arbitrate when the clinical decision (PHQ-8 item ≥ 2 indicates symptom presence) is ambiguous

### 7.4 Meta-Judge Prompt

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

## 9. Data Schemas (Pydantic)

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

    # Total scores (updated with posterior convolution)
    total_mode: int = Field(ge=0, le=24)
    total_expected: float
    total_std: float
    total_posterior: dict[int, float] = Field(default_factory=dict)  # P(total=k) for k in 0..24
    total_ci_90: tuple[int, int] | None = None  # 90% credible interval
    severity_bucket: Literal["0-4", "5-9", "10-14", "15-19", "20-24"]
    severity_bucket_probs: dict[str, float] = Field(default_factory=dict)  # P(severity=bucket)

    # Consensus metadata
    triggered_arbitration: bool
    arbitration_items: list[str] = Field(default_factory=list)
    arbitration_reasons: dict[str, str] = Field(default_factory=dict)  # item -> reason

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

## 10. Repository Structure

```
vibe-check/
├── README.md
├── pyproject.toml
├── uv.lock                           # Lockfile for reproducibility
├── .python-version                   # Python version pin (e.g., 3.11)
├── .env.example
├── .pre-commit-config.yaml           # Pre-commit hooks (ruff + mypy)
├── Makefile                          # Developer convenience commands
├── CLAUDE.md
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI workflow
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
│       │   ├── client_extractor.py   # Build dialogue views (client_only, client_qa)
│       │   └── cleaner.py            # Normalization + CJK detection (optional filtering)
│       │
│       └── export/
│           ├── __init__.py
│           ├── jsonl.py
│           ├── csv.py
│           └── embeddings.py
│
├── scripts/
│   ├── score_corpus.py               # Main CLI entry point
│   ├── compute_diagnostics.py        # Phase 0: sanity diagnostics (no DAIC-WOZ)
│   ├── generate_embeddings.py        # Create retrieval corpus
│   └── evaluate_transfer.py          # Local-only sim-to-real metrics (no vendor APIs)
│
├── tests/
│   ├── conftest.py                   # Shared fixtures (mock clients, sample data)
│   ├── fixtures/                     # Reusable test fixtures
│   │   ├── __init__.py
│   │   ├── mock_llm.py               # Mock LLM client
│   │   └── sample_dialogues.py       # Sample test dialogues
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

## 11. Developer Experience (2026 Best Practices)

This section defines production-grade Python DevEx based on 2026 best practices with uv, ruff, mypy strict mode, and GitHub Actions.

### 10.1 Complete pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vibe-check"
version = "0.1.0"
description = "Multi-agent PHQ-8 scoring for synthetic therapy dialogues"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "CLARITY-DIGITAL-TWIN" }]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Orchestration
    "langgraph>=1.0.0",
    "langgraph-checkpoint-postgres>=2.0.0",  # For production checkpointing
    "psycopg[binary,pool]>=3.2.0",           # REQUIRED for PostgresSaver

    # Agents (PydanticAI for structured outputs)
    "pydantic-ai>=1.0.0",

    # Provider clients
    "langchain-openai>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-google-genai>=2.1.0",

    # Data
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
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
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=8.3.5",
    "pytest-asyncio>=0.25.0",
    "pytest-cov>=7.0.0",
    "pytest-xdist>=3.5.0",
    "pytest-sugar>=1.0.0",
    "httpx>=0.28.0",
    "respx>=0.22.0",
    # Linting & formatting
    "ruff>=0.9.2",
    # Type checking
    "mypy>=1.15.0",
    # Pre-commit
    "pre-commit>=4.1.0",
]

[project.scripts]
vibe-check = "vibe_check.cli:main"

# ─────────────────────────────────────────────────────────────
# Ruff Configuration (linter + formatter)
# ─────────────────────────────────────────────────────────────
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "RUF",    # Ruff-specific rules
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # function call in default argument (needed for FastAPI)
]

[tool.ruff.lint.isort]
known-first-party = ["vibe_check"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

# ─────────────────────────────────────────────────────────────
# Mypy Configuration (strict mode)
# ─────────────────────────────────────────────────────────────
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unreachable = true
show_error_codes = true
pretty = true

# Per-module overrides for external libs
[[tool.mypy.overrides]]
module = [
    "datasets.*",
    "langgraph.*",
    "langchain_openai.*",
    "langchain_anthropic.*",
    "langchain_google_genai.*",
    "scipy.*",
    "sklearn.*",
]
ignore_missing_imports = true

# ─────────────────────────────────────────────────────────────
# Pytest Configuration
# ─────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "-ra",
    "--tb=short",
]
markers = [
    "unit: Unit tests (fast, no external deps)",
    "integration: Integration tests (may use mocks)",
    "e2e: End-to-end tests (requires API keys)",
    "slow: Slow tests (>10s)",
]
filterwarnings = [
    "ignore::DeprecationWarning:pydantic.*:",
]

# ─────────────────────────────────────────────────────────────
# Coverage Configuration
# ─────────────────────────────────────────────────────────────
[tool.coverage.run]
source = ["src/vibe_check"]
branch = true
parallel = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "@abstractmethod",
]
fail_under = 80
show_missing = true
skip_covered = true

[tool.coverage.html]
directory = "htmlcov"
```

### 10.2 Pre-commit Configuration (.pre-commit-config.yaml)

```yaml
# .pre-commit-config.yaml
# Modern Python pre-commit using uv + ruff (2026)
# Install: uv run pre-commit install

repos:
  # ─────────────────────────────────────────────────────────────
  # Basic file hygiene
  # ─────────────────────────────────────────────────────────────
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  # ─────────────────────────────────────────────────────────────
  # Ruff (linting + formatting, replaces black/isort/flake8)
  # ─────────────────────────────────────────────────────────────
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.2
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # ─────────────────────────────────────────────────────────────
  # Mypy (type checking via uv for deps)
  # ─────────────────────────────────────────────────────────────
  - repo: local
    hooks:
      - id: mypy
        name: mypy (strict)
        entry: uv run mypy
        language: system
        types: [python]
        args: [src, tests, scripts]
        pass_filenames: false
        require_serial: true
```

### 10.3 GitHub Actions CI Workflow (.github/workflows/ci.yml)

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Install dependencies
        run: uv sync --locked --all-extras --dev

      - name: Ruff lint
        run: uv run ruff check . --output-format=github

      - name: Ruff format check
        run: uv run ruff format --check .

  typecheck:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Install dependencies
        run: uv sync --locked --all-extras --dev

      - name: Mypy
        run: uv run mypy src tests scripts --strict

  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set Python version
        run: uv python pin ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --locked --all-extras --dev

      - name: Run tests with coverage
        run: |
          uv run pytest tests/ \
            -n auto \
            --cov=src/vibe_check \
            --cov-report=xml \
            --cov-report=term-missing \
            --cov-fail-under=80

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          files: coverage.xml
          fail_ci_if_error: false
```

### 10.4 Makefile

```makefile
# Makefile - Developer convenience commands
.PHONY: help install dev test lint format typecheck ci clean

# Default target
help:
	@echo "vibe-check development commands:"
	@echo ""
	@echo "  make dev          Install all dependencies + pre-commit hooks"
	@echo "  make install      Install production dependencies only"
	@echo "  make test         Run all tests with coverage"
	@echo "  make test-unit    Run unit tests only (fast)"
	@echo "  make test-parallel Run tests in parallel"
	@echo "  make lint         Run ruff linter"
	@echo "  make lint-fix     Auto-fix linting issues"
	@echo "  make format       Format code with ruff"
	@echo "  make typecheck    Run mypy strict type checking"
	@echo "  make ci           Full CI: lint + typecheck + test"
	@echo "  make clean        Remove build artifacts"

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
install:
	uv sync --locked

dev:
	uv sync --locked --all-extras --dev
	uv run pre-commit install

# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ --cov=src/vibe_check --cov-report=term-missing --cov-fail-under=80

test-unit:
	uv run pytest tests/unit/ -v

test-parallel:
	uv run pytest tests/ -n auto --dist=loadscope

# ─────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src tests scripts --strict

# ─────────────────────────────────────────────────────────────
# CI
# ─────────────────────────────────────────────────────────────
ci: format-check lint typecheck test

# ─────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
```

---

## 12. Configuration

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
# Preprocessing (Dialogue Views) - Updated per Senior Review
# ─────────────────────────────────────────────────────────────
# What jurors see (default: preserve minimal question context for short answers)
SCORING_DIALOGUE_VIEW=client_qa
# What embeddings/indexing use (CHANGED: client_qa to avoid semantic void)
# Options: client_qa (recommended), client_contextualized (best quality, +$5)
EMBEDDING_DIALOGUE_VIEW=client_qa

# ─────────────────────────────────────────────────────────────
# Concurrency Limits (Infrastructure)
# ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_DIALOGUES=50

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
from typing import Literal

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

    # Preprocessing (updated per senior review - client_qa for embeddings)
    scoring_dialogue_view: Literal["client_qa", "client_only"] = "client_qa"
    embedding_dialogue_view: Literal["client_qa", "client_contextualized", "client_only"] = "client_qa"

    # Concurrency
    max_concurrent_dialogues: int = 50

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

## 13. Validation Protocol

### 12.1 Phase 0: Labeler Sanity Checks (SQPsychConv-Only)

This repo is designed to score **synthetic** data (SQPsychConv) with frontier APIs. To avoid DAIC-WOZ licensing/EULA violations, **Phase 0 must not require DAIC-WOZ** and must not send restricted transcripts to vendor APIs.

**Goals (no ground truth required)**:

- Validate end-to-end correctness: preprocessing → structured output → aggregation → checkpoint/resume → export.
- Detect prompt/JSON/schema brittleness before paying for a full 2,090-dialogue run.
- Produce diagnostics that make failures actionable (without storing transcript text in logs/artifacts).

**Procedure**:

1. Run a small stratified sample (e.g., 50 dialogues) end-to-end.
2. Compute internal diagnostics: agreement/entropy distributions, arbitration rate, condition separation (MDD > control), parse failure rate.
3. Optionally run a clinician/human audit slice for calibration (recommended if you want any tuned thresholds without DAIC-WOZ).

Example:

```bash
uv run python scripts/score_corpus.py \
    --input AIMH/SQPsychConv_qwq \
    --limit 50 \
    --output data/outputs/scored_sqpsychconv_smoke.jsonl
```

**Success Criteria** (sanity-only):

- 0% unrecoverable parse failures after the JSON/structured-output fallback chain.
- Checkpoint/resume works (kill the process mid-run; restart; no duplicates).
- Arbitration rate is within an expected band (e.g., 10–30% on the smoke slice).
- Directional validity: mean PHQ-8 total for `mdd` > `control`.

### 12.1.1 Threshold Tuning (Optional, No DAIC-WOZ)

Because SQPsychConv has no public ground truth severity labels, avoid “tuning” arbitration/uncertainty thresholds on synthetic data in a way that could silently overfit to generator artifacts.

If tuning is required, prefer one of:

- **Human audit slice** (within SQPsychConv) with blinded clinician scoring.
- **Downstream local-only evaluation** in `ai-psychiatrist` using DAIC-WOZ (no external egress), treating this as a separate integration project and decision.

### 12.2 Phase 1: Score SQPsychConv

```bash
uv run python scripts/score_corpus.py \
    --input AIMH/SQPsychConv_qwq \
    --output data/outputs/scored_sqpsychconv.jsonl
```

**Internal Sanity Checks**:

- Record corpus integrity: `file_id` uniqueness, duplicate detection, and a deterministic split manifest (do not trust HF split names).
- Record dialogue-view diagnostics: empty/near-empty `client_only_text`, frequency of short/ambiguous answers (`yes/no/ok/8`) that require `client_qa_text` context.
- Record speaker parsing integrity: unknown/missing role-prefix rate (must be 0 or explicitly warned + skipped).
- Record text artifact stats: CJK character count per variant (detect by default; removal only via explicit ablation).
- MDD dialogues should have higher mean PHQ-8 than control
- Cronbach's α > 0.7 (internal consistency)
- Arbitration rate < 30% (most items reach consensus)

### 12.3 Phase 2: Generate Embeddings

```bash
uv run python scripts/generate_embeddings.py \
    --input data/outputs/scored_sqpsychconv.jsonl \
    --output data/outputs/embeddings/
```

### 12.4 Phase 3: Sim-to-Real Evaluation (Local-Only; No Vendor APIs)

This phase uses DAIC-WOZ and must be executed in a local-only environment where transcripts never leave the machine/network that is authorized to hold them.

`vibe-check` does not require DAIC-WOZ to function; treat this phase as a downstream integration run (recommended) that can be executed from `ai-psychiatrist` or an equivalent local evaluation harness.

**Non-negotiable**: do not call OpenAI/Anthropic/Google APIs with DAIC-WOZ transcripts.

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
| client_only vs dialogue_clean | Embedding strategy (prompt-leakage ablation) |
| total_mode vs total_expected | Label to use |
| low-entropy filter | Only retrieve confident exemplars |

**Success Criteria**:

| Metric | Target | Rationale |
|--------|--------|-----------|
| DAIC-WOZ MAE | < 6.0 | SOTA range is 5.0-6.0 |
| Binary AUC | > 0.75 | Useful discrimination |
| Δ vs zero-shot | > 5% relative | Proves transfer value |

---

## 14. Structured Output Contract (Per Provider)

**Critical**: Do not rely on "JSON in plaintext" prompting. Use provider-specific structured output modes.

### 13.1 Provider Contracts

| Provider | Structured Output Mode | Fallback Strategy |
|----------|----------------------|-------------------|
| **OpenAI** | `response_format: { type: "json_schema", json_schema: {...} }` | JSON repair → fix-JSON reprompt |
| **Anthropic** | Tool use with single-tool schema OR strict JSON mode | JSON repair → fix-JSON reprompt |
| **Google** | `response_mime_type: "application/json"` + `response_schema` | JSON repair → fix-JSON reprompt |

### 13.2 JSON Repair Fallback Chain

Do not retry blindly at `temperature=0`—deterministic failures will repeat.

```python
from pydantic import ValidationError
import json_repair  # e.g., json-repair library

async def parse_with_fallback(
    raw_response: str,
    schema: type[BaseModel],
    model_id: str,
    max_repair_attempts: int = 2
) -> BaseModel | None:
    """Parse LLM response with fallback chain.

    Returns:
        Parsed model or None if all attempts fail.
    """
    # 1. Direct parse
    try:
        return schema.model_validate_json(raw_response)
    except ValidationError:
        pass

    # 2. Tolerant JSON repair
    try:
        repaired = json_repair.repair_json(raw_response)
        return schema.model_validate_json(repaired)
    except (ValidationError, json.JSONDecodeError):
        pass

    # 3. Fix-JSON reprompt (include the invalid JSON)
    for attempt in range(max_repair_attempts):
        fix_prompt = f"""The following JSON is invalid. Fix it to match this schema:
{schema.model_json_schema()}

Invalid JSON:
{raw_response}

Return ONLY the corrected JSON, no explanation."""

        fixed_response = await call_model(fix_prompt, model_id, temperature=0.1)
        try:
            return schema.model_validate_json(fixed_response)
        except ValidationError:
            continue

    # 4. Fail closed
    return None
```

### 13.3 Error Tracking

Track parse failures by provider and error type:

```python
class ParseErrorStats(BaseModel):
    provider: str
    error_type: Literal["invalid_json", "schema_mismatch", "repair_failed"]
    count: int
    sample_error: str | None  # First error message (no transcript text)
```

---

## 15. Job Ledger (Batch Operations)

**Problem**: LangGraph checkpointing tracks graph state, but for 2,090-dialogue batches you also need a job-level view for:
- Partial re-runs
- Progress monitoring
- Error analysis
- Audit trails

### 14.1 Job Ledger Schema

```sql
CREATE TABLE job_ledger (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL,  -- pending, running, succeeded, failed, skipped
    attempts INT DEFAULT 0,
    last_error_code VARCHAR(100),  -- rate_limit, parse_fail, refusal, timeout
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    models_used JSONB,  -- {"gpt": "gpt-5.2", "claude": "...", ...}
    prompt_hash VARCHAR(64),  -- SHA256 of prompt template
    config_hash VARCHAR(64),  -- SHA256 of scoring config
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for partial re-runs
CREATE INDEX idx_job_ledger_status ON job_ledger(status);
CREATE INDEX idx_job_ledger_error ON job_ledger(last_error_code);
```

### 14.2 Ledger Operations

```python
class JobLedger:
    """Minimal job tracking for batch operations."""

    async def mark_started(self, file_id: str, models: dict[str, str]) -> None:
        """Mark job as running."""

    async def mark_succeeded(self, file_id: str) -> None:
        """Mark job as complete."""

    async def mark_failed(self, file_id: str, error_code: str) -> None:
        """Mark job as failed with error category."""

    async def get_pending(self) -> list[str]:
        """Get file_ids not yet processed."""

    async def get_failed(self, error_code: str | None = None) -> list[str]:
        """Get file_ids that failed, optionally filtered by error type."""

    async def reset_failed(self, error_code: str | None = None) -> int:
        """Reset failed jobs to pending for retry."""
```

### 14.3 Restart Semantics

```bash
# Resume from last checkpoint (LangGraph handles node-level state)
uv run python scripts/score_corpus.py --resume

# Retry only failed jobs
uv run python scripts/score_corpus.py --retry-failed

# Retry only rate-limited jobs (after waiting)
uv run python scripts/score_corpus.py --retry-failed --error-code rate_limit
```

---

## 16. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **API rate limits** | `aiolimiter` + exponential backoff with jitter + global semaphore |
| **Batch job crash** | LangGraph checkpointing + job ledger; resume from exact node |
| **Invalid JSON** | Structured output modes + tolerant JSON repair + fix-JSON reprompt (Section 13) |
| **Model refuses** | "Clinical research assistant" role framing; self-harm as separate stage |
| **Prompt leakage (therapist protocol bias)** | Use bias-aware dialogue views (`client_qa` for embeddings; `client_qa` for scoring); run required ablations |
| **Semantic void (embeddings)** | Use `client_qa` or `client_contextualized` for embeddings (Section 5.3.1) |
| **Synthetic circularity** | Cross-vendor scorers; validate on DAIC-WOZ |
| **Cost overrun** | Hidden token budget (Section 2.3); Gemini 3 Flash is cheapest; batch API discounts |
| **Redistribution/license risk** | Data Governance section (Section 3); SQPsychConv license UNKNOWN until author confirmation |
| **File descriptor exhaustion** | Global semaphore (Section 4.4); MAX_CONCURRENT_DIALOGUES=50 |
| **Hidden thinking tokens** | 3x budget multiplier for GPT-5.2 (Section 2.3) |

---

## 17. Success Criteria Summary

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

## 18. Next Steps

### Immediate (Before Implementation)

1. [ ] Senior review of this spec
2. [ ] Approve model selection and costs
3. [ ] Set up API keys and verify access
4. [ ] Finalize prompt templates

### Phase 1: Scaffold (3-5 days)

1. [ ] Initialize repo with `uv init`
2. [ ] Implement Pydantic schemas
3. [ ] Implement preprocessing (dialogue views: `client_only`, `client_qa`)
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

## 19. References

### Research Papers

- [Traub et al. (2024): AUGRC for selective classification](https://arxiv.org/abs/2407.01032)
- [Amazon CollabEval: Multi-agent LLM-as-Judge](https://www.amazon.science/publications/enhancing-llm-as-a-judge-via-multi-agent-collaboration)
- [PHQ-8 vs PHQ-9 equivalency](https://stacks.cdc.gov/view/cdc/84248)
- [SQPsychConv dataset](https://arxiv.org/abs/2510.25384)
- [Burdiss[o] et al. (2024): Validity of therapist prompts in DAIC-WOZ](https://aclanthology.org/2024.clinicalnlp-1.8/)

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
