# SPEC: sqpsychconv-scored

**Repository**: `sqpsychconv-scored` (new repo, separate from ai-psychiatrist)
**Version**: 1.0-draft
**Date**: 2026-01-02
**Status**: DRAFT - Awaiting Senior Review

---

## 1. Executive Summary

This specification defines the architecture for a multi-agent LLM system that scores the SQPsychConv synthetic therapy dataset with PHQ-8 depression severity labels. The scored dataset enables deployable few-shot retrieval without DAIC-WOZ licensing restrictions.

### The Problem

The `ai-psychiatrist` pipeline achieves strong PHQ-8 prediction but cannot be deployed because:
- DAIC-WOZ has restrictive academic licensing
- Embeddings derived from DAIC-WOZ cannot be redistributed
- Few-shot retrieval requires reference examples with ground truth scores

### The Solution

Score SQPsychConv (2,090 synthetic therapy dialogues) with PHQ-8 labels using frontier LLM consensus, creating a freely redistributable retrieval corpus validated against DAIC-WOZ ground truth.

### Why This Is NOT Circular

- **Training corpus**: SQPsychConv (LLM-scored synthetic data)
- **Validation corpus**: DAIC-WOZ (real clinical PHQ-8 ground truth)
- Ground truth comes from DAIC-WOZ, not from the scoring LLMs

---

## 2. Definitive Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Scoring Metric** | PHQ-8 + self-harm boolean tag | DAIC-WOZ alignment; avoids PHQ-9 safety refusals; self-harm tag preserves utility |
| **Framework** | LangGraph | Fault-tolerant checkpointing for 2k+ dialogues; native map-reduce support |
| **Aggregation** | Distributional posterior | More principled than mean/mode; provides calibrated uncertainty |
| **Disagreement Threshold** | Range ≥ 2 per item | Simple, interpretable; triggers arbitration on real disagreement |
| **Model Jury** | GPT-4o + Claude 3.5 Sonnet + Gemini 1.5 Pro | Heterogeneous families reduce correlated errors |
| **Judge Model** | Claude Opus 4 (or GPT o1) | Stronger model, different family from majority of jurors |
| **Runs per Model** | 2 runs × 3 models = 6 passes | Balances cost vs. stability; expand adaptively if high variance |
| **Preprocessing** | Participant/Client utterances only | Avoids therapist prompt bias documented in DAIC-WOZ research |
| **Checkpoint Storage** | SQLite (dev) / PostgreSQL (prod) | LangGraph native; enables exact resume after failures |

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE PHASE: SYNTHETIC INDEXING                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐   │
│  │ DATA SOURCE    │     │  PREPROCESS    │     │  CONSENSUS SCORING │   │
│  │ SQPsychConv    │────▶│  Extract       │────▶│  3 Models × 2 Runs │   │
│  │ (HuggingFace)  │     │  Client Text   │     │  + Judge Arbiter   │   │
│  └────────────────┘     └────────────────┘     └─────────┬──────────┘   │
│                                                          │               │
│                                                          ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        SCORED DATASET                             │   │
│  │  • file_id, condition, dialogue                                   │   │
│  │  • PHQ8_* item scores (0-3) + evidence quotes                     │   │
│  │  • total_score, severity_bucket                                   │   │
│  │  • item_entropy, vote_distribution (uncertainty)                  │   │
│  │  • mentions_self_harm_or_death (boolean + evidence)               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Export to ai-psychiatrist
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ONLINE PHASE: EVALUATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐   │
│  │ TARGET DOMAIN  │     │  RETRIEVAL &   │     │    EVALUATION      │   │
│  │ DAIC-WOZ       │────▶│  AGGREGATION   │────▶│  vs Ground Truth   │   │
│  │ Real Clinical  │     │  k-NN from     │     │  MAE, AUC, F1      │   │
│  └────────────────┘     │  Synthetic     │     └────────────────────┘   │
│                         └────────────────┘                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Consensus Scoring Engine (LangGraph)

```
                         ┌─────────────────────┐
                         │  Input Dialogue     │
                         │  (Client text only) │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
     ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
     │ Juror A        │   │ Juror B        │   │ Juror C        │
     │ GPT-4o         │   │ Claude 3.5     │   │ Gemini 1.5 Pro │
     │ (2 runs)       │   │ Sonnet (2 runs)│   │ (2 runs)       │
     └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    CONSENSUS CHECK      │
                    │    Per-item analysis    │
                    └───────────┬─────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │  LOW VARIANCE    │              │  HIGH VARIANCE   │
    │  All items       │              │  Any item has    │
    │  range < 2       │              │  range ≥ 2       │
    └────────┬─────────┘              └────────┬─────────┘
             │                                 │
             ▼                                 ▼
    ┌──────────────────┐              ┌──────────────────┐
    │  AGGREGATE       │              │  META-JUDGE      │
    │  Distributional  │              │  Claude Opus 4   │
    │  posterior       │              │  Arbitrate items │
    └────────┬─────────┘              └────────┬─────────┘
             │                                 │
             └────────────────┬────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │    FINAL SCORE      │
                    │    + Uncertainty    │
                    └─────────────────────┘
```

---

## 4. Scoring Metric: PHQ-8 + Self-Harm Tag

### 4.1 Why PHQ-8 (NOT PHQ-9)

| Factor | PHQ-8 | PHQ-9 |
|--------|-------|-------|
| DAIC-WOZ alignment | Direct match | Scale mismatch (0-24 vs 0-27) |
| LLM safety refusals | None | High risk (~10-74% refusal rate) |
| Suicidality coverage in SQPsychConv | N/A | Only 0.3% explicit mentions |
| Construct validity | r = 0.996 with PHQ-9 | Full scale |
| Liability | Lower | Higher (suicide detection is high-stakes) |

**Decision**: PHQ-8 is mandatory. PHQ-9 Item 9 is NOT scored.

### 4.2 PHQ-8 Items (0-3 Scale)

| Item | Name | PHQ Key | Description |
|------|------|---------|-------------|
| 1 | Anhedonia | `PHQ8_NoInterest` | Little interest or pleasure in doing things |
| 2 | Depressed Mood | `PHQ8_Depressed` | Feeling down, depressed, or hopeless |
| 3 | Sleep | `PHQ8_Sleep` | Trouble falling/staying asleep, or sleeping too much |
| 4 | Fatigue | `PHQ8_Tired` | Feeling tired or having little energy |
| 5 | Appetite | `PHQ8_Appetite` | Poor appetite or overeating |
| 6 | Guilt | `PHQ8_Failure` | Feeling bad about yourself—or that you are a failure |
| 7 | Concentration | `PHQ8_Concentrating` | Trouble concentrating on things |
| 8 | Psychomotor | `PHQ8_Moving` | Moving or speaking slowly / being fidgety or restless |

**Score Anchors**:
- `0` = Not at all
- `1` = Several days
- `2` = More than half the days
- `3` = Nearly every day

### 4.3 Self-Harm Boolean Tag (NOT PHQ-9 Item 9)

Instead of scoring PHQ-9 Item 9 numerically, store a binary tag:

```json
{
  "mentions_self_harm_or_death": true,
  "self_harm_evidence": [
    "I keep replaying past failures, wondering if anyone'd notice if I just vanished"
  ],
  "disclaimer": "NOT validated for suicide risk assessment. For flagging purposes only."
}
```

**Rationale**:
- SQPsychConv has only 0.3% explicit suicidality mentions (6/2,090 dialogues)
- Numeric 0-3 scoring is unreliable with such sparse signal
- Binary tag preserves utility for retrieval filtering without high-stakes implications

### 4.4 Severity Buckets

| PHQ-8 Total | Severity |
|-------------|----------|
| 0-4 | Minimal |
| 5-9 | Mild |
| 10-14 | Moderate |
| 15-19 | Moderately Severe |
| 20-24 | Severe |

---

## 5. Consensus Architecture

### 5.1 Model Selection (Heterogeneous Jury)

| Role | Model | Provider | Rationale |
|------|-------|----------|-----------|
| Juror A | GPT-4o | OpenAI | Reasoning specialist; strong instruction following |
| Juror B | Claude 3.5 Sonnet | Anthropic | Nuance specialist; human-like judgment in clinical tasks |
| Juror C | Gemini 1.5 Pro | Google | Context specialist; massive context window for long dialogues |
| Judge | Claude Opus 4 | Anthropic | Strongest available; different primary provider from jurors |

**Heterogeneity Requirement**: At minimum, use 2 different model families (e.g., OpenAI + Anthropic). Ideally use 3 families to minimize correlated errors.

### 5.2 Scoring Runs

- **Default**: 2 runs per model × 3 models = 6 independent scores per dialogue
- **Adaptive**: If aggregate uncertainty is high after 6 runs, add 1 run per model (9 total) OR escalate directly to judge

### 5.3 Distributional Aggregation

For each PHQ-8 item:

1. **Collect votes**: `v_{i,1}, v_{i,2}, ..., v_{i,6}` from all scorer runs
2. **Compute posterior** over {0,1,2,3} with Dirichlet smoothing:
   ```
   p_i(s) ∝ α + #{v_{i,*} = s}    where α = 0.5 (weak prior)
   ```
3. **Final item score**: `argmax p_i` (mode) AND `E[p_i]` (expected value)
4. **Item uncertainty**: `H(p_i)` (entropy of distribution)

**Why distributional?** Simple mean/mode discards disagreement information. The posterior preserves uncertainty for downstream selective prediction.

### 5.4 Disagreement Threshold

Trigger Meta-Judge arbitration if ANY of:

1. **Range ≥ 2** for any item (e.g., scores {0, 2, 1} have range 2)
2. **Insufficient evidence** flagged by ≥2 scorers for any item
3. **Total score std ≥ 2.0** across all scorer runs

**Why Range ≥ 2?** This catches real disagreements (e.g., "symptom absent" vs "symptom severe") while allowing natural variance within adjacent scores.

### 5.5 Meta-Judge Arbitration

When triggered, the judge receives:

1. The **client-only dialogue text** (relevant portion for contested items)
2. All **scorer outputs** with scores and evidence quotes
3. The **PHQ-8 scoring rubric** with severity anchors

The judge outputs:
- Final score for each contested item
- **Statement of Resolution** explaining the decision
- Confidence level

---

## 6. Framework: LangGraph

### 6.1 Why LangGraph (Not Custom asyncio)

| Feature | Custom asyncio | LangGraph |
|---------|----------------|-----------|
| Fault tolerance | Manual implementation | Native checkpointing |
| Resume after crash | Custom state management | `compile(checkpointer=...)` |
| Parallel fan-out | `asyncio.gather()` | `Send()` API |
| Conditional routing | if/else chains | Conditional edges |
| Observability | Custom logging | LangSmith integration |
| Batch processing | Manual rate limiting | Built-in throttling |

For 2,090 dialogues × 6 runs × potential retries, fault tolerance is critical. LangGraph's checkpointing to SQLite/Postgres ensures exact resume after API failures, rate limits, or crashes.

### 6.2 Graph Structure

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# State schema
class ScoringState(TypedDict):
    file_id: str
    dialogue: str
    client_text: str  # Preprocessed
    jury_results: List[PHQ8Report]
    needs_arbitration: bool
    final_output: AggregatedPHQ8

# Nodes
workflow = StateGraph(ScoringState)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("jury", jury_scoring_node)
workflow.add_node("aggregate", aggregation_node)
workflow.add_node("arbitrate", meta_judge_node)

# Edges
workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "jury")
workflow.add_edge("jury", "aggregate")

# Conditional routing
def route_after_aggregate(state):
    return "arbitrate" if state["needs_arbitration"] else END

workflow.add_conditional_edges("aggregate", route_after_aggregate)
workflow.add_edge("arbitrate", END)

# Compile with persistence
checkpointer = SqliteSaver.from_conn_string("scores.sqlite")
app = workflow.compile(checkpointer=checkpointer)
```

### 6.3 Map-Reduce for Batch Processing

Use LangGraph's `Send()` API to fan-out across dialogues:

```python
from langgraph.constants import Send

def orchestrator_node(state):
    """Fan-out to score all dialogues in parallel."""
    return [
        Send("score_single", {"file_id": fid, "dialogue": dlg})
        for fid, dlg in state["dialogues"]
    ]
```

---

## 7. Data Schema

### 7.1 Input: SQPsychConv (HuggingFace)

```csv
file_id,condition,client_model,therapist_model,dialogue
active436,mdd,qwq_qwen,qwq_qwen,"Therapist: Good morning!..."
control1328,control,qwq_qwen,qwq_qwen,"Therapist: Hello..."
```

- **Corpus size**: 2,090 dialogues (train = test in HF export)
- **Distribution**: 56.4% control, 43.6% MDD
- **Length**: 2,487–12,446 chars (mean ~5,953)

### 7.2 Output: Scored Dataset

#### Per-Item Schema (Pydantic)

```python
class PHQItem(BaseModel):
    score: Literal[0, 1, 2, 3]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    insuff_evidence: bool = False
    vote_distribution: Dict[str, int]  # {"0": 1, "1": 3, "2": 2, "3": 0}
    entropy: float
```

#### Full Report Schema

```python
class ScoredDialogue(BaseModel):
    # Identity
    file_id: str
    condition: Literal["mdd", "control"]
    client_model: str
    therapist_model: str

    # PHQ-8 Items
    PHQ8_NoInterest: PHQItem
    PHQ8_Depressed: PHQItem
    PHQ8_Sleep: PHQItem
    PHQ8_Tired: PHQItem
    PHQ8_Appetite: PHQItem
    PHQ8_Failure: PHQItem
    PHQ8_Concentrating: PHQItem
    PHQ8_Moving: PHQItem

    # Aggregates
    total_score_mode: int = Field(ge=0, le=24)
    total_score_expected: float
    severity_bucket: Literal["0-4", "5-9", "10-14", "15-19", "20-24"]

    # Safety tag (NOT PHQ-9 Item 9)
    mentions_self_harm_or_death: bool = False
    self_harm_evidence: List[str] = Field(default_factory=list)

    # Uncertainty
    triggered_arbitration: bool
    total_vote_std: float
    per_model_reports: List[PHQ8Report]

    # Provenance
    scoring_models: List[str]
    judge_model: Optional[str]
    prompt_version: str
    scored_at: datetime
```

### 7.3 Output Files

| File | Description |
|------|-------------|
| `scored_sqpsychconv.csv` | Main scored dataset (1 row per dialogue) |
| `scored_sqpsychconv.jsonl` | Full structured output with uncertainty |
| `scoring_metadata.json` | Model versions, prompt hashes, run config |
| `validation_report.json` | Inter-model agreement, arbitration stats |
| `embeddings/sqpsychconv_scored.npz` | Vector embeddings for retrieval |
| `embeddings/sqpsychconv_scored.json` | Sidecar with texts and metadata |

---

## 8. Preprocessing

### 8.1 Client-Only Text Extraction

**Rationale**: DAIC-WOZ research shows models can exploit therapist prompts as shortcuts. Score participant/client utterances only.

```python
def extract_client_text(dialogue: str) -> str:
    """Extract only Client: utterances from dialogue."""
    lines = dialogue.split("\n")
    client_lines = [
        line.replace("Client:", "").strip()
        for line in lines
        if line.startswith("Client:")
    ]
    return " ".join(client_lines)
```

### 8.2 Quality Filters

| Issue | Detection | Action |
|-------|-----------|--------|
| CJK code-switching | `re.search(r'[\u4e00-\u9fff]', text)` | Flag in metadata; score anyway (models handle it) |
| Short dialogues | `len(client_text) < 500` | Flag low confidence |
| Missing [/END] marker | `not text.endswith("[/END]")` | Include anyway |

---

## 9. Validation Protocol

### 9.1 Phase 0: Scorer Competence (Pre-Validation)

Before scoring SQPsychConv, validate scorer ensemble on DAIC-WOZ:

1. Run ensemble on DAIC-WOZ train/dev transcripts (where ground truth exists)
2. Report metrics:
   - **Total score**: MAE, RMSE, Pearson r, Spearman ρ
   - **Binary (PHQ-8 ≥ 10)**: AUC, sensitivity, specificity, F1
   - **Item-level**: MAE per item, weighted κ per item
   - **Reliability**: ICC(2,k), Krippendorff's α (ordinal)

**Success Criteria**:
- Total score MAE ≤ 3.0
- Binary AUC > 0.80
- ICC ≥ 0.8 for inter-model agreement

### 9.2 Phase 1: Score SQPsychConv

- Process all 2,090 dialogues
- Store per-item scores, uncertainty, evidence

**Internal Sanity Checks**:
- Score distribution by condition (MDD should score higher than control)
- Cronbach's α over 8 items (internal consistency)
- Flag outliers: control with PHQ-8 ≥ 15, MDD with PHQ-8 ≤ 4

### 9.3 Phase 2: Build Retrieval Corpus

- Generate embeddings for scored dialogues
- Index in vector store with PHQ-8 metadata

### 9.4 Phase 3: Sim-to-Real Evaluation

Test three configurations on DAIC-WOZ:

| Baseline | Description |
|----------|-------------|
| A (No retrieval) | LLM predicts PHQ-8 directly from DAIC-WOZ text |
| B (kNN transfer) | Retrieve k synthetic, predict as weighted average |
| C (RAG few-shot) | Retrieve k synthetic as exemplars, LLM predicts |

**Ablations**:
- k ∈ {3, 5, 10, 20}
- Participant-only vs full transcript embeddings
- total_mode vs total_expected
- Uncertainty gating (only retrieve low-entropy exemplars)

**Success Criteria**:
- MAE < 6.0 (state-of-the-art on DAIC-WOZ is 5.0-6.0)
- Binary AUC > 0.75
- Improvement over Baseline A demonstrates transfer value

### 9.5 Phase 4: Human Validation (Optional)

- Sample 50-100 dialogues (stratified: high-uncertainty + low-uncertainty)
- Clinical expert scores PHQ-8 from transcript
- Report Cohen's κ / ICC vs ensemble

---

## 10. Cost Estimation

### 10.1 Token Estimates

| Component | Tokens |
|-----------|--------|
| Average dialogue | ~1,500 input |
| Scoring prompt | ~500 input |
| Structured output | ~400 output |
| **Per dialogue per model** | ~2,000 input / ~400 output |

### 10.2 Pricing (January 2026)

| Model | Input (/1M) | Output (/1M) |
|-------|-------------|--------------|
| GPT-4o | $2.50 | $10.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Claude Opus 4 (judge) | $15.00 | $75.00 |

### 10.3 Total Cost Estimate

| Scenario | Cost |
|----------|------|
| 3 models × 2 runs × 2,090 dialogues | ~$75-100 |
| + Judge on 20% (arbitration) | +$25-40 |
| + Batch API discounts (50%) | -50% |
| **Realistic total** | **$60-100** |

For validation runs (2-3 passes): **$150-250 total**

---

## 11. Configuration

### 11.1 Environment Variables

```env
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Model Selection
SCORER_MODELS=gpt-4o,claude-3-5-sonnet-20241022,gemini-1.5-pro
JUDGE_MODEL=claude-3-opus-20240229
RUNS_PER_MODEL=2

# Scoring Parameters
DISAGREEMENT_THRESHOLD_RANGE=2
ARBITRATION_TOTAL_STD=2.0

# Rate Limiting (per minute)
OPENAI_RPM=60
ANTHROPIC_RPM=60
GOOGLE_RPM=60

# Checkpointing
CHECKPOINT_DB=sqlite:///scores.sqlite
RESUME_FROM_CHECKPOINT=true

# Output
OUTPUT_DIR=./outputs
PROMPT_VERSION=v1.0
```

### 11.2 Scoring Prompt Template

```text
You are a clinical research assistant analyzing therapy transcripts for a retrospective study.
You are NOT interacting with a patient. Your task is strictly data classification.

Score the following transcript on the PHQ-8 depression scale.
Base scoring ONLY on what the CLIENT/PARTICIPANT says about THEIR OWN symptoms.

For each of the 8 items, provide:
1. Score (0-3): 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
2. Confidence (0.0-1.0): How confident are you in this score?
3. Evidence: 1-3 direct quotes from the transcript supporting this score
4. insuff_evidence: true if the transcript lacks clear information for this item

PHQ-8 Items:
1. PHQ8_NoInterest: Little interest or pleasure in doing things
2. PHQ8_Depressed: Feeling down, depressed, or hopeless
3. PHQ8_Sleep: Trouble falling/staying asleep, or sleeping too much
4. PHQ8_Tired: Feeling tired or having little energy
5. PHQ8_Appetite: Poor appetite or overeating
6. PHQ8_Failure: Feeling bad about yourself—or that you are a failure
7. PHQ8_Concentrating: Trouble concentrating on things
8. PHQ8_Moving: Moving or speaking slowly / being fidgety or restless

Also detect: Does the transcript contain any mentions of self-harm, death wishes, or suicidal thoughts?
If yes, set mentions_self_harm_or_death=true and provide evidence quotes.

Client transcript:
{client_text}

Respond ONLY with valid JSON matching the schema. Do not include explanations outside JSON.
```

---

## 12. Repository Structure

```
sqpsychconv-scored/
├── README.md
├── pyproject.toml
├── .env.example
├── CLAUDE.md                     # AI assistant instructions
│
├── src/
│   └── sqpsychconv_scored/
│       ├── __init__.py
│       ├── config.py             # Pydantic settings
│       ├── schemas.py            # Data models (PHQItem, ScoredDialogue)
│       │
│       ├── data/
│       │   ├── loader.py         # HuggingFace data loading
│       │   └── preprocessor.py   # Client text extraction
│       │
│       ├── scoring/
│       │   ├── prompts.py        # Scoring prompt templates
│       │   ├── juror.py          # Single-model scorer
│       │   ├── aggregator.py     # Distributional aggregation
│       │   └── judge.py          # Meta-judge arbitration
│       │
│       ├── graph/
│       │   ├── nodes.py          # LangGraph node implementations
│       │   ├── edges.py          # Conditional routing
│       │   └── workflow.py       # Main graph definition
│       │
│       └── export/
│           ├── csv_writer.py
│           ├── jsonl_writer.py
│           └── embedding_generator.py
│
├── scripts/
│   ├── score_corpus.py           # Main entry point
│   ├── validate_scorers.py       # Phase 0: scorer competence
│   ├── generate_embeddings.py    # Phase 2: retrieval corpus
│   └── evaluate_transfer.py      # Phase 3: sim-to-real eval
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
│   ├── raw/                      # SQPsychConv from HF
│   └── outputs/                  # Scored datasets
│
└── docs/
    ├── architecture.md
    ├── prompts.md
    └── validation-protocol.md
```

---

## 13. Dependencies

```toml
[project]
name = "sqpsychconv-scored"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",
    "langchain-google-genai>=2.0.0",
    "pydantic>=2.0.0",
    "datasets>=2.0.0",          # HuggingFace
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",      # Metrics
    "tenacity>=8.0.0",          # Retries
    "aiolimiter>=1.1.0",        # Rate limiting
    "structlog>=24.0.0",        # Logging
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
```

---

## 14. Risk Mitigation

### 14.1 LLM Safety Refusals

| Risk | Mitigation |
|------|------------|
| Model refuses to score depression content | Use "clinical research assistant" role framing |
| Model outputs help resources instead of scores | Explicitly instruct: "Do not provide help resources; output only JSON" |
| Suicidality content triggers refusal | Score PHQ-8 only; use boolean tag for self-harm |

### 14.2 Synthetic Circularity

| Risk | Mitigation |
|------|------------|
| LLM recognizes LLM-generated style | Use cross-vendor scorers (not the generator family) |
| Inflated accuracy on synthetic data | Validate ONLY on DAIC-WOZ ground truth |
| Style bias in embeddings | Score based on symptom evidence, not linguistic style |

### 14.3 Technical Failures

| Risk | Mitigation |
|------|------------|
| API rate limits | aiolimiter + exponential backoff with jitter |
| Batch job crash | LangGraph checkpointing; resume from exact position |
| Invalid JSON output | Pydantic validation + retry with structured output mode |

---

## 15. Success Criteria

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Scorer ICC | ≥ 0.80 | High inter-model reliability |
| Krippendorff's α | ≥ 0.70 | Ordinal agreement accounting for chance |
| DAIC-WOZ MAE | < 6.0 | State-of-the-art range |
| DAIC-WOZ AUC (binary) | > 0.75 | Useful discriminative power |
| Improvement over zero-shot | > 5% relative MAE reduction | Demonstrates transfer value |
| Arbitration rate | < 30% | Most items reach consensus |
| Processing completion | 100% | All 2,090 dialogues scored |

---

## 16. Next Steps

### Immediate (Before Implementation)

1. [ ] Senior review of this spec
2. [ ] Approve definitive decisions (PHQ-8, LangGraph, model selection)
3. [ ] Finalize prompt template with clinical input
4. [ ] Set up API keys and rate limits

### Phase 1: Scaffold

1. [ ] Initialize repository with pyproject.toml
2. [ ] Implement Pydantic schemas
3. [ ] Implement data loader (HuggingFace)
4. [ ] Implement client text preprocessor

### Phase 2: Core Pipeline

1. [ ] Implement single-model scorer (juror)
2. [ ] Implement distributional aggregator
3. [ ] Implement meta-judge arbitration
4. [ ] Build LangGraph workflow with checkpointing

### Phase 3: Validation

1. [ ] Run scorer competence on DAIC-WOZ
2. [ ] Score full SQPsychConv corpus
3. [ ] Generate embeddings
4. [ ] Run sim-to-real evaluation

---

## 17. References

### Research Papers
- Traub et al. (2024): AUGRC for selective classification - https://arxiv.org/abs/2407.01032
- Amazon CollabEval: Multi-agent LLM-as-Judge - https://www.amazon.science/publications/enhancing-llm-as-a-judge-via-multi-agent-collaboration
- PHQ-8 vs PHQ-9 equivalency meta-analysis - https://stacks.cdc.gov/view/cdc/84248
- SQPsychConv paper - https://arxiv.org/abs/2510.25384

### Technical Documentation
- LangGraph checkpointing - https://langchain-ai.github.io/langgraph/concepts/persistence/
- PydanticAI - https://ai.pydantic.dev/
- OpenAI structured outputs - https://platform.openai.com/docs/guides/structured-outputs

### Parent Project
- ai-psychiatrist repo - https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist
- GitHub Issue #38 - Cross-dataset validation using SQPsychConv

---

*This specification synthesizes research from Gemini Deep Research and GPT Deep Research, taking the best aspects of each to produce a definitive, implementable architecture.*
