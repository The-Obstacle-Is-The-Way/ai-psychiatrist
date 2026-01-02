# Deep Research Prompt: Multi-Agent Depression Scoring System for SQPsychConv

**Purpose**: This prompt is designed for an external research agent to conduct comprehensive investigation and architectural design for a new repository that implements a multi-agent LLM system for scoring synthetic therapy conversations.

**Date**: 2026-01-02
**Parent Project**: [ai-psychiatrist](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist)

---

## 1. Executive Summary

We need to design and implement a **separate repository** containing a multi-agent LLM system that:

1. Takes SQPsychConv synthetic therapy transcripts as input
2. Assigns depression severity labels (PHQ-8, PHQ-9, or an alternative metric—to be determined)
3. Uses frontier LLM APIs (OpenAI, Anthropic, Google) configured via `.env`
4. Produces a scored dataset with embeddings for downstream use

### Why a Separate Repository?

| ai-psychiatrist (existing) | sqpsychconv-scored (new) |
|---------------------------|--------------------------|
| Depression prediction pipeline | Dataset annotation/creation |
| Uses local Ollama models | Uses frontier API models |
| Validated on DAIC-WOZ | Creates training data for DAIC-WOZ validation |
| Restricted by DAIC-WOZ licensing | Fully open, deployable |

---

## 2. Background Context

### 2.1 The DAIC-WOZ Licensing Problem

Our existing `ai-psychiatrist` system achieves strong PHQ-8 prediction performance but is **blocked from production deployment** because:

- DAIC-WOZ dataset has restrictive academic licensing
- Embeddings derived from DAIC-WOZ cannot be distributed
- Few-shot retrieval requires reference examples with ground truth scores
- Zero-shot works but few-shot (with RAG) provides better accuracy + interpretability

### 2.2 SQPsychConv Dataset

[SQPsychConv](https://arxiv.org/abs/2510.25384) is a synthetic therapy conversation dataset:

- **Cohort size**: 2,090 underlying client profiles (1,178 control / 912 MDD) conditioned on structured questionnaire data (Kircher et al., 2019; not included in the public release)
- **Generator variants**: 7 open-weight LLM variants (mistral, command, qwen2.5, llama3.3, nemotron, qwq, gemma) with separate releases (e.g., `AIMH/SQPsychConv_qwq`)
- **Public labels only**: `mdd` (major depressive disorder) vs `control` (no HAMD/BDI severity scores in the public dataset release)
- **Schema (HF qwq variant)**: `file_id`, `condition`, `client_model`, `therapist_model`, `dialogue`
- **Dialogue structure (HF qwq variant; measured from local export)**: 2,487–12,446 chars (mean ~5,953) and ~35 utterances per dialogue on average (≈18 therapist + ≈18 client)
- **Quality issue (HF qwq variant; measured)**: 4,019 CJK characters due to code-switching

### 2.3 The Opportunity

If we can **LLM-score SQPsychConv dialogues with severity labels**, we create:

1. A **deployable reference corpus** (no DAIC-WOZ restrictions; verify SQPsychConv license terms for redistribution)
2. **Embeddings for RAG** that enable few-shot prediction on new transcripts
3. **Benchmarking data** to validate against DAIC-WOZ ground truth
4. A **novel scored dataset** contribution to the research community

### 2.4 Circularity Avoidance

**Critical**: LLM-scoring SQPsychConv is NOT circular if:

- Ground truth validation comes from DAIC-WOZ (real clinical data)
- The scoring LLM is different from the generation LLM
- We use multiple frontier models and aggregate scores

---

## 3. Research Questions

Please investigate and provide recommendations for each:

### 3.1 Optimal Depression Metric (CRITICAL OPEN QUESTION)

**Question**: Should we score PHQ-8, PHQ-9, or a different subjective self-report metric?

#### Option A: PHQ-8 (Current Pipeline)

- 8 items, 0-3 scores per item (0-24 total)
- **Pro**: Aligns with DAIC-WOZ labels (direct comparison possible)
- **Pro**: Excludes suicidality item (ethical simplicity)
- **Con**: Less clinically common than PHQ-9

#### Option B: PHQ-9 (Most Common Clinical Screener)

- 9 items including "thoughts of self-harm or suicide"
- **Pro**: Most widely used depression screener globally
- **Pro**: Better clinical interpretability for downstream use
- **Con**: Item 9 (suicidality) is ethically complex for synthetic data
- **Con**: Misaligns with DAIC-WOZ validation target

#### Option C: Other Self-Report Instruments

Research whether these are more appropriate:

| Instrument | Items | Type | Notes |
|------------|-------|------|-------|
| **PHQ-2** | 2 | Self-report | Ultra-brief screening (items 1-2 of PHQ-9) |
| **GAD-7** | 7 | Self-report | Anxiety, often comorbid with depression |
| **BDI-II** | 21 | Self-report | Beck Depression Inventory, more comprehensive |
| **QIDS-SR16** | 16 | Self-report | Quick Inventory of Depressive Symptomatology |
| **CES-D** | 20 | Self-report | Center for Epidemiologic Studies Depression Scale |
| **K10/K6** | 10/6 | Self-report | Kessler Psychological Distress Scale |

#### Option D: Clinician-Rated Scales (for reference)

- **HAMD-17**: What SQPsychConv was originally conditioned on
- **MADRS**: Montgomery-Åsberg Depression Rating Scale
- **Note**: These require clinical expertise—can LLMs reliably simulate clinician ratings?

#### Option E: Binary/Categorical Severity Only

- minimal / mild / moderate / moderately severe / severe
- **Pro**: Simpler, higher inter-rater agreement expected
- **Con**: Less granular, less useful for RAG retrieval

#### Our Empirical Findings (January 2026): PHQ-8 vs PHQ-9 Feasibility

We analyzed the SQPsychConv dataset to determine whether PHQ-9 (with suicidality item) is feasible:

**Source Questionnaires Used in Generation**:

The SQPsychConv dialogues were generated from Kircher et al. (2019) data using:
- **BDI (Beck Depression Inventory)** - 21 items, including BDI9 (suicidality)
- **HAM-D (Hamilton Depression Rating Scale)**
- **HAM-A (Hamilton Anxiety Rating Scale)**

**Suicidality Content IS Present, But Sparse**:

We searched the qwq variant (2,090 dialogues per split) for suicidality-related content:

| Search Pattern | Train Matches | Test Matches |
|----------------|---------------|--------------|
| `suicid\|kill myself\|end my life\|want to die\|better off dead` | 6 | 6 |
| `vanish` (passive ideation) | 5 | 5 |

**Examples found**:
- Therapist: "I noticed you mentioned occasional guilt and fleeting thoughts about suicide"
- Client: "I keep replaying past failures, wondering if anyone'd notice if I just vanished"
- BDI9 item example: "Question BDI9: Answer: I don't think about doing anything to myself"

**Critical Issues with PHQ-9 Item 9**:

1. **Coverage too sparse**: Only ~0.3% of dialogues (6/2,090) contain explicit suicidality mentions
2. **Not systematically probed**: The LLM therapist didn't consistently ask about suicidal thoughts
3. **Inconsistent signal**: Some BDI9 answers indicate no ideation ("I don't think about doing anything to myself")
4. **Score extraction unreliable**: Can't map free-text to PHQ-9 Item 9's 0-3 scale without generating unreliable labels

**PHQ-8 vs PHQ-9 Decision Matrix**:

| Factor | PHQ-8 | PHQ-9 |
|--------|-------|-------|
| DAIC-WOZ alignment | ✓ Compatible | ✗ Different scale (0-24 vs 0-27) |
| Suicidality coverage | N/A | ✗ Only 0.3% explicit mentions |
| Score reliability | Higher | Lower (item 9 unreliable) |
| Clinical interpretability | Severity only | Severity + suicidality screen |
| Liability concerns | Lower | Higher (suicide detection is high-stakes) |
| Downstream benchmarking | ✓ Direct comparison possible | ✗ Requires score transformation |

**Our Preliminary Recommendation: PHQ-8**

Based on empirical analysis, PHQ-8 is more defensible because:
1. **DAIC-WOZ benchmarking requires PHQ-8** for direct comparison
2. **PHQ-9 Item 9 cannot be reliably scored** from dialogues where suicidality was rarely discussed
3. **Liability is lower** for severity-only scoring

**Alternative Hybrid Approach**:

If suicidality detection is desired, consider:
```
Primary: PHQ-8 total (0-24) → severity classification
Secondary: Suicidality flag (boolean) → from explicit mentions if present
```

This preserves DAIC-WOZ compatibility while still flagging high-risk content.

**Key Research Questions**:

1. Which self-report instrument is most reliably scored by LLMs from conversation text?
2. Does PHQ-9 item 9 (suicidality) present insurmountable ethical/safety concerns for synthetic data?
3. Is there literature on LLM-based scoring of specific depression instruments?
4. Should we score multiple instruments and let downstream users choose?
5. **NEW**: Is the hybrid PHQ-8 + suicidality flag approach clinically valid and useful?
6. **NEW**: Are there better ways to handle sparse suicidality signal in synthetic therapy data?

### 3.2 Multi-Agent Architecture

**Question**: What multi-agent framework should we use?

#### Our Preliminary Research (January 2026)

We conducted initial research on framework options. **This is NOT a final recommendation**—we want deeper investigation.

**Key Finding: Microsoft AutoGen → Agent Framework Transition**

- Microsoft [retired AutoGen in October 2025](https://venturebeat.com/ai/microsoft-retires-autogen-and-debuts-agent-framework-to-unify-and-govern)
- Merged with Semantic Kernel into unified **Microsoft Agent Framework**
- GA scheduled for Q1 2026
- If starting fresh, Agent Framework is the path forward (not AutoGen)

**Framework Options Identified**:

| Framework | Multi-Provider | Batch Suitability | Our Initial Assessment |
|-----------|---------------|-------------------|------------------------|
| Microsoft Agent Framework | Yes (OpenAI/Azure native; others via adapters) | Designed for orchestration, may be overkill | Investigate further |
| LangGraph | Yes | Good for DAG workflows | Viable alternative |
| PydanticAI | Yes (OpenAI, Anthropic, Gemini native) | Lightweight, structured outputs | **Promising for our use case** |
| CrewAI | Yes | Role-based, rapid prototyping | May be too simple |
| Custom asyncio | Yes | Full control | More code, but no framework lock-in |

**PydanticAI Emerged as Interesting** ([docs](https://ai.pydantic.dev/)):

- Native multi-model support (OpenAI, Anthropic, Gemini, others)
- Pydantic validation guarantees structured output schema compliance
- Lightweight—no heavy orchestration abstractions
- Built by Pydantic team (FastAPI-like ergonomics)

**Open Question**: Is PydanticAI sufficient, or do we need heavier orchestration (Agent Framework, LangGraph) for the consensus/judge pattern described below?

#### Option Summary

| Option | Description | Investigate? |
|--------|-------------|--------------|
| **A. Microsoft Agent Framework** | Enterprise-grade, production orchestration | Yes—but check if overkill |
| **B. LangGraph** | Graph-based workflows, LangChain ecosystem | Yes—for complex DAG needs |
| **C. PydanticAI** | Lightweight structured outputs, multi-model | Yes—**strong candidate** |
| **D. Custom asyncio + httpx** | Minimal deps, full control | Yes—as baseline comparison |
| **E. CrewAI** | Role-based agent collaboration | Maybe—may be too simple |

### 3.3 Multi-Model Consensus Strategy (CollabEval Pattern)

**Question**: How should we combine scores from multiple frontier models?

#### Our Preliminary Research: CollabEval Framework

Amazon's research on [multi-agent LLM-as-Judge](https://www.amazon.science/publications/enhancing-llm-as-a-judge-via-multi-agent-collaboration) shows that **collaborative consensus** outperforms both single-model judging AND simple averaging.

**Key Finding**: Multi-agent ensembles achieved 65.1% vs GPT-4's 57.5% alone on AlpacaEval benchmarks.

**Proposed 3-Phase Architecture** (to be validated by research):

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: INDEPENDENT SCORING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Dialogue ──┬──▶ GPT-5.2 ────▶ Score + Confidence + Evidence          │
│              │                                                          │
│              ├──▶ Claude Sonnet 4.5 ─▶ Score + Confidence + Evidence   │
│              │                                                          │
│              └──▶ Gemini 3 Flash ────▶ Score + Confidence + Evidence   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     PHASE 2: CONSENSUS CHECK                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   IF all 3 agree (within ±1 per item) ──▶ ACCEPT, return median        │
│                                                                         │
│   ELSE ──▶ Proceed to Phase 3                                          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                     PHASE 3: JUDGE ARBITRATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Present disagreements to a 4th model (different from scorers):       │
│                                                                         │
│   "GPT-5.2 scored PHQ8_Depressed=3 citing 'everything feels pointless' │
│    Claude scored PHQ8_Depressed=2 citing 'some days feel clearer'      │
│    Gemini scored PHQ8_Depressed=3 citing 'the smoke is still thick'    │
│                                                                         │
│    Which scoring is most justified by the full dialogue?"              │
│                                                                         │
│   Judge (e.g., Claude Opus 4) ──▶ Selects winner with justification    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Open Questions for Research**:

1. Is the 3-phase CollabEval pattern the right approach, or is there a better consensus methodology?
2. What should the disagreement threshold be (±1? exact match? per-item vs total score)?
3. Should the judge be a stronger model (Opus 4) or same-tier with different training (Gemini 3 Pro)?
4. How do we handle cases where the judge also disagrees with all scorers?

#### Validation Metrics for Consensus

From [LLM-as-Judge research](https://arxiv.org/html/2412.05579v2):

| Metric | Target | What It Measures |
|--------|--------|------------------|
| ICC(2,k) | ≥0.80 | Intraclass correlation across scorers |
| Krippendorff's α | ≥0.70 | Inter-rater reliability accounting for chance |
| Cohen's κ (vs human) | ≥0.60 | Agreement with clinical expert validation |

### 3.4 Preprocessing Requirements

**Question**: What preprocessing/filtering does SQPsychConv need?

Based on the DAIC-WOZ validity paper (attached), there are critical preprocessing concerns:

1. **Chinese character code-switching**: The qwq variant contains 4,019 CJK characters mid-sentence (measured in our local export)
   - Option: Filter affected dialogues OR use regex cleanup

2. **Therapist prompt bias**: The validity paper shows models can exploit therapist prompts as shortcuts
   - Bias concentrates in specific prompt regions (e.g., mental-health-history question blocks)
   - SQPsychConv is CBT-structured, so bias patterns may differ
   - Recommendation: Score participant utterances only? Or full dialogue?

3. **Dialogue length normalization**: Dialogues range from 2,487 to 12,446 chars
   - Consider chunking strategy for long dialogues

4. **MDD vs Control distribution**: 43.6% MDD, 56.4% control
   - May need stratified sampling for balanced scoring

5. **Model variant selection**: 7 LLM variants generated the data
   - Should we score all variants or select highest-quality (gemma/qwen2.5 per expert evaluation)?

---

## 4. Proposed Architecture

Please evaluate, refine, or redesign this proposed architecture:

### 4.1 High-Level Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     sqpsychconv-scored Repository                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  Data Loader    │────▶│  Preprocessor   │────▶│  Chunk Manager  │   │
│  │  (HuggingFace)  │     │  (cleanup/filter)│     │  (if needed)    │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│                                                          │              │
│                                                          ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Multi-Agent Scoring Engine (CollabEval)              │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │ GPT-5.2     │  │ Claude      │  │ Gemini 3    │               │   │
│  │  │ Scorer      │  │ Sonnet 4.5  │  │ Flash       │               │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │   │
│  │         │                │                │                       │   │
│  │         └────────────────┼────────────────┘                       │   │
│  │                          ▼                                        │   │
│  │              ┌─────────────────────┐                              │   │
│  │              │  Consensus Check    │                              │   │
│  │              │  (Phase 2)          │                              │   │
│  │              └──────────┬──────────┘                              │   │
│  │                         │                                         │   │
│  │            ┌────────────┴────────────┐                            │   │
│  │            ▼                         ▼                            │   │
│  │    ┌──────────────┐          ┌──────────────┐                     │   │
│  │    │ Agreement    │          │ Disagreement │                     │   │
│  │    │ → Accept     │          │ → Judge      │                     │   │
│  │    └──────────────┘          └──────┬───────┘                     │   │
│  │                                     │                             │   │
│  │                                     ▼                             │   │
│  │                          ┌─────────────────────┐                  │   │
│  │                          │  Judge Agent        │                  │   │
│  │                          │  (Claude Opus 4?)   │                  │   │
│  │                          └─────────────────────┘                  │   │
│  │                                                                   │   │
│  └───────────────────────────────┬───────────────────────────────────┘   │
│                                  ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Output Generation                             │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  • scored_sqpsychconv.csv (file_id, condition, PHQ8_* scores,   │    │
│  │    per-item evidence + confidence, agreement_rate)              │    │
│  │  • embeddings/<prefix>.npz + .json sidecar                      │    │
│  │  • validation_report.json (inter-model agreement, judge stats)  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Scoring Prompt Design

Each scorer agent should use a structured prompt that:

1. Provides item definitions with severity anchors
2. Instructs evidence-based scoring (quote supporting text)
3. Allows N/A for insufficient evidence
4. Requests confidence level per item

**Example prompt structure** (instrument TBD from research):

```text
You are a clinical psychologist scoring a therapy transcript for PHQ-8 depression symptoms.

For each of the 8 PHQ-8 items, provide:
1. Score (0-3): 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
2. Evidence: Direct quote from transcript supporting this score
3. Confidence: Low/Medium/High (0.0-1.0)
4. If insufficient evidence, respond with score=null and explain why

PHQ-8 Items:
1. Little interest or pleasure in doing things (Anhedonia)
2. Feeling down, depressed, or hopeless
3. Trouble falling/staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself—or that you are a failure
7. Trouble concentrating on things
8. Moving or speaking slowly / being fidgety or restless

Transcript:
{dialogue}

Respond in JSON format with Pydantic-compatible schema.
```

### 4.3 Checkpointing Strategy

Given ~2,090 dialogues × 3 models × potential retries:

- Checkpoint after each dialogue (not after each model)
- Store intermediate results in SQLite or JSON lines
- Support resume from last successful dialogue
- Rate limit aware (respect API quotas)

### 4.4 Configuration

```env
# .env file structure
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Model selection (January 2026 frontier models)
SCORER_MODELS=gpt-5.2,claude-sonnet-4-5,gemini-3-flash
JUDGE_MODEL=claude-opus-4

# Scoring parameters
DISAGREEMENT_THRESHOLD=1  # Max score difference before judge needed
CONFIDENCE_THRESHOLD=0.7  # Minimum confidence to include

# Rate limiting
OPENAI_RPM=60
ANTHROPIC_RPM=60
GOOGLE_RPM=60

# Checkpointing
CHECKPOINT_DIR=./checkpoints
RESUME_FROM_CHECKPOINT=true
```

---

## 5. Cost Estimation (January 2026 Pricing)

### 5.1 Token Estimates

| Component | Tokens |
|-----------|--------|
| Average dialogue | ~1,500 input tokens |
| Scoring prompt | ~500 input tokens |
| Structured output | ~400 output tokens |
| **Total per dialogue per model** | ~2,000 input / ~400 output |

### 5.2 Current Frontier Model Pricing

| Model | Input (per 1M) | Output (per 1M) | Source |
|-------|----------------|-----------------|--------|
| GPT-5.2 | $1.75 | $14.00 | [OpenAI](https://openai.com/api/pricing/) |
| Gemini 3 Flash | $0.50 | $3.00 | [Google AI](https://ai.google.dev/gemini-api/docs/pricing) |
| Claude Sonnet 4.5 | $3.00 | $15.00 | [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models) |
| Claude Opus 4 (judge) | ~$15.00 | ~$75.00 | Estimate |

### 5.3 Estimated Total Cost

| Scenario | Cost |
|----------|------|
| 3 scorers, 2,090 dialogues, 1 pass | ~$50-60 |
| + Judge on 20% of items (disagreements) | +$15-25 |
| + Batch API discounts (50%) | -50% |
| **Realistic total** | **$40-75** |

For multiple validation runs (2-3 passes): **$100-200 total**

---

## 6. Validation Strategy

### 6.1 Internal Consistency

- Inter-model agreement (Fleiss' kappa, ICC across 3 scorers)
- Intra-model consistency (score same dialogue twice, measure drift)
- Score distribution analysis (should roughly match MDD/control split)

### 6.2 External Validation (DAIC-WOZ Benchmarking)

After creating scored SQPsychConv:

1. Generate embeddings for scored dialogues
2. Use as retrieval corpus in ai-psychiatrist few-shot pipeline
3. Evaluate PHQ-8 prediction on DAIC-WOZ test set
4. Compare to DAIC-WOZ-only baseline

**Success criteria**:

- AURC/AUGRC within 20% of DAIC-WOZ-only baseline
- Coverage (Cmax) should be similar to baseline
- If synthetic corpus performs comparably, it validates the approach

### 6.3 Human Validation Subset

- Randomly sample 50-100 scored dialogues
- Have clinical expert rate on same scale
- Compute inter-rater reliability (Cohen's kappa) between LLM ensemble and human

---

## 7. Deliverables

The new repository should produce:

### 7.1 Code Artifacts

- [ ] Python package with CLI for scoring pipeline
- [ ] Multi-agent orchestration (framework TBD from research)
- [ ] Preprocessing utilities for SQPsychConv cleanup
- [ ] Embedding generation script (compatible with ai-psychiatrist)
- [ ] Evaluation scripts for internal consistency

### 7.2 Data Artifacts

- [ ] `scored_sqpsychconv.csv`: Full scored dataset (use ai-psychiatrist-compatible column names if PHQ-8: `PHQ8_NoInterest`, `PHQ8_Depressed`, `PHQ8_Sleep`, `PHQ8_Tired`, `PHQ8_Appetite`, `PHQ8_Failure`, `PHQ8_Concentrating`, `PHQ8_Moving`)
- [ ] Reference embeddings artifact compatible with ai-psychiatrist (NPZ + JSON sidecar; see `scripts/generate_embeddings.py`)
- [ ] `scoring_metadata.json`: Model versions, timestamps, parameters
- [ ] `validation_report.json`: Agreement statistics, confidence distributions

### 7.3 Documentation

- [ ] README with quickstart
- [ ] Architecture decision records (ADRs)
- [ ] Prompt templates and versioning
- [ ] Reproducibility guide

---

## 8. Research Tasks

Please investigate and provide findings on:

### 8.1 Framework Comparison (Priority: HIGH)

1. Compare: Microsoft Agent Framework vs LangGraph vs PydanticAI vs custom asyncio
2. Evaluate based on:
   - Multi-provider LLM support (OpenAI, Anthropic, Google)
   - Batch processing suitability
   - Checkpointing/resume capabilities
   - Rate limiting handling
   - Structured output validation
   - Observability/debugging
3. **Specific question**: Is PydanticAI sufficient for the CollabEval consensus pattern, or do we need heavier orchestration?
4. Provide code examples for the recommended approach

### 8.2 Scoring Metric Decision (Priority: HIGH)

1. **Deep dive**: PHQ-8 vs PHQ-9 vs other self-report instruments (BDI-II, QIDS-SR16, CES-D, K10)
2. Research: Which instruments have been successfully scored by LLMs from conversation text?
3. Ethical analysis: Is PHQ-9 item 9 (suicidality) appropriate for synthetic data scoring?
4. Clinical validity: Which instrument is most clinically interpretable for downstream users?
5. **Recommendation**: Should we score multiple instruments, or pick one?

### 8.3 LLM-as-Judge Best Practices (Priority: HIGH)

1. Validate or refine the CollabEval 3-phase pattern
2. Research: What is the optimal disagreement threshold?
3. Research: Should the judge be a stronger model or a different model family?
4. Research: How to handle cascading disagreements (judge disagrees with all scorers)?
5. Review [Awesome LLM-as-Judge](https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge) for latest patterns

### 8.4 Prompt Engineering (Priority: MEDIUM)

1. Research best practices for clinical scoring prompts
2. Investigate chain-of-thought vs direct scoring
3. Test structured output formats (JSON with Pydantic validation)
4. Evaluate calibration techniques for confidence estimation

### 8.5 Preprocessing Pipeline (Priority: MEDIUM)

1. Develop regex patterns for Chinese character cleanup
2. Determine participant-only vs full dialogue scoring
3. Design chunking strategy for long dialogues
4. Recommend model variant filtering (if any)

### 8.6 Validation Protocol (Priority: HIGH)

1. Design inter-model agreement metrics (ICC, Krippendorff's α)
2. Develop DAIC-WOZ benchmarking protocol
3. Create human validation sampling strategy
4. Define success/failure criteria

---

## 9. Attached Documents

When presenting this prompt to the external research agent, attach:

### Required Attachments

1. **SQPsychConv Paper** (`SQPsychConv.md` or PDF)
   - Location: `_literature/markdown/SQPsychConv/SQPsychConv.md`
   - Purpose: Full methodology and dataset details

2. **DAIC-WOZ Validity Paper** (`daic-woz-prompts.md` or PDF)
   - Location: `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md`
   - Purpose: Critical preprocessing concerns about therapist prompt bias

3. **Cross-Validation Spec** (`spec-sqpsychconv-cross-validation.md`)
   - Location: `docs/research/spec-sqpsychconv-cross-validation.md`
   - Purpose: Detailed feasibility analysis and Path G recommendation

4. **Microsoft Agent Framework Notes** (`microsoft-agent-framework-integration.md`)
   - Location: `docs/_brainstorming/microsoft-agent-framework-integration.md`
   - Purpose: Initial framework evaluation (note: AutoGen is now deprecated)

5. **Sample Data** (train_sample.csv, test_sample.csv)
   - Location: `data/sqpsychconv/`
   - Purpose: Local HF `AIMH/SQPsychConv_qwq` exports (2,090 rows per split) for schema + reproducibility

### Optional Context

6. **CLAUDE.md** (project conventions)
7. **GitHub Issue #38** (original research proposal)
8. **ai-psychiatrist Compatibility Contract** (recommended)
   - `scripts/generate_embeddings.py` (embedding artifact format)
   - `src/ai_psychiatrist/services/reference_store.py` (PHQ-8 column names + reference store expectations)

---

## 10. Success Criteria

The research is successful if it produces:

1. **Clear framework recommendation** with justification and code examples
2. **Optimal metric decision** (PHQ-8/PHQ-9/other) with clinical rationale
3. **Validated consensus pattern** (CollabEval or alternative) with implementation guidance
4. **Preprocessing pipeline design** addressing all quality issues
5. **Validation protocol** that can definitively answer: "Does synthetic→real transfer work?"
6. **Architecture blueprint** that can be implemented in <2 weeks

---

## 11. Timeline and Scope

This research should focus on **design and architecture**, not implementation. The goal is to produce a comprehensive blueprint that enables rapid implementation.

**Scope boundaries**:

- IN: Framework selection, metric decision, consensus pattern design, validation protocol
- OUT: Full implementation, running actual scoring, generating final dataset

**Expected output format**: Technical specification document with:

- Decision matrices for each research question
- Code snippets for recommended approaches
- Pseudocode for critical algorithms
- Risk analysis and mitigation strategies

---

## 12. Additional Context

### 12.1 Why This Matters

If this works, we solve the **production deployment problem** for clinical NLP:

- Current: Restricted datasets → restricted models → no production use
- Future: Synthetic data → open models → deployable clinical AI

### 12.2 Related Work

- SQPsychConv paper shows synthetic data can train effective counseling models
- DAIC-WOZ validity paper shows preprocessing is critical
- Our ai-psychiatrist shows PHQ-8 prediction is achievable with proper methodology
- Amazon CollabEval shows multi-agent consensus outperforms single-model judging

### 12.3 Ethical Considerations

- Synthetic data scoring must not be used for actual clinical decisions
- The goal is research validation, not patient care
- Any suicide-related scoring (PHQ-9 item 9) requires extreme care
- Output should include uncertainty quantification, not just point estimates

### 12.4 Key References from Our Preliminary Research

- [Microsoft Agent Framework Overview](https://learn.microsoft.com/en-us/agent-framework/overview/agent-framework-overview)
- [VentureBeat: Microsoft retires AutoGen](https://venturebeat.com/ai/microsoft-retires-autogen-and-debuts-agent-framework-to-unify-and-govern)
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Amazon: Enhancing LLM-as-Judge via Multi-Agent Collaboration](https://www.amazon.science/publications/enhancing-llm-as-a-judge-via-multi-agent-collaboration)
- [Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594)
- [Awesome LLM-as-Judge Resources](https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge)

---

*This prompt was generated from the ai-psychiatrist repository context. For questions or clarifications, refer to the attached documents or the GitHub issue.*
