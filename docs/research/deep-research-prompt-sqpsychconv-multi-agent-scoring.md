# Deep Research Prompt: Multi-Agent Depression Scoring System for SQPsychConv

**Purpose**: This prompt is designed for an external research agent to conduct comprehensive investigation and architectural design for a new repository that implements a multi-agent LLM system for scoring synthetic therapy conversations.

**Date**: 2026-01-02
**Parent Project**: [ai-psychiatrist](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist)

---

## 1. Executive Summary

We need to design and implement a **separate repository** containing a multi-agent LLM system that:

1. Takes SQPsychConv synthetic therapy transcripts as input
2. Assigns depression severity labels (PHQ-8, PHQ-9, or an alternative metric—to be determined)
3. Uses frontier LLM APIs (OpenAI, Anthropic, Gemini) configured via `.env`
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

- **2,090 patients** across 7 LLM variants (qwq, gemma, llama3.3, mistral, command, nemotron, qwen2.5)
- **Binary labels only**: `mdd` (major depressive disorder) vs `control`
- **NO severity scores** in the public release (HAMD/BDI scores were used for generation but stripped due to FOR2107 data governance)
- **Rich symptom content**: PHQ-8-relevant keywords appear in 38-100% of MDD dialogues
- **CBT-structured**: ~15-24 turns, dual-agent (therapist + client) generation
- **Quality issues**: ~4,000 Chinese characters from code-switching in qwq variant

### 2.3 The Opportunity

If we can **LLM-score SQPsychConv dialogues with PHQ-8 labels**, we create:

1. A **deployable reference corpus** (no licensing restrictions)
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

### 3.1 Optimal Depression Metric

**Question**: Should we score PHQ-8, PHQ-9, or a different metric?

Consider:
- PHQ-8 (our current pipeline uses this, 8 items, 0-3 scores)
- PHQ-9 (adds suicidality item—ethically complex for synthetic data)
- HAMD-17 (clinician-rated, what SQPsychConv was conditioned on)
- BDI-II (21 items, self-report)
- Binary severity (mild/moderate/severe/minimal)
- Custom composite score

**Factors to weigh**:
- Alignment with DAIC-WOZ labels (PHQ-8)
- Clinical interpretability
- Ethical considerations (suicidality scoring)
- Inter-rater reliability of LLM scorers
- Downstream use cases (screening vs diagnosis support)

### 3.2 Multi-Agent Architecture

**Question**: What multi-agent framework should we use?

#### Option A: Microsoft Agent Framework (AutoGen)

**Pros**:
- Production-grade orchestration (GroupChat, Sequential, Handoff patterns)
- Built-in checkpointing for long-running scoring jobs
- OpenTelemetry observability
- Native support for multiple LLM backends

**Cons**:
- Framework is in preview (`--pre`)
- May be over-engineered for a scoring pipeline
- Learning curve for a new repo

**Reference**: See attached `microsoft-agent-framework-integration.md`

#### Option B: LangGraph / LangChain

**Pros**:
- Widely adopted, extensive documentation
- Graph-based workflow definition
- Good for multi-step reasoning chains

**Cons**:
- Heavier dependency footprint
- Abstractions may fight our domain-specific needs

#### Option C: Lightweight Custom Implementation

**Pros**:
- Minimal dependencies
- Full control over orchestration logic
- Easier to debug and modify

**Cons**:
- Must implement retry logic, rate limiting, checkpointing ourselves
- Less portable to other use cases

#### Option D: Claude Agent SDK

**Pros**:
- Native Anthropic integration
- Designed for agentic workflows

**Cons**:
- Anthropic-specific (we want multi-provider)

**Recommendation Request**: Given that this is a **batch scoring job** (not interactive), which framework provides the best balance of robustness, simplicity, and multi-provider support?

### 3.3 Multi-Model Aggregation Strategy

**Question**: How should we combine scores from multiple frontier models?

Consider:
- Simple averaging (mean of GPT-4o, Claude, Gemini scores)
- Weighted averaging (based on model-specific calibration)
- Majority voting for categorical severity
- Uncertainty quantification (std dev across models as confidence)
- Ensemble with learned weights

**Key insight from ai-psychiatrist**: We use AURC/AUGRC metrics that account for selective prediction (abstention when uncertain). The scoring system should produce confidence estimates, not just point predictions.

### 3.4 Preprocessing Requirements

**Question**: What preprocessing/filtering does SQPsychConv need?

Based on the [DAIC-WOZ validity paper](attached), there are critical preprocessing concerns:

1. **Chinese character code-switching**: The qwq variant contains ~4,000 CJK characters mid-sentence
   - Option: Filter affected dialogues OR use regex cleanup

2. **Therapist prompt bias**: The validity paper shows models can exploit therapist prompts as shortcuts
   - The second half of DAIC-WOZ interviews contains biased mental health questions
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

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     sqpsychconv-scored Repository                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │  Data Loader    │────▶│  Preprocessor   │────▶│  Chunk Manager  │   │
│  │  (HuggingFace)  │     │  (cleanup/filter)│     │  (sliding window)│   │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
│                                                          │              │
│                                                          ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Multi-Agent Scoring Engine                     │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │   │
│  │  │ GPT-4o      │  │ Claude      │  │ Gemini 2.0  │               │   │
│  │  │ Scorer      │  │ Scorer      │  │ Scorer      │               │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │   │
│  │         │                │                │                       │   │
│  │         └────────────────┼────────────────┘                       │   │
│  │                          ▼                                        │   │
│  │              ┌─────────────────────┐                              │   │
│  │              │  Score Aggregator   │                              │   │
│  │              │  (ensemble + conf)  │                              │   │
│  │              └─────────────────────┘                              │   │
│  │                          │                                        │   │
│  └──────────────────────────┼────────────────────────────────────────┘   │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Output Generation                             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  • scored_sqpsychconv.csv (file_id, condition, phq8_*, conf)    │   │
│  │  • embeddings/*.npz (for RAG retrieval)                         │   │
│  │  • validation_report.json (inter-model agreement stats)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Scoring Prompt Design

Each scorer agent should use a structured prompt that:

1. Provides PHQ-8 item definitions with severity anchors
2. Instructs evidence-based scoring (quote supporting text)
3. Allows N/A for insufficient evidence
4. Requests confidence level per item

**Example prompt structure** (to be refined):

```
You are a clinical psychologist scoring a therapy transcript for PHQ-8 depression symptoms.

For each of the 8 PHQ-8 items, provide:
1. Score (0-3): 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
2. Evidence: Direct quote from transcript supporting this score
3. Confidence: Low/Medium/High
4. If insufficient evidence, respond "N/A" with explanation

PHQ-8 Items:
1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling/staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself
7. Trouble concentrating
8. Moving/speaking slowly or being fidgety/restless

Transcript:
{dialogue}

Respond in JSON format:
{
  "items": [
    {"item": 1, "score": 2, "evidence": "...", "confidence": "high"},
    ...
  ],
  "total_score": 14,
  "severity": "moderate"
}
```

### 4.2 Checkpointing Strategy

Given ~2,090 dialogues × 3 models × potential retries:

- Checkpoint after each dialogue (not after each model)
- Store intermediate results in SQLite or JSON lines
- Support resume from last successful dialogue
- Rate limit aware (respect API quotas)

### 4.3 Configuration

```env
# .env file structure
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# Model selection
SCORER_MODELS=gpt-4o,claude-sonnet-4-20250514,gemini-2.0-flash

# Scoring parameters
MIN_SCORER_AGREEMENT=2  # Minimum models that must agree
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

## 5. Validation Strategy

### 5.1 Internal Consistency

- Inter-model agreement (Fleiss' kappa across 3 scorers)
- Intra-model consistency (score same dialogue twice, measure drift)
- Score distribution analysis (should roughly match MDD/control split)

### 5.2 External Validation (DAIC-WOZ Benchmarking)

After creating scored SQPsychConv:

1. Generate embeddings for scored dialogues
2. Use as retrieval corpus in ai-psychiatrist few-shot pipeline
3. Evaluate PHQ-8 prediction on DAIC-WOZ test set
4. Compare to DAIC-WOZ-only baseline

**Success criteria**:
- AURC/AUGRC within 20% of DAIC-WOZ-only baseline
- Coverage (Cmax) should be similar to baseline
- If synthetic corpus performs comparably, it validates the approach

### 5.3 Human Validation Subset

- Randomly sample 50-100 scored dialogues
- Have clinical expert rate on same PHQ-8 scale
- Compute inter-rater reliability (Cohen's kappa) between LLM ensemble and human

---

## 6. Deliverables

The new repository should produce:

### 6.1 Code Artifacts

- [ ] Python package with CLI for scoring pipeline
- [ ] Multi-agent orchestration (framework TBD from research)
- [ ] Preprocessing utilities for SQPsychConv cleanup
- [ ] Embedding generation script (compatible with ai-psychiatrist)
- [ ] Evaluation scripts for internal consistency

### 6.2 Data Artifacts

- [ ] `scored_sqpsychconv.csv`: Full scored dataset
- [ ] `scored_sqpsychconv_embeddings.npz`: Vector embeddings
- [ ] `scoring_metadata.json`: Model versions, timestamps, parameters
- [ ] `validation_report.json`: Agreement statistics, confidence distributions

### 6.3 Documentation

- [ ] README with quickstart
- [ ] Architecture decision records (ADRs)
- [ ] Prompt templates and versioning
- [ ] Reproducibility guide

---

## 7. Research Tasks

Please investigate and provide findings on:

### 7.1 Framework Comparison (Priority: HIGH)

1. Compare Microsoft Agent Framework vs LangGraph vs custom implementation
2. Evaluate based on:
   - Multi-provider LLM support (OpenAI, Anthropic, Google)
   - Batch processing suitability
   - Checkpointing/resume capabilities
   - Rate limiting handling
   - Observability/debugging
3. Provide code examples for the recommended approach

### 7.2 Scoring Metric Decision (Priority: HIGH)

1. Research PHQ-8 vs PHQ-9 vs alternatives
2. Consider ethical implications of suicidality scoring
3. Evaluate alignment with downstream use cases
4. Recommend optimal metric with justification

### 7.3 Prompt Engineering (Priority: MEDIUM)

1. Research best practices for clinical scoring prompts
2. Investigate chain-of-thought vs direct scoring
3. Test structured output formats (JSON, XML, etc.)
4. Evaluate calibration techniques for confidence estimation

### 7.4 Preprocessing Pipeline (Priority: MEDIUM)

1. Develop regex patterns for Chinese character cleanup
2. Determine participant-only vs full dialogue scoring
3. Design chunking strategy for long dialogues
4. Recommend model variant filtering (if any)

### 7.5 Validation Protocol (Priority: HIGH)

1. Design inter-model agreement metrics
2. Develop DAIC-WOZ benchmarking protocol
3. Create human validation sampling strategy
4. Define success/failure criteria

---

## 8. Attached Documents

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
   - Purpose: Initial framework evaluation

5. **Sample Data** (train_sample.csv, test_sample.csv)
   - Location: `data/sqpsychconv/`
   - Purpose: Example dialogue structure and format

### Optional Context

6. **CLAUDE.md** (project conventions)
7. **GitHub Issue #38** (original research proposal)

---

## 9. Success Criteria

The research is successful if it produces:

1. **Clear framework recommendation** with justification and code examples
2. **Optimal metric decision** (PHQ-8/PHQ-9/other) with clinical rationale
3. **Preprocessing pipeline design** addressing all quality issues
4. **Validation protocol** that can definitively answer: "Does synthetic→real transfer work?"
5. **Architecture blueprint** that can be implemented in <2 weeks

---

## 10. Timeline and Scope

This research should focus on **design and architecture**, not implementation. The goal is to produce a comprehensive blueprint that enables rapid implementation.

**Scope boundaries**:
- IN: Framework selection, metric decision, architecture design, validation protocol
- OUT: Full implementation, running actual scoring, generating final dataset

**Expected output format**: Technical specification document with:
- Decision matrices for each research question
- Code snippets for recommended approaches
- Pseudocode for critical algorithms
- Risk analysis and mitigation strategies

---

## 11. Additional Context

### 11.1 Why This Matters

If this works, we solve the **production deployment problem** for clinical NLP:

- Current: Restricted datasets → restricted models → no production use
- Future: Synthetic data → open models → deployable clinical AI

### 11.2 Related Work

- SQPsychConv paper shows synthetic data can train effective counseling models
- DAIC-WOZ validity paper shows preprocessing is critical
- Our ai-psychiatrist shows PHQ-8 prediction is achievable with proper methodology

### 11.3 Ethical Considerations

- Synthetic data scoring must not be used for actual clinical decisions
- The goal is research validation, not patient care
- Any suicide-related scoring (PHQ-9 item 9) requires extreme care
- Output should include uncertainty quantification, not just point estimates

---

*This prompt was generated from the ai-psychiatrist repository context. For questions or clarifications, refer to the attached documents or the GitHub issue.*
