# SPEC: SQPsychConv Cross-Dataset Validation

**Status**: Research Specification
**Related Issue**: [#38 - Cross-dataset validation using SQPsychConv](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/38)
**Created**: 2026-01-01

---

## Executive Summary

This spec analyzes the feasibility of using the SQPsychConv synthetic therapy dataset for cross-dataset validation with our DAIC-WOZ-based PHQ-8 prediction pipeline. After deep analysis, we identify **significant challenges** but also **viable pathways forward**.

---

## 1. Dataset Analysis

### 1.1 SQPsychConv Reality vs. Expectations

| Aspect | Issue #38 Assumed | Actual Dataset |
|--------|-------------------|----------------|
| Labels | HAMD/BDI severity scores | **Binary**: `mdd` vs `control` only |
| Severity Levels | Continuous/ordinal | **None** (classification only) |
| Scale Type | Clinician-rated (HAMD) | No scale scores in data |

**Critical Finding**: The SQPsychConv dataset (e.g., `AIMH/SQPsychConv_qwq`) contains:
- `file_id`: Unique identifier (e.g., "active436", "control1328")
- `condition`: Binary label (`mdd` or `control`)
- `client_model`: Generator model identifier for the client role (e.g., `qwq_qwen`)
- `therapist_model`: Generator model identifier for the therapist role (e.g., `qwq_qwen`)
- `dialogue`: Full therapy transcript (mean ~5,953 chars; ~35 utterances on average)

There are **no HAMD, BDI, or severity scores** in the downloadable dataset.

### 1.2 Dataset Statistics (Verified from Actual Data)

```
Train split: 2,090 conversations
  - control: 1,178 (56.4%)
  - mdd: 912 (43.6%)

Test split: 2,090 conversations
  - control: 1,178 (56.4%)
  - mdd: 912 (43.6%)

Dialogue structure:
  - Length: 2,487–12,446 chars (mean ~5,953)
  - Utterances per dialogue: ~35 on average (≈18 therapist + ≈18 client)
  - Format: "Therapist: ..." / "Client: ..." alternating
  - End markers: [/END] tags
```

**Important note (HF `SQPsychConv_qwq`)**: In our local cache, `data/sqpsychconv/train_sample.csv` and `data/sqpsychconv/test_sample.csv` are byte-identical (same `file_id` set and same dialogues). Treat them as a single corpus unless you intentionally re-split.

### 1.3 Data Quality Issues

**Chinese Character Code-Switching**: 4,019 CJK characters found in dialogues (qwq variant; measured).
The qwq_qwen model occasionally code-switches mid-sentence:

```
"Overthink every word,怕说错话。End up silent, then恨自己不够好。"
Translation: "...afraid of saying wrong things...hate myself for not being good enough"
```

This is a quality issue but not a blocker for PHQ-8 scoring.

### 1.4 PHQ-8 Symptom Coverage Analysis

Keyword analysis of 50 MDD dialogues shows strong symptom coverage:

| PHQ-8 Item | Keyword Coverage | Example Phrases |
|------------|------------------|-----------------|
| Item 1: Anhedonia | 38% | "pointless", "no interest", "can't enjoy" |
| Item 2: Depressed mood | 94% | "sad", "hopeless", "empty", "hollow" |
| Item 3: Sleep | 62% | "insomnia", "awake", "tossing", "exhausted" |
| Item 4: Fatigue | 54% | "tired", "drained", "energy" |
| Item 5: Appetite | 98% | "appetite", "eat", "food" (often mentioned) |
| Item 6: Guilt | 90% | "guilt", "failure", "worthless", "burden" |
| Item 7: Concentration | 100% | "concentrate", "focus", "distracted" |
| Item 8: Psychomotor | 48% | "slow", "restless", "agitated" |

**Key Finding**: The dialogues contain rich PHQ-8-relevant symptom content, making LLM-based scoring feasible.

### 1.5 Dialogue Structure

SQPsychConv dialogues follow a CBT-structured format:
```
Therapist: Good morning! I notice you've described feeling deeply sad...
Client: Lately, everything feels pointless. Even small tasks tire me out...
Therapist: It sounds overwhelming to feel everything lacks purpose...
Client: I guess I just... fear they'll suffer without me pushing...
[~30+ more turns]
Therapist: Thank you for sharing so openly. Let's recap today...
```

Compare to DAIC-WOZ (our current pipeline):
```
Ellie: where are you from originally
Participant: im from los angeles
Ellie: how are you doing today
Participant: im doing okay i guess <laughter> um a bit tired
```

**Key Differences**:
- SQPsychConv: CBT-structured, deep therapeutic dialogue
- DAIC-WOZ: Semi-structured interview, more natural/hesitant speech

---

## 2. Scale Mapping Research

### 2.1 HAMD ↔ PHQ-9 Correlation

From [PMC8599822](https://pmc.ncbi.nlm.nih.gov/articles/PMC8599822/):
- **Pearson correlation**: r = 0.61–0.72 (moderate-strong)
- **ICC**: 0.594 (moderate consistency)
- **Kappa for severity**: 0.248 (fair agreement only)

**Severity Cutoffs**:
| Scale | Mild | Moderate | Severe |
|-------|------|----------|--------|
| HAMD-17 | 8–16 | 17–23 | ≥24 |
| PHQ-9 | 5–9 | 10–14 | ≥15 |

### 2.2 BDI ↔ PHQ-9 Correlation

From [PMC5515387](https://pmc.ncbi.nlm.nih.gov/articles/PMC5515387/):
- **Pearson correlation**: r = 0.77–0.88 (strong)
- Thresholds for moderate depression are similar between BDI-II and PHQ-9

**Mapped Cutoffs** (approximate):
| PHQ-9 Score | BDI-II Equivalent |
|-------------|-------------------|
| 6–8 | Mild |
| 9–14 | Moderate |
| ≥15 | Severe |

### 2.3 Critical Limitation

**No validated item-to-item mapping exists.** The scales measure related but not identical constructs:
- PHQ-8: 8 items (anhedonia, depressed mood, sleep, fatigue, appetite, guilt, concentration, psychomotor)
- HAMD-17: 17 items (includes anxiety, somatic symptoms, insight)
- BDI-II: 21 items (includes punishment feelings, suicidal ideation)

**Implication**: Even if SQPsychConv had HAMD/BDI scores, we cannot directly map to PHQ-8 item scores (0-3).

---

## 3. Architecture Analysis

### 3.1 Current Pipeline (DAIC-WOZ)

```
┌─────────────────────────────────────────────────────────────────┐
│ Data Layer                                                       │
├─────────────────────────────────────────────────────────────────┤
│ data/transcripts/{pid}_P/{pid}_TRANSCRIPT.csv                   │
│ data/paper_splits/paper_split_train.csv (PHQ8_* columns)        │
│ data/embeddings/*.npz + .json + .tags.json + .chunk_scores.json │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Embedding Generation (scripts/generate_embeddings.py)           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load transcript via TranscriptService                        │
│ 2. Create sliding chunks (chunk_size=8, step=2)                 │
│ 3. Generate embeddings via EmbeddingClient                      │
│ 4. Write .npz + .json + .meta.json + .tags.json                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Reference Store (services/reference_store.py)                   │
├─────────────────────────────────────────────────────────────────┤
│ • Loads embeddings, texts, tags, chunk_scores                   │
│ • Provides get_score(participant_id, PHQ8Item) → 0-3 or None    │
│ • PHQ8_COLUMN_MAP: PHQ8Item → CSV column name                   │
│ • Ground truth loaded from train/dev CSVs                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Embedding Service (services/embedding.py)                       │
├─────────────────────────────────────────────────────────────────┤
│ • build_reference_bundle(evidence_dict) → ReferenceBundle       │
│ • Cosine similarity search against reference store              │
│ • Returns top-k chunks with reference_score (0-3)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Quantitative Agent (agents/quantitative.py)                     │
├─────────────────────────────────────────────────────────────────┤
│ • Extracts evidence per PHQ-8 item                              │
│ • Builds reference bundle via EmbeddingService                  │
│ • Formats prompt: "<Reference Examples>\n(PHQ8_Sleep Score: 2)" │
│ • LLM predicts 0-3 score per item                               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Coupling Points

The pipeline is tightly coupled to PHQ-8 in these locations:

| Component | PHQ-8 Coupling |
|-----------|----------------|
| `domain/enums.py` | `PHQ8Item` enum (8 items) |
| `reference_store.py` | `PHQ8_COLUMN_MAP`, `get_score()` returns 0-3 |
| `embedding.py` | `ReferenceBundle.format_for_prompt()` uses "PHQ8_*" keys |
| `quantitative.py` | `PHQ8_KEY_MAP`, expects 0-3 scores |
| `agents/prompts/quantitative.py` | Prompts reference PHQ-8 specifically |
| Ground truth CSVs | Columns named `PHQ8_NoInterest`, `PHQ8_Depressed`, etc. |

---

## 4. Feasibility Assessment

### 4.1 Approach Matrix

| Approach | Requires | Feasibility | Engineering Effort |
|----------|----------|-------------|-------------------|
| **A. Scale Mapping** | HAMD/BDI item scores | ❌ Not available | N/A |
| **B. Adapt Pipeline** | Rewrite for HAMD/BDI | ❌ No scores exist | N/A |
| **C. Qualitative-Only** | Binary labels only | ✅ Works | Low |
| **D. Total Score Binary** | Binary labels only | ✅ Works | Low |
| **E. LLM-Derived Scores** | LLM annotation | ⚠️ Circular if evaluated on SQPsychConv | Medium |
| **F. Severity-Conditioned** | Regenerate with severity | ⚠️ Requires authors | High |
| **G. LLM-Scored + DAIC-WOZ Eval** | Frontier LLM + DAIC-WOZ | ✅ **Recommended** | Medium-High |

### 4.2 Viable Paths

#### Path C: Qualitative Evidence Retrieval (Low Effort)

Use SQPsychConv dialogues as a **retrieval corpus only**, without using labels:

```python
# Conceptual: Use SQPsychConv chunks for semantic similarity
# but ignore condition labels during retrieval

class SQPsychConvReferenceStore:
    """Store that provides chunks without scores."""

    def get_score(self, participant_id, item) -> None:
        return None  # No ground truth available
```

**Pros**:
- Tests if synthetic conversations contain similar linguistic patterns
- Measures retrieval quality without label dependency

**Cons**:
- Cannot use for few-shot scoring (no reference scores)
- Only tests embedding quality, not prediction

#### Path D: Binary Classification (Low Effort)

Convert our pipeline to predict depressed/not-depressed:

```python
# Map PHQ-8 total to binary:
# PHQ-8 total ≥ 10 → depressed (matches clinical cutoff)

def is_depressed(phq8_total: int) -> bool:
    return phq8_total >= 10

# SQPsychConv labels:
# mdd → depressed=True
# control → depressed=False
```

**Pros**:
- Simple, no scale mapping needed
- Can compute AUC, sensitivity, specificity

**Cons**:
- Loses item-level granularity
- Binary classification is much easier than item scoring

#### Path E: LLM-Derived Scores (Circular Risk)

Have an LLM score SQPsychConv dialogues on PHQ-8 scale:

```python
# 1. For each SQPsychConv dialogue
# 2. Run our QuantitativeAssessmentAgent in zero-shot mode
# 3. Store predicted scores as "ground truth"
# 4. Use these for few-shot retrieval

# DANGER: Same model scoring what it retrieves = circular
```

**Pros**:
- Creates item-level "scores" for few-shot prompting

**Cons**:
- **Circular reasoning**: LLM's biases get amplified
- Scores are predictions, not ground truth
- Inflates apparent accuracy

#### Path F: Author Collaboration (Best, Highest Effort)

Request severity-annotated dialogues from SQPsychConv authors:

```
Contact: AIMH (AI for Mental Health) team
Request: HAMD/BDI scores used for dialogue generation
Offer: Collaboration on cross-dataset validation
```

**Pros**:
- True severity labels enable proper few-shot prompting
- Scientific collaboration value

**Cons**:
- Depends on author response
- May take weeks/months (and likely blocked by FOR2107 data governance)

#### Path G: LLM-Derived PHQ-8 with DAIC-WOZ Validation (Recommended)

**Key Insight**: Circularity is avoided if ground truth comes from a separate dataset.

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Score SQPsychConv with Frontier LLM                     │
├─────────────────────────────────────────────────────────────────┤
│ SQPsychConv dialogue                                            │
│     → GPT-4/Claude (multi-run, averaged)                        │
│     → pseudo-PHQ-8 scores (0-3 per item)                        │
│     → SQPsychConv-Scored dataset                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Use as Retrieval Corpus                                 │
├─────────────────────────────────────────────────────────────────┤
│ Generate embeddings for SQPsychConv-Scored                      │
│ Use pseudo-PHQ-8 scores for severity-matched retrieval          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Evaluate on DAIC-WOZ (Real Ground Truth)                │
├─────────────────────────────────────────────────────────────────┤
│ DAIC-WOZ test case → Retrieve from SQPsychConv-Scored           │
│     → Few-shot predict PHQ-8                                    │
│     → Compare to DAIC-WOZ real PHQ-8 labels                     │
│     → Report MAE, correlation, item-level metrics               │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Is NOT Circular**:
- Training corpus: SQPsychConv (LLM-scored)
- Evaluation corpus: DAIC-WOZ (real clinical scores)
- Ground truth comes from DAIC-WOZ, not from the LLM that scored SQPsychConv

**Pros**:
- Enables severity-matched retrieval from synthetic corpus
- Validates whether synthetic data + imputed scores can improve real prediction
- Empirical validation: if DAIC-WOZ metrics improve, the pipeline works
- Creates a novel scored dataset as a side effect

**Cons**:
- Noise in LLM scoring propagates to retrieval
- Requires multi-agent orchestration for stable scoring
- API costs for ~4k dialogues × multiple runs

**Stochasticity Mitigation**:
```python
# Multi-run scoring for stability
dialogue → [Run 1, Run 2, Run 3, Run 4, Run 5]
         → [PHQ-8: 14, 12, 15, 13, 14]
         → Mean: 13.6, Std: 1.1
```

---

## 5. Implementation Spec (Path D: Binary Classification)

### 5.1 Required Changes

#### New Configuration

```python
# config.py addition
class CrossDatasetSettings(BaseSettings):
    """Cross-dataset validation settings."""

    sqpsychconv_dir: Path = Field(
        default=Path("data/sqpsychconv"),
        description="Path to SQPsychConv dataset"
    )
    prediction_target: Literal["binary", "severity", "phq8"] = "binary"
    depressed_threshold: int = 10  # PHQ-8 total ≥ 10
```

#### New Data Loader

```python
# services/sqpsychconv_loader.py
class SQPsychConvLoader:
    """Load SQPsychConv dialogues for evaluation."""

    def load_dialogue(self, file_id: str) -> Transcript:
        """Load a single dialogue as a Transcript."""
        # Parse from train_sample.csv or Arrow format

    def get_label(self, file_id: str) -> bool:
        """Get depression label (True for 'mdd', False for 'control')."""
```

#### New Evaluation Script

```python
# scripts/evaluate_cross_dataset.py
"""
Evaluate DAIC-WOZ-trained model on SQPsychConv.

Usage:
    python scripts/evaluate_cross_dataset.py \
        --mode zero_shot \
        --prediction binary \
        --output data/outputs/cross_dataset_eval.json
"""
```

### 5.2 Evaluation Metrics

For binary classification:
- **AUC-ROC**: Area under receiver operating characteristic
- **Sensitivity**: True positive rate (detecting MDD)
- **Specificity**: True negative rate (detecting control)
- **F1 Score**: Harmonic mean of precision/recall

### 5.3 Baseline Experiment

```bash
# 1. Run zero-shot on SQPsychConv test set
python scripts/evaluate_cross_dataset.py \
    --mode zero_shot \
    --prediction binary \
    --split test

# 2. Run few-shot (using DAIC-WOZ embeddings)
python scripts/evaluate_cross_dataset.py \
    --mode few_shot \
    --prediction binary \
    --embeddings data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz \
    --split test
```

---

## 6. Circularity Concerns

### 6.1 The Synthetic Data Problem

SQPsychConv dialogues are **LLM-generated** conditioned on:
- Client profiles (possibly with severity indicators)
- Therapeutic structure (CBT)

**Risk**: An LLM evaluating LLM-generated conversations may recognize stylistic patterns rather than clinical content.

### 6.2 Mitigation Strategies

1. **Use different model families**: If SQPsychConv used Qwen, evaluate with Gemma/LLaMA
2. **Report synthetic-vs-real gap**: Always compare to DAIC-WOZ baseline
3. **Human validation subset**: Annotate 50-100 SQPsychConv dialogues with human raters
4. **Ablation by generation model**: SQPsychConv has variants (gemma, llama3, qwq) - compare

---

## 7. Recommended Next Steps

### Phase 1: Stabilize DAIC-WOZ Baseline (Current)
- Complete participant-only preprocessing evaluation
- Finalize chunk-level scoring
- Document reproducible baseline metrics

### Phase 2: Binary Cross-Validation (Low Effort)
1. Implement `SQPsychConvLoader`
2. Create `evaluate_cross_dataset.py`
3. Run zero-shot binary classification on SQPsychConv
4. Report AUC, sensitivity, specificity

### Phase 3: Investigate Severity Signals (Medium Effort)
1. Contact SQPsychConv authors for severity metadata
2. Explore if `file_id` numbers encode severity
3. Try LLM-derived severity (with circularity caveats)

### Phase 4: Publication-Ready Validation (High Effort)
1. Full cross-dataset protocol with statistical tests
2. Human annotation for subset validation
3. Multiple model family comparison

---

## 8. References

### Scale Mapping Research
- [PMC8599822: PHQ-9 vs HAMD](https://pmc.ncbi.nlm.nih.gov/articles/PMC8599822/)
- [PMC5515387: BDI/CES-D/PHQ-9 Common Metric](https://pmc.ncbi.nlm.nih.gov/articles/PMC5515387/)
- [PMC2148236: PHQ-9 vs HADS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2148236/)

### SQPsychConv
- [arXiv:2510.25384](https://arxiv.org/abs/2510.25384)
- [HuggingFace: AIMH/SQPsychConv](https://huggingface.co/collections/AIMH/sqpsychconv)

### DAIC-WOZ
- [Dataset Access](https://dcapswoz.ict.usc.edu/)

---

## 9. Decision Summary

| Question | Answer |
|----------|--------|
| Can we use SQPsychConv for item-level PHQ-8 few-shot? | **No** - no item scores in dataset |
| Can we use it for severity classification? | **No** - only binary labels |
| Can we use it for binary classification? | **Yes** - depressed/control |
| Can we use it for embedding quality testing? | **Yes** - semantic similarity |
| Is direct HAMD→PHQ-8 mapping possible? | **No** - no validated item mapping |
| Can we LLM-score it for PHQ-8? | **Yes** - dialogues have rich symptom content |
| Is LLM-scoring circular? | **No** - if evaluated on DAIC-WOZ ground truth |
| Should we proceed? | **Yes, with Path G (LLM-scored + DAIC-WOZ eval)** |

---

## 10. Implementation Recommendation: Separate Repository

### 10.1 Why Separate Repo?

The SQPsychConv scoring work is a **distinct dataset creation project**, not a depression prediction project:

| Aspect | ai-psychiatrist (this repo) | sqpsychconv-scored (new repo) |
|--------|----------------------------|-------------------------------|
| **Purpose** | Depression prediction pipeline | Dataset annotation/creation |
| **Primary output** | PHQ-8 predictions | Scored dataset + methodology |
| **Dependencies** | Ollama, local LLMs | Frontier APIs (OpenAI, Anthropic) |
| **Evaluation** | DAIC-WOZ ground truth | Internal consistency, human validation |
| **Reusability** | Specific to this paper | General mental health NLP |

### 10.2 Proposed Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ sqpsychconv-scored (new repo)                                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load SQPsychConv from HuggingFace                            │
│ 2. Multi-agent LLM scoring pipeline (GPT-4/Claude)              │
│ 3. Generate pseudo-PHQ-8 scores with uncertainty                │
│ 4. Optional: Human validation on 50-100 samples                 │
│ 5. Output: scored_sqpsychconv.csv + methodology docs            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (vendor or download)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ ai-psychiatrist (this repo)                                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load scored dataset from sqpsychconv-scored                  │
│ 2. Generate embeddings                                          │
│ 3. Use as retrieval corpus                                      │
│ 4. Evaluate on DAIC-WOZ                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 Novel Contribution

If you create `sqpsychconv-scored`, you've produced:

1. **A new dataset**: SQPsychConv with PHQ-8 item scores
2. **A methodology paper**: LLM-based clinical scoring of synthetic dialogues
3. **Validation data**: Calibration against DAIC-WOZ real scores
4. **Community resource**: Others can use for retrieval, fine-tuning, etc.

This is potentially a larger contribution than the original cross-validation experiment.

---

## 11. FOR2107 Data Governance Context

### 11.1 Why Scores Were Stripped

From the arXiv paper:

> "Although the clinical questionnaire dataset from Kircher et al. (2019) is anonymized, the sensitivity of the data and its access terms trigger strict requirements. Specifically, data privacy regulations restrict the use of clinical questionnaire data to controlled, audited environments."

The SQPsychConv authors were bound by German research ethics (FOR2107/MACS consortium) and could not release the conditioning HAMD/BDI scores publicly.

### 11.2 Accessing the Source Data

The FOR2107 consortium (for2107.de) has formal data access procedures:
- Principal investigators: Prof. Tilo Kircher (Marburg), Prof. Udo Dannlowski (Münster)
- Requires institutional collaboration agreement
- Likely multi-month process

**Recommendation**: Do not wait for this. Path G (LLM-scoring) is faster and creates equivalent value.

---

*This spec will be updated as we learn more about SQPsychConv's hidden metadata and author collaboration opportunities.*
