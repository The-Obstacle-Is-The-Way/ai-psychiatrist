# Pipeline

This document explains how the four-agent pipeline works to assess depression from clinical interview transcripts.

---

## Overview

The AI Psychiatrist pipeline processes a transcript through four specialized agents, with an iterative refinement loop to ensure quality:

```text
Transcript → Qualitative → [Judge ↔ Refinement] → Quantitative → Meta-Review → Severity
```

Each agent serves a specific purpose, and their outputs feed into subsequent stages.

---

## Pipeline Stages

### Stage 1: Qualitative Assessment

**Agent:** `QualitativeAssessmentAgent`
**Model:** Gemma 3 27B (default)
**Paper Reference:** Section 2.3.1

The qualitative agent analyzes the transcript to identify clinical factors across four domains:

| Domain | Description | Example Findings |
|--------|-------------|------------------|
| **PHQ-8 Symptoms** | Symptom presence and frequency | "Reports low energy nearly every day" |
| **Social Factors** | Relationships, support systems | "Limited social support, lives alone" |
| **Biological Factors** | Medical history, family history | "Family history of depression" |
| **Risk Factors** | Stressors, warning signs | "Recent job loss, financial stress" |

**Output:** `QualitativeAssessment` entity with structured sections and supporting quotes.

**Prompt Structure:**
```text
System: You are a clinical psychologist analyzing interview transcripts...
User: <transcript>
{transcript_text}
</transcript>

Please analyze this interview and provide:
1. Overall assessment
2. PHQ-8 symptom analysis with frequencies
3. Social factors
4. Biological factors
5. Risk factors
```

---

### Stage 2: Judge Evaluation

**Agent:** `JudgeAgent`
**Model:** Gemma 3 27B (default, temperature=0.0)
**Paper Reference:** Section 2.3.1, Appendix B

The judge agent evaluates the qualitative assessment on four quality metrics:

| Metric | Description | Scoring Guide |
|--------|-------------|---------------|
| **Coherence** | Logical consistency | 5=No contradictions, 1=Major logical errors |
| **Completeness** | Symptom coverage | 5=All symptoms addressed, 1=Major gaps |
| **Specificity** | Concrete vs vague | 5=Specific quotes/frequencies, 1=Generic statements |
| **Accuracy** | PHQ-8/DSM-5 alignment | 5=Clinically correct, 1=Major misinterpretations |

**Scoring:** 1-5 Likert scale per metric

**Decision Logic:**
- If ALL scores ≥ 4: Assessment is acceptable, proceed to quantitative
- If ANY score ≤ 3: Trigger refinement loop

---

### Stage 3: Feedback Loop (Iterative Refinement)

**Service:** `FeedbackLoopService`
**Paper Reference:** Section 2.3.1

When judge scores are below threshold, the feedback loop refines the assessment:

```text
┌─────────────────────────────────────────────────────────────┐
│                     FEEDBACK LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐                                       │
│   │   Qualitative   │◄──────────────────────────────┐       │
│   │     Agent       │                               │       │
│   └────────┬────────┘                               │       │
│            │                                        │       │
│            ▼                                        │       │
│   ┌─────────────────┐    Low scores?    ┌────────-──┴──────┐│
│   │   Judge Agent   │─────Yes──────────►│ Extract Feedback ││
│   │  (Evaluate)     │                   │ for low metrics  ││
│   └────────┬────────┘                   └──────────────────┘│
│            │                                                │
│            │ All scores ≥ 4?                                │
│            │ OR max iterations?                             │
│            │                                                │
│            ▼ Yes                                            │
│   ┌─────────────────┐                                       │
│   │     EXIT        │                                       │
│   │  (Proceed to    │                                       │
│   │  Quantitative)  │                                       │
│   └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
- `max_iterations`: 10 (paper Section 2.3.1)
- `score_threshold`: 3 (scores ≤ 3 trigger refinement)

**Refinement Prompt:**
```text
The judge evaluated your assessment and found issues:

Coherence: Scored 2/5. "The assessment contradicts itself..."
Specificity: Scored 3/5. "More specific quotes needed..."

Please revise your assessment addressing these concerns:
<original_assessment>
{previous_assessment}
</original_assessment>

<transcript>
{transcript_text}
</transcript>
```

**Paper Results (Figure 2, 142 participants)**: The paper reports mean ± SD improvements after
the feedback loop:

| Metric | Before | After |
|--------|--------|-------|
| Coherence | 4.96 ± 0.20 | 5.00 ± 0.00 |
| Specificity | 4.37 ± 0.62 | 4.38 ± 0.58 |
| Accuracy | 4.33 ± 0.53 | 4.36 ± 0.48 |
| Completeness | 3.61 ± 0.85 | 3.72 ± 0.61 |

---

### Stage 4: Quantitative Assessment

**Agent:** `QuantitativeAssessmentAgent`
**Model:** Gemma 3 27B (default)
**Paper Reference:** Section 2.3.2, Section 2.4.2

The quantitative agent predicts PHQ-8 item scores (0-3) for each symptom.

#### Evidence Extraction

First, the agent extracts evidence quotes for each PHQ-8 item:

```json
{
  "PHQ8_NoInterest": ["i don't enjoy anything anymore", "nothing seems fun"],
  "PHQ8_Depressed": ["i feel really down most days"],
  "PHQ8_Sleep": ["i can't fall asleep until 3am"],
  ...
}
```

**Keyword Backfill (Optional):** If enabled, keyword matching can supplement extraction when the
LLM misses evidence. By default (paper-text parity), backfill is OFF; see
`docs/concepts/backfill-explained.md` and `docs/archive/bugs/analysis-027-paper-implementation-comparison.md`.

#### Few-Shot Reference Retrieval

For each item with evidence:

1. **Embed** the evidence text using qwen3-embedding:8b
2. **Search** the reference store for similar chunks
3. **Retrieve** top-k (default: 2) most similar references with known scores

```text
Query: "i don't enjoy anything anymore, nothing seems fun"
              │
              ▼ Embedding + Similarity Search
┌─────────────────────────────────────────────────┐
│ Reference 1 (similarity: 0.89, score: 2)        │
│ "haven't felt like doing my hobbies lately"     │
├─────────────────────────────────────────────────┤
│ Reference 2 (similarity: 0.85, score: 3)        │
│ "nothing brings me joy anymore"                 │
└─────────────────────────────────────────────────┘
```

#### Scoring

The agent generates scores with reasoning:

```json
{
  "PHQ8_NoInterest": {
    "evidence": "i don't enjoy anything anymore",
    "reason": "Clear anhedonia, consistent with nearly every day",
    "score": 3
  },
  "PHQ8_Sleep": {
    "evidence": "i can't fall asleep until 3am",
    "reason": "Significant sleep onset insomnia",
    "score": 2
  },
  "PHQ8_Appetite": {
    "evidence": "No relevant evidence found",
    "reason": "Transcript does not discuss eating habits",
    "score": "N/A"
  }
}
```

**Output:** `PHQ8Assessment` with all 8 item scores, total score (0-24), and severity level.

**Paper Results:**
- Zero-shot MAE: 0.796
- Few-shot MAE: 0.619 (22% lower item-level MAE vs zero-shot)
- MedGemma few-shot MAE: 0.505 (Appendix F alternative; better MAE but fewer predictions overall)

---

### Stage 5: Meta-Review

**Agent:** `MetaReviewAgent`
**Model:** Gemma 3 27B (default)
**Paper Reference:** Section 2.3.3

The meta-review agent integrates all previous outputs to determine final severity:

**Inputs:**
1. Original transcript
2. Qualitative assessment (social, biological, risk factors)
3. Quantitative scores (PHQ-8 item scores)

**Output:**
- Final severity level (0-4: MINIMAL, MILD, MODERATE, MOD_SEVERE, SEVERE)
- Explanation of determination
- MDD indicator (true if severity ≥ MODERATE)

**Prompt Structure:**
```text
You are integrating multiple assessments to determine depression severity.

<transcript>
{transcript_text}
</transcript>

<qualitative_assessment>
{qualitative_text}
</qualitative_assessment>

<quantitative_scores>
{phq8_scores}
</quantitative_scores>

Provide:
<severity>0-4</severity>
<explanation>Your integrated reasoning...</explanation>
```

**Paper Results:** 78% accuracy on severity prediction, comparable to human experts.

---

## Complete Pipeline Flow

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  INPUT                                                                 │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Transcript: "Ellie: How are you? Participant: I feel down..."     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  QUALITATIVE (Gemma 3 27B)                                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Overall: Participant shows signs of depression...                 │ │
│  │ PHQ-8: Anhedonia (several days), low mood (most days)...          │ │
│  │ Social: Limited support network...                                │ │
│  │ Biological: No family history mentioned...                        │ │
│  │ Risk: Recent stressors...                                         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  JUDGE (Gemma 3 27B, temp=0)                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Coherence: 4/5  |  Completeness: 3/5  |  Specificity: 4/5  |      │ │
│  │ Accuracy: 4/5   |  → Completeness low, trigger refinement         │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  FEEDBACK LOOP (1 iteration)                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Refined assessment with better completeness...                    │ │
│  │ Judge re-evaluation: All scores ≥ 4 ✓                             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  QUANTITATIVE (Gemma 3 27B)                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ PHQ8_NoInterest: 2  |  PHQ8_Depressed: 2  |  PHQ8_Sleep: 1        │ │
│  │ PHQ8_Tired: 2       |  PHQ8_Appetite: N/A |  PHQ8_Failure: 1      │ │
│  │ PHQ8_Concentrating: 1  |  PHQ8_Moving: N/A                        │ │
│  │ Total: 9 → MILD severity                                          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  META-REVIEW (Gemma 3 27B)                                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ Severity: 1 (MILD)                                                │ │
│  │ Explanation: While the participant reports several symptoms,      │ │
│  │ their frequency is mostly "several days" rather than daily.       │ │
│  │ The qualitative assessment notes limited but present coping...    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│                              ▼                                         │
│  OUTPUT                                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ FullAssessment {                                                  │ │
│  │   severity: MILD                                                  │ │
│  │   is_mdd: false                                                   │ │
│  │   phq8_total: 9                                                   │ │
│  │   ...                                                             │ │
│  │ }                                                                 │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Timing

The paper reports the full pipeline runs in **~1 minute** on a MacBook Pro with an Apple
M3 Pro chipset (Section 2.3.5 / Discussion). Real-world timing varies significantly with:

- backend (Ollama vs HuggingFace),
- model quantization / device (CPU/GPU),
- and whether the feedback loop triggers refinements.

Note: The paper text emphasizes consumer hardware (M3 Pro / no GPU requirement), but the public repo
also includes SLURM scripts configured for A100 GPUs (`_reference/slurm/job_ollama.sh`). We cannot
determine what hardware/precision produced the reported metrics from the paper text alone.

For local reproduction runtime measurements, see `docs/results/reproduction-notes.md`.

---

## Configuration Impact

| Setting | Effect on Pipeline |
|---------|-------------------|
| `FEEDBACK_ENABLED=false` | Skip refinement loop entirely |
| `FEEDBACK_MAX_ITERATIONS=5` | Cap refinement attempts |
| `EMBEDDING_TOP_K_REFERENCES=4` | More reference examples per item |
| `LLM_BACKEND=huggingface` + `MODEL_QUANTITATIVE_MODEL=medgemma:27b` | Use Appendix F alternative (official weights via HuggingFace; may reduce prediction availability) |

---

## See Also

- [Architecture](architecture.md) - System design details
- [PHQ-8](phq8.md) - Understanding the assessment scale
- [Configuration](../reference/configuration.md) - All settings
