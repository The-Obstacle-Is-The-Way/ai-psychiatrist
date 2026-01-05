# Clinical Understanding: How This System Works

**Audience**: Clinicians, researchers, non-CS folks
**Last Updated**: 2026-01-03

---

## The Big Picture

This system reads interview transcripts (like DAIC-WOZ clinical interviews) and **selectively** infers PHQ-8 depression item scores when the transcript contains sufficient evidence. When it cannot justify an item score from transcript evidence, it returns **`N/A`** (abstention).

PHQ-8 item scores are defined by **2-week frequency**, but DAIC-WOZ transcripts are not structured as PHQ administration. This creates a real validity constraint for transcript-only item scoring; see: `docs/clinical/task-validity.md`.

---

## Key Concepts Explained

### 1. The PHQ-8 Structure

The PHQ-8 has 8 items (questions), each scored 0-3:
- **0** = Not at all
- **1** = Several days
- **2** = More than half the days
- **3** = Nearly every day

**Total score** ranges 0-24. The items are:
1. Little interest or pleasure (Anhedonia)
2. Feeling down, depressed, hopeless
3. Sleep problems
4. Low energy, fatigue
5. Appetite changes
6. Feeling bad about yourself
7. Trouble concentrating
8. Psychomotor changes (moving/speaking slower or restless)

---

### 2. What "Evidence Extraction" Means

**Analogy**: Imagine you're reading a patient's interview transcript. Before you score each PHQ-8 item, you first **highlight passages** that are relevant to each symptom.

That's what evidence extraction does:
1. The LLM reads the entire transcript
2. For each PHQ-8 item, it finds and extracts **quotes** (evidence) from the interview that relate to that symptom
3. Examples:
   - For "sleep problems": might extract "I've been waking up at 3am every night"
   - For "low interest": might extract "I used to love painting but haven't touched it in months"

**Why it matters**: The more evidence found, the more confident the system can be about scoring. If no evidence is found for an item, the system often returns "N/A" (can't assess).

---

### 3. What "Coverage" Means

**Coverage** = What percentage of the 8 items got actual scores (vs N/A)

**Examples**:
- If 4 out of 8 items were scored and 4 were N/A → **50% coverage**
- If 6 out of 8 items were scored → **75% coverage**
- If all 8 items were scored → **100% coverage**

**Clinical parallel**: Sometimes a clinical interview doesn't touch on every symptom domain. If the patient never discussed sleep, you can't really score the sleep item. Same logic here.

---

### 4. What the LLM Actually Does

The system makes **multiple LLM calls** per patient:

#### Step 1: Evidence Extraction
- LLM reads transcript
- Outputs JSON with quotes for each PHQ-8 item
- Output is schema-validated and evidence-grounded (rejected quotes are logged without transcript text)
- If parsing/validation fails, the participant evaluation fails loudly (no silent fallbacks)

#### Step 2: Few-Shot Retrieval
- Uses the extracted evidence to find **similar patients** from the training data
- "This patient talks about sleep like Patient X did, who had score 2 on sleep"

#### Step 3: Scoring
- LLM sees: the transcript, the evidence, and examples from similar patients
- Outputs: a score (0-3) or "N/A" for each item, plus reasoning

---

### 5. What MAE (Mean Absolute Error) Means

**MAE** is how far off the predictions are, on average.

**Simple example**:
- Patient's true score on Item 1: **2**
- System predicted: **1**
- Error = |2 - 1| = **1**

Do this for all items across all patients, average the errors → **MAE**

**Paper's reported MAE**: 0.619 (few-shot mode)

**What this means clinically**: On average, the system is off by about 0.6 points per item. On a 0-3 scale, that's reasonably accurate but not perfect.

---

### 6. How It All Connects

```
Interview Transcript
        ↓
   Evidence Extraction (find relevant quotes)
        ↓
   Similar Patient Retrieval (few-shot examples)
        ↓
   LLM Scoring (predict 0-3 or N/A per item)
        ↓
   MAE Calculation (compare to ground truth)
```

**Key relationships**:

| Factor | Affects | How |
|--------|---------|-----|
| Evidence quality | Coverage | Better evidence → fewer N/A items |
| Coverage | MAE calculation | N/A items are excluded from MAE |
| Few-shot examples | Score accuracy | Similar patients help calibrate predictions |
| Interview richness | Everything | Sparse interviews → sparse evidence → low coverage |

---

## Why We're Seeing What We're Seeing

### The Core Driver: Evidence Availability (Not “Model Knowledge”)

Many DAIC-WOZ interviews do not contain explicit PHQ-8 frequency language for each item. The system is designed to abstain (`N/A`) when evidence is insufficient rather than hallucinate frequency.

### Variable Coverage (Often ~50% on DAIC-WOZ)
Coverage varies across participants and items. This depends on:
- What symptoms the patient discussed
- Whether extracted quotes can be grounded in the transcript
- How explicit the symptom mentions were

### The Paper's Approach
The paper excludes N/A items from MAE calculation. This is valid because:
1. It matches clinical reality (can't score what wasn't discussed)
2. It focuses accuracy metrics on what the system actually predicted
3. Coverage is reported separately so you know how much was skipped

---

## What This Means for Going Forward

### Potential Improvements

1. **Better Evidence Extraction**
   - Reduce malformed JSON rates via prompt tightening and/or an explicit repair step
   - Could improve coverage by reducing empty-evidence cases

2. **Prompt Engineering**
   - Adjust how we ask the LLM to extract evidence
   - Be more explicit about valid output formats

### What the Results Will Tell Us

When you run a reproduction/evaluation, you'll see:
- **MAE_item**: Average error per item (compare to paper's 0.619)
- **Coverage**: Percentage of items with predictions
- **By-participant breakdown**: Which patients were harder to assess

If our MAE is close to 0.619 with reasonable coverage, we've successfully reproduced the paper's methodology.

---

## Summary

**In one sentence**: The system extracts symptom-related quotes from interviews, optionally retrieves similar examples, predicts 0-3 scores per PHQ-8 item (or `N/A` if insufficient evidence), and we evaluate accuracy and abstention jointly via coverage-aware metrics (AURC/AUGRC) plus item-level MAE on predicted items.

**Known limitation**: Item-level PHQ-8 scoring from transcript-only evidence is often underdetermined because PHQ-8 is a 2-week frequency instrument. This is a dataset/task constraint, not just an engineering issue; see `docs/clinical/task-validity.md`.

---

## Technical Appendix: Paper-Specified Parameters

From the paper (Section 2.4.2 and Appendix D):

### LLM Calls Per Participant

| Step | Model | Purpose |
|------|-------|---------|
| 1. Evidence Extraction | Gemma 3 27B | Find relevant quotes for each PHQ-8 item |
| 2. Scoring | Gemma 3 27B | Predict 0-3 scores using evidence + examples |

**Total: 2 LLM calls per participant** (plus embedding calls)

### Few-Shot Hyperparameters (Paper Appendix D)

| Parameter | Optimal Value | What It Means |
|-----------|---------------|---------------|
| N_example | 2 | Number of similar examples per PHQ-8 item |
| N_chunk | 8 | Lines per transcript chunk |
| Step size | 2 | Sliding window overlap |
| Dimension | 4096 | Embedding vector size |

**Maximum reference chunks per participant**: 2 examples × 8 items = **16 chunks**

### How Similar Examples Are Found

1. Training transcripts are pre-chunked (8 lines each, sliding by 2)
2. Each chunk is pre-embedded using Qwen 3 8B Embedding (4096 dimensions)
3. For a new patient:
   - Evidence extracted by LLM is embedded
   - Cosine similarity finds the 2 most similar training chunks per item
   - Those chunks + their ground truth scores become the "few-shot examples"

### Paper Results (Section 3.2)

| Mode | MAE | Notes |
|------|-----|-------|
| Zero-shot | 0.796 | No examples, just prompt |
| Few-shot | 0.619 | With 2 similar examples per item |
| Few-shot + MedGemma | 0.505 | Better MAE but fewer predictions |

The paper reports that few-shot reduced MAE by **22%** compared to zero-shot; reproduction results may differ depending on model/backend and retrieval configuration.
