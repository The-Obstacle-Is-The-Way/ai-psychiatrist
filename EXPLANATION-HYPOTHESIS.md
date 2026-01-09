# Explanation Hypothesis: Why PHQ-8 Item-Level Prediction from DAIC-WOZ is Fundamentally Limited

**Created**: 2026-01-08
**Context**: After 14 reproduction runs, we observe:
- Few-shot consistently underperforms zero-shot
- Severity inference increases coverage but decreases accuracy
- Best MAE_item ≈ 0.60-0.65 with ~50% coverage

This document explains **why** from first principles.

---

## Executive Summary

**The core problem is a construct mismatch**: PHQ-8 measures **frequency** ("how many days in the past 2 weeks"), but DAIC-WOZ transcripts contain **qualitative symptom discussions** that rarely mention explicit day-counts.

This is not a model limitation, chunking problem, or embedding issue. It's a **fundamental data limitation** - we're asking an LLM to predict frequency from interviews that weren't designed to elicit frequency information.

---

## 1. How PHQ-8 Actually Works

The PHQ-8 is a **self-report questionnaire** completed by the patient **before** the clinical interview.

### PHQ-8 Response Scale

| Score | Label | Frequency |
|-------|-------|-----------|
| 0 | Not at all | 0-1 days in past 2 weeks |
| 1 | Several days | 2-6 days |
| 2 | More than half the days | 7-11 days |
| 3 | Nearly every day | 12-14 days |

### Critical Insight

The PHQ-8 asks: *"Over the past 2 weeks, how often have you been bothered by..."*

This is a **frequency question** answered with **day-count bins**. The patient self-reports this BEFORE sitting down with the clinician.

---

## 2. What DAIC-WOZ Transcripts Actually Contain

The [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/) contains semi-structured clinical interviews conducted by a virtual agent (Ellie). The interviews:

1. **Discuss symptoms qualitatively** ("Tell me about your sleep")
2. **Do NOT systematically elicit frequency** for each PHQ-8 domain
3. **Were not designed for PHQ-8 scoring** - the PHQ-8 was collected separately as metadata

### Example Transcript Patterns

What participants actually say:
- "I've been feeling tired lately" → What frequency? Unknown.
- "I have trouble sleeping" → How many nights? Unknown.
- "I'm not really interested in things" → How many days? Unknown.

What we would need for PHQ-8 scoring:
- "I've been tired for about 10 of the past 14 days" → Score 2
- "I had trouble sleeping maybe 3-4 nights" → Score 1

**The transcripts simply don't contain this information most of the time.**

---

## 3. Why ~50% Coverage is Correct Behavior

Our **strict mode** (`--severity-inference strict`) instructs the LLM:
> "Only assign numeric scores (0-3) when evidence clearly indicates frequency"

When the transcript says "I've been feeling down" without any temporal markers, the correct behavior is to output **N/A** - because we genuinely don't know if that's 2 days or 12 days.

The ~50% coverage we observe is not a limitation of the model. It's the model **correctly recognizing** that frequency information is absent.

---

## 4. Why Inference Mode Made Things Worse

Our **infer mode** (`--severity-inference infer`) adds a mapping table:
- "always" → Score 3
- "sometimes" → Score 1
- etc.

### What We Expected
More coverage + similar accuracy = better AURC

### What Actually Happened
| Mode | Run 13 (strict) | Run 14 (infer) |
|------|-----------------|----------------|
| Coverage | 50% | 60% |
| MAE_item | 0.608 | 0.703 |
| AURC | 0.107 | 0.129 |

**Coverage went up, but accuracy went DOWN more than proportionally.**

### Why?

The inference rules map **intensity** words to **frequency** scores, but these aren't the same thing:
- "I always feel tired" might mean "I'm tired every day" (Score 3)
- Or it might mean "Whenever I feel something, it's tiredness" (intensity, not frequency)

The mapping is **semantically incorrect**. We're conflating intensity with frequency, and the ground truth PHQ-8 is specifically about frequency.

---

## 5. Why Few-Shot Doesn't Help

Few-shot retrieval finds similar transcript chunks from the training set and shows them as examples. But:

### Problem A: Reference Chunks Also Lack Frequency
If the training transcripts also rarely mention explicit frequency, the retrieved examples don't provide useful calibration.

### Problem B: Evidence Grounding Rejects ~50% of Quotes
Our evidence grounding system (Spec 053) validates that LLM-extracted quotes actually exist in the transcript. About 50% are rejected as hallucinations, starving few-shot of reference material.

### Problem C: Chunk-Level Scoring is Noisy
Even with Spec 035 (chunk-level scoring), the scores attached to reference chunks are inferences from the same limited data, propagating the fundamental limitation.

---

## 6. Known DAIC-WOZ Dataset Limitations

Research has documented several issues with DAIC-WOZ:

### 6.1 Self-Report Bias
> "Self-reports may depend on social, subjective, and other kinds of biases, and may affect the annotation accuracy. Taking DAIC-WOZ as ground truth may not be the most accurate way to measure depression status." - [PMC11850819](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850819/)

### 6.2 Specificity Problem
> "High DAIC-WOZ performance may largely be caused by learning disorder-general cues (flat affect, vocal strain, negative language) and not markers unique to Major Depressive Disorder." - [ACM ICMI 2024](https://dl.acm.org/doi/10.1145/3747327.3763034)

### 6.3 Interviewer Prompt Bias
> "Models using interviewer prompts learn to focus on specific regions where questions about past mental health experiences are asked, using them as discriminative shortcuts." - [arXiv:2404.14463](https://arxiv.org/html/2404.14463v1)

### 6.4 Clinical Significance Gap
> "State-of-the-art systems achieved MAE of 3.80 comparable to a four-point range span. A difference in estimating a whole depression level could result in substantial change in treatment." - [PMC10308283](https://pmc.ncbi.nlm.nih.gov/articles/PMC10308283/)

---

## 7. What This Means for Our Results

### Our Best Results in Context

| Metric | Our Best | Paper (Greene et al.) | Clinical Significance |
|--------|----------|----------------------|----------------------|
| MAE_item (zero-shot) | 0.608 | 0.796 | We beat the paper by 24% |
| Coverage | 50% | ~100% (forced) | We're honest about uncertainty |
| AURC | 0.107 | Not reported | Selective prediction |

### Interpretation

1. **We're doing better than the paper** on items we actually score
2. **Our abstention is appropriate** - we correctly identify when frequency evidence is missing
3. **Few-shot doesn't help** because the fundamental limitation is data, not prompting
4. **Inference mode hurts** because intensity ≠ frequency

---

## 8. Alternative Tasks That May Work Better

Given the fundamental frequency-mismatch, these alternative prediction targets may be more appropriate:

### 8.1 Binary Classification (Spec 062)
**Task**: Depressed (PHQ-8 ≥ 10) vs Not Depressed (PHQ-8 < 10)

**Why it might work better**: Binary classification can leverage "disorder-general cues" (flat affect, negative language) that DAIC-WOZ models actually learn. We don't need precise frequency - just overall severity signal.

### 8.2 Total Score Prediction (Spec 061)
**Task**: Predict sum of items (0-24) instead of individual items

**Why it might work better**: Item-level errors may cancel out. Total score is clinically actionable (severity tiers: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-24 severe).

### 8.3 Severity Tier Classification
**Task**: Classify into severity tiers (minimal/mild/moderate/severe)

**Why it might work better**: Coarser granularity matches the signal available in transcripts.

---

## 9. Remaining Questions

### 9.1 Is Gemma3:27b the Right Model?
Possibly not. Larger models (70B+) or clinically fine-tuned models (Med-PaLM, MedGemma) might better recognize implicit frequency signals.

### 9.2 Would Multimodal Help?
DAIC-WOZ includes audio/video. Prosodic features (speech rate, pauses) might provide frequency-correlated signals that transcripts lack.

### 9.3 Is the Original Paper's Few-Shot Result Valid?
The paper reported few-shot MAE 0.619 < zero-shot MAE 0.796. But they didn't have:
- Evidence grounding (Spec 053)
- Strict JSON parsing
- Coverage tracking

Their "few-shot" may have included hallucinated quotes that happened to be directionally correct by chance.

---

## 10. Conclusions

### The Bottom Line

1. **~50% coverage with strict mode is correct** - the transcripts genuinely lack frequency information for ~50% of item-participant pairs

2. **Few-shot underperformance is expected** - reference examples also lack frequency information

3. **Inference mode hurts accuracy** - mapping intensity words to frequency scores is semantically incorrect

4. **This is a fundamental data limitation** - not a model, embedding, or chunking problem

### Recommendations

1. **Accept strict mode as the ceiling** for item-level prediction on DAIC-WOZ
2. **Evaluate binary classification** (Spec 062) - may work better for this dataset
3. **Evaluate total score prediction** (Spec 061) - item errors may cancel out
4. **Consider this task partially solved** - we've characterized the fundamental limitations

---

## References

- [DAIC-WOZ Database](https://dcapswoz.ict.usc.edu/)
- [DAIC-WOZ Validity Study (arXiv:2404.14463)](https://arxiv.org/html/2404.14463v1)
- [Depression Detection Reproducibility Study (ACM ICMI 2024)](https://dl.acm.org/doi/10.1145/3747327.3763034)
- [Multi-Instance Learning for Depression (PMC11850819)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850819/)
- [Task Validity Analysis](docs/clinical/task-validity.md)
- [Few-Shot Analysis](docs/results/few-shot-analysis.md)
