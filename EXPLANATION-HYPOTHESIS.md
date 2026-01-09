# Explanation Hypothesis: Why PHQ-8 Item-Level Prediction from DAIC-WOZ is Fundamentally Limited

**Created**: 2026-01-08
**Status**: Hypothesis (supported by ablations; not a proof)

**Context (clean post-confound runs)**:
- Post BUG-035, few-shot underperforms zero-shot in the clean comparative runs we have (Run 13 strict; Run 14 infer)
- Severity inference (Spec 063, `infer`) increased coverage but decreased accuracy in Run 14 (single ablation so far)
- Best observed item-level MAE (mean-of-items; `MAE_i`) is ≈ 0.60–0.65 at ~50% coverage under strict mode

**Primary evidence artifacts**:
- Run 13 (strict): `data/outputs/both_paper-test_20260107_134730.json` (git `01d3124`)
- Run 14 (infer): `data/outputs/both_paper-test_20260108_114058.json` (git `e55c00f`)

This document explains **why** from first principles.

---

## Executive Summary

**The primary bottleneck is a construct mismatch**: PHQ-8 measures **frequency** (“how many days in the past 2 weeks”), while DAIC-WOZ interviews primarily contain **qualitative symptom discussion** and often do not elicit item-by-item 2-week frequency bins.

This does not mean model/prompt/retrieval improvements are irrelevant. It means there is a strong **information availability limit**: when the transcript does not contain enough time-windowed frequency evidence, any item score becomes an inference with non-trivial subjectivity (or should abstain).

---

## 1. How PHQ-8 Actually Works

The PHQ-8 is a **self-report frequency instrument** about the **past two weeks**. In DAIC-WOZ, PHQ-8 is collected as metadata; the interview transcript is not a PHQ administration script.

### PHQ-8 Response Scale

| Score | Label | Frequency |
|-------|-------|-----------|
| 0 | Not at all | 0-1 days in past 2 weeks |
| 1 | Several days | 2-6 days |
| 2 | More than half the days | 7-11 days |
| 3 | Nearly every day | 12-14 days |

### Critical Insight

The PHQ-8 asks: *"Over the past 2 weeks, how often have you been bothered by..."*

This is a **frequency question** answered with **day-count bins**. In DAIC-WOZ, those bins come from questionnaire metadata, not from an explicit PHQ-style administration embedded in the transcript.

---

## 2. What DAIC-WOZ Transcripts Actually Contain

The [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/) contains semi-structured clinical interviews conducted by a virtual agent (Ellie). The interviews:

1. **Discuss symptoms qualitatively** ("Tell me about your sleep")
2. **Do not systematically elicit PHQ-8-style frequency bins** for each item
3. **Were not designed for item-level PHQ-8 scoring**; PHQ-8 is collected separately as metadata

Additional constraint (by design in this repo): the validated baseline uses **participant-only** transcripts (`data/transcripts_participant_only`), which removes interviewer question context. This reduces protocol leakage but further reduces explicit temporal anchoring (“over the past two weeks…”).

### Example Transcript Patterns

What participants actually say:
- "I've been feeling tired lately" → What frequency? Unknown.
- "I have trouble sleeping" → How many nights? Unknown.
- "I'm not really interested in things" → How many days? Unknown.

What we would need for PHQ-8 scoring:
- "I've been tired for about 10 of the past 14 days" → Score 2
- "I had trouble sleeping maybe 3-4 nights" → Score 1

**Often, the transcript does not contain enough information to map a symptom mention into a 2-week day-count bin.**

---

## 3. Why ~50% Coverage is Correct Behavior

Our **strict mode** (`--severity-inference strict`) instructs the LLM:
> "Only assign numeric scores (0-3) when evidence clearly indicates frequency"

When the transcript says "I've been feeling down" without any temporal markers, the correct behavior is to output **N/A** - because we genuinely don't know if that's 2 days or 12 days.

The ~50% coverage we observe in Run 13 is not a pipeline bug. It reflects a conservative policy: **abstain unless the transcript supports a frequency bin**. It is consistent with the broader task-validity argument in `docs/clinical/task-validity.md`.

---

## 4. Why Inference Mode Made Things Worse

Our **infer mode** (`--severity-inference infer`) adds a heuristic **FREQUENCY INFERENCE GUIDE** to the scoring prompt (see `src/ai_psychiatrist/agents/prompts/quantitative.py`). In particular, it:
- Maps ambiguous temporal/intensity phrases (e.g., “always”, “all the time”, “lately”) to day-count bins
- Permits scoring from impact statements (“can’t function”) when no explicit temporal markers exist
- Explicitly discourages N/A unless the symptom is truly unmentioned

### What We Expected
More coverage + similar accuracy = better AURC

### What Actually Happened (zero-shot; paper-test)
| Metric | Run 13 (strict) | Run 14 (infer) |
|------|-----------------|----------------|
| Coverage | 50.0% | 60.1% |
| MAE_i (mean-of-items) | 0.6079 | 0.7030 |
| AURC_full (confidence=`llm`) | 0.1066 | 0.1292 |

Notes:
- MAE_i is `item_mae_by_item` in the output JSON (mean of per-item MAEs).
- AURC_full is computed by `scripts/evaluate_selective_prediction.py` from run outputs (loss=`abs_norm`).

**Coverage went up, but accuracy went DOWN more than proportionally.**

### Why?

The key failure mode is **unvalidated inference**: the prompt is now authorizing the model to map weak signals into exact 2-week bins. This includes:
- **Intensity/valence → frequency conflation** (“always”, “all the time” can be rhetorical/intensity, not a count)
- **Vague recency markers** (“lately”, “recently”) that do not imply a PHQ-8 day-count
- **Time-window mismatch** (the interview may discuss months/years; PHQ-8 is strictly past 2 weeks)

These are plausible reasons Run 14 increased coverage but injected enough noise to degrade both MAE and risk-coverage metrics.

---

## 5. Why Few-Shot Doesn't Help

Few-shot retrieval finds similar transcript chunks from the training set and shows them as examples. But:

### Problem A: Reference Chunks Also Lack Frequency
If the training transcripts also often lack explicit PHQ-8-style frequency bins, the retrieved examples may not provide useful calibration.

### Problem B: Evidence Grounding Rejects a Large Fraction of Extracted Quotes
Our evidence grounding system (Spec 053) validates that LLM-extracted quotes exist verbatim in the transcript (substring mode). In Runs 13–14, about **60%** of extracted quotes were rejected by substring grounding (validated ≈ 40%), which can starve retrieval of usable query evidence.

Operational estimate (from run logs): Run 13 rejected 296/494 extracted quotes; Run 14 rejected 291/486 (both ≈ 60%).

Important nuance: “rejected by substring grounding” is not identical to “hallucinated”. It also includes near-miss formatting issues (punctuation/whitespace) and partial-quote mismatches. The rejection rate is still operationally relevant: rejected quotes do not contribute to retrieval.

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

| Metric | Our Best | Published baseline (Greene et al.) | Caveat |
|--------|----------|----------------------|----------------------|
| MAE_i (item-level; mean-of-items) | 0.608 | 0.796 | Not directly comparable if coverage differs |
| Coverage | 50% | ~100% (forced) | Selective prediction vs forced scoring |
| AURC_full (confidence=`llm`) | 0.107 | Not reported | Coverage-aware risk metric |

### Interpretation

1. On scored items, **our MAE_i is lower than the published baseline**, but the baseline uses forced coverage; treat MAE comparisons as meaningful only at similar coverage.
2. **Abstention is methodologically appropriate** when the transcript does not support a 2-week frequency bin.
3. **Few-shot has not helped in clean comparative runs so far**, plausibly because retrieval is bottlenecked by sparse/low-quality frequency evidence and quote validation rejects many extracted quotes.
4. **Inference mode (as implemented) hurts**, likely because it authorizes unvalidated mappings from weak cues (recency/intensity/impact) into PHQ-8 day-count bins.

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

Because of these differences, their few-shot comparison is not directly comparable to this repo’s conservative, evidence-grounded selective prediction setting. Post BUG-035, our clean runs show few-shot underperforming zero-shot under the validated baseline configuration.

---

## 10. Conclusions

### The Bottom Line

1. **Strict mode produces conservative coverage** (Run 13: ~50% on paper-test), consistent with the claim that transcript-only item-frequency scoring is often underdetermined.
2. **Few-shot has not helped in clean comparative runs so far**, plausibly due to sparse frequency evidence + high quote rejection + retrieval noise/anchoring.
3. **Infer mode (as implemented in Spec 063) degraded quality in Run 14**: coverage rose, but both MAE_i and AURC_full worsened. The prompt authorizes unvalidated mappings from weak cues into PHQ-8 bins.
4. **The primary limitation is information availability**, though model/prompt/retrieval improvements can still matter at the margins or under different transcript variants (e.g., with question context, or multimodal signals).

### Recommendations

1. Treat strict mode as the **current best conservative baseline** for item-level scoring on participant-only transcripts.
2. When experimenting with inference, prefer policies that are **explicitly conservative** (e.g., restrict to unambiguous temporal markers; separate “impact” from “frequency”; preserve abstention when time window is unclear).
3. Prioritize evaluating **total score** and **binary** tasks (Specs 061/062 are implemented) because they better match the available signal.
4. Keep reporting **coverage-aware metrics** (Cmax + AURC/AUGRC) and treat MAE comparisons as valid only at similar coverage.

---

## References

- [DAIC-WOZ Database](https://dcapswoz.ict.usc.edu/)
- [DAIC-WOZ Documentation PDF](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)
- [PHQ-8 Description / Validation (PubMed 18752852)](https://pubmed.ncbi.nlm.nih.gov/18752852/)
- [DAIC-WOZ Validity Study (arXiv:2404.14463)](https://arxiv.org/html/2404.14463v1)
- [Depression Detection Reproducibility Study (ACM ICMI 2024)](https://dl.acm.org/doi/10.1145/3747327.3763034)
- [Multi-Instance Learning for Depression (PMC11850819)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850819/)
- [Task Validity Analysis](docs/clinical/task-validity.md)
- [Few-Shot Analysis](docs/results/few-shot-analysis.md)
