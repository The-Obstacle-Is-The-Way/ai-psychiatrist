# Coverage Explained: What It Is and Why It Matters

**Audience**: Anyone trying to understand what "coverage" means in PHQ-8 assessment
**Last Updated**: 2026-01-02

---

## What is Coverage?

**Coverage** is the percentage of PHQ-8 items that received an actual score (0, 1, 2, or 3) instead of "N/A" (Not Applicable / Cannot Assess).

### Simple Example

The PHQ-8 has 8 items. If a patient's assessment looks like this:

| Item | Score |
|------|-------|
| No Interest | 2 |
| Depressed | 1 |
| Sleep | 1 |
| Tired | N/A |
| Appetite | N/A |
| Failure | 0 |
| Concentrating | N/A |
| Moving | N/A |

**Coverage = 4/8 = 50%**

Only 4 items got scores; 4 were marked "N/A" (cannot assess).

---

## Why Does Coverage Happen?

The system says "N/A" when it **cannot find enough evidence** in the interview transcript to make a prediction.

### Reasons for N/A

1. **Symptom not discussed**: If the patient never mentioned sleep, the system can't score the sleep item
2. **Vague mentions**: "I've been okay" doesn't give enough information
3. **Evidence extraction failed**: Sometimes the LLM fails to extract relevant quotes
4. **Conservative thresholds**: Some models are more cautious about making predictions

### Clinical Parallel

This mirrors real clinical practice. If a patient never discussed their appetite during an interview, a clinician wouldn't score that item either—they'd mark it as "not assessed."

---

## Coverage vs. Accuracy Tradeoff

This is the key insight:

```
Higher coverage → More predictions → Includes harder items → Potentially higher MAE
Lower coverage → Fewer predictions → Only "easy" items → Potentially lower MAE
```

### Example Run vs. Paper

The paper reports item-level MAE and notes that **in ~50% of cases** the model could not
provide a prediction due to insufficient evidence. The paper does not fully specify what
the denominator for “cases” is (item-level vs subject-level), but it is clearly describing
substantial abstention due to missing evidence.

This repository also computes **item-level MAE excluding N/A**, but the exact
coverage/MAE depends on model weights/quantization, backend, and prompt behavior.

| Metric | Paper (reported) | Example Run (paper-test, few-shot, participant-only transcripts) |
|--------|------------------|----------------------------------------------|
| Coverage (Cmax) | ~50% abstention (“unable to provide a prediction”) | 50.9% |
| MAE_item | 0.619 | 0.609 |

Run details and metric definitions live in:
- `docs/results/run-history.md`
- `docs/results/reproduction-results.md`
- `docs/statistics/metrics-and-evaluation.md`

Interpretation: higher coverage often increases MAE because the model attempts more
items (including harder-to-evidence symptoms). This is a general tradeoff; attributing
cause requires ablations (e.g., retrieval thresholds, validation, model choice).

---

## Per-Item Coverage Patterns

Not all PHQ-8 items are created equal. Some are discussed more often in interviews:

| Item | Typical Coverage Pattern | Why |
|------|--------------|-----|
| Depressed | High | Often directly discussed |
| Sleep | High | Common topic, clear evidence |
| Appetite | Low | Often not discussed explicitly |
| Moving | Low | Hard to infer from text alone (psychomotor change) |

The paper confirms this:
> "PHQ-8-Appetite had no successfully retrieved reference chunks"

Note: this quote is about **few-shot reference retrieval** (no retrieved reference chunks),
not “coverage” directly.

And:
> "For symptoms such as poor appetite and moving slowly, MAE performance was highly variable due to substantially fewer subjects with available scores"

---

## What's Better: High or Low Coverage?

**It depends on your goal:**

### High Coverage is Better If:
- You want to assess as many symptoms as possible
- You're willing to accept some error on harder items
- Clinical utility matters (a partial assessment is better than no assessment)

### Low Coverage is Better If:
- You only want high-confidence predictions
- You prefer to say "I don't know" rather than risk being wrong
- You're measuring MAE and want it to look good

### Our Approach

We prioritize **pure LLM measurement**—the system only scores items where the LLM found sufficient evidence. This matches the paper's methodology and provides a clean measure of model capability.

---

## How Coverage Affects MAE Calculation

**MAE (Mean Absolute Error)** is only calculated on items that have scores.

### Example

| Item | Ground Truth | Prediction | Error |
|------|--------------|------------|-------|
| No Interest | 2 | 1 | 1 |
| Depressed | 1 | 2 | 1 |
| Sleep | 1 | 1 | 0 |
| Tired | 2 | N/A | (excluded) |
| Appetite | 0 | N/A | (excluded) |
| Failure | 0 | 0 | 0 |
| Concentrating | 1 | N/A | (excluded) |
| Moving | 0 | N/A | (excluded) |

**MAE = (1 + 1 + 0 + 0) / 4 = 0.5**

Note: Only 4 items counted because 4 were N/A.

### The Trick

If the system skips hard items (where it would have made errors) and only predicts easy items (where it's accurate), MAE looks artificially good.

---

## What Drives Coverage in Our System?

### 1. Evidence Extraction
The LLM reads the transcript and extracts quotes for each PHQ-8 item. If it finds quotes, it can make a prediction.

### 2. Model Confidence
The LLM decides when to say "N/A". Some models are more conservative than others.

### 3. Transcript Richness
Longer, more detailed interviews → more evidence → higher coverage.

---

## Why Our Coverage May Differ from the Paper

Paper Section 3.2 explicitly notes that **subjects without sufficient evidence were
excluded**, and that in **~50% of cases** the model was unable to provide a prediction
due to insufficient evidence.

Plausible contributors to coverage differences include:

1. Prompt wording and parsing behavior differences
2. Model weights and quantization differences (paper does not specify quantization)
3. Backend/runtime differences (Ollama vs HuggingFace)

---

## Summary

| Concept | Definition |
|---------|------------|
| **Coverage** | % of PHQ-8 items that got scores (not N/A) |
| **N/A** | Item not scored due to insufficient evidence |
| **Tradeoff** | Higher coverage → more items → may include harder predictions |
| **MAE impact** | Only scored items count; N/A items are excluded |

**Key takeaway**: Coverage and MAE must be interpreted together. A system with 0.619 MAE at ~50% abstention is not directly comparable to a system with ~0.78 MAE at ~69% coverage—they’re making different tradeoffs.

---

## Related Documentation

- [Clinical Understanding](../clinical/clinical-understanding.md) - How the system works
- [Reproduction Results](../results/reproduction-results.md) - Historical run notes
- [Agent Sampling Registry](../configs/agent-sampling-registry.md) - Sampling parameters (paper leaves some unspecified)
- [Metrics and Evaluation](metrics-and-evaluation.md) - Exact metric definitions + output schema
