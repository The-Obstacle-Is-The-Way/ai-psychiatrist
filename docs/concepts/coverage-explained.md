# Coverage Explained: What It Is and Why It Matters

**Audience**: Anyone trying to understand what "coverage" means in PHQ-8 assessment
**Last Updated**: 2025-12-23

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

The paper reports item-level MAE and notes that **in ~50% of cases** the model could
not provide a prediction due to insufficient evidence.

This repository also computes **item-level MAE excluding N/A**, but the exact
coverage/MAE depends on model weights/quantization, backend, and prompt behavior.

| Metric | Paper | Example Run (paper-style split, few-shot) |
|--------|-------|--------------------------------------|
| Coverage | ~50% cases had no prediction | 74.1% |
| Item MAE | 0.619 | 0.757 (weighted across predicted items) |

See `docs/results/reproduction-notes.md` for the concrete run details and the raw
artifact under `data/outputs/`.

Interpretation: higher coverage often increases MAE because the model attempts more
items (including harder-to-evidence symptoms). This is a general tradeoff; attributing
cause requires ablations (e.g., keyword backfill on/off).

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

We lean toward **higher coverage** because:
1. Clinically, a 74% assessment is more useful than 50%
2. Even imperfect predictions provide signal
3. Items marked N/A provide no information

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

### 2. Keyword Backfill
When LLM extraction fails, we search for keywords like "sleep", "tired", "appetite" to find relevant sentences. This INCREASES coverage.

### 3. Model Confidence
The LLM decides when to say "N/A". Some models are more conservative than others.

### 4. Transcript Richness
Longer, more detailed interviews → more evidence → higher coverage.

---

## Why Our Coverage May Differ from the Paper

Paper Section 3.2 explicitly notes that **subjects without sufficient evidence were
excluded**, and that in **~50% of cases** the model was unable to provide a prediction
due to insufficient evidence.

In this repository, one plausible contributor to higher coverage is the presence of a
rule-based keyword backfill step (`QuantitativeAssessmentAgent._keyword_backfill`) that
adds evidence when the initial LLM extraction misses it.

Other plausible contributors include:

1. Prompt wording and parsing behavior differences
2. Model weights and quantization differences (paper does not specify quantization)
3. Backend/runtime differences (Ollama vs HuggingFace)

To turn this from a hypothesis into a conclusion, we need an ablation run with keyword
backfill disabled, using the same split and model backend.

---

## Summary

| Concept | Definition |
|---------|------------|
| **Coverage** | % of PHQ-8 items that got scores (not N/A) |
| **N/A** | Item not scored due to insufficient evidence |
| **Tradeoff** | Higher coverage → more items → may include harder predictions |
| **MAE impact** | Only scored items count; N/A items are excluded |

**Key takeaway**: Coverage and MAE must be interpreted together. A system with 0.6 MAE at 50% coverage is NOT directly comparable to 0.75 MAE at 74% coverage—they're making different tradeoffs.

---

## Related Documentation

- [clinical-understanding.md](./clinical-understanding.md) - How the system works
- [reproduction-notes.md](../results/reproduction-notes.md) - Actual results and analysis
- [gap-001-paper-unspecified-parameters.md](../bugs/gap-001-paper-unspecified-parameters.md) - Why results may differ
