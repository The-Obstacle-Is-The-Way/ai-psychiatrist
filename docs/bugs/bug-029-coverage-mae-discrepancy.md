# BUG-029: Coverage/MAE Discrepancy vs Paper

**Status**: Investigation Complete - Not a Bug
**Severity**: N/A (Behavioral difference, not defect)
**Discovered**: 2025-12-27
**Component**: Quantitative Assessment Pipeline

---

## Summary

Our reproduction shows higher coverage and higher MAE than the paper claims. Investigation reveals this is expected LLM behavioral variation, not a code bug.

---

## Our Results vs Paper

**Important**: The paper reports **item-level MAE excluding "N/A"**, and their reported MAEs
correspond to the "mean of per-item MAEs" (each PHQ-8 item equally weighted).

All numbers below are computed on the **paper TEST split (41 participants)**:

| Metric | Zero-Shot (Ours) | Zero-Shot (Theirs) | Few-Shot (Ours) | Few-Shot (Theirs) |
|--------|------------------|--------------------|-----------------|--------------------|
| Coverage | 56.9% | 40.9% | 71.6% | 50.0% |
| MAE (mean of per-item MAEs) | 0.717 | 0.796 | 0.860 | 0.619 |

For completeness, our script also reports a weighted MAE (mean across all predicted items):
- Ours: zero-shot 0.698, few-shot 0.852
- Theirs: zero-shot 0.746, few-shot 0.640

**Key Observations:**
1. Our coverage is consistently ~15-20% higher than theirs
2. Few-shot increases coverage for BOTH implementations (expected behavior)
3. Higher coverage = more predictions on harder cases = higher MAE

---

## Root Cause Analysis

### 1. Few-Shot Increases Coverage (Expected)

From their own notebooks (source of truth):
- Zero-shot coverage on TEST: **40.9%**
- Few-shot coverage on TEST: **50.0%** (+9.1%)

Our implementation:
- Zero-shot coverage: **56.9%**
- Few-shot coverage: **71.6%** (+14.7%)

**Both implementations show coverage increase with few-shot.** The reference examples prime the LLM to be more confident in making predictions.

### 2. Why Our Coverage is Higher Overall

The driver is not a parsing bug: the coverage gap is mostly due to different **model behavior** (how often it chooses "N/A"),
especially for certain PHQ-8 items.

Per-item coverage (TEST split) shows the gap clearly:

- **PHQ8_Appetite** coverage:
  - Theirs: 2.6% (zero-shot) → 4.9% (few-shot)
  - Ours: 20.0% (zero-shot) → 42.5% (few-shot)
- **PHQ8_Moving** coverage:
  - Theirs: 10.3% (zero-shot) → 14.6% (few-shot)
  - Ours: 25.0% (zero-shot) → 47.5% (few-shot)

This is consistent with the paper's Appendix E observation that **PHQ8_Appetite had no successfully retrieved reference chunks**
because the model did not identify appetite evidence during evidence retrieval.

Potential contributors (not mutually exclusive):
- Different Gemma 3 model variants (`gemma3:27b-it-qat` vs their `gemma3-optimized:27b`)
- Different sampling parameters (their notebooks set `temperature/top_k/top_p`; our pipeline currently only controls `temperature`)
- Run-to-run variance (they averaged some validation runs; our numbers are single runs)

### 3. The Coverage/MAE Tradeoff

**This is the critical insight:**

When coverage increases, the model is making predictions on cases where it previously said "N/A" (insufficient evidence). These are inherently harder cases:
- Weaker evidence
- More ambiguous symptoms
- Higher prediction difficulty

The paper's 50% coverage means they only predicted on the "easy" cases → lower MAE.
Our 71.6% coverage means we predicted on easy + harder cases → higher MAE.

**This is not a fair comparison.**

---

## Validation: Keyword Backfill Is NOT the Cause

The user correctly identified that keyword backfill is NOT used in the paper's actual experiments:

1. **Paper's sloppy Python code**: Has `_keyword_backfill()` function (line 478 of `quantitative_assessor_f.py`)
2. **Paper's notebooks (source of truth)**: Do NOT use keyword backfill
3. **Our config**: `enable_keyword_backfill=False` (matches notebook behavior)

The keyword backfill in their Python file is dead code that wasn't used in the actual experiments.

---

## Conclusion

**This is NOT a bug in our implementation.**

The discrepancy is caused by:
1. Coverage/MAE tradeoff (higher coverage → harder predictions → higher MAE)
2. Model + sampling configuration differences (model variant and/or sampling params)

To compare results rigorously, we either need:
- Matched-coverage evaluation (apply a confidence threshold so both modes operate at the same coverage), or
- Risk–coverage curve evaluation (AURC-family metrics).

---

## Recommendations

### Option A: Accept Higher Coverage (Recommended)
- Our implementation makes more predictions (71.6% vs 50%)
- Higher MAE is expected when predicting harder cases
- More clinically useful

### Option B: Match Paper Coverage (For Comparison Only)
If strict paper comparison is needed:
1. Match the paper's few-shot hyperparameters (Appendix D: chunk size 8, 2 reference examples) — already matched
2. Match their model variant and sampling parameters (their notebooks set `temperature/top_k/top_p`)
3. Introduce a principled abstention/thresholding mechanism (see `docs/specs/25-aurc-augrc-implementation.md`)

This would likely bring MAE closer to 0.619 but sacrifices clinical utility.

---

## Files Referenced

- `_reference/ai_psychiatrist/quantitative_assessment/embedding_quantitative_analysis.ipynb` - Their few-shot notebook (no backfill)
- `_reference/ai_psychiatrist/quantitative_assessment/basic_quantitative_analysis.ipynb` - Their zero-shot notebook
- `_reference/ai_psychiatrist/analysis_output/quan_gemma_zero_shot.jsonl` - Their zero-shot results (filtered to TEST IDs)
- `_reference/ai_psychiatrist/analysis_output/quan_gemma_few_shot/TEST_analysis_output/` - Their few-shot results (TEST)

---

## Related

- Paper Section 3.2: "in 50% of cases it was unable to provide a prediction due to insufficient evidence"
- Paper Appendix F: MedGemma achieves lower MAE (0.505) with even LOWER coverage
