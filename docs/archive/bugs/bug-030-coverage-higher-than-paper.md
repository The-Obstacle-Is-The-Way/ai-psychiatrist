# BUG-030: Our Coverage is Significantly Higher Than Paper's

> **📦 ARCHIVED**: 2025-12-30
> **Resolution**: Covered by BUG-029 investigation - higher coverage is due to model behavioral variance, not a bug.
> **Action Taken**: None required - see BUG-029 for full analysis.

**Status**: ✅ CLOSED - Superseded by BUG-029
**Severity**: N/A (Not a bug)
**Discovered**: 2025-12-27
**Component**: Quantitative Assessment Pipeline

---

## Summary

Our implementation shows significantly higher coverage than the paper reports. This could indicate:
1. A bug in our implementation (including too many cases)
2. A difference in their implementation we haven't replicated
3. LLM behavioral variance (model version, temperature, etc.)

**This requires careful code audit before concluding it's just variance.**

---

## Coverage Comparison

| Method | Their Coverage | Our Coverage | Difference |
|--------|----------------|--------------|------------|
| Zero-Shot | 40.9% | 56.9% | +16.0% |
| Few-Shot | 50.0% | 71.6% | +21.6% |

Our coverage is **30-40% higher** (relative) than theirs.

**Note**: Their zero-shot coverage above is computed on the **paper TEST split (41 participants)** by filtering
`_reference/ai_psychiatrist/analysis_output/quan_gemma_zero_shot.jsonl` to the 41 test IDs. (The oft-quoted 43.8%
figure is the coverage over all 142 participants.)

---

## Questions to Investigate

### 1. Are we including cases we shouldn't?

- [ ] Check if we're counting N/A responses differently
- [ ] Verify our parsing logic for "insufficient evidence" responses
- [ ] Compare exact prompts used (ours vs their notebooks)

### 2. Are we using different thresholds?

- [ ] Check if there's an abstention threshold we're missing
- [ ] Verify confidence/evidence requirements match their implementation

### 3. Model differences?

- [ ] We use `gemma3:27b-it-qat`, they use `gemma3-optimized:27b` in notebooks/output artifacts
- [ ] Sampling parameters: their few-shot notebook sets `temperature/top_k/top_p`; our pipeline currently controls only `temperature`

### 4. Prompt differences?

- [ ] Compare our quantitative prompts to their notebook prompts verbatim
- [ ] Check for any system prompt differences

---

## Code Locations to Audit

1. `src/ai_psychiatrist/agents/quantitative.py` - Scoring logic
2. `src/ai_psychiatrist/agents/prompts/quantitative.py` - Prompts
3. Compare with `_reference/ai_psychiatrist/quantitative_assessment/embedding_quantitative_analysis.ipynb`

---

## Related

- `docs/archive/bugs/bug-029-coverage-mae-discrepancy.md` - Coverage/MAE analysis
- `docs/paper-reproduction-analysis.md` - Comprehensive analysis
- `docs/specs/25-aurc-augrc-implementation.md` - Selective prediction evaluation suite

---

## Notes

This is a **pending investigation**, not a confirmed bug. The higher coverage could be:
- Legitimate improvement (our implementation is less conservative)
- A bug (we're including invalid predictions)
- Behavioral variance (model/temperature differences)

We need to rule out implementation bugs before concluding it's variance.
