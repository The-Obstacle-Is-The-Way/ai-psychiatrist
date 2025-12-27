# BUG-030: Our Coverage is Significantly Higher Than Paper's

**Status**: Open - Investigation Pending
**Severity**: Medium (Potential implementation issue)
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
| Zero-Shot | 43.8% | 56.9% | +13.1% |
| Few-Shot | 50.0% | 71.6% | +21.6% |

Our coverage is **30-40% higher** (relative) than theirs.

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

- [ ] We use gemma3:27b, they use gemma3-optimized:27b
- [ ] We use temperature 0.0, they use 0.2 in notebooks
- [ ] Embedding model differences (qwen3-embedding:8b vs dengcao/Qwen3-Embedding-8B:Q8_0)

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

- `docs/bugs/bug-029-coverage-mae-discrepancy.md` - Coverage/MAE analysis
- `docs/paper-reproduction-analysis.md` - Comprehensive analysis
- `docs/specs/spec-024-aurc-metric.md` - AURC implementation spec

---

## Notes

This is a **pending investigation**, not a confirmed bug. The higher coverage could be:
- Legitimate improvement (our implementation is less conservative)
- A bug (we're including invalid predictions)
- Behavioral variance (model/temperature differences)

We need to rule out implementation bugs before concluding it's variance.
