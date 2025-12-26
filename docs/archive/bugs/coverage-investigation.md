# Coverage Investigation: Why 74% vs Paper's 50%

**Date**: 2025-12-23
**Status**: RESOLVED - Implemented in [SPEC-003](../specs/SPEC-003-backfill-toggle.md)
**GitHub Issue**: [#49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
**Severity**: LOW - Coverage tradeoff, not necessarily a bug

---

## Summary

Historical reproduction runs in this repo have shown higher item-level coverage than the
paper’s reported abstention rate (Section 3.2: “in ~50% of cases it was unable to provide a
prediction due to insufficient evidence”).

Initial hypothesis (still plausible, but incomplete): coverage can be higher here because we
implemented an optional, rule-based **keyword backfill** step that can add evidence when the
initial LLM evidence extraction misses it.

**Update (SSOT, 2025-12-24):** backfill is **not the whole story**. We observed **69.2% coverage
with backfill OFF** (paper-text parity), so other differences (prompt formatting, parsing,
model/runtime/quantization, evaluation denominator) also contribute. See:
- `docs/bugs/analysis-027-paper-implementation-comparison.md`
- `docs/bugs/investigation-026-reproduction-mae-divergence.md`

---

## Root Cause: Keyword Backfill

### What It Does

When the LLM fails to extract evidence for a PHQ-8 item, we search the transcript for keywords related to that symptom and add matching sentences as evidence.

**Code location**: `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._find_keyword_hits`, `QuantitativeAssessmentAgent._merge_evidence`)

```python
# Keyword backfill is implemented as two helpers:
# - _find_keyword_hits(): find matching sentences per PHQ8_* key
# - _merge_evidence(): inject matches into scorer evidence (up to cap)
#
# Backfill injection is controlled by:
#   QUANTITATIVE_ENABLE_KEYWORD_BACKFILL (default: false / paper-text parity)
hits = self._find_keyword_hits(transcript, cap=cap)
enriched = self._merge_evidence(current_evidence, hits, cap=cap)
```

### Keyword List

We use a hand-curated keyword list (with collision-avoidance heuristics) at:
`src/ai_psychiatrist/resources/phq8_keywords.yaml`

Examples per domain:
- **Sleep**: "can't sleep", "insomnia", "trouble sleeping", "wake up tired"
- **Tired**: "exhausted", "no energy", "drained", "feeling tired"
- **Depressed**: "depressed", "hopeless", "crying", "feeling down"

### How It Affects Coverage

1. LLM fails to extract evidence for "Tired" domain
2. Keyword backfill finds "I feel exhausted" in transcript
3. Evidence is added → scoring is attempted
4. Score produced instead of N/A → coverage increases

---

## Is This a Bug?

**No.** This is intentional functionality that improves clinical utility.

### Arguments FOR Higher Coverage:
- More items get assessed → more clinical signal
- Keyword backfill catches evidence LLM missed
- A higher-coverage assessment can be more clinically useful than a lower-coverage one
- Even imperfect predictions provide information

### Arguments AGAINST (paper's approach):
- Only high-confidence predictions
- Prefers "I don't know" over potential errors
- Lower MAE on fewer items

---

## Does Paper Use Keyword Backfill?

### What the Paper TEXT Says

The paper text does **not** explicitly describe a keyword backfill mechanism. It describes
LLM-based evidence extraction and records missing/insufficient evidence as “N/A”.

### What the Paper REPO Does (Verified)

The public repo does include and execute keyword backfill in the few-shot agent:
- `_reference/agents/quantitative_assessor_f.py:29-38` defines `DOMAIN_KEYWORDS`
- `_reference/agents/quantitative_assessor_f.py:84-102` defines `_keyword_backfill(...)`
- `_reference/agents/quantitative_assessor_f.py:478` calls `_keyword_backfill(...)` unconditionally

This creates an ambiguity we now treat as **two parity targets**:
- **Paper-text parity** (methodology as written): backfill not described → keep backfill OFF.
- **Paper-repo parity** (match public implementation): backfill ON *and* match the repo’s keyword list/behavior.

If paper used pure LLM extraction without backfill:
- More extraction failures → more N/A → lower coverage
- Could explain part of the historical 74% (backfill ON) vs paper-text high abstention, but our observed 69.2% with backfill OFF indicates additional factors are at play (prompts, model/runtime, denominator).

---

## Item-by-Item Coverage Comparison

| Item | Historical Run Coverage (backfill ON) | Paper Notes |
|------|--------------|-------------|
| Depressed | 100% | Always discussed |
| Sleep | 98% | Common topic |
| Failure | 95% | Clear evidence |
| NoInterest | 88% | Usually discussed |
| Tired | 83% | Common complaint |
| Concentrating | 51% | Less often discussed |
| Moving | 44% | Hard to detect from text |
| Appetite | 34% | Rarely discussed |

These values come from a historical run recorded in `docs/results/reproduction-notes.md` (that run is
explicitly **invalidated** for paper-text parity because it used backfill ON).

**Paper confirms** (Appendix E):
> "PHQ-8-Appetite had no successfully retrieved reference chunks during inference"

Note: The paper statement about appetite refers to retrieval (“no successfully retrieved reference
chunks”), not coverage directly. Attribution of our higher appetite coverage to keyword backfill is
plausible but unproven without an ablation run.

---

## Should We Change This?

### Option 1: Keep As-Is (Recommended)
- Keep backfill as an opt-in feature
- Default remains **paper-text parity** (backfill OFF)
- Enable for higher coverage when clinical utility is prioritized

### Option 2: Disable Keyword Backfill
- Not applicable (already the default as of SPEC-003)
- Run with `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` (explicit) to match paper-text parity

### Option 3: Make Configurable
- ✅ Implemented in SPEC-003 via environment variable
- Allow users to choose their coverage/accuracy tradeoff:
  - `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` (default, paper-text parity)
  - `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true` (higher coverage)

---

## Conclusion

The coverage difference is:
1. **Plausibly influenced** by keyword backfill (hypothesis)
2. **Not a bug** - intentional functionality
3. **Clinically beneficial** - more items assessed
4. **Documented** - users can understand the tradeoff

Higher coverage can be a feature, but the exact tradeoff should be validated with an ablation run.

## Next Step to Confirm Root Cause

Add an ablation mode to disable keyword backfill and rerun:

1. same split
2. same model + backend
3. compare coverage/MAE deltas

---

## Resolution

This investigation led to [SPEC-003: Make Keyword Backfill Optional](../specs/SPEC-003-backfill-toggle.md), which:

1. **Adds config flag**: `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` to enable/disable keyword backfill
2. **Tracks N/A reasons**: Understand why items return N/A
3. **Defaults to paper-text parity**: Backfill is OFF by default

After implementation, users can:
- Run with defaults (or `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false`) to match paper-text parity
- Run `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true` for higher coverage
- Run ablation studies comparing backfill ON vs OFF
- Track which items benefit most from backfill

---

## Related Documentation

- [SPEC-003: Backfill Toggle](../specs/SPEC-003-backfill-toggle.md) - Implementation specification
- [Backfill Explained](../concepts/backfill-explained.md) - How backfill works
- [Paper Parity Guide](../guides/paper-parity-guide.md) - How to reproduce paper results
- [Coverage Explained](../concepts/coverage-explained.md) - Plain-language explanation
- [Reproduction Notes](../results/reproduction-notes.md) - Results and methodology
- `src/ai_psychiatrist/resources/phq8_keywords.yaml` - Keyword list
