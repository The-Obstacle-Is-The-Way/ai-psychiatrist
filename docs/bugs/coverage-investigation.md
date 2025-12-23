# Coverage Investigation: Why 74% vs Paper's 50%

**Date**: 2025-12-23
**Status**: SPEC'D - Resolution in [SPEC-003](../specs/SPEC-003-backfill-toggle.md)
**GitHub Issue**: [#49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
**Severity**: LOW - Coverage tradeoff, not necessarily a bug

---

## Summary

An example reproduction run achieved **~74% coverage** vs the paper’s report that
**in ~50% of cases** the model could not provide a prediction due to insufficient
evidence (Section 3.2).

Hypothesis: coverage is higher in this repository partly because it includes a
rule-based keyword backfill step that can add evidence when the initial LLM extraction
misses it.

---

## Root Cause: Keyword Backfill

### What It Does

When the LLM fails to extract evidence for a PHQ-8 item, we search the transcript for keywords related to that symptom and add matching sentences as evidence.

**Code location**: `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._keyword_backfill`)

```python
def _keyword_backfill(
    self,
    transcript: str,
    current: dict[str, list[str]],
    cap: int = 3,
) -> dict[str, list[str]]:
    """Add keyword-matched sentences when LLM misses evidence."""
    parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
    sentences = [p.strip() for p in parts if p and len(p.strip()) > 0]

    out = {k: list(v) for k, v in current.items()}

    for key, keywords in DOMAIN_KEYWORDS.items():
        need = max(0, cap - len(out.get(key, [])))
        if need == 0:
            continue

        hits: list[str] = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                hits.append(sent)
            if len(hits) >= need:
                break
        if hits:
            existing = set(out.get(key, []))
            merged = out.get(key, []) + [h for h in hits if h not in existing]
            out[key] = merged[:cap]

    return out
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

**Unknown.** The paper does not explicitly describe this mechanism.

Paper Section 2.3.2 only says:
> "extracts symptom-related evidence directly from the text"

No mention of fallback keyword matching.

If paper used pure LLM extraction without backfill:
- More extraction failures → more N/A → lower coverage
- Explains the 50% vs 74% difference

---

## Item-by-Item Coverage Comparison

| Item | Example Run Coverage | Paper Notes |
|------|--------------|-------------|
| Depressed | 100% | Always discussed |
| Sleep | 98% | Common topic |
| Failure | 95% | Clear evidence |
| NoInterest | 88% | Usually discussed |
| Tired | 83% | Common complaint |
| Concentrating | 51% | Less often discussed |
| Moving | 44% | Hard to detect from text |
| Appetite | 34% | Rarely discussed |

**Paper confirms** (Appendix E):
> "PHQ-8-Appetite had no successfully retrieved reference chunks during inference"

Note: The paper statement about appetite refers to retrieval (“no successfully retrieved reference
chunks”), not coverage directly. Attribution of our higher appetite coverage to keyword backfill is
plausible but unproven without an ablation run.

---

## Should We Change This?

### Option 1: Keep As-Is (Recommended)
- Higher coverage = more clinical utility
- MAE difference is expected given tradeoff
- Document the difference

### Option 2: Disable Keyword Backfill
- Would lower coverage toward paper's 50%
- Would likely improve MAE (fewer hard predictions)
- Reduces clinical utility

### Option 3: Make Configurable
- Add flag: `--no-keyword-backfill`
- Allow users to choose their coverage/accuracy tradeoff

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

1. **Adds config flag**: `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` to disable backfill for paper parity
2. **Tracks N/A reasons**: Understand why items return N/A
3. **Maintains backwards compatibility**: Default behavior unchanged

After implementation, users can:
- Run `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` to match paper methodology
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
