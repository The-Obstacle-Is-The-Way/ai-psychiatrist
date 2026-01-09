# Why Few-Shot May Not Beat Zero-Shot: Analysis

**Created**: 2026-01-05
**Last Updated**: 2026-01-08
**Status**: Living document; updated with Run 13 baseline + Run 14 severity inference ablation

---

> **BUG-035 CONTEXT (Fixed 2026-01-06)**: Runs 1–12 are confounded because few-shot prompts could
> still differ from zero-shot when retrieval returned zero references (a sentinel wrapper containing
> “No valid evidence found”). Run 13 is the first clean post-fix comparative run; use it for any
> zero-shot vs few-shot claims.
> See [BUG-035](../_archive/bugs/BUG-035_FEW_SHOT_PROMPT_CONFOUND.md) and `docs/results/run-history.md`.

---

## Executive Summary

Run 13 (the first clean post-BUG-035 comparative run) confirms that **zero-shot outperforms few-shot**
on MAE_item (0.6079 vs 0.6571) at similar coverage (~50.0% vs ~48.5%). Few-shot underperformance is
therefore attributable to retrieval/reference quality issues and evidence bottlenecks, not prompt
confounding. This document explains why few-shot can be neutral or harmful from first principles.

Run 12 shows the same directional pattern, but is pre-fix and should be treated as historical context.

---

## Run 13 Results (Post BUG-035; Clean Comparative Baseline)

From `data/outputs/both_paper-test_20260107_134730.json` (Run 13):

| Mode | N_eval | MAE_item | Coverage |
|------|--------|----------|----------|
| Zero-shot | 40/41 | **0.6079** | 50.0% |
| Few-shot | 41/41 | 0.6571 | 48.5% |

Key confirmation: zero-shot still beats few-shot after the confound fix, so retrieval quality (not prompt contamination) is the bottleneck.

---

## Run 14: Severity Inference (`infer`) Ablation (Spec 063)

From `data/outputs/both_paper-test_20260108_114058.json` (Run 14; `--severity-inference infer`):

| Mode | N_eval | MAE_item | Coverage |
|------|--------|----------|----------|
| Zero-shot | 41/41 | 0.7030 | 60.1% |
| Few-shot | 40/41 | 0.7843 | 57.5% |

Selective prediction (abs_norm, 1,000 bootstrap resamples):
- Zero-shot best AURC/AUGRC: `hybrid_consistency` (AURC 0.1258, AUGRC 0.0377)
- Few-shot best AURC: `consistency_inverse_std` (AURC 0.1391)
- Few-shot best AUGRC: `token_energy` (AUGRC 0.0394)

**Interpretation**:
- Severity inference increases coverage substantially vs Run 13 (~+8–11 points Cmax), but **degrades AURC/AUGRC** relative to the strict baseline.
- Few-shot still does not outperform zero-shot under `infer`; the evidence/reference bottleneck remains, and additional “inferred” predictions appear to be harder cases the strict prompt would have abstained on.

---

## Historical Context: Run 12 (Pre BUG-035 Fix; Confounded)

| Mode | N_eval | MAE_item | Coverage | AURC | AUGRC |
|------|--------|----------|----------|------|-------|
| Zero-shot | 41/41 | **0.5715** | 48.5% | 0.1019 | 0.0252 |
| Few-shot | 41/41 | 0.6159 | 46.0% | 0.1085 | 0.0242 |

**Paired comparison** (few − zero, default confidence):
- ΔAURC = +0.0066, 95% CI: [-0.014, +0.026] — includes zero
- ΔAUGRC = -0.001, 95% CI: [-0.007, +0.004] — includes zero

**Neither difference is statistically significant.**

---

## Core Reason: Evidence-Limited, Not Knowledge-Limited

PHQ-8 scoring requires **frequency over 2 weeks** (0-3 scale based on how many days). DAIC-WOZ transcripts often don't state frequency explicitly.

**The bottleneck is "missing evidence", not "missing rubric knowledge".**

- Few-shot can't add evidence to the test transcript
- It only changes how the model interprets what's already there
- If the transcript doesn't support a frequency claim, the best behavior is still N/A

---

## Why Few-Shot May Not Improve (Or Can Hurt)

### 1. Embedding Similarity ≠ Severity Similarity

"Sleep problems" matches "sleep problems", but not necessarily score 0 vs 3. Retrieved examples can be semantically on-topic but severity-mismatched.

### 2. Reference-Label Noise

Even with chunk-level scoring (Spec 35), chunk scores can be wrong or underspecified. One misleading exemplar can anchor the model to the wrong score.

### 3. Anchoring Dominates Evidence

Showing explicit scores in references can cause the scorer to overweight them versus the test participant's actual quotes.

### 4. Reference Bundle Too Small / Narrow

After guardrails + truncation, many items get 0-1 exemplars. The "few-shot" is brittle and can be worse than none.

**Run 12 data**: Only 15.2% of item assessments (50 of 328) had any references retrieved; those 50 items received 52 references total.

### 5. Guardrails Reduce Effective Retrieval

Item-tag filtering + similarity thresholds can make references sparse/uneven. Some items get (possibly noisy) references; others get none.

### 6. Prompt Interference / Cognitive Load

More instructions + more context can reduce attention on the primary evidence, especially when references are not perfectly aligned.

### 7. Coverage Can Legitimately Shift

Even with identical evidence extraction, the scorer sees different context in few-shot and can become more conservative (or overconfident). Similar coverage is reassuring; different coverage isn't automatically a bug.

### 8. Ceiling/Headroom

If zero-shot is already strong (e.g., consistency sampling + better prompting), few-shot has little room to help and more ways to hurt.

### 9. Small-N + Heterogeneity

With 41 participants, true effects can be small relative to variance. You can see "no clear win" even if few-shot helps a subset and hurts another subset.

### 10. RAG Helps Only for Specific Failure Modes

Few-shot helps most when the model is miscalibrated about what a 1 vs 2 vs 3 looks like for a symptom. If your main failure mode is "no frequency evidence", RAG won't move the needle.

---

## Run 8 vs Run 12: What Changed?

| Factor | Run 8 (few-shot won) | Run 12 (zero-shot won) |
|--------|---------------------|------------------------|
| Evidence grounding | Not enabled | Enabled (Spec 053) |
| Items with LLM evidence | 61.3% | **32.0%** |
| Items with references | Higher (inferred) | **15.2%** |
| Consistency sampling | No | Yes (n=5, temp=0.2) |

**Key finding**: Evidence grounding (Spec 053) rejects ~60% of extracted quotes in recent runs (e.g., Run 13/14; substring match fails). This is often a quote-extraction fidelity problem (near-miss/paraphrase/formatting), not necessarily a true hallucination, but it still breaks the few-shot retrieval chain:

1. Less evidence → fewer embedding queries
2. Fewer queries → fewer references retrieved
3. Fewer references → few-shot operates like zero-shot
4. Consistency sampling improves zero-shot dramatically (26% MAE reduction)
5. Consistency sampling doesn't help few-shot (already like zero-shot)

---

## Per-Item Analysis (Run 8 → Run 12)

### Zero-shot: Improved across the board

| Item | Run 8 MAE | Run 12 MAE | Change |
|------|-----------|------------|--------|
| NoInterest | 0.632 | 0.444 | -0.187 ↓ |
| Tired | 0.960 | 0.800 | -0.160 ↓ |
| Appetite | 1.200 | 0.333 | -0.867 ↓ |
| Concentrating | 0.824 | 0.588 | -0.235 ↓ |

### Few-shot: Mixed results

| Item | Run 8 MAE | Run 12 MAE | Change |
|------|-----------|------------|--------|
| NoInterest | 0.571 | 0.462 | -0.110 ↓ |
| **Tired** | 0.885 | 0.957 | +0.072 ↑ |
| **Appetite** | 0.000 | 0.333 | +0.333 ↑ |
| **Failure** | 0.750 | 0.833 | +0.083 ↑ |
| **Moving** | 0.500 | 0.571 | +0.071 ↑ |

**Consistency sampling improved zero-shot universally but had mixed effects on few-shot.**

---

## Evidence from Optimal Metrics

The selective prediction metrics files include `aurc_optimal` and `augrc_optimal` (the best achievable AURC/AUGRC with oracle confidence). Comparing actual vs optimal:

| Mode | AURC | AURC_optimal | e-AURC (gap) |
|------|------|--------------|--------------|
| Zero-shot | 0.102 | 0.033 | 0.069 (212%) |
| Few-shot | 0.109 | 0.038 | 0.071 (189%) |

**Key insight**: Few-shot's optimal AURC is *worse* than zero-shot's optimal (0.038 vs 0.033). This means even with perfect confidence estimation, few-shot predictions are inherently less accurate than zero-shot predictions in Run 12.

This points to **retrieval/reference noise + anchoring** as the primary cause, not a confidence-signal artifact.

---

## What Would Validate "Few-Shot Truly Helps"

To claim few-shot is genuinely beneficial, we would need:

1. **Item-level deltas**: Some items improve consistently (e.g., sleep/energy) while others don't
2. **Retrieval sanity**: Retrieved reference scores correlate with test ground truth conditional on item (not just semantic similarity)
3. **Ablations**:
   - References without scores (tests anchoring harm)
   - Random references vs retrieved references (tests whether retrieval does anything)
   - More exemplars per item (tests "k too small")
   - Reranking (tests "similarity ≠ severity")

---

## Implications for Future Work

### Option A: Accept Current State

- Evidence grounding is methodologically correct (prevents hallucination contamination)
- Few-shot with strict grounding ≈ zero-shot
- Zero-shot + consistency is the recommended approach

### Option B: Tune Evidence Grounding

- Try `QUANTITATIVE_EVIDENCE_QUOTE_VALIDATION_MODE="fuzzy"` with lower threshold
- Accept more quotes, risk more hallucination contamination
- Re-run to see if few-shot recovers

### Option C: Improve Reference Quality

- Better chunk scoring (higher-quality labels)
- Rerank by severity similarity, not just semantic similarity
- Increase reference count per item (currently sparse after filtering)

### Option D: Disable Evidence Grounding for Comparison

- `QUANTITATIVE_EVIDENCE_QUOTE_VALIDATION_ENABLED=false`
- Replicate Run 8 conditions
- Use only for research comparison, not production

---

## Summary

**The codebase is correct.** The finding that few-shot ≈ zero-shot (or worse) is a valid research result, not a bug. It reflects:

1. **Evidence grounding** correctly rejecting hallucinated quotes, but starving few-shot of data
2. **Consistency sampling** benefiting zero-shot more than few-shot
3. **Fundamental limitations** of RAG for this task (evidence-limited, not knowledge-limited)

For DAIC-WOZ PHQ-8 scoring with strict evidence grounding, **zero-shot with consistency sampling is the recommended approach**.

---

## Related Documentation

- [RAG Design Rationale](../rag/design-rationale.md) — Participant-level score problem, zero-shot inflation hypothesis
- [Chunk-level Scoring](../rag/chunk-scoring.md) — Spec 35 implementation
- [Configuration](../configs/configuration.md) — Evidence grounding settings
- [Run History](run-history.md) — Full run details
