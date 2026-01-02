# AUGRC Improvement Techniques for Few-Shot PHQ-8 Assessment

**Research Date**: January 2026
**Author**: AI Research Assistant
**Status**: Research Complete - Awaiting Implementation Decision

---

## Executive Summary

**The Problem**: In Run 8, few-shot prompting achieved better MAE_item (0.609 vs 0.776) than zero-shot, but **selective prediction quality is statistically indistinguishable** between modes under the current confidence signal (evidence counts).

**Root Cause**: Confidence is currently based on **evidence count** (number of quotes extracted by the LLM), which is independent of the few-shot retrieval system. The retrieval similarity scores (cosine similarity 0-1) are computed but **not used** as confidence signals.

**Key Insight**: Few-shot provides better examples for scoring, but the confidence mechanism ignores all retrieval-related signals.

---

## Part 1: Current System Analysis

### 1.1 How Confidence is Currently Generated

```
Transcript
  ↓
Evidence Extraction (LLM finds quotes per PHQ-8 item)
  ↓
confidence = llm_evidence_count (or llm + keyword evidence)  ← CURRENT SIGNAL
  ↓
AUGRC computed by sorting predictions by confidence
```

**File**: `src/ai_psychiatrist/agents/quantitative.py:257-259`
```python
final_items[phq_item] = ItemAssessment(
    llm_evidence_count=llm_counts.get(legacy_key, 0),
    keyword_evidence_count=keyword_added_counts.get(legacy_key, 0),
)
```

**File**: `src/ai_psychiatrist/metrics/selective_prediction.py:87-91`
```python
# Confidence = evidence count, sorted descending for risk-coverage curve
predictions.sort(key=lambda x: x[1], reverse=True)
```

### 1.2 Unused Signals in Few-Shot

The few-shot pipeline computes **retrieval similarity** but discards it after ranking:

**File**: `src/ai_psychiatrist/services/embedding.py:231-237`
```python
raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])
sim = (1.0 + raw_cos) / 2.0  # Transform to [0, 1]
```

This similarity is stored in `SimilarityMatch.similarity` but **never flows to confidence**.

### 1.3 Run 8 Results (Ground Truth)

Run 8 artifacts (SSOT):
- `docs/results/run-history.md` (run narrative)
- `data/outputs/selective_prediction_metrics_20260102T132843Z.json` (zero-shot)
- `data/outputs/selective_prediction_metrics_20260102T132902Z.json` (few-shot)
- `data/outputs/selective_prediction_metrics_20260102T132930Z_paired.json` (paired overlap N=40)

**Single-mode selective prediction metrics** (`--loss abs_norm`, confidence=`llm`):

| Mode | MAE_item | AURC_full | AUGRC_full | Cmax |
|------|----------|----------:|-----------:|-----:|
| Zero-shot | 0.776 | 0.141432 | 0.031276 | 48.78% |
| Few-shot | 0.609 | 0.124751 | 0.030575 | 50.94% |

**Paired comparison (overlap N=40, `--intersection-only`)**:
- ΔAURC_full (few − zero) = `-0.0197` (CI95 `[-0.0529, +0.0139]`)
- ΔAUGRC_full (few − zero) = `-0.0021` (CI95 `[-0.0127, +0.0079]`)

Interpretation: differences are small relative to uncertainty; **confidence ranking quality does not materially change**.

**Interpretation**:
- MAE improved 21.5% with few-shot (0.776 → 0.609)
- AUGRC is effectively unchanged (overlapping CIs; paired ΔAUGRC CI includes 0)
- Few-shot predicts better but does not meaningfully improve confidence ranking under the current signal

---

## Part 2: Candidate Techniques (Recent UQ + Selective Prediction Literature)

This section summarizes external literature that is directly relevant to improving **risk–coverage performance** (AURC/AUGRC) via better uncertainty/confidence signals in RAG-like LLM systems.

### 2.1 Retrieval-Grounded Signals (RAG-specific)

Key idea: if the answer depends on retrieved context, the *quality of retrieval* is a strong confidence proxy.

Signals that exist in this repo today (or can be added with low risk):
- mean/max retrieval similarity per PHQ-8 item (already computed in few-shot)
- reference score agreement / dispersion across retrieved examples (available when chunk scores are enabled)
- reference validator pass/fail rate (Spec 36; optional)

Relevant references:
- Google Research (ICLR 2025): “Sufficient Context” shows retrieval **relevance alone** can be misleading, and proposes combining a context-sufficiency signal with model confidence for selective generation: https://arxiv.org/abs/2411.06037
- ACL 2025: “Why UE Methods Fall Short in RAG” argues off-the-shelf UE can fail in RAG and motivates calibration functions tailored to retrieval settings: https://aclanthology.org/2025.findings-acl.852/

### 2.2 Post-Hoc Confidence Calibration (Black-box friendly)

Key idea: treat confidence estimation as its own supervised problem.
- Fit a calibrator on a held-out split (e.g., `paper-val`) mapping observable signals → probability of correctness.
- Use the calibrated score as the ranking signal for the RC curve.

Practical calibrators that are easy to ship and audit:
- logistic regression / Platt scaling
- isotonic regression (monotone, avoids “probability” assumptions)

Relevant references:
- UniCR (2025): a unified framework that fuses heterogeneous evidence (including retrieval compatibility) into a calibrated probability and applies conformal risk control for refusal: https://arxiv.org/abs/2509.01455
- “Calibrating Verbalized Probabilities for LLMs” (2024) motivates post-hoc calibration when using self-reported confidences from black-box models: https://arxiv.org/abs/2410.06707

### 2.3 Disagreement-Based Uncertainty (Ensembles / Self-Consistency)

Key idea: if multiple “independent” prompts disagree, uncertainty is higher.

Implementation options:
- multiple prompt templates (deterministic) and compare predicted scores
- multiple models (if available) and compare scores

Trade-off: higher runtime cost.

### 2.4 Conformal Risk Control (Optional, Guarantees-focused)

Key idea: calibrate a thresholding rule with finite-sample guarantees (risk or coverage). This is most useful if we want to *choose an operating point* (e.g., “risk ≤ 0.2”) rather than optimize integrated area metrics.

Relevant reference:
- UniCR (2025) explicitly uses conformal risk control for distribution-free guarantees: https://arxiv.org/abs/2509.01455

## Part 3: Recommended Improvements (Ranked by Effort/Impact)

### 3.1 Immediate: Use Retrieval Similarity as Confidence Signal

**Effort**: Low (code change only)
**Expected Impact**: Moderate AUGRC improvement
**Risk**: Low

**Current State**: Similarity computed but discarded after ranking.

**Proposed Change**: Persist per-item retrieval similarity statistics into `item_signals`, and add a new confidence variant that uses them.

```python
# Option A: Replace evidence count with average retrieval similarity
confidence = mean([match.similarity for match in top_k_matches])

# Option B: Combine evidence count + similarity (weighted)
confidence = alpha * evidence_count + (1 - alpha) * mean_similarity

# Option C: Product (joint signal)
confidence = evidence_count * mean_similarity
```

**Implementation Location**:
- `src/ai_psychiatrist/agents/quantitative.py:250-259` (store similarity in ItemAssessment)
- `scripts/evaluate_selective_prediction.py:182-192` (parse new confidence)

**Rationale (first principles)**: if the retrieved examples are semantically close to the participant’s evidence, we expect the scoring prompt to be better grounded.

### 3.2 Short-Term: Add Verbalized Confidence Extraction

**Effort**: Medium
**Expected Impact**: Moderate-High AUGRC improvement
**Risk**: Medium (LLM may be overconfident)

**Concept**: Ask the LLM to rate its own confidence (1-5 or percentage) alongside the score.

**Prompt Modification**:
```
For each PHQ-8 item, provide:
1. Score (0-3)
2. Confidence (1-5): How confident are you in this score?
3. Evidence quotes
```

**Calibration Requirement**: LLM verbalized confidence is often overconfident. Apply temperature scaling or isotonic regression on a validation set.

### 3.3 Medium-Term: Multi-Evidence Ensemble (UniCR-Style)

**Effort**: High
**Expected Impact**: High AUGRC improvement
**Risk**: Medium (requires validation data for calibration head)

**Concept**: Train a small calibration head that takes multiple signals and outputs calibrated confidence.

**Input Signals**:
1. `llm_evidence_count` (current)
2. `retrieval_similarity_mean` (from few-shot)
3. `retrieval_similarity_max` (best match quality)
4. `reference_score_variance` (disagreement among retrieved examples)
5. `llm_verbalized_confidence` (if extracted)

**Calibration**: Use paper-train split to learn weights (logistic regression or small MLP).

**Coverage**: Apply Conformal Risk Control for coverage guarantees at a target risk level.

### 3.4 Long-Term: Semantic Entropy via Multiple Sampling

**Effort**: Very High (compute cost)
**Expected Impact**: Highest AUGRC improvement
**Risk**: High (latency, cost)

**Concept**: Sample N responses from the LLM, cluster by meaning, compute entropy.

**Implementation**:
1. Set temperature > 0 for LLM scoring
2. Sample 5-10 responses per item
3. Cluster by semantic similarity (embedding-based)
4. Compute entropy over cluster distribution

**Trade-off**: 5-10x increase in LLM calls per evaluation.

**Hybrid Approach**: Apply semantic entropy only to low-confidence items (second-pass refinement).

---

## Part 4: Ablation Study Design

To validate any improvement, run the following ablations:

### Ablation A: Retrieval Similarity Confidence

| Experiment | Confidence Signal | Expected Outcome |
|------------|-------------------|------------------|
| Baseline | `llm_evidence_count` | Current AUGRC ≈ 0.031 (Run 8) |
| A1 | `retrieval_similarity_mean` | Tests whether retrieval strength is a good ranking signal |
| A2 | `hybrid_evidence_similarity` | Deterministic combined signal (see Spec 046) |
| A3 | `retrieval_similarity_max` | Tests whether “best match” is better than mean |

### Ablation B: Verbalized Confidence

| Experiment | Confidence Signal | Expected Outcome |
|------------|-------------------|------------------|
| B1 | `llm_verbalized_confidence` (raw) | May be overconfident |
| B2 | `llm_verbalized_confidence` (temp-scaled) | Calibrated version |
| B3 | `evidence_count * verbalized_confidence` | Combined signal |

### Ablation C: Multi-Signal Ensemble

| Experiment | Signals Combined | Expected Outcome |
|------------|------------------|------------------|
| C1 | evidence + similarity + verbalized | Best if properly weighted |
| C2 | + reference_score_variance | Penalize disagreeing examples |

---

## Part 5: Implementation Roadmap

### Phase 1: Quick Win (Week 1)

1. **Spec 046: Retrieval Similarity Confidence**
   - See: `docs/_specs/spec-046-selective-prediction-confidence-signals.md`
   - Add `retrieval_similarity_mean` to `ItemAssessment`
   - Update `evaluate_selective_prediction.py` to use it as confidence
   - Run ablation on paper-test split
   - Compare AUGRC vs baseline

2. **Data Collection**: Log all similarity scores in Run 9 output JSON for post-hoc analysis.

### Phase 2: Verbalized Confidence (Week 2-3)

1. **Spec (proposed): LLM Confidence Extraction**
   - Modify quantitative scoring prompt to request confidence
   - Parse confidence from LLM output
   - Apply temperature scaling calibration

2. **Calibration Data**: Use paper-train to learn temperature.

### Phase 3: Ensemble Calibration (Week 4+)

1. **Spec (proposed): Multi-Signal Calibrator**
   - Train logistic regression on [evidence, similarity, verbalized] → correctness
   - Output calibrated probability as confidence
   - Evaluate on paper-test

---

## Part 6: Key References

All links below were validated as reachable on 2026-01-02:

1. Traub et al. (2024): AUGRC proposal for selective classification evaluation: https://arxiv.org/abs/2407.01032
2. “Population AURC characterization” (2024): https://arxiv.org/abs/2410.15361
3. Vashurin et al. (TACL 2025): LM-Polygraph benchmark for LLM UQ: https://arxiv.org/abs/2406.15627
4. Farquhar et al. (Nature 2024): semantic entropy for hallucination detection: https://www.nature.com/articles/s41586-024-07421-0
5. Nguyen et al. (ACL Findings 2025): beyond semantic entropy via pairwise semantic similarity: https://aclanthology.org/2025.findings-acl.234/
6. Soudani et al. (ACL Findings 2025): UE pitfalls in RAG + calibration function: https://aclanthology.org/2025.findings-acl.852/
7. Google Research (ICLR 2025): sufficient context for RAG systems: https://arxiv.org/abs/2411.06037
8. UniCR (2025): calibrated probability + conformal risk control for refusal: https://arxiv.org/abs/2509.01455
9. QuCo-RAG (2025): corpus-grounded uncertainty signals for dynamic RAG: https://arxiv.org/abs/2512.19134
10. UQLM toolkit paper (2025): black-box/white-box/judge/ensemble UQ: https://arxiv.org/abs/2504.19254
11. Xie et al. (EMNLP 2024): Adaptive temperature scaling: https://arxiv.org/abs/2409.19817

---

## Appendix A: The Fundamental Issue Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                        CURRENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Transcript ──┬──> Evidence Extraction ──> evidence_count ───┐   │
│               │                                               │   │
│               └──> Retrieval ──> similarity (DISCARDED) ──X   │   │
│                          │                                    │   │
│                          v                                    v   │
│                    Reference Examples               Confidence    │
│                          │                          (evidence     │
│                          v                           only!)       │
│                    LLM Scoring ──> Score                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                       PROPOSED SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Transcript ──┬──> Evidence Extraction ──> evidence_count ───┐   │
│               │                                               │   │
│               └──> Retrieval ──> similarity ─────────────────┤   │
│                          │                                    │   │
│                          v                                    v   │
│                    Reference Examples            Confidence =     │
│                          │                      f(evidence,       │
│                          v                        similarity,     │
│                    LLM Scoring ──> Score          verbalized)    │
│                          │                                       │
│                          └──> verbalized_confidence ────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Expected AUGRC Improvement Estimates

We do **not** have a defensible a-priori estimate for AUGRC gains without running ablations in this codebase. Treat improvement magnitude as an empirical question.

---

## Conclusion

The AUGRC parity between zero-shot and few-shot reveals a **confidence signal gap**. Few-shot retrieval provides valuable information (similarity scores, reference agreement) that is currently discarded.

**Immediate recommendation**: Implement Spec 046 (retrieval similarity as confidence) and run an ablation. This is a low-effort, low-risk change with clear motivation from recent RAG uncertainty work (e.g., sufficient-context and RAG-specific calibration results).

**Medium-term**: Add verbalized confidence extraction and train a simple calibrator on paper-train.

The literature is clear: multi-signal ensembles with proper calibration outperform any single signal. Our system has the signals—we just need to use them.
