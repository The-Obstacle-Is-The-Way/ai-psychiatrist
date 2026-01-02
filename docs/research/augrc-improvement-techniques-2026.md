# AUGRC Improvement Techniques for Few-Shot PHQ-8 Assessment

**Research Date**: January 2026
**Author**: AI Research Assistant
**Status**: Research Complete - Awaiting Implementation Decision

---

## Executive Summary

**The Problem**: In Run 8, few-shot prompting achieved better MAE (0.609 vs 0.776) than zero-shot, but AUGRC remained identical (0.031 for both modes). This means **few-shot improves prediction accuracy but doesn't improve the model's ability to know when to abstain**.

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
confidence = len(llm_evidence_quotes)  ← CURRENT SIGNAL
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

### 1.3 Run 8 Results (The Evidence)

| Mode | MAE_item | AURC | AUGRC | Cmax |
|------|----------|------|-------|------|
| Zero-shot | 0.776 | 0.141 | **0.031** | 48.8% |
| Few-shot | 0.609 | 0.125 | **0.031** | 50.9% |

**Interpretation**:
- MAE improved 21.5% with few-shot (0.776 → 0.609)
- AUGRC is **identical** (0.031 for both)
- Few-shot predicts better but doesn't "know" which predictions to trust more

---

## Part 2: 2025-2026 State-of-the-Art Techniques

### 2.1 LM-Polygraph Framework (TACL 2025)

**Source**: [Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737)

Key findings from the benchmark:
- **Semantic Entropy** outperforms naive entropy for text generation
- **SAR (Shifting Attention to Relevance)** is consistently effective across tasks
- **Normalization as calibration**: Transform raw uncertainty → expected quality metric

**Applicable Methods**:
1. Token-level entropy (requires logit access)
2. Self-consistency sampling (multiple LLM calls, aggregate agreement)
3. Normalized confidence via binned calibration

### 2.2 Semantic Entropy (Nature 2024, ACL 2025 Extensions)

**Source**: [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)

**Concept**: Cluster responses by meaning, compute entropy over clusters. Low semantic entropy = high confidence in the meaning.

**2025 Extension - Beyond Semantic Entropy**:
- [Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity](https://aclanthology.org/2025.findings-acl.234/)
- Addresses intra-cluster spread and inter-cluster distance
- Better calibration for longer responses

**Applicability**: Would require multiple LLM samples per item (expensive but effective).

### 2.3 UniCR Framework (Risk-Controlled Refusal, 2025)

**Source**: [Trusted Uncertainty in Large Language Models: A Unified Framework for Confidence Calibration and Risk-Controlled Refusal](https://arxiv.org/html/2509.01455)

**Components**:
1. Multi-evidence uncertainty signals (retrieval quality + LLM confidence)
2. Calibration head with temperature scaling
3. Conformal Risk Control (CRC) for coverage guarantees

**Key Innovation**: Combines heterogeneous evidence into a single calibrated probability.

### 2.4 QuCo-RAG (Corpus-Based Uncertainty)

**Source**: [QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation](https://arxiv.org/html/2512.19134v1)

**Key Insight**: LLM internal confidence (logits, token probability) is poorly calibrated. Corpus statistics provide better-grounded confidence.

**For our system**: Retrieval similarity is a corpus-grounded signal that should improve calibration.

### 2.5 Sufficient Context Detection (ICLR 2025)

**Source**: [Deeper insights into retrieval augmented generation: The role of sufficient context](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)

**Key Finding**: It's possible to know when an LLM has enough context to answer correctly.

**Selective Generation**: Combines sufficient context signal + model's self-rated confidence for abstention decisions.

### 2.6 Ensemble Methods (UQLM Toolkit, 2025)

**Source**: [Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers](https://arxiv.org/html/2504.19254)

**Tunable Ensemble**: Combine multiple confidence signals (evidence count, retrieval similarity, LLM verbalized confidence) with learned weights.

**Key Result**: Tunable ensemble generally outperforms individual components.

### 2.7 Temperature Scaling / Post-Hoc Calibration

**Source**: [Calibrating Language Models With Adaptive Temperature Scaling](https://openreview.net/forum?id=BgfGqNpoMi)

**Concept**: Learn a single temperature parameter to rescale logits, minimizing calibration error on held-out data.

**Adaptive Temperature Scaling**: Predict temperature per token based on features.

---

## Part 3: Recommended Improvements (Ranked by Effort/Impact)

### 3.1 Immediate: Use Retrieval Similarity as Confidence Signal

**Effort**: Low (code change only)
**Expected Impact**: Moderate AUGRC improvement
**Risk**: Low

**Current State**: Similarity computed but discarded after ranking.

**Proposed Change**: Incorporate retrieval similarity into confidence.

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

**Rationale**: Retrieval similarity is a corpus-grounded signal (QuCo-RAG finding). High similarity = retrieved examples are truly relevant = higher confidence in the prediction.

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

**Reference**: [Calibrating Verbalized Probabilities for Large Language Models](https://arxiv.org/html/2410.06707v1)

### 3.3 Medium-Term: Multi-Evidence Ensemble (UniCR-Style)

**Effort**: High
**Expected Impact**: High AUGRC improvement
**Risk**: Medium (requires validation data for calibration head)

**Concept**: Train a small calibration head that takes multiple signals and outputs calibrated confidence.

**Input Signals**:
1. `llm_evidence_count` (current)
2. `mean_retrieval_similarity` (from few-shot)
3. `max_retrieval_similarity` (best match quality)
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
| Baseline | `evidence_count` | Current AUGRC: 0.031 |
| A1 | `mean_similarity` | May improve if similarity is well-calibrated |
| A2 | `evidence_count * mean_similarity` | Joint signal |
| A3 | `evidence_count + alpha * mean_similarity` | Weighted combination |

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

1. **Spec 40: Retrieval Similarity Confidence**
   - Add `mean_retrieval_similarity` to `ItemAssessment`
   - Update `evaluate_selective_prediction.py` to use it as confidence
   - Run ablation on paper-test split
   - Compare AUGRC vs baseline

2. **Data Collection**: Log all similarity scores in Run 9 output JSON for post-hoc analysis.

### Phase 2: Verbalized Confidence (Week 2-3)

1. **Spec 41: LLM Confidence Extraction**
   - Modify quantitative scoring prompt to request confidence
   - Parse confidence from LLM output
   - Apply temperature scaling calibration

2. **Calibration Data**: Use paper-train to learn temperature.

### Phase 3: Ensemble Calibration (Week 4+)

1. **Spec 42: Multi-Signal Calibrator**
   - Train logistic regression on [evidence, similarity, verbalized] → correctness
   - Output calibrated probability as confidence
   - Evaluate on paper-test

---

## Part 6: Key References

1. **LM-Polygraph Benchmark** (TACL 2025): [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737)
2. **Semantic Entropy** (Nature 2024): [Nature](https://www.nature.com/articles/s41586-024-07421-0)
3. **Beyond Semantic Entropy** (ACL 2025): [ACL Anthology](https://aclanthology.org/2025.findings-acl.234/)
4. **UniCR Framework** (2025): [arXiv](https://arxiv.org/abs/2509.01455)
5. **QuCo-RAG** (2025): [arXiv](https://arxiv.org/abs/2512.19134)
6. **Sufficient Context** (ICLR 2025): [Google Research Blog](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
7. **UQLM Toolkit** (2025): [arXiv](https://arxiv.org/abs/2504.19254)
8. **Adaptive Temperature Scaling** (2024): [OpenReview](https://openreview.net/forum?id=BgfGqNpoMi)
9. **Verbalized Probability Calibration** (2024): [arXiv](https://arxiv.org/abs/2410.06707)
10. **KDD 2025 UQ Survey**: [ACM DL](https://dl.acm.org/doi/10.1145/3711896.3736569)
11. **Why UE Methods Fall Short in RAG** (ACL 2025): [ACL Anthology](https://aclanthology.org/2025.findings-acl.852/) - Axiomatic analysis of UE deficiencies
12. **AUGRC Definition** (Traub et al. 2024): [arXiv](https://arxiv.org/abs/2407.01032) - Foundational metric paper
13. **AURC Population Characterization** (2024): [arXiv](https://arxiv.org/abs/2410.15361) - Theoretical foundations

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

Based on literature and our system characteristics:

| Technique | Expected AUGRC Reduction | Confidence |
|-----------|--------------------------|------------|
| Retrieval similarity alone | 10-20% | Medium |
| + Verbalized confidence | 20-40% | Medium-High |
| + Multi-signal calibration | 30-50% | High |
| + Semantic entropy | 40-60% | High (if affordable) |

**Baseline**: AUGRC = 0.031 (Run 8)

**Target**: AUGRC < 0.020 would represent a significant improvement in selective prediction quality.

---

## Conclusion

The AUGRC parity between zero-shot and few-shot reveals a **confidence signal gap**. Few-shot retrieval provides valuable information (similarity scores, reference agreement) that is currently discarded.

**Immediate recommendation**: Implement Spec 40 (retrieval similarity as confidence) and run an ablation. This is a low-effort, low-risk change with clear theoretical motivation from the 2025 literature.

**Medium-term**: Add verbalized confidence extraction and train a simple calibrator on paper-train.

The literature is clear: multi-signal ensembles with proper calibration outperform any single signal. Our system has the signals—we just need to use them.
