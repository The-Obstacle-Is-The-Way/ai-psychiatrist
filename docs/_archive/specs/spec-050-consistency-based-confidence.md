# Spec 050: Consistency-Based Confidence (Multi-Sample)

**Status**: Implemented (2026-01-03)
**Priority**: Medium (alternative to verbalized confidence)
**Depends on**: None (standalone)
**Estimated effort**: Medium-High
**Research basis**: [CoCoA (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737), [Semantic Entropy (2024)](https://arxiv.org/abs/2302.09664)

## 0. Problem Statement

LLMs exhibit **aleatoric uncertainty** (irreducible randomness) and **epistemic uncertainty** (model uncertainty). When an LLM is uncertain about a prediction, running multiple inference passes with `temperature > 0` will yield **inconsistent outputs**.

**Hypothesis**: Predictions with high consistency across samples are more likely to be correct.

This approach is validated by:
- [CoCoA (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737): "Confidence-Consistency Aggregation yields best overall reliability"
- [Semantic Entropy (2024)](https://arxiv.org/abs/2302.09664): Clustering semantically equivalent outputs captures uncertainty better than token-level entropy
- [LM-Polygraph Benchmark](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737): Consistency-based methods among top performers

### Trade-off

**Cost**: N inference passes per item (N typically 3-10)
**Benefit**: Strong uncertainty signal independent of LLM self-assessment

## 1. Goals / Non-Goals

### 1.1 Goals

- Implement **multi-sample inference** for quantitative assessment
- Compute **consistency metrics** across samples:
  - Score agreement (exact match rate)
  - Score variance (standard deviation)
  - Modal score (most common prediction)
  - Modal confidence (frequency of modal score)
- Persist consistency signals in run artifacts
- Add **consistency-based confidence variants** in evaluation
- Support configurable sample count (N) and temperature

### 1.2 Non-Goals

- Semantic clustering of evidence (would require NLP pipeline)
- Monte Carlo Dropout (requires model modifications, not applicable to Ollama)
- Changing the single-pass inference mode (consistency mode is opt-in)

## 2. Proposed Solution

### 2.1 Multi-Sample Inference Mode

Add `--consistency-samples` flag to `scripts/reproduce_results.py`:

```bash
# Standard inference (N=1)
uv run python scripts/reproduce_results.py --split paper-test

# Consistency mode (N=5)
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --consistency-samples 5 \
  --temperature 0.3
```

**Implementation in `QuantitativeAssessmentAgent`:**

```python
def assess_with_consistency(
    self,
    transcript: str,
    qualitative: QualitativeAssessment,
    *,
    n_samples: int = 5,
    temperature: float = 0.3,
) -> PHQ8Assessment:
    """Run multiple inference passes and aggregate."""
    samples = []
    for _ in range(n_samples):
        # Each call uses temperature > 0 for diversity
        sample = self._assess_single(transcript, qualitative, temperature=temperature)
        samples.append(sample)

    # Aggregate across samples
    return self._aggregate_samples(samples)
```

### 2.2 Consistency Metrics

For each PHQ-8 item, compute:

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `consistency_modal_score` | Mode of predicted scores | 0-3 or None | Most common prediction |
| `consistency_modal_count` | Count of modal score | 1-N | How many samples agreed |
| `consistency_modal_confidence` | modal_count / N | 0-1 | Confidence from agreement |
| `consistency_score_std` | Std of scores | 0-1.5 | Lower = more consistent |
| `consistency_na_rate` | Fraction of N/A predictions | 0-1 | High = uncertain |

**Final prediction selection:**
- If `consistency_modal_confidence >= 0.5`: Use modal score
- Else: Use single-sample fallback or N/A

### 2.3 Run Artifact Schema Extension

```json
{
  "item_signals": {
    "Sleep": {
      "llm_evidence_count": 2,
      "retrieval_similarity_mean": 0.82,
      "verbalized_confidence": 4,
      "consistency_modal_score": 2,
      "consistency_modal_count": 4,
      "consistency_modal_confidence": 0.8,
      "consistency_score_std": 0.45,
      "consistency_na_rate": 0.0,
      "consistency_samples": [2, 2, 2, 2, 1]
    }
  },
  "provenance": {
    "consistency_enabled": true,
    "consistency_n_samples": 5,
    "consistency_temperature": 0.3
  }
}
```

### 2.4 New Confidence Variants

Add to `scripts/evaluate_selective_prediction.py`:

```python
CONFIDENCE_VARIANTS = {
    # Existing...

    # NEW (Spec 050)
    "consistency",           # = consistency_modal_confidence
    "consistency_inverse_std",  # = 1 / (1 + consistency_score_std)
    "hybrid_consistency",    # Combine consistency with other signals
}
```

**Formula for `consistency_inverse_std`:**
```text
confidence = 1 / (1 + consistency_score_std)
```

**Formula for `hybrid_consistency`:**
```text
c = consistency_modal_confidence
e = min(llm_evidence_count, 3) / 3
s = retrieval_similarity_mean or 0.0

confidence = 0.4 * c + 0.3 * e + 0.3 * s
```

### 2.5 Configuration

Environment variables / `.env`:

```bash
# Consistency mode (default: disabled)
CONSISTENCY_ENABLED=false
CONSISTENCY_N_SAMPLES=5
CONSISTENCY_TEMPERATURE=0.3
```

## 3. Implementation Plan

### Phase 1: Multi-Sample Infrastructure

1. Add `--consistency-samples` and `--temperature` flags to `reproduce_results.py`
2. Implement `assess_with_consistency()` in `QuantitativeAssessmentAgent`
3. Implement sample aggregation logic
4. Add `ConsistencyMetrics` dataclass

### Phase 2: Run Artifact Changes

5. Extend `item_signals` schema with consistency fields
6. Persist `consistency_samples` array for debugging
7. Add provenance fields for consistency configuration

### Phase 3: Evaluation Support

8. Add `consistency`, `consistency_inverse_std`, `hybrid_consistency` variants
9. Handle missing consistency signals gracefully (error if variant requested but missing)

## 4. Test Plan

### 4.1 Unit Tests

- `test_consistency_aggregation`: Modal score computation
- `test_consistency_metrics`: Std, NA rate calculations
- `test_consistency_edge_cases`: All N/A, single sample, tie-breaking

### 4.2 Integration Tests

- Mock LLM with deterministic outputs, verify aggregation
- Mock LLM with varying outputs, verify consistency metrics

### 4.3 Ablation Run

```bash
# Generate consistency run
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --consistency-samples 5 \
  --temperature 0.3 \
  2>&1 | tee data/outputs/run_consistency.log

# Evaluate
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run_consistency.json \
  --mode few_shot \
  --confidence consistency

uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run_consistency.json \
  --mode few_shot \
  --confidence hybrid_consistency
```

## 5. Expected Outcomes

Based on CoCoA and semantic entropy literature:

| Confidence Signal | Expected AUGRC | vs Baseline |
|-------------------|----------------|-------------|
| `llm` (baseline) | 0.031 | — |
| `consistency` | ~0.025 | -19% |
| `consistency_inverse_std` | ~0.024 | -23% |
| `hybrid_consistency` | ~0.020 | **-35%** |

**Trade-off**: 5x inference time for potential 20-35% AUGRC improvement

## 6. Acceptance Criteria

- [ ] `--consistency-samples` flag works in `reproduce_results.py`
- [ ] Consistency metrics computed and persisted in run artifacts
- [ ] `evaluate_selective_prediction.py` supports consistency variants
- [ ] Provenance tracks consistency configuration
- [ ] Documentation in `docs/statistics/metrics-and-evaluation.md`
- [ ] Tests pass: `make ci`

## 7. Performance Considerations

### 7.1 Inference Time

| N Samples | Estimated Time (paper-test) | vs Single-Pass |
|-----------|----------------------------|----------------|
| 1 | ~2 hours | 1x |
| 3 | ~6 hours | 3x |
| 5 | ~10 hours | 5x |
| 10 | ~20 hours | 10x |

**Recommendation**: Use N=5 for evaluation runs, N=3 for development.

### 7.2 Parallelization

Samples for the same participant can be parallelized if Ollama has sufficient GPU memory. Consider adding `--consistency-parallel` flag.

### 7.3 Caching

Cache intermediate samples to allow resumption if run is interrupted.

## 8. File Changes

### New Files

- `src/ai_psychiatrist/domain/value_objects.py` (extend with `ConsistencyMetrics`)
- `tests/unit/agents/test_consistency.py`

### Modified Files

- `scripts/reproduce_results.py` (add consistency flags)
- `src/ai_psychiatrist/agents/quantitative.py` (add `assess_with_consistency`)
- `scripts/evaluate_selective_prediction.py` (add consistency variants)
- `src/ai_psychiatrist/config.py` (add consistency settings)

## 9. References

- [CoCoA (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737)
- [Semantic Entropy (2024)](https://arxiv.org/abs/2302.09664)
- [LM-Polygraph Benchmark](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737)
- [fd-shifts MCD methods](../../_reference/fd-shifts/fd_shifts/analysis/confid_scores.py)
