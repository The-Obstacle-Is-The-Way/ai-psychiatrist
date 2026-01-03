# Spec 048: Verbalized Confidence for AUGRC Improvement

**Status**: Implemented (2026-01-03)
**Priority**: High (next AUGRC improvement lever)
**Depends on**: Spec 046 (retrieval signals)
**Estimated effort**: Medium
**Research basis**: [LLM Uncertainty Survey 2025](https://arxiv.org/abs/2503.15850), [ICLR 2025 - Do LLMs Estimate Uncertainty Well?](https://proceedings.iclr.cc/paper_files/paper/2025/)

## 0. Problem Statement

Run 9 (Spec 046) achieved **5.4% AURC improvement** using `retrieval_similarity_mean` as a confidence signal, but **AUGRC remains at ~0.031** (target: <0.020).

Current confidence signals are **external to the LLM's reasoning**:
- Evidence count (how many quotes extracted)
- Retrieval similarity (how similar the references were)

Neither signal captures the LLM's **internal uncertainty** about its own prediction. Research shows that asking LLMs to verbalize their confidence—while imperfect—provides complementary signal that improves calibration.

### Key Research Findings

| Source | Finding |
|--------|---------|
| [LLM Uncertainty Survey 2025](https://arxiv.org/abs/2503.15850) | Verbalized confidence is overconfident (80-100% range) but still useful when calibrated |
| [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/ef472869c217bf693f2d9bbde66a6b07-Paper-Conference.pdf) | `normalized p(true)` is a reliable uncertainty method across settings |
| [CoCoA (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737) | Hybrid confidence-consistency aggregation yields best overall reliability |

**Expected improvement**: 20-40% AUGRC reduction (literature-based estimate)

## 1. Goals / Non-Goals

### 1.1 Goals

- Add **verbalized confidence** field to LLM output schema (per item)
- Persist verbalized confidence in run artifacts (`item_signals`)
- Add **new confidence variants** in `evaluate_selective_prediction.py`:
  - `verbalized`: raw verbalized confidence
  - `verbalized_calibrated`: temperature-scaled verbalized confidence
  - `hybrid_verbalized`: combination of verbalized + retrieval + evidence signals
- Provide **calibration infrastructure** to fit temperature scaling on paper-train
- Maintain backward compatibility with existing run artifacts

### 1.2 Non-Goals

- Changing the scoring logic (this spec targets confidence/ranking quality only)
- Ensemble methods requiring multiple inference passes (see Spec 050)
- Training a full ML calibrator (see Spec 049)

## 2. Proposed Solution

### 2.1 Extend LLM Output Schema

Current `ItemAssessment` output from the LLM:

```json
{
  "item": "Sleep",
  "score": 2,
  "evidence": ["Quote 1...", "Quote 2..."],
  "explanation": "..."
}
```

New output with verbalized confidence:

```json
{
  "item": "Sleep",
  "score": 2,
  "confidence": 4,
  "evidence": ["Quote 1...", "Quote 2..."],
  "explanation": "..."
}
```

Where `confidence` is an integer from 1-5:
- 1 = Very uncertain (guessing)
- 2 = Somewhat uncertain
- 3 = Moderately confident
- 4 = Fairly confident
- 5 = Very confident

### 2.2 Prompt Modification

Add to the quantitative assessment prompt (after the scoring instructions):

```text
For each item, also provide a confidence rating from 1 to 5:
- 1: Very uncertain - I am guessing based on minimal evidence
- 2: Somewhat uncertain - Evidence is weak or ambiguous
- 3: Moderately confident - Some supporting evidence
- 4: Fairly confident - Clear supporting evidence
- 5: Very confident - Strong, unambiguous evidence

If you cannot assess an item (N/A), do not include a confidence rating for that item.
```

### 2.3 Domain Model Changes

Extend `ItemAssessment` in `src/ai_psychiatrist/domain/value_objects.py`:

```python
@dataclass(frozen=True)
class ItemAssessment:
    item: PHQ8Item
    score: int | None
    evidence: tuple[str, ...]
    explanation: str
    na_reason: str | None = None

    # Existing (Spec 046)
    retrieval_reference_count: int | None = None
    retrieval_similarity_mean: float | None = None
    retrieval_similarity_max: float | None = None

    # NEW (Spec 048)
    verbalized_confidence: int | None = None  # 1-5 scale
```

### 2.4 Export in Run Artifacts

Add to `item_signals` in run output JSON:

```json
{
  "item_signals": {
    "Sleep": {
      "llm_evidence_count": 2,
      "retrieval_reference_count": 1,
      "retrieval_similarity_mean": 0.82,
      "retrieval_similarity_max": 0.82,
      "verbalized_confidence": 4
    }
  }
}
```

### 2.5 New Confidence Variants

Add to `scripts/evaluate_selective_prediction.py`:

```python
CONFIDENCE_VARIANTS = {
    # Existing
    "llm",
    "total_evidence",
    "retrieval_similarity_mean",
    "retrieval_similarity_max",
    "hybrid_evidence_similarity",

    # NEW (Spec 048)
    "verbalized",
    "verbalized_calibrated",
    "hybrid_verbalized",
}
```

**Formula for `verbalized`:**
```text
confidence = (verbalized_confidence - 1) / 4  # Normalize to [0, 1]
```

**Formula for `verbalized_calibrated`:**
```text
# Temperature scaling learned from paper-train (probability-space temperature scaling)
p = (verbalized_confidence - 1) / 4   # Normalize to [0, 1] (use 0.5 if null)
confidence = sigmoid(logit(p) / T)
# where T > 0 is fit by minimizing binary negative log-likelihood
```

**Formula for `hybrid_verbalized`:**
```text
e = min(llm_evidence_count, 3) / 3
s = retrieval_similarity_mean or 0.0
v = (verbalized_confidence - 1) / 4 if verbalized_confidence else 0.5

confidence = 0.4 * v + 0.3 * e + 0.3 * s
```

### 2.6 Calibration Infrastructure

New script: `scripts/calibrate_verbalized_confidence.py`

```bash
# Fit temperature scaling on paper-train
uv run python scripts/calibrate_verbalized_confidence.py \
  --input data/outputs/run_paper_train.json \
  --mode few_shot \
  --output data/calibration/temperature_scaling.json

# Apply calibration to evaluation
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run_paper_test.json \
  --mode few_shot \
  --confidence verbalized_calibrated \
  --calibration data/calibration/temperature_scaling.json
```

**Calibration artifact schema:**
```json
{
  "method": "temperature_scaling",
  "temperature": 2.3,
  "fitted_on": {
    "run_id": "...",
    "mode": "few_shot",
    "n_samples": 464
  },
  "metrics": {
    "nll_before": 1.23,
    "nll_after": 0.98,
    "ece_before": 0.15,
    "ece_after": 0.08
  }
}
```

## 3. Implementation Plan

### Phase 1: Prompt & Schema Changes

1. Update `src/ai_psychiatrist/agents/prompts/quantitative.py` with confidence instructions
2. Update `ItemAssessment` dataclass with `verbalized_confidence` field
3. Update `QuantitativeAssessmentAgent` to parse and validate confidence from LLM response
4. Update `scripts/reproduce_results.py` to export `verbalized_confidence` in `item_signals`

### Phase 2: Evaluation Support

5. Add `verbalized` confidence variant to `evaluate_selective_prediction.py`
6. Add CLI flag `--calibration` to load calibration artifact
7. Add `verbalized_calibrated` and `hybrid_verbalized` variants

### Phase 3: Calibration Script

8. Create `scripts/calibrate_verbalized_confidence.py`
9. Implement temperature scaling optimization (scipy.optimize or sklearn)
10. Add unit tests for calibration fitting and application

## 4. Test Plan

### 4.1 Unit Tests

- `test_verbalized_confidence_parsing`: Validates 1-5 range, handles missing gracefully
- `test_verbalized_confidence_normalization`: Verifies [0, 1] output
- `test_temperature_scaling_calibration`: Verifies NLL reduction
- `test_hybrid_verbalized_formula`: Verifies bounded output

### 4.2 Integration Tests

- Mock LLM response with confidence field
- Verify end-to-end flow from assessment to evaluation

### 4.3 Ablation Run

After implementation, run on paper-test and compare:

```bash
# Baseline
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run10.json --mode few_shot --confidence llm

# Verbalized raw
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run10.json --mode few_shot --confidence verbalized

# Verbalized calibrated
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run10.json --mode few_shot --confidence verbalized_calibrated \
  --calibration data/calibration/temperature_scaling.json

# Hybrid
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run10.json --mode few_shot --confidence hybrid_verbalized
```

## 5. Expected Outcomes

Based on literature:

| Confidence Signal | Expected AUGRC | vs Current |
|-------------------|----------------|------------|
| `llm` (baseline) | 0.031 | — |
| `verbalized` (raw) | ~0.028 | -10% |
| `verbalized_calibrated` | ~0.024 | -23% |
| `hybrid_verbalized` | ~0.020 | **-35%** |

**Target**: AUGRC < 0.020 with `hybrid_verbalized`

## 6. Acceptance Criteria

- [ ] LLM outputs include `confidence` field (1-5)
- [ ] `ItemAssessment` has `verbalized_confidence` field
- [ ] Run artifacts include `verbalized_confidence` in `item_signals`
- [ ] `evaluate_selective_prediction.py` supports `verbalized`, `verbalized_calibrated`, `hybrid_verbalized`
- [ ] `calibrate_verbalized_confidence.py` produces valid calibration artifact
- [ ] Documentation updated in `docs/statistics/metrics-and-evaluation.md`
- [ ] Tests pass: `make ci`

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| LLM ignores confidence instruction | Medium | Add examples in prompt; validate output |
| Verbalized confidence too noisy | Medium | Calibration reduces noise; hybrid signal provides fallback |
| Calibration overfits paper-train | Low | Use proper train/val split; check generalization |

## 8. References

- [LLM Uncertainty Survey 2025](https://arxiv.org/abs/2503.15850)
- [ICLR 2025 - Do LLMs Estimate Uncertainty Well?](https://proceedings.iclr.cc/paper_files/paper/2025/)
- [CoCoA (TACL 2025)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737)
- [Can LLMs Express Their Uncertainty?](https://arxiv.org/html/2306.13063v2)
- [Uncertainty Distillation (2025)](https://arxiv.org/html/2503.14749)
