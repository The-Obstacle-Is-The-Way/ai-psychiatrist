# Statistical Methodology: AURC/AUGRC for Selective Prediction

**Purpose**: Explains why AURC/AUGRC are the correct metrics for evaluating selective prediction systems, and why naive MAE comparisons are invalid.

**Last Updated**: 2025-12-28

---

## The Problem: Comparing MAE at Different Coverages

When a model can abstain (say "N/A" or "I don't know"), comparing raw MAE values is **statistically invalid** if the models have different coverage rates.

### Example of Invalid Comparison

| Model | Coverage | MAE | Conclusion? |
|-------|----------|-----|-------------|
| Zero-Shot | 55% | 0.64 | ? |
| Few-Shot | 72% | 0.80 | ? |

**Q: Is Few-Shot worse because MAE is higher?**

**A: We can't tell!** Few-Shot is predicting on 17% more items. Those additional items might be harder cases that Zero-Shot abstained from. The higher MAE could simply reflect Few-Shot's willingness to attempt difficult predictions.

This is like comparing:
- A surgeon who only operates on easy cases (low mortality rate)
- A surgeon who takes on hard cases (higher mortality rate)

The second surgeon isn't necessarily worse—they're just not refusing difficult patients.

---

## The Solution: Risk-Coverage Analysis

Instead of comparing single MAE values, we analyze the **entire risk-coverage curve**.

### Key Concepts

1. **Coverage (c)**: Fraction of items where the model makes a prediction (doesn't abstain)
2. **Risk at coverage c**: Average loss on the items the model is most confident about, up to coverage c
3. **Risk-Coverage Curve**: Plot of risk vs. coverage as we increase the coverage threshold

### The Metrics

#### AURC (Area Under Risk-Coverage Curve)

```
AURC = ∫₀^Cmax Risk(c) dc
```

- **Lower is better**
- Measures total "cost" of predictions across all coverage levels
- A model with AURC=0.10 accumulates less error than one with AURC=0.20

#### AUGRC (Area Under Generalized Risk-Coverage Curve)

```
AUGRC = (1/Cmax) × ∫₀^Cmax Risk(c) dc
```

- **Lower is better**
- AURC normalized by maximum coverage
- Useful when comparing models with very different Cmax values
- Interpretation: "average risk per unit coverage"

---

## Why This Matters for Depression Assessment

Our system predicts PHQ-8 item scores (0-3 scale) from clinical interview transcripts. The model can abstain when it lacks sufficient evidence.

### The Tradeoff

| High Coverage | Low Coverage |
|---------------|--------------|
| More predictions | Fewer predictions |
| Potentially higher error | Lower error on attempted items |
| More clinical utility? | Less clinical utility? |

**AURC/AUGRC capture this tradeoff** by integrating over all coverage levels.

### Clinical Interpretation

- **Low AURC**: Model has good calibration—when it's confident, it's usually right
- **High AURC**: Model is overconfident—high-confidence predictions are often wrong
- **Similar AUGRC, different Cmax**: Models have similar quality but different "willingness to predict"

---

## Our Implementation

### Confidence Signals

We use two confidence signals for ranking predictions:

1. **`llm_evidence_count`**: Number of evidence spans the LLM cited
2. **`keyword_evidence_count`**: Number of keyword matches found
3. **`total_evidence`**: Sum of both (composite signal)

### Bootstrap Confidence Intervals

We compute 95% CIs using participant-level bootstrap (1000 resamples):

```python
# Pseudocode
for b in range(1000):
    resampled_participants = bootstrap_sample(participants)
    metrics[b] = compute_aurc(resampled_participants)
ci_low, ci_high = percentile(metrics, [2.5, 97.5])
```

This accounts for participant-level clustering (each participant has 8 PHQ-8 items).

---

## Comparison to Paper Methodology

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| Primary metric | MAE | AURC, AUGRC |
| Coverage handling | Ignored (different coverages compared) | Integrated over curve |
| Statistical inference | None reported | Bootstrap 95% CIs |
| Conclusion validity | Questionable | Statistically sound |

### Why the Paper's Comparison Was Invalid

The paper reported:
- Zero-shot MAE: 0.796
- Few-shot MAE: 0.619

And concluded few-shot is better. But if:
- Zero-shot coverage was ~50%
- Few-shot coverage was ~50%

Then the comparison is fair. However, if coverages differed, **the conclusion is unsupported**.

---

## Running the Evaluation

```bash
# Single mode evaluation
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/your_output.json \
  --mode zero_shot \
  --seed 42

# Compare two modes (paired analysis)
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/your_output.json \
  --mode zero_shot \
  --input data/outputs/your_output.json \
  --mode few_shot \
  --seed 42
```

### Output Interpretation

```
Confidence: llm
  Cmax:       0.5549 [0.4726, 0.6402]   # Max coverage achieved
  AURC:       0.1342 [0.0944, 0.1758]   # Lower = better
  AUGRC:      0.0368 [0.0239, 0.0532]   # Lower = better
```

---

## References

1. **Geifman & El-Yaniv (2017)**: "Selective Classification for Deep Neural Networks" - Original selective prediction framework
2. **Jaeger et al. (2023)**: "A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification" - AURC/AUGRC formalization
3. **fd-shifts library**: Reference implementation we validated against

---

## Related Documentation

- [Spec 25: AURC/AUGRC Implementation](../archive/specs/25-aurc-augrc-implementation.md) - Technical implementation details
- Evaluation Script: `scripts/evaluate_selective_prediction.py` - CLI tool for computing metrics
