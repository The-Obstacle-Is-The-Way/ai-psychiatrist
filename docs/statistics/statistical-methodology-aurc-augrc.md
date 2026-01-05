# Statistical Methodology: AURC/AUGRC for Selective Prediction

**Purpose**: Explains why AURC/AUGRC are the correct metrics for evaluating selective prediction systems, and why naive MAE comparisons are invalid.

**Last Updated**: 2026-01-03

---

## The Problem: Comparing MAE at Different Coverages

When a model can abstain (say "N/A" or "I don't know"), comparing raw MAE values is **not coverage-adjusted** if the models have different coverage rates.

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

We also compute a **generalized risk** (a.k.a. *joint risk*) curve:

```
GeneralizedRisk(c) = (1/N) × Σ(loss_i for accepted items up to coverage c)
```

At each working point, this equals:

```
GeneralizedRisk(c) = Coverage(c) × Risk(c)
```

```
AUGRC = ∫₀^Cmax GeneralizedRisk(c) dc
```

- **Lower is better**
- Penalizes both error **and** abstention (because generalized risk scales with coverage)
- Optional normalized variant (sometimes reported): `nAUGRC = AUGRC / Cmax` when `Cmax > 0`

#### Optimal and Excess Metrics (Spec 052)

Beyond raw AURC/AUGRC, we compute **excess** metrics that measure distance from the theoretical optimum:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **AURC_optimal** | AURC with oracle ranking (sort by loss ascending) | Theoretical lower bound |
| **AUGRC_optimal** | AUGRC with oracle ranking | Theoretical lower bound |
| **e-AURC** | `AURC - AURC_optimal` | How much room for improvement |
| **e-AUGRC** | `AUGRC - AUGRC_optimal` | How much room for improvement |
| **AURC_achievable** | AURC of lower convex hull | Best achievable by threshold selection |

**Why excess metrics matter**: A CSF with `e-AURC = 0` perfectly ranks predictions by correctness. The `aurc_gap_pct = (e-AURC / AURC_optimal) × 100` shows percentage improvement possible.

SSOT: `compute_eaurc()`, `compute_eaugrc()`, `compute_aurc_optimal()`, `compute_augrc_optimal()`, `compute_aurc_achievable()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

---

## Why This Matters for Depression Assessment

Our system predicts PHQ-8 item scores (0-3 scale) from clinical interview transcripts. The model can abstain (`N/A`) when it lacks sufficient evidence.

Important: PHQ-8 items are defined by **2-week frequency**, but DAIC-WOZ transcripts are not structured as PHQ administration. Transcript-only item scoring is often underdetermined, so abstention is expected and must be treated as part of the evaluation signal. See: `docs/clinical/task-validity.md`.

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
- **Low AUGRC**: Model accumulates low joint loss as it increases coverage (good accuracy without reckless over-prediction)

---

## Our Implementation

### Confidence Scoring Functions (CSFs)

CSFs are functions that produce a scalar confidence value for each prediction (higher = more confident). They live in `src/ai_psychiatrist/confidence/csf_registry.py` and support composition via the `secondary:` prefix.

| CSF Name | Source | Description |
|----------|--------|-------------|
| `llm` / `total_evidence` | Spec 046 | Number of evidence spans cited by LLM |
| `retrieval_similarity_mean` | Spec 046 | Mean similarity of retrieved references |
| `retrieval_similarity_max` | Spec 046 | Max similarity of retrieved references |
| `hybrid_evidence_similarity` | Spec 046 | 0.5 × evidence + 0.5 × similarity |
| `verbalized` | Spec 048 | LLM's self-reported confidence (1-5 scale) |
| `verbalized_calibrated` | Spec 048 | Temperature-scaled verbalized confidence |
| `hybrid_verbalized` | Spec 048 | 0.4 × verbalized + 0.3 × evidence + 0.3 × similarity |
| `token_msp` | Spec 051 | Mean Maximum Softmax Probability over tokens |
| `token_pe` | Spec 051 | Predictive entropy (inverted: 1/(1+entropy)) |
| `token_energy` | Spec 051 | Energy score from logsumexp of logprobs |
| `consistency` | Spec 050 | Modal confidence from multi-sample scoring |
| `consistency_inverse_std` | Spec 050 | 1/(1+std) of score distribution |
| `hybrid_consistency` | Spec 050 | 0.4 × consistency + 0.3 × evidence + 0.3 × similarity |
| `calibrated` | Spec 049 | Supervised calibrator output (logistic/isotonic) |

**Combining CSFs**: Use `secondary:<csf1>+<csf2>:<average|product>` syntax:
```bash
--confidence secondary:token_msp+retrieval_similarity_mean:average
```

### Bootstrap Confidence Intervals

We compute 95% CIs using participant-level bootstrap (default: 10,000 resamples; configurable via `--bootstrap-resamples`):

```python
# Pseudocode
for b in range(10_000):
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

- Exact definitions + output schema: [metrics-and-evaluation.md](./metrics-and-evaluation.md)
- Evaluation Script: `scripts/evaluate_selective_prediction.py` - CLI tool for computing metrics
