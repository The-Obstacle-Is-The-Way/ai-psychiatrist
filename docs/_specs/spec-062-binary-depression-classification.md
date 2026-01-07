# Spec 062: Binary Depression Classification

**Status**: IMPLEMENTED
**Created**: 2026-01-05
**Implemented**: 2026-01-07
**Rationale**: Binary classification (PHQ-8 >= 10) may be more defensible than item-level frequency scoring on DAIC-WOZ.

---

## Motivation

### The Frequency Problem

PHQ-8 item scores (0-3) require 2-week frequency estimation. DAIC-WOZ doesn't elicit frequency. Binary classification sidesteps this:

> "Does this participant show signs of clinical depression?"

This is closer to what psychiatrists actually assess from interviews.

### Clinical Threshold

PHQ-8 >= 10 is the standard screening threshold for major depression ([Kroenke et al., 2009](https://pubmed.ncbi.nlm.nih.gov/18752852/)):

| Total Score | Severity | Clinical Action |
|-------------|----------|-----------------|
| 0-4 | Minimal | None |
| 5-9 | Mild | Watchful waiting |
| **10-14** | **Moderate** | **Treatment consideration** |
| 15-19 | Moderately Severe | Active treatment |
| 20-24 | Severe | Immediate treatment |

Binary classification asks: "Is this person at or above the treatment threshold?"

### Prior Art

- The paper reports 78% accuracy on binary classification (Meta-Review agent)
- Multiple DAIC-WOZ studies use binary depression detection
- This is a more established task than item-level frequency scoring

---

## Design

### Prediction Mode

```bash
uv run python scripts/reproduce_results.py --prediction-mode binary
```

### Binary Classification Strategies

#### Strategy A: Threshold on Predicted Total (Default)

```python
def classify_binary(total_score: int | None, threshold: int = 10) -> str | None:
    if total_score is None:
        return None  # Abstain
    return "depressed" if total_score >= threshold else "not_depressed"
```

#### Strategy B: Direct Binary Prompt

New prompt that classifies without predicting item scores:

```text
Based on this clinical interview transcript, determine whether the
participant shows signs of clinical depression.

Consider:
- Expressed mood and affect
- Behavioral indicators (withdrawal, anhedonia)
- Sleep, energy, appetite mentions
- Self-perception and hopelessness
- Concentration difficulties

Output: "depressed" or "not_depressed"
Also output your confidence (1-5) and reasoning.

If there is truly insufficient evidence to make any determination, output "N/A".
```

#### Strategy C: Holistic Assessment (Meta-Review Style)

Leverage the existing Meta-Review agent which already does binary classification:

```python
# Meta-Review agent already outputs:
{
  "final_assessment": {
    "is_depressed": true,
    "confidence": 0.8,
    "reason": "Multiple indicators of moderate depression..."
  }
}
```

---

## Implementation

### Implemented Scope (2026-01-07)

- Phase 1 (Threshold-Based): Implemented via `PREDICTION_MODE=binary` / `--prediction-mode binary`.
- Uncertainty handling: abstain when total-score bounds straddle the threshold.
- Phase 2/3 (Direct / Ensemble): Deferred (not implemented). `BINARY_STRATEGY=direct|ensemble` fails loudly.

### Phase 1: Threshold-Based (Trivial)

1. Add `--prediction-mode binary` flag
2. Compute total from items (Spec 061 sum-of-items)
3. Apply threshold (default 10)
4. Output binary label

### Phase 2: Direct Classification (Medium)

1. Add binary classification prompt
2. Optionally bypass item-level scoring entirely
3. Add dedicated evaluation script

### Phase 3: Multi-Strategy Ensemble (Optional)

Combine strategies for higher accuracy:
- Strategy A (threshold) vote
- Strategy B (direct prompt) vote
- Strategy C (meta-review) vote
- Majority wins

---

## Evaluation

### Metrics for Binary Classification

| Metric | Formula | Notes |
|--------|---------|-------|
| Accuracy | `(TP + TN) / N` | Primary metric |
| Precision | `TP / (TP + FP)` | Avoid false positives |
| Recall | `TP / (TP + FN)` | Catch true depression |
| F1 | `2 * (P * R) / (P + R)` | Balance P and R |
| AUROC | Area under ROC curve | Threshold-independent |

### Confusion Matrix Output

```json
{
  "binary_metrics": {
    "accuracy": 0.78,
    "precision": 0.75,
    "recall": 0.82,
    "f1": 0.78,
    "confusion_matrix": {
      "true_positive": 15,
      "true_negative": 17,
      "false_positive": 5,
      "false_negative": 4
    }
  }
}
```

### Coverage for Binary

Binary classification can still abstain:
- If sum-of-items has <50% item coverage
- If direct prompt outputs N/A

Report coverage alongside accuracy.

---

## Configuration

### New Settings

```bash
# .env
PREDICTION_MODE=binary  # item | total | binary
BINARY_THRESHOLD=10  # PHQ-8 total score threshold
BINARY_STRATEGY=threshold  # threshold | direct | ensemble
```

### CLI Override

```bash
uv run python scripts/reproduce_results.py \
  --prediction-mode binary \
  --binary-threshold 10 \
  --binary-strategy direct
```

---

## Output Schema Changes

Add to participant results:

```json
{
  "participant_id": "303",
  "prediction_mode": "binary",
  "binary_classification": {
    "predicted": "depressed",
    "actual": "depressed",
    "correct": true,
    "strategy": "threshold",
    "threshold_used": 10,
    "total_score_predicted": 12,
    "confidence": 0.75
  }
}
```

---

## Testing

1. Unit tests for threshold classification
2. Integration test with `--prediction-mode binary`
3. Verify confusion matrix computation
4. Compare accuracy to paper's reported 78%

---

## Comparison to Meta-Review

The Meta-Review agent already does binary classification. Key differences:

| Aspect | Meta-Review | Spec 062 Binary |
|--------|-------------|-----------------|
| Input | Full pipeline output | Transcript (or total score) |
| Interpretability | High (uses item scores) | Lower (direct) or High (threshold) |
| Speed | Requires full pipeline | Can bypass items |
| Coverage | Depends on item coverage | Can be higher |

Consider Spec 062 as an **alternative path** when item-level scoring has low coverage.

---

## Dependencies

- Spec 061 (Total Score) for threshold-based strategy
- Existing Meta-Review agent can be reused for ensemble

---

## Related

- [Spec 061: Total PHQ-8 Score Prediction](spec-061-total-phq8-score-prediction.md)
- [Spec 063: Severity Inference Prompt Policy](spec-063-severity-inference-prompt-policy.md)
- [Task Validity](../clinical/task-validity.md)
- [PHQ-8 Documentation](../clinical/phq8.md)
