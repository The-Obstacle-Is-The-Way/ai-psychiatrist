# Spec 061: Total PHQ-8 Score Prediction (0-24)

**Status**: PROPOSED
**Created**: 2026-01-05
**Rationale**: Item-level PHQ-8 frequency scoring (0-3 per item) is often underdetermined from DAIC-WOZ transcripts. Total score prediction (0-24) may be more defensible.

---

## Motivation

### Task Validity Problem

PHQ-8 item scores (0-3) encode **2-week frequency** (0-1, 2-6, 7-11, 12-14 days). DAIC-WOZ interviews are not structured to elicit frequency information. This creates a fundamental construct mismatch (see `docs/clinical/task-validity.md`).

**Run 12 evidence**:
- Only 32% of item assessments have any grounded evidence
- ~50% abstention rate (N/A) is expected behavior
- Coverage stabilizes around 46-49%

### Why Total Score May Be More Valid

1. **Error averaging**: Item-level errors partially cancel when summed
2. **Fewer degrees of freedom**: 1 prediction vs 8 predictions per participant
3. **Prior art**: Text-only PHQ-8 total regression exists ([PubMed 37398577](https://pubmed.ncbi.nlm.nih.gov/37398577/))
4. **Clinical utility**: Total score determines severity tier (0-4, 5-9, 10-14, 15-19, 20-24)

---

## Design

### Prediction Modes

Add a new configuration option and CLI flag:

```python
# config.py
class PredictionSettings(BaseSettings):
    prediction_mode: Literal["item", "total", "binary"] = "item"
```

```bash
# CLI usage
uv run python scripts/reproduce_results.py --prediction-mode total
```

### Mode Behaviors

| Mode | Output | Coverage Handling | Evaluation Metric |
|------|--------|-------------------|-------------------|
| `item` | 8 scores (0-3) or N/A per item | Per-item abstention | MAE_item, AURC |
| `total` | 1 score (0-24) per participant | Participant-level abstention | MAE_total, RMSE |
| `binary` | 1 label (depressed/not) | Participant-level abstention | Accuracy, F1 |

### Total Score Prediction Strategy

#### Option A: Sum of Item Predictions (Default)

Use existing item-level pipeline, sum non-N/A scores:
```python
def predict_total_score(item_scores: dict[str, int | None]) -> int | None:
    scored_items = [v for v in item_scores.values() if v is not None]
    if len(scored_items) < 4:  # Require at least 50% coverage
        return None  # Abstain
    return sum(scored_items)  # Partial sum (underestimate)
```

**Note**: Partial sums underestimate true total when items are missing.

#### Option B: Direct Total Prediction (Optional)

Add a new prompt that predicts total score directly without item decomposition:
```text
Based on this clinical interview transcript, estimate the participant's
overall PHQ-8 depression severity score (0-24).

Consider all observable indicators of depression symptoms:
- Mood and affect
- Sleep and energy
- Interest and pleasure
- Self-perception
- Concentration

Output a single integer 0-24, or "N/A" if insufficient evidence.
```

**Trade-off**: Less interpretable (no item breakdown) but avoids compounding item abstentions.

---

## Implementation

### Phase 1: Sum-of-Items (Low Effort)

1. Add `--prediction-mode` CLI flag to `reproduce_results.py`
2. In output generation, compute total from item scores
3. Add `total_score` and `total_score_predicted` fields to output JSON
4. Update evaluation script to compute MAE_total when mode=total

### Phase 2: Direct Prediction (Medium Effort)

1. Add new prompt template in `agents/prompts/quantitative.py`
2. Add `DirectTotalAgent` or extend `QuantitativeAgent` with mode switch
3. Output format: `{"total_score": int | "N/A", "confidence": float, "reason": str}`

---

## Evaluation

### Metrics for Total Score

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE_total | `mean(|predicted - actual|)` | Primary metric |
| RMSE | `sqrt(mean((predicted - actual)^2))` | Penalizes large errors |
| Correlation | Pearson r | Linear relationship |
| Severity Tier Accuracy | `sum(tier_pred == tier_actual) / N` | Clinically meaningful |

### Severity Tiers (PHQ-8)

| Tier | Range | Label |
|------|-------|-------|
| 0 | 0-4 | Minimal/None |
| 1 | 5-9 | Mild |
| 2 | 10-14 | Moderate |
| 3 | 15-19 | Moderately Severe |
| 4 | 20-24 | Severe |

---

## Configuration

### New Settings

```bash
# .env
PREDICTION_MODE=total  # item | total | binary
TOTAL_SCORE_MIN_COVERAGE=0.5  # Minimum item coverage for sum-of-items
```

### CLI Override

```bash
uv run python scripts/reproduce_results.py \
  --prediction-mode total \
  --total-min-coverage 0.5
```

---

## Output Schema Changes

Add to participant results:

```json
{
  "participant_id": "303",
  "prediction_mode": "total",
  "total_score": {
    "predicted": 12,
    "actual": 14,
    "method": "sum_of_items",
    "items_covered": 6,
    "confidence": 0.75
  },
  "severity_tier": {
    "predicted": 2,
    "actual": 2,
    "correct": true
  }
}
```

---

## Testing

1. Unit tests for total score computation from items
2. Integration test with `--prediction-mode total`
3. Verify output JSON schema includes total fields
4. Compare MAE_total to MAE_item on same run

---

## Dependencies

- None (uses existing pipeline)
- Phase 2 requires new prompt design

---

## Related

- [Spec 062: Binary Depression Classification](spec-062-binary-depression-classification.md)
- [Spec 063: Severity Inference Prompt Policy](spec-063-severity-inference-prompt-policy.md)
- [Task Validity](../clinical/task-validity.md)
- [HYPOTHESES-FOR-IMPROVEMENT.md](../../HYPOTHESES-FOR-IMPROVEMENT.md) Section 10
