# Spec 063: Severity Inference Prompt Policy

**Status**: PROPOSED
**Created**: 2026-01-05
**Rationale**: PHQ-8 scores 0-3 represent severity gradation. Prompts should allow inference from temporal/intensity markers without requiring explicit frequency statements.

---

## Motivation

### The Core Insight

PHQ-8 scores (0-3) are defined by day-counts, but they fundamentally represent **severity gradation**:

| Score | Days | Severity Meaning |
|-------|------|------------------|
| 0 | 0-1 | Not present / minimal |
| 1 | 2-6 | Mild / occasional |
| 2 | 7-11 | Moderate / frequent |
| 3 | 12-14 | Severe / persistent |

A skilled clinician doesn't require patients to count days. They infer severity from:
- Temporal markers ("lately", "recently", "since [event]")
- Intensity qualifiers ("always", "sometimes", "occasionally")
- Impact statements ("I can't function", "it's been hard")
- Context patterns (repeated mentions across interview)

### Current Prompt Behavior

From `src/ai_psychiatrist/agents/prompts/quantitative.py`:
```text
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency
```

This is **methodologically conservative** but causes ~50% abstention because most transcripts lack explicit frequency statements.

### Hypothesis 2A (from HYPOTHESES-FOR-IMPROVEMENT.md)

> When explicit frequency is not stated, you may infer approximate frequency from:
> - Temporal language ("lately" → several days, "always" → nearly every day)
> - Intensity markers ("sometimes" → several days)
> - Functional impact ("can't work" → more than half the days)
> Document your inference in the reason field.

---

## Design

### Prompt Policy Modes

Add a configuration option for prompt policy:

```python
# config.py
class PromptPolicySettings(BaseSettings):
    severity_inference: Literal["strict", "infer"] = "strict"
```

| Mode | Behavior |
|------|----------|
| `strict` | Current behavior: require explicit frequency evidence |
| `infer` | Allow inference from temporal/intensity markers |

### Inference Rules

When `severity_inference=infer`:

```text
FREQUENCY INFERENCE GUIDE:

When explicit day-counts are not stated, infer approximate frequency:

| Language Pattern | Inferred Frequency | Score |
|------------------|-------------------|-------|
| "every day", "constantly", "all the time", "always" | 12-14 days | 3 |
| "most days", "usually", "often", "frequently" | 7-11 days | 2 |
| "sometimes", "occasionally", "lately", "recently" | 2-6 days | 1 |
| "once", "rarely", "not really", "never" | 0-1 days | 0 |

For symptom mentions without temporal markers:
- If impact is severe (can't function) → Score 2-3
- If impact is mentioned but manageable → Score 1
- If mentioned casually without distress → Score 0

IMPORTANT: Document your inference reasoning in the 'reason' field.
```

### Transparency Requirements

Every inference must be documented:

```json
{
  "item": "PHQ8_Tired",
  "score": 2,
  "evidence_count": 1,
  "inference_used": true,
  "inference_type": "intensity_marker",
  "reason": "Participant said 'I'm always exhausted' - 'always' implies frequency of 7+ days (Score 2)"
}
```

---

## Implementation

### Phase 1: Prompt Variant

1. Add `--severity-inference` CLI flag (`strict` or `infer`)
2. Create alternate prompt template with inference rules
3. Add `inference_used` and `inference_type` to output schema
4. Track inference rate in metrics

### Prompt Template Changes

**Strict mode** (current):
```text
Only assign numeric scores (0-3) when evidence clearly indicates frequency.
If frequency is ambiguous or unstated, output N/A.
```

**Infer mode** (new):
```text
Assign numeric scores (0-3) based on evidence.

When explicit frequency is not stated, use these inference rules:
- "always", "every day", "constantly" → Score 3
- "usually", "most days", "often" → Score 2
- "sometimes", "lately", "occasionally" → Score 1
- "rarely", "never", "not really" → Score 0

Document your inference in the reason field.
Only output N/A if there is truly no mention of the symptom.
```

### Output Schema Additions

```python
class ItemAssessment(BaseModel):
    # Existing fields...
    inference_used: bool = False
    inference_type: str | None = None  # "temporal_marker" | "intensity_marker" | "impact_statement"
    inference_marker: str | None = None  # The actual word/phrase triggering inference
```

---

## Evaluation

### Ablation Design

Run both modes on the same test set:

```bash
# Strict mode (baseline)
uv run python scripts/reproduce_results.py \
  --prediction-mode item \
  --severity-inference strict

# Infer mode (intervention)
uv run python scripts/reproduce_results.py \
  --prediction-mode item \
  --severity-inference infer
```

### Expected Outcomes

| Metric | Strict Mode | Infer Mode | Expected Change |
|--------|-------------|------------|-----------------|
| Coverage | ~48% | ~70-80% | Increase |
| MAE | ~0.57 | TBD | May increase (more predictions) |
| AURC | ~0.10 | TBD | May decrease (more calibrated?) |

### Key Questions to Answer

1. **Does inference improve coverage?** (Expected: yes)
2. **Does inference maintain accuracy?** (Needs measurement)
3. **Does inference improve AURC?** (Depends on calibration)
4. **Is inference consistent?** (Check inter-run variance)

---

## Configuration

### New Settings

```bash
# .env
SEVERITY_INFERENCE_MODE=strict  # strict | infer
```

### CLI Override

```bash
uv run python scripts/reproduce_results.py \
  --severity-inference infer
```

---

## Risk Mitigation

### Over-Inference Risk

Inference mode might assign scores where abstention is appropriate.

**Mitigation**:
- Require `inference_used=true` flag for transparency
- Track inference rate per item
- Compare MAE for inferred vs non-inferred items

### Anchoring Risk

The inference rules might anchor the LLM to specific mappings.

**Mitigation**:
- Allow model to override with reasoning
- Log when model deviates from inference rules

### Consistency Risk

Different models may interpret inference rules differently.

**Mitigation**:
- Test with multiple models
- Report inter-model variance

---

## Testing

1. Unit tests for prompt template selection
2. Integration test with `--severity-inference infer`
3. Verify `inference_used` field appears in output
4. Compare coverage between modes
5. Manual audit of inference reasoning quality

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Coverage increase | >= 20 percentage points |
| MAE degradation | < 0.15 |
| Inference rate | Document actual rate |
| Reasoning quality | Manual audit passes |

---

## Dependencies

- None (prompt-only change)
- Benefits Specs 061 and 062 (higher item coverage → better totals)

---

## Related

- [Spec 061: Total PHQ-8 Score Prediction](spec-061-total-phq8-score-prediction.md)
- [Spec 062: Binary Depression Classification](spec-062-binary-depression-classification.md)
- [HYPOTHESES-FOR-IMPROVEMENT.md](../../HYPOTHESES-FOR-IMPROVEMENT.md) Hypothesis 2A
- [Task Validity](../clinical/task-validity.md)
