# SPEC-024: Risk–Coverage Evaluation (AURC-Family) for PHQ-8 Selective Prediction

**Status**: Proposed
**Priority**: High
**Created**: 2025-12-27

---

## Problem Statement

Comparing zero-shot vs few-shot MAE at different coverage levels is not apples-to-apples.

| Method | Coverage | MAE | Fair? |
|--------|----------|-----|-------|
| Zero-Shot | 56.9% | 0.717 | ❌ |
| Few-Shot | 71.6% | 0.860 | ❌ |

Higher coverage means predicting on harder cases, which naturally increases MAE.

---

## Solution: Risk–Coverage Curve + AURC-Family Metrics

AURC-family metrics evaluate the **entire tradeoff curve**, not just a single point.

In selective prediction/abstention settings, a model can choose to abstain on low-confidence items.
We need to evaluate not only "how accurate" but also "how accurate at what coverage".

### Algorithm

```python
def compute_aurc(items: list[dict]) -> float:
    \"\"\"Compute AURC for regression risk=MAE.

    Each item is one PHQ-8 item prediction for one participant.

    Required fields:
      - score: int | None   (None means abstained / N/A)
      - ground_truth: int
      - confidence: float   (higher = more confident)

    Notes:
      - Sort predicted items by confidence.
      - Risk at a coverage level is MAE over the kept items.
      - Coverage is computed as kept_items / total_items (including abstentions).
    \"\"\"
    # Keep only predicted items, ranked by confidence.
    predicted = [x for x in items if x[\"score\"] is not None]
    predicted.sort(key=lambda x: x[\"confidence\"], reverse=True)

    risks: list[float] = []
    coverages: list[float] = []

    total = len(items)
    for k in range(1, len(predicted) + 1):
        subset = predicted[:k]
        coverages.append(k / total)
        risks.append(
            sum(abs(x[\"score\"] - x[\"ground_truth\"]) for x in subset) / len(subset)
        )

    # Trapezoidal rule.
    aurc = 0.0
    for i in range(1, len(coverages)):
        aurc += (coverages[i] - coverages[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return aurc
```

### Interpretation

Lower is better, but absolute values depend on the task/risk scale.
Interpret AURC comparatively (method A < method B).

### Comparison

```
Method        | Coverage | MAE   | AURC  | Fair Comparison
--------------|----------|-------|-------|----------------
Zero-Shot     | 56.9%    | 0.698 | ???   | ✅ Comparable
Few-Shot      | 71.6%    | 0.852 | ???   | ✅ Comparable
```

---

## Implementation Requirements

1. **Define a confidence signal** (required): AURC needs a way to rank items by confidence.

2. **Capture the signal in outputs**: Current experiment outputs record only predicted score vs N/A.

3. **Decide the abstention mechanism**: We currently abstain only when the model outputs "N/A".
   AURC becomes useful when we can also abstain based on a threshold over a confidence signal.

### Recommended Confidence Signals (Practical)

- **Few-shot**: `top_similarity` from the embedding retrieval for that PHQ-8 item.
  - If similarity is low, abstain.
- **Both modes**: evidence strength from the evidence retrieval step:
  - number of extracted evidence quotes (`llm_evidence_count`)
  - whether evidence quotes are exact substrings of the transcript (anti-hallucination filter)

These enable a principled risk–coverage curve without changing the scoring model.

---

## Alternative: Matched Coverage Comparison

If implementing AURC is too complex, a simpler approach:

1. Choose a target coverage (e.g., 50%)
2. Apply a threshold over the confidence signal to match that coverage
3. Compare MAE at the same coverage level

```python
# Example: Force both to 50% coverage
# Compare MAE on the 50% highest-confidence predictions from each
```

---

## Research References

1. [Overcoming Common Flaws in the Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032v1) - 2024
2. [A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)](https://arxiv.org/abs/2410.15361) - 2024
3. [Know Your Limits: A Survey of Abstention in Large Language Models](https://arxiv.org/abs/2407.18418) - 2024

---

## Related

- `docs/bugs/bug-029-coverage-mae-discrepancy.md` - Analysis of coverage difference
- Paper Section 3.2 - "50% of cases unable to provide prediction"
