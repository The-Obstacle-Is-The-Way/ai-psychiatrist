# SPEC-024: AURC Metric for Fair Zero-Shot vs Few-Shot Comparison

**Status**: Proposed
**Priority**: High
**Created**: 2025-12-27

---

## Problem Statement

Comparing zero-shot vs few-shot MAE at different coverage levels is not apples-to-apples:

| Method | Coverage | MAE | Fair? |
|--------|----------|-----|-------|
| Zero-Shot | 56.9% | 0.698 | ❌ |
| Few-Shot | 71.6% | 0.852 | ❌ |

Higher coverage means predicting on harder cases, which naturally increases MAE.

---

## Solution: Area Under Risk-Coverage Curve (AURC)

AURC evaluates the **entire tradeoff curve**, not just a single point.

### Algorithm

```python
def compute_aurc(predictions: list[dict]) -> float:
    """
    Compute Area Under Risk-Coverage Curve for regression.

    Args:
        predictions: List of dicts with:
            - 'score': predicted score (int 0-3) or None for N/A
            - 'ground_truth': actual score (int 0-3)
            - 'confidence': model confidence (float 0-1)

    Returns:
        AURC value (lower is better)
    """
    # Filter to predictions with scores (not N/A)
    valid = [p for p in predictions if p['score'] is not None]

    # Sort by confidence (descending)
    valid.sort(key=lambda x: x['confidence'], reverse=True)

    # Compute risk (MAE) at each coverage level
    risks = []
    coverages = []

    for i in range(1, len(valid) + 1):
        subset = valid[:i]
        coverage = i / len(predictions)  # fraction of ALL predictions
        mae = sum(abs(p['score'] - p['ground_truth']) for p in subset) / len(subset)

        risks.append(mae)
        coverages.append(coverage)

    # Compute area under curve (trapezoidal rule)
    aurc = 0.0
    for i in range(1, len(coverages)):
        width = coverages[i] - coverages[i-1]
        height = (risks[i] + risks[i-1]) / 2
        aurc += width * height

    return aurc
```

### Interpretation

| AURC Value | Meaning |
|------------|---------|
| 0.0 | Perfect (all predictions correct, ordered by confidence) |
| ~0.5 | Poor calibration |
| ~1.0+ | Worse than random |

### Comparison

```
Method        | Coverage | MAE   | AURC  | Fair Comparison
--------------|----------|-------|-------|----------------
Zero-Shot     | 56.9%    | 0.698 | ???   | ✅ Comparable
Few-Shot      | 71.6%    | 0.852 | ???   | ✅ Comparable
```

---

## Implementation Requirements

1. **Capture confidence scores**: Currently we only capture score or N/A. We need to also capture the model's confidence for each prediction.

2. **For Pydantic AI**: The model already provides reasoning. We could:
   - Use token probabilities (if available)
   - Use a heuristic based on evidence quality
   - Add a confidence prompt to the model

3. **For N/A predictions**: These are implicitly "low confidence" - the model chose to abstain.

---

## Alternative: Matched Coverage Comparison

If implementing AURC is too complex, a simpler approach:

1. Set a threshold to match coverage between zero-shot and few-shot
2. Compare MAE at the same coverage level

```python
# Example: Force both to 50% coverage
# Compare MAE on the 50% highest-confidence predictions from each
```

---

## Research References

1. [Know Your Limits: A Survey of Abstention in LLMs](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754) - MIT Press, Dec 2024
2. [Overcoming Common Flaws in Evaluation of Selective Classification](https://arxiv.org/html/2407.01032v1) - Jul 2024
3. [A Novel Characterization of Population AURC](https://arxiv.org/abs/2410.15361) - Oct 2024
4. [TorchUncertainty Library](https://torch-uncertainty.github.io/) - AURC implementation for classification

---

## Related

- `docs/bugs/bug-029-coverage-mae-discrepancy.md` - Analysis of coverage difference
- Paper Section 3.2 - "50% of cases unable to provide prediction"
