# Spec 25: AURC Implementation for Rigorous Selective Prediction Evaluation

> **STATUS**: Proposed
>
> **Priority**: High — Required for valid statistical comparison of zero-shot vs few-shot
>
> **GitHub Issue**: #66
>
> **Created**: 2025-12-27
>
> **Resolves**: Paper uses invalid statistical comparison (MAE at different coverage levels)

---

## Problem Statement

The paper compares MAE between zero-shot and few-shot at different coverage levels (40.9% vs 50.0%). This is statistically invalid for selective prediction systems because lower coverage means the model only predicts on "easy" cases, artificially lowering MAE.

**Current state**: Our outputs only save `predicted_items[item] = score | None`. We have no confidence signal to rank predictions.

**Required state**: Capture `llm_evidence_count` per item, enabling AURC computation and matched-coverage comparison.

---

## Solution Overview

1. **Capture confidence signal** in `reproduce_results.py` outputs
2. **Implement AURC computation** in `src/ai_psychiatrist/metrics/aurc.py`
3. **Add matched-coverage comparison** utility
4. **Integrate into experiment output** with backward compatibility

---

## TDD Implementation Plan

### Phase 1: Unit Tests for AURC Computation

**File**: `tests/unit/metrics/test_aurc.py`

```python
"""Unit tests for AURC metric computation."""

import pytest

from ai_psychiatrist.metrics.aurc import (
    compute_aurc,
    compute_risk_at_coverage,
    compute_risk_coverage_curve,
    ItemPrediction,
)


class TestComputeAURC:
    """Tests for compute_aurc function."""

    def test_perfect_ranking_low_aurc(self) -> None:
        """Perfect confidence ranking (correct predictions ranked highest) yields low AURC."""
        # All predictions correct and ranked by confidence
        items = [
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),  # correct, highest conf
            ItemPrediction(score=1, ground_truth=1, confidence=0.9),  # correct
            ItemPrediction(score=0, ground_truth=0, confidence=0.8),  # correct
            ItemPrediction(score=None, ground_truth=1, confidence=0.0),  # abstained
        ]
        aurc = compute_aurc(items)
        assert aurc == 0.0  # Perfect predictions = 0 risk at all coverage levels

    def test_random_ranking_higher_aurc(self) -> None:
        """Random confidence ranking yields higher AURC than perfect ranking."""
        # Errors ranked higher than correct predictions (bad calibration)
        items = [
            ItemPrediction(score=3, ground_truth=0, confidence=1.0),  # error=3, highest conf
            ItemPrediction(score=2, ground_truth=2, confidence=0.5),  # correct
            ItemPrediction(score=1, ground_truth=1, confidence=0.3),  # correct
        ]
        aurc = compute_aurc(items)
        assert aurc > 0.0  # Errors at high confidence = higher AURC

    def test_all_abstained_returns_zero(self) -> None:
        """All abstained predictions yield AURC of 0 (no predictions to rank)."""
        items = [
            ItemPrediction(score=None, ground_truth=1, confidence=0.0),
            ItemPrediction(score=None, ground_truth=2, confidence=0.0),
        ]
        aurc = compute_aurc(items)
        assert aurc == 0.0

    def test_single_prediction(self) -> None:
        """Single prediction yields AURC equal to its absolute error times coverage."""
        items = [
            ItemPrediction(score=2, ground_truth=1, confidence=1.0),  # error = 1
            ItemPrediction(score=None, ground_truth=0, confidence=0.0),  # abstained
        ]
        aurc = compute_aurc(items)
        # At coverage 0.5, risk is 1.0 (error of 1). Area = 0.5 * 1.0 = 0.5
        assert aurc == pytest.approx(0.5, abs=0.01)

    def test_coverage_denominator_includes_abstentions(self) -> None:
        """Coverage should be predicted / total, including abstentions."""
        items = [
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),  # correct
            ItemPrediction(score=None, ground_truth=1, confidence=0.0),  # abstained
            ItemPrediction(score=None, ground_truth=0, confidence=0.0),  # abstained
        ]
        curve = compute_risk_coverage_curve(items)
        assert len(curve) == 1
        coverage, risk = curve[0]
        assert coverage == pytest.approx(1 / 3)  # 1 prediction / 3 total
        assert risk == 0.0  # correct prediction


class TestComputeRiskAtCoverage:
    """Tests for matched-coverage comparison."""

    def test_risk_at_50_percent(self) -> None:
        """Compute MAE at exactly 50% coverage."""
        items = [
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),  # correct
            ItemPrediction(score=3, ground_truth=1, confidence=0.8),  # error=2
            ItemPrediction(score=1, ground_truth=1, confidence=0.5),  # correct
            ItemPrediction(score=0, ground_truth=2, confidence=0.3),  # error=2
        ]
        risk = compute_risk_at_coverage(items, target_coverage=0.5)
        # Top 50% by confidence: first 2 items. MAE = (0 + 2) / 2 = 1.0
        assert risk == pytest.approx(1.0)

    def test_risk_at_100_percent(self) -> None:
        """Risk at 100% coverage equals overall MAE."""
        items = [
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),
            ItemPrediction(score=1, ground_truth=0, confidence=0.5),
        ]
        risk = compute_risk_at_coverage(items, target_coverage=1.0)
        # All items: MAE = (0 + 1) / 2 = 0.5
        assert risk == pytest.approx(0.5)

    def test_insufficient_coverage_returns_none(self) -> None:
        """Return None if not enough predictions to reach target coverage."""
        items = [
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),
            ItemPrediction(score=None, ground_truth=1, confidence=0.0),
            ItemPrediction(score=None, ground_truth=0, confidence=0.0),
            ItemPrediction(score=None, ground_truth=1, confidence=0.0),
        ]
        risk = compute_risk_at_coverage(items, target_coverage=0.5)
        # Only 25% coverage available (1/4), cannot reach 50%
        assert risk is None


class TestRiskCoverageCurve:
    """Tests for risk-coverage curve generation."""

    def test_curve_is_sorted_by_coverage(self) -> None:
        """Curve points should be sorted by increasing coverage."""
        items = [
            ItemPrediction(score=1, ground_truth=1, confidence=0.5),
            ItemPrediction(score=2, ground_truth=2, confidence=1.0),
            ItemPrediction(score=0, ground_truth=1, confidence=0.3),
        ]
        curve = compute_risk_coverage_curve(items)
        coverages = [c for c, r in curve]
        assert coverages == sorted(coverages)

    def test_curve_length_equals_predictions(self) -> None:
        """Curve has one point per prediction (not per abstention)."""
        items = [
            ItemPrediction(score=1, ground_truth=1, confidence=1.0),
            ItemPrediction(score=None, ground_truth=2, confidence=0.0),
            ItemPrediction(score=2, ground_truth=2, confidence=0.5),
        ]
        curve = compute_risk_coverage_curve(items)
        assert len(curve) == 2  # 2 predictions, 1 abstention
```

### Phase 2: AURC Module Implementation

**File**: `src/ai_psychiatrist/metrics/__init__.py`

```python
"""Metrics for evaluation and model analysis."""

from ai_psychiatrist.metrics.aurc import (
    ItemPrediction,
    compute_aurc,
    compute_risk_at_coverage,
    compute_risk_coverage_curve,
)

__all__ = [
    "ItemPrediction",
    "compute_aurc",
    "compute_risk_at_coverage",
    "compute_risk_coverage_curve",
]
```

**File**: `src/ai_psychiatrist/metrics/aurc.py`

```python
"""AURC (Area Under Risk-Coverage Curve) computation for selective prediction.

Paper Reference:
    - A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)
      https://arxiv.org/abs/2410.15361
    - Overcoming Common Flaws in Evaluation of Selective Classification Systems
      https://arxiv.org/abs/2407.01032

This module provides coverage-adjusted evaluation metrics for selective prediction,
where the model can abstain (output N/A) on low-confidence items.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ItemPrediction:
    """A single item prediction with confidence for AURC computation.

    Attributes:
        score: Predicted score (0-3) or None if abstained (N/A).
        ground_truth: Actual score (0-3).
        confidence: Confidence signal (higher = more confident). Used to rank predictions.
    """

    score: int | None
    ground_truth: int
    confidence: float

    @property
    def is_predicted(self) -> bool:
        """True if a prediction was made (not abstained)."""
        return self.score is not None

    @property
    def absolute_error(self) -> int:
        """Absolute error between prediction and ground truth. Returns 0 if abstained."""
        if self.score is None:
            return 0
        return abs(self.score - self.ground_truth)


def compute_risk_coverage_curve(
    items: list[ItemPrediction],
) -> list[tuple[float, float]]:
    """Compute the risk-coverage curve.

    Ranks predictions by confidence (descending) and computes cumulative
    MAE (risk) at each coverage level.

    Args:
        items: List of predictions with confidence signals.

    Returns:
        List of (coverage, risk) tuples, sorted by increasing coverage.
        Coverage is relative to total items (including abstentions).
        Risk is MAE over the included predictions.
    """
    # Filter to predictions only, sorted by confidence (high to low)
    predicted = [x for x in items if x.is_predicted]
    predicted.sort(key=lambda x: x.confidence, reverse=True)

    if not predicted:
        return []

    total = len(items)
    curve: list[tuple[float, float]] = []
    cumulative_error = 0

    for k, item in enumerate(predicted, start=1):
        cumulative_error += item.absolute_error
        coverage = k / total
        risk = cumulative_error / k  # MAE over first k predictions
        curve.append((coverage, risk))

    return curve


def compute_aurc(items: list[ItemPrediction]) -> float:
    """Compute Area Under Risk-Coverage Curve.

    Lower AURC is better. AURC = 0 means perfect predictions ranked by confidence.

    Uses trapezoidal integration over the risk-coverage curve.

    Args:
        items: List of predictions with confidence signals.

    Returns:
        AURC value (lower is better). Returns 0.0 if no predictions.
    """
    curve = compute_risk_coverage_curve(items)

    if len(curve) < 2:
        # 0 or 1 point: area is risk * coverage for single point, or 0
        if len(curve) == 1:
            coverage, risk = curve[0]
            return coverage * risk
        return 0.0

    # Trapezoidal integration
    aurc = 0.0
    for i in range(1, len(curve)):
        c_prev, r_prev = curve[i - 1]
        c_curr, r_curr = curve[i]
        # Trapezoid area = width * average height
        aurc += (c_curr - c_prev) * (r_prev + r_curr) / 2

    return aurc


def compute_risk_at_coverage(
    items: list[ItemPrediction],
    target_coverage: float,
) -> float | None:
    """Compute risk (MAE) at a specific coverage level.

    Useful for matched-coverage comparison between methods.

    Args:
        items: List of predictions with confidence signals.
        target_coverage: Target coverage (0.0-1.0).

    Returns:
        MAE at the target coverage, or None if insufficient predictions.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")

    curve = compute_risk_coverage_curve(items)

    if not curve:
        return None

    # Find the point at or just above target coverage
    for coverage, risk in curve:
        if coverage >= target_coverage - 1e-9:  # Small epsilon for float comparison
            return risk

    # Not enough predictions to reach target coverage
    return None
```

### Phase 3: Extend `reproduce_results.py` to Capture Confidence Signal

**File**: `scripts/reproduce_results.py`

Add `evidence_counts` to `EvaluationResult` and output serialization:

```python
# In EvaluationResult dataclass, add:
evidence_counts: dict[PHQ8Item, int] = field(default_factory=dict)
"""Number of LLM-extracted evidence quotes per item (confidence signal)."""

# In evaluate_participant(), after getting assessment:
evidence_counts = {
    item: assessment.items[item].llm_evidence_count
    for item in PHQ8Item.all_items()
}

# In EvaluationResult construction:
evidence_counts=evidence_counts,

# In ExperimentResults.to_dict(), add to each result:
"evidence_counts": {
    item.value: count for item, count in r.evidence_counts.items()
},
```

### Phase 4: Integration Tests

**File**: `tests/integration/test_aurc_pipeline.py`

```python
"""Integration tests for AURC computation from experiment outputs."""

import json
from pathlib import Path

import pytest

from ai_psychiatrist.metrics.aurc import ItemPrediction, compute_aurc, compute_risk_at_coverage


class TestAURCFromExperimentOutput:
    """Test AURC computation from saved experiment JSON."""

    @pytest.fixture
    def sample_output(self, tmp_path: Path) -> Path:
        """Create a sample experiment output with evidence_counts."""
        data = {
            "experiments": [
                {
                    "results": {
                        "results": [
                            {
                                "participant_id": 300,
                                "success": True,
                                "predicted_items": {
                                    "NoInterest": 2,
                                    "Depressed": None,
                                    "Sleep": 1,
                                },
                                "ground_truth_items": {
                                    "NoInterest": 2,
                                    "Depressed": 1,
                                    "Sleep": 2,
                                },
                                "evidence_counts": {
                                    "NoInterest": 3,
                                    "Depressed": 0,
                                    "Sleep": 1,
                                },
                            }
                        ]
                    }
                }
            ]
        }
        path = tmp_path / "output.json"
        path.write_text(json.dumps(data))
        return path

    def test_load_and_compute_aurc(self, sample_output: Path) -> None:
        """Load experiment output and compute AURC."""
        data = json.loads(sample_output.read_text())
        results = data["experiments"][0]["results"]["results"]

        items: list[ItemPrediction] = []
        for r in results:
            for item_name in r["predicted_items"]:
                pred = r["predicted_items"][item_name]
                gt = r["ground_truth_items"][item_name]
                conf = r["evidence_counts"][item_name]
                items.append(ItemPrediction(score=pred, ground_truth=gt, confidence=conf))

        aurc = compute_aurc(items)
        assert aurc >= 0.0

        # Matched coverage at 50%
        risk_50 = compute_risk_at_coverage(items, target_coverage=0.5)
        # 2/3 items predicted, so 66% max coverage. 50% should work.
        if risk_50 is not None:
            assert risk_50 >= 0.0
```

### Phase 5: Add CLI Option to `reproduce_results.py`

Add `--compute-aurc` flag to compute and report AURC after experiments:

```python
# In argument parser:
parser.add_argument(
    "--compute-aurc",
    action="store_true",
    help="Compute and report AURC after experiments (requires evidence_counts in output)",
)

# After experiment completion, if --compute-aurc:
if args.compute_aurc:
    from ai_psychiatrist.metrics.aurc import ItemPrediction, compute_aurc, compute_risk_at_coverage

    items: list[ItemPrediction] = []
    for exp in all_experiments:
        for r in exp.results:
            if not r.success:
                continue
            for item in PHQ8Item.all_items():
                pred = r.predicted_items.get(item)
                gt = r.ground_truth_items.get(item)
                conf = r.evidence_counts.get(item, 0)
                if gt is not None:
                    items.append(ItemPrediction(score=pred, ground_truth=gt, confidence=conf))

    aurc = compute_aurc(items)
    risk_50 = compute_risk_at_coverage(items, target_coverage=0.5)

    logger.info(
        "AURC metrics computed",
        aurc=round(aurc, 4),
        risk_at_50_pct=round(risk_50, 4) if risk_50 else "N/A",
        total_items=len(items),
    )
```

---

## Acceptance Criteria

- [ ] `src/ai_psychiatrist/metrics/aurc.py` exists with `compute_aurc`, `compute_risk_at_coverage`, `compute_risk_coverage_curve`
- [ ] `ItemPrediction` dataclass defined with `score`, `ground_truth`, `confidence`
- [ ] Unit tests in `tests/unit/metrics/test_aurc.py` pass (all 10 tests)
- [ ] `reproduce_results.py` outputs `evidence_counts` per item per participant
- [ ] Integration test verifies AURC computation from saved experiment JSON
- [ ] `--compute-aurc` flag works in `reproduce_results.py`
- [ ] AURC is reported in experiment summary when flag is used
- [ ] No regressions in existing tests
- [ ] Documentation updated in `docs/specs/24-aurc-metric.md` to reference this implementation

---

## Files to Create

```
src/ai_psychiatrist/metrics/__init__.py
src/ai_psychiatrist/metrics/aurc.py
tests/unit/metrics/__init__.py
tests/unit/metrics/test_aurc.py
tests/integration/test_aurc_pipeline.py
```

## Files to Modify

```
scripts/reproduce_results.py  # Add evidence_counts output + --compute-aurc flag
docs/specs/24-aurc-metric.md  # Update status to reference this spec
```

---

## Output Schema Extension

Current output (per participant):
```json
{
  "participant_id": 300,
  "predicted_items": {"NoInterest": 2, "Depressed": null, ...},
  "ground_truth_items": {"NoInterest": 2, "Depressed": 1, ...}
}
```

New output (backward compatible):
```json
{
  "participant_id": 300,
  "predicted_items": {"NoInterest": 2, "Depressed": null, ...},
  "ground_truth_items": {"NoInterest": 2, "Depressed": 1, ...},
  "evidence_counts": {"NoInterest": 3, "Depressed": 0, ...}
}
```

---

## References

1. [A Novel Characterization of the Population AURC](https://arxiv.org/abs/2410.15361) - 2024
2. [Overcoming Common Flaws in SC Evaluation](https://arxiv.org/abs/2407.01032) - 2024
3. [Know Your Limits: Abstention in LLMs](https://arxiv.org/abs/2407.18418) - 2024
4. GitHub Issue #66 - Paper uses invalid statistical comparison

---

## Related

- `docs/specs/24-aurc-metric.md` - Original AURC proposal (now superseded by this implementation spec)
- `docs/bugs/bug-029-coverage-mae-discrepancy.md` - Coverage/MAE discrepancy analysis
- `docs/paper-reproduction-analysis.md` - Full reproduction analysis
