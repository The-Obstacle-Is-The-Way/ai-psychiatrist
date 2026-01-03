# Spec 049: Supervised Confidence Calibrator (Multi-Signal Ensemble)

**Status**: Ready to Implement
**Priority**: High (Phase 3 of AUGRC improvement)
**Depends on**: Spec 046 (retrieval signals), Spec 048 (verbalized confidence)
**Estimated effort**: Medium-High
**Research basis**: [fd-shifts NeurIPS 2024](https://arxiv.org/abs/2407.01032), [On Calibration of Modern Neural Networks (Guo et al. 2017)](https://arxiv.org/abs/1706.04599), [Platt Scaling (1999)](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)

## 0. Problem Statement

Individual confidence signals (evidence count, retrieval similarity, verbalized confidence) each capture different aspects of prediction quality. A **supervised calibrator** can learn the optimal combination of these signals to predict whether a prediction is correct.

This is the standard approach in the selective classification literature and is implemented in the [fd-shifts benchmark](https://github.com/IML-DKFZ/fd-shifts).

### Current State

From Run 9 + Spec 046:
- `llm` (evidence count): AURC 0.135, AUGRC 0.035
- `retrieval_similarity_mean`: AURC 0.128, AUGRC 0.034

No signal alone achieves AUGRC < 0.020.

### Research Support

From [fd-shifts confid_scores.py](../../_reference/fd-shifts/fd_shifts/analysis/confid_scores.py):

```python
# Secondary combinations - combining multiple confidence signals
_combine_opts = {
    "average": lambda x, y: (x + y) / 2,
    "product": lambda x, y: x * y,
}
```

The calibration literature (Guo et al. 2017, Platt 1999) demonstrates that post-hoc calibration on a validation set significantly improves uncertainty estimation. Multi-signal calibration (e.g., using logistic regression on multiple features) is a standard extension to capture complementary uncertainty information.

## 1. Goals / Non-Goals

### 1.1 Goals

- Implement a **supervised calibrator** that learns to predict prediction correctness from multiple signals
- Support multiple calibration methods:
  - **Temperature scaling** (single parameter, like Spec 048)
  - **Platt scaling** (logistic regression on a single signal)
  - **Multi-signal logistic regression** (combine all signals)
  - **Isotonic regression** (non-parametric, single signal)
- Train on `paper-train` or `paper-val`, evaluate on `paper-test`
- Output calibrated probabilities that can be used as confidence scores
- Enable **risk-controlled refusal** at inference time (optional)

### 1.2 Non-Goals

- Deep learning calibrators (keep it simple: logistic/isotonic regression)
- Cross-validation hyperparameter tuning (use sensible defaults)
- Ensemble methods requiring multiple inference passes (see Spec 050)

## 2. Proposed Solution

### 2.1 Calibrator Training Pipeline

New script: `scripts/train_confidence_calibrator.py`

```bash
# Train a multi-signal logistic regression calibrator
uv run python scripts/train_confidence_calibrator.py \
  --input data/outputs/run_paper_train.json \
  --mode few_shot \
  --method logistic \
  --features llm_evidence_count,retrieval_similarity_mean,verbalized_confidence \
  --target correctness \
  --output data/calibration/logistic_calibrator.json
```

**Supported methods:**
- `temperature`: Temperature scaling (single T parameter)
- `platt`: Logistic regression on a single signal
- `logistic`: Multi-signal logistic regression
- `isotonic`: Isotonic regression (single signal, non-parametric)

**Supported targets:**
- `correctness`: Binary `1{abs_error == 0}`
- `near_correct`: Binary `1{abs_error <= 1}`
- `loss`: Regression on normalized absolute error

### 2.2 Calibrator Artifact Schema

```json
{
  "method": "logistic",
  "version": "1.0",
  "features": ["llm_evidence_count", "retrieval_similarity_mean", "verbalized_confidence"],
  "target": "correctness",
  "model": {
    "coefficients": [0.23, 0.45, 0.32],
    "intercept": -1.2,
    "scaler": {
      "mean": [1.5, 0.75, 3.2],
      "std": [0.8, 0.12, 0.9]
    }
  },
  "training_metadata": {
    "run_id": "abc123",
    "mode": "few_shot",
    "n_samples": 464,
    "positive_rate": 0.65
  },
  "validation_metrics": {
    "auc_roc": 0.78,
    "brier_score": 0.18,
    "ece": 0.05
  }
}
```

### 2.3 Calibrator Application

Extend `scripts/evaluate_selective_prediction.py`:

```bash
# Apply calibrator to evaluation
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/run_paper_test.json \
  --mode few_shot \
  --confidence calibrated \
  --calibration data/calibration/logistic_calibrator.json
```

**New confidence variant: `calibrated`**

When `--calibration` is provided:
1. Load calibrator artifact
2. For each item, extract features from `item_signals`
3. Apply calibrator to get `p_correct`
4. Use `p_correct` as confidence for AURC/AUGRC computation

### 2.4 Implementation Details

**Feature extraction (`CalibratorFeatureExtractor`):**

```python
class CalibratorFeatureExtractor:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def extract(self, item_signals: dict) -> np.ndarray:
        features = []
        for name in self.feature_names:
            value = item_signals.get(name)
            if value is None:
                # Handle missing with sensible defaults
                if "similarity" in name:
                    value = 0.0
                elif "confidence" in name:
                    value = 3  # middle of 1-5 scale
                else:
                    value = 0
            features.append(float(value))
        return np.array(features)
```

**Calibrator classes:**

```python
class TemperatureScalingCalibrator:
    """Single temperature parameter."""

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        # Minimize NLL: -sum(y * log(softmax(c/T)) + (1-y) * log(1 - softmax(c/T)))
        from scipy.optimize import minimize_scalar
        ...

class PlattScalingCalibrator:
    """Logistic regression on a single signal."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(penalty=None)
        self.model.fit(X.reshape(-1, 1), y)

class MultiSignalLogisticCalibrator:
    """Logistic regression on multiple signals."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(penalty="l2", C=1.0)
        self.model.fit(X_scaled, y)

class IsotonicCalibrator:
    """Non-parametric isotonic regression."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(X.flatten(), y)
```

### 2.5 Risk-Controlled Refusal (Optional)

If a user wants runtime refusal based on calibrated confidence:

```python
class RiskController:
    def __init__(self, calibrator, error_budget: float):
        self.calibrator = calibrator
        self.error_budget = error_budget  # e.g., 0.1 for 10% expected error
        self.threshold = None

    def fit_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit refusal threshold using conformal prediction."""
        p_correct = self.calibrator.predict_proba(X)
        # Find threshold τ such that E[loss | p > τ] <= error_budget
        from sklearn.isotonic import IsotonicRegression
        ...

    def should_refuse(self, p_correct: float) -> bool:
        return p_correct < self.threshold
```

This enables a runtime policy: "Only predict when calibrated confidence exceeds threshold τ."

## 3. Implementation Plan

### Phase 1: Calibrator Training Script

1. Create `scripts/train_confidence_calibrator.py`
2. Implement feature extraction from run artifacts
3. Implement calibrator classes (temperature, Platt, logistic, isotonic)
4. Implement calibrator serialization/deserialization

### Phase 2: Evaluation Integration

5. Add `--calibration` flag to `evaluate_selective_prediction.py`
6. Add `calibrated` confidence variant
7. Implement calibrator loading and application

### Phase 3: Risk Controller (Optional)

8. Implement conformal threshold fitting
9. Add `--risk-budget` flag for inference-time refusal

## 4. Test Plan

### 4.1 Unit Tests

- `test_feature_extraction`: Handles missing signals gracefully
- `test_temperature_scaling_fit`: T > 1 for overconfident, T < 1 for underconfident
- `test_logistic_calibrator_fit`: Coefficients are reasonable
- `test_isotonic_calibrator_monotonic`: Output is monotonically increasing

### 4.2 Integration Tests

- Train calibrator on synthetic data, verify serialization roundtrip
- Verify `evaluate_selective_prediction.py --calibration` works end-to-end

### 4.3 Ablation

Compare on paper-test:

| Confidence Signal | Method | AUGRC |
|-------------------|--------|-------|
| `llm` | — | 0.031 |
| `verbalized` | temperature | ~0.024 |
| `llm + retrieval + verbalized` | logistic | ~0.018 |
| `llm + retrieval + verbalized` | isotonic | ~0.018 |

## 5. Expected Outcomes

Based on fd-shifts and calibration literature:

| Method | Expected AUGRC | vs Baseline |
|--------|----------------|-------------|
| Temperature scaling (single signal) | ~0.024 | -23% |
| Platt scaling (single signal) | ~0.023 | -26% |
| Logistic (multi-signal) | ~0.018 | **-42%** |
| Isotonic (multi-signal fallback) | ~0.019 | -39% |

**Target**: AUGRC < 0.020 with multi-signal logistic calibrator

## 6. Acceptance Criteria

- [ ] `scripts/train_confidence_calibrator.py` trains calibrators from run artifacts
- [ ] Calibrator artifacts are JSON-serializable with full metadata
- [ ] `evaluate_selective_prediction.py` supports `--calibration` flag
- [ ] `calibrated` confidence variant works correctly
- [ ] Documentation in `docs/statistics/metrics-and-evaluation.md`
- [ ] Tests pass: `make ci`

## 7. File Changes

### New Files

- `scripts/train_confidence_calibrator.py`
- `src/ai_psychiatrist/calibration/__init__.py`
- `src/ai_psychiatrist/calibration/calibrators.py`
- `src/ai_psychiatrist/calibration/feature_extraction.py`
- `tests/unit/calibration/test_calibrators.py`

### Modified Files

- `scripts/evaluate_selective_prediction.py` (add `--calibration`, `calibrated` variant)
- `docs/statistics/metrics-and-evaluation.md` (document calibration)

## 8. References

- [fd-shifts benchmark](https://github.com/IML-DKFZ/fd-shifts)
- [Platt Scaling (1999)](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)
- [On Calibration of Modern Neural Networks (Guo et al. 2017)](https://arxiv.org/abs/1706.04599)
