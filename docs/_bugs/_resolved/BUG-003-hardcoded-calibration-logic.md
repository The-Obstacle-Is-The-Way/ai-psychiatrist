# BUG-003: Hardcoded Calibration Logic in Evaluation Script

**Severity**: P3 (Architecture / OCP Violation)
**Status**: Closed (Won't Fix - Design Concern, Not Bug)
**Resolved**: 2026-01-03
**Resolution**: Valid architectural observation but not a bug. Code works correctly. Tech debt for future consideration.
**Created**: 2026-01-03
**File**: `scripts/evaluate_selective_prediction.py`

## Description

The `evaluate_selective_prediction.py` script hardcodes the logic for applying calibration models inside the `_compute_confidence` function and `parse_items` loop:

```python
if confidence_key == "verbalized_calibrated":
    # ... checks instance of TemperatureScalingCalibrator ...
    # ... applies calibration ...
elif confidence_key == "calibrated":
    # ... checks instance of SupervisedCalibration ...
    # ... applies prediction ...
else:
    # ... uses standard CSF registry ...
```

This violates the Open-Closed Principle. If we want to add a new calibrated variant (e.g., `token_msp_calibrated`), we must modify this script.

## Impact

-   **Maintenance**: Adding new calibration types requires editing the evaluation script logic.
-   **Coupling**: The evaluation script is tightly coupled to specific string keys (`"verbalized_calibrated"`) and specific calibration classes.
-   **Inconsistency**: "Standard" CSFs are handled via `csf_registry`, but "Calibrated" CSFs are handled via special `if` blocks.

## Recommended Fix

Refactor `csf_registry` or the evaluation loop to support "Wrapping" CSFs with a Calibrator dynamically.

Example idea:
`confidence="calibrated:verbalized"` -> Registry returns a composed function that calls `verbalized` then applies the loaded calibrator (passed via context).
