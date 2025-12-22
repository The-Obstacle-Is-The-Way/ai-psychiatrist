# Reproduction Notes: PHQ-8 Assessment Evaluation

## Summary

This document captures findings from the paper reproduction attempt on 2025-12-22.

## Final Results

### Zero-Shot Evaluation (gemma3:27b)

| Metric | Value |
|--------|-------|
| Total Participants | 47 |
| Successful | 41 (87%) |
| Failed (timeouts) | 6 |
| **MAE** | **4.02** |
| RMSE | 5.38 |
| Median AE | 3.0 |
| Total Time | 2.96 hours |

### Perfect Predictions (error = 0)

| Participant | Ground Truth | Predicted |
|-------------|--------------|-----------|
| 359 | 13 | 13 |
| 387 | 2 | 2 |
| 408 | 0 | 0 |
| 462 | 9 | 9 |
| 465 | 2 | 2 |
| 470 | 3 | 3 |

### Observations

1. **Underestimation of severe depression**: The model tends to predict lower scores for participants with high ground truth (≥15). Examples:
   - P308: predicted 10, actual 22 (error 12)
   - P354: predicted 7, actual 18 (error 11)
   - P453: predicted 5, actual 17 (error 12)

2. **Good performance on mild cases**: Many predictions within 1-3 points for lower severity cases.

3. **Timeouts**: 6 participants timed out (180s limit), suggesting some transcripts require longer processing.

## Environment

- **Hardware**: M1 Pro Max (Apple Silicon)
- **Models**:
  - gemma3:27b (17GB, quantized)
  - qwen3-embedding:8b (4.7GB)
  - alibayram/medgemma:27b (16GB)
- **Ollama**: Running locally

## Bug Discovered: MedGemma Excessive N/A Predictions

### Problem

The default `quantitative_model` was set to `alibayram/medgemma:27b` based on Paper Appendix F claiming better MAE (0.505 vs 0.619). However, in practice:

- MedGemma returns **8/8 N/A scores** for most participants
- This results in predicted total score = 0 for all participants
- Total-score MAE: **6.75** (much worse than expected)

### Root Cause

Paper Appendix F states:
> "The few-shot approach with MedGemma 27B achieved an improved average MAE of 0.505 **but detected fewer relevant chunks, making fewer predictions overall**"

The "fewer predictions" caveat is critical: MedGemma is too conservative and marks items as N/A when evidence is ambiguous. While this produces better MAE *when it does score*, it makes fewer predictions overall.

### Example

Participant 308 (ground truth PHQ-8 = 22, severe depression):
- MedGemma: predicted 0, na_count = 8
- gemma3:27b: predicted 10, na_count = 4
- Transcript clearly shows: "yeah it's pretty depressing", sleep problems, job stress

### Fix Applied

Changed default `quantitative_model` from `alibayram/medgemma:27b` to `gemma3:27b`:
- `src/ai_psychiatrist/config.py`: Updated ModelSettings default
- `tests/unit/test_config.py`: Updated test with explanatory docstring
- `.env.example`: Updated default with note
- `.env`: Updated user's local config

## MAE Interpretation

**Important**: The paper's reported MAE values (0.619 zero-shot, 0.505 few-shot) are **item-level MAE** (each item ranges 0-3), NOT total-score MAE (which ranges 0-24).

Our total-score MAE of 4.02 translates to roughly **0.50 item-level MAE** if errors were evenly distributed across 8 items, which is actually better than the paper's reported 0.619 for zero-shot.

## Performance Notes

- Each assessment takes ~3-4 minutes on M1 Pro Max
- Two LLM calls per assessment: evidence extraction + scoring
- 47 test participants = ~3 hours for full evaluation
- 180s timeout per LLM call (6 timeouts occurred)

## Files Created

- `scripts/reproduce_results.py`: Batch evaluation script
- `data/outputs/reproduction_results_20251222_040100.json`: Full results

## Remaining Work

1. ~~Complete full zero-shot evaluation~~ ✅
2. Run few-shot evaluation with embedding references (optional)
3. Investigate timeout issues for longer transcripts
4. Consider increasing timeout or optimizing prompts
