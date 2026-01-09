# Next Steps

**Status**: Run 16 complete; item-level baseline + total-score tested; ready for binary classification ablation
**Last Updated**: 2026-01-09

---

## Current Status Summary

### What's Done ✅

| Milestone | Run | Status |
|-----------|-----|--------|
| BUG-035 fix (prompt confound) | Run 13 | ✅ Fixed |
| BUG-048 fix (NaN in JSON) | Post-Run 13 | ✅ Fixed |
| Confidence suite (Specs 048-052) | Run 12 | ✅ Validated |
| Chunk-level scoring (Spec 35) | Run 7+ | ✅ Enabled |
| Participant-only preprocessing | Run 8+ | ✅ Enabled |
| Evidence grounding (Spec 053) | Run 11+ | ✅ Enabled |
| **Severity inference (Spec 063)** | **Run 14** | ✅ **Tested - DOES NOT HELP** |
| **Total score metrics (Spec 061)** | **Run 16** | ✅ **Tested** |
| Binary classification (Spec 062) | Implemented | ❌ Not yet tested |

### Key Results Summary

| Run | Mode | MAE_item | AURC | Coverage | Notes |
|-----|------|----------|------|----------|-------|
| **13** | Zero-shot (strict) | **0.608** | **0.107** | 50% | **Best item-level baseline** |
| 13 | Few-shot (strict) | 0.657 | 0.115 | 49% | Worse than zero-shot |
| 14 | Zero-shot (infer) | 0.703 | 0.129 | 60% | Coverage ↑, accuracy ↓ |
| 14 | Few-shot (infer) | 0.784 | 0.147 | 58% | Even worse |

### Total Score Results (Spec 061; `--prediction-mode total`, `total_score_min_coverage=0.5`)

**Run 16 output**: `data/outputs/both_paper-test_20260109_110840.json` (run_id `b2fd1ce3`, git `e3eb011`)

| Run | Mode | N_pred | MAE_total | RMSE | r | TierAcc |
|-----|------|--------|-----------|------|---|---------|
| **16** | Zero-shot (strict) | 26/41 (63.4%) | **4.8846** | 6.3941 | 0.4123 | **0.3462** |
| 16 | Few-shot (strict) | 22/41 (53.7%) | 5.5455 | 6.5989 | 0.5037 | 0.2727 |

**Caveat**: Coverage differs between modes; on the overlap set where both produced a total (N=21), MAE_total is nearly identical (zero-shot 5.5238 vs few-shot 5.5714).

### Key Finding: Fundamental Task Limitation

**See: [EXPLANATION-HYPOTHESIS.md](EXPLANATION-HYPOTHESIS.md)**

The ~50% coverage with strict mode is **correct behavior**, not a limitation:
- PHQ-8 measures **frequency** ("how many days")
- DAIC-WOZ transcripts discuss symptoms **qualitatively** without day-counts
- We cannot infer frequency from intensity ("always tired" ≠ "tired 12/14 days")
- The inference mode failed because it conflates intensity with frequency

---

## Final Ablation Plan

We have three remaining experiments to complete the analysis:

### Run 15 (Optional): Item-Level Baseline Re-run (Post Spec 061/062 + BUG-048)

**Purpose**: If we want a dedicated post-PR#96 item-level baseline artifact, re-run `--prediction-mode item` under strict mode.

**Status**: Not executed yet (we ran Run 16 total-score first).

```bash
tmux new -s run15

uv run python scripts/reproduce_results.py \
  --split paper-test \
  --severity-inference strict \
  2>&1 | tee data/outputs/run15_baseline_$(date +%Y%m%d_%H%M%S).log
```

**Expected**: Similar item-level behavior to Run 13/Run 16 (allowing for LLM variance).

**Runtime**: ~20 hours (both modes)

### Run 16: Total Score Prediction (Spec 061) ✅ Completed

**Purpose**: Test if sum-of-items is more predictable than individual items (errors may cancel).

```bash
tmux new -s run16

uv run python scripts/reproduce_results.py \
  --split paper-test \
  --prediction-mode total \
  --total-min-coverage 0.5 \
  2>&1 | tee data/outputs/run16_total_$(date +%Y%m%d_%H%M%S).log
```

**What to evaluate**:
- Total score MAE (0-24 scale)
- Coverage (what % of participants get a total score)
- Correlation with ground truth

**Results**:
- Output: `data/outputs/both_paper-test_20260109_110840.json`
- Zero-shot: MAE_total 4.8846 @ 63.4% coverage (26/41)
- Few-shot: MAE_total 5.5455 @ 53.7% coverage (22/41)

**Runtime**: ~13.7 hours (both modes; per-mode ~6.5h / ~7.1h)

### Run 17: Binary Classification (Spec 062)

**Purpose**: Test if depressed/not-depressed (PHQ-8 ≥ 10) is easier than item-level scoring.

```bash
tmux new -s run17

uv run python scripts/reproduce_results.py \
  --split paper-test \
  --prediction-mode binary \
  --binary-threshold 10 \
  2>&1 | tee data/outputs/run17_binary_$(date +%Y%m%d_%H%M%S).log
```

**What to evaluate**:
- Accuracy, Precision, Recall, F1
- ROC-AUC (if confidence scores available)
- Comparison to DAIC-WOZ state-of-art

**Runtime**: ~20 hours (both modes)

---

## Complete Ablation Matrix

After all runs complete:

| Run | Prediction Mode | Severity Inference | Purpose |
|-----|-----------------|-------------------|---------|
| 13 | item | strict | Original baseline (pre-BUG-048 fix) |
| 14 | item | infer | Severity inference ablation ❌ |
| **15** | item | strict | Optional clean item baseline (post PR#96) |
| **16** | total | strict | Total score prediction ✅ |
| **17** | binary | strict | Binary classification |

---

## Preflight Checklist

```bash
# 1. Verify HF deps
uv run python -c "import torch; print(torch.__version__)"

# 2. Verify transcripts exist
ls -d data/transcripts_participant_only/*_P | wc -l  # Should be ~189

# 3. Verify embeddings + sidecars exist
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.tags.json
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json

# 4. Dry-run to confirm config
uv run python scripts/reproduce_results.py --split paper-test --dry-run
```

---

## Definition of Done

| Milestone | Status |
|-----------|--------|
| Paper MAE_item parity | ✅ few-shot 0.609 vs paper 0.619 (Run 8) |
| BUG-035 fixed | ✅ Run 13 is clean baseline |
| BUG-048 fixed | ✅ Strict JSON output |
| Confidence suite validated | ✅ Run 12 |
| **Spec 063 tested** | ✅ Run 14 - **Does not help** |
| **Spec 061 tested** | ✅ Run 16 |
| Spec 062 tested | ❌ Pending Run 17 |
| Task limitations documented | ✅ EXPLANATION-HYPOTHESIS.md |

---

## Conclusions So Far

1. **Item-level PHQ-8 prediction from DAIC-WOZ is fundamentally limited** by the mismatch between frequency-based scoring and qualitative transcripts.

2. **Strict mode (~50% coverage, MAE 0.608) is the ceiling** for item-level prediction. The model correctly abstains when frequency is unknown.

3. **Few-shot does not beat zero-shot** because reference examples also lack frequency information.

4. **Binary classification and total score may work better** because they don't require item-level frequency precision.

---

## Related Documentation

- [EXPLANATION-HYPOTHESIS.md](EXPLANATION-HYPOTHESIS.md) - **Why this task is fundamentally limited**
- [Run History](docs/results/run-history.md) - All run details and metrics
- [Few-Shot Analysis](docs/results/few-shot-analysis.md) - Why few-shot may not beat zero-shot
- [Task Validity](docs/clinical/task-validity.md) - Construct mismatch analysis
- [Spec 061](docs/_specs/spec-061-total-phq8-score-prediction.md) - Total score prediction
- [Spec 062](docs/_specs/spec-062-binary-depression-classification.md) - Binary classification
