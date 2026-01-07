# Next Steps

**Status**: Run 13 complete (valid baseline); ready for Run 14 (new specs)
**Last Updated**: 2026-01-07

---

## Current Status Summary

### What's Done ✅

| Milestone | Run | Status |
|-----------|-----|--------|
| BUG-035 fix (prompt confound) | Run 13 | ✅ Fixed |
| Confidence suite (Specs 048-052) | Run 12 | ✅ Validated |
| Chunk-level scoring (Spec 35) | Run 7+ | ✅ Enabled |
| Participant-only preprocessing | Run 8+ | ✅ Enabled |
| Evidence grounding (Spec 053) | Run 11+ | ✅ Enabled |
| Severity inference (Spec 063) | Post-Run 13 | ✅ **Implemented (not yet tested)** |
| Total score metrics (Spec 061) | Post-Run 13 | ✅ **Implemented (not yet tested)** |
| Binary classification (Spec 062) | Post-Run 13 | 🔶 Threshold strategy only |

### Run 13 Baseline (Current SSOT)

| Mode | MAE_item | AURC (llm) | Best AUGRC | Coverage |
|------|----------|------------|------------|----------|
| **Zero-shot** | **0.6079** | 0.107 | 0.024 (`consistency_inverse_std`) | 50.0% |
| Few-shot | 0.6571 | 0.115 | 0.025 (`token_pe`) | 48.5% |

**Key finding**: Zero-shot beats few-shot (valid after BUG-035 fix). This is a valid research result, not a bug.

---

## What's NEW Since Run 13

Three specs were implemented **after** Run 13 (commit `01d3124`):

### 1. Spec 063: Severity Inference (Implemented 2026-01-06)

**What it does**: Allows scoring from temporal/intensity markers without requiring explicit day-counts.

| Mode | Coverage (expected) | MAE (expected) |
|------|---------------------|----------------|
| `strict` (current) | ~48% | ~0.60 |
| `infer` (new) | ~70-80% | TBD (needs measurement) |

**CLI**: `--severity-inference infer`

**Why this matters**: The ~50% abstention rate is because transcripts lack explicit frequency. Inference mode may recover 20-30% more coverage.

### 2. Spec 061: Total Score Prediction (Implemented 2026-01-07)

**What it does**: Sum-of-items total score (0-24) with coverage gating.

**CLI**: `--prediction-mode total`

**Why this matters**: Item-level errors may average out; clinically useful for severity tiers.

### 3. Spec 062: Binary Classification (Partial)

**What it does**: PHQ-8 >= 10 threshold for depression screening.

**CLI**: `--prediction-mode binary`

**Status**: Only `threshold` strategy implemented; `direct` and `ensemble` fail loudly.

---

## Run 14: What To Test

### Option A: Severity Inference Ablation (Recommended First)

Test whether `--severity-inference infer` improves coverage without destroying accuracy.

```bash
tmux new -s run14

# Strict mode (baseline - should match Run 13)
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --severity-inference strict \
  --zero-shot-only \
  2>&1 | tee data/outputs/run14_strict_$(date +%Y%m%d_%H%M%S).log

# Infer mode (intervention)
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --severity-inference infer \
  --zero-shot-only \
  2>&1 | tee data/outputs/run14_infer_$(date +%Y%m%d_%H%M%S).log
```

**What to look for**:
- Coverage increase (target: >= 20 percentage points)
- MAE degradation (acceptable: < 0.15 increase)
- Inference rate (document in results)

### Option B: Total Score Evaluation

If you want to evaluate sum-of-items total score:

```bash
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --prediction-mode total \
  --total-min-coverage 0.5 \
  --zero-shot-only \
  2>&1 | tee data/outputs/run14_total_$(date +%Y%m%d_%H%M%S).log
```

### Option C: Full Few-Shot + Zero-Shot (Long Run)

If you want a complete new baseline with post-Run-13 code:

```bash
uv run python scripts/reproduce_results.py \
  --split paper-test \
  2>&1 | tee data/outputs/run14_both_$(date +%Y%m%d_%H%M%S).log
```

**Runtime**: ~20 hours (both modes)

---

## Preflight Checklist (Don't Skip)

```bash
# 1. Verify HF deps (this was Run 10's failure mode)
uv run python -c "import torch; print(torch.__version__)"

# 2. Verify transcripts exist
ls -d data/transcripts_participant_only/*_P | wc -l  # Should be ~189

# 3. Verify embeddings + sidecars exist
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.tags.json
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json

# 4. Dry-run to confirm config
uv run python scripts/reproduce_results.py --split paper-test --dry-run
# Verify header shows:
#   - Embeddings: huggingface_qwen3_8b_paper_train_participant_only.npz
#   - Tags: FOUND
#   - Chunk Scores: FOUND
#   - Reference Score Source: chunk
#   - Item Tag Filter: True
```

---

## Configuration Reference

### Key `.env` Settings

```bash
# Spec 063: Severity inference
QUANTITATIVE_SEVERITY_INFERENCE_MODE=strict  # or "infer"

# Spec 061: Prediction mode
PREDICTION_MODE=item  # or "total" or "binary"
TOTAL_SCORE_MIN_COVERAGE=0.5

# Spec 062: Binary threshold
BINARY_THRESHOLD=10
BINARY_STRATEGY=threshold
```

### CLI Overrides

| Setting | CLI Flag |
|---------|----------|
| Severity inference | `--severity-inference strict\|infer` |
| Prediction mode | `--prediction-mode item\|total\|binary` |
| Total min coverage | `--total-min-coverage 0.5` |

---

## Definition of Done

| Milestone | Status |
|-----------|--------|
| Paper MAE_item parity | ✅ few-shot 0.609 vs paper 0.619 (Run 8) |
| BUG-035 fixed | ✅ Run 13 is clean baseline |
| Confidence suite validated | ✅ Run 12 |
| AUGRC < 0.020 target | ❌ Best: 0.0216 (`token_energy`, Run 12) |
| Spec 063 tested | ❌ **Pending Run 14** |
| Spec 061 tested | ❌ **Pending Run 14** |

---

## Historical Context

### Why We Abandoned "Paper-Parity"

The original paper has **severe methodological flaws** (see closed issues #81, #69, #66, #47, #46, #45). We built a robust, independent implementation.

### Why Few-Shot ≤ Zero-Shot

Evidence grounding (Spec 053) rejects ~50% of quotes, starving few-shot of reference data. With strict grounding, zero-shot + consistency sampling is recommended.

See: `docs/results/few-shot-analysis.md`

---

## Related Documentation

- [Run History](docs/results/run-history.md) - All run details and metrics
- [Few-Shot Analysis](docs/results/few-shot-analysis.md) - Why few-shot may not beat zero-shot
- [Spec 061](docs/_specs/spec-061-total-phq8-score-prediction.md) - Total score prediction
- [Spec 063](docs/_specs/spec-063-severity-inference-prompt-policy.md) - Severity inference
- [Task Validity](docs/clinical/task-validity.md) - Construct mismatch analysis
