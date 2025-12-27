# Paper Reproduction Results

**Last Updated**: 2025-12-27
**Status**: Validated zero-shot + few-shot runs complete (paper split).

---

## Executive Summary

| Mode | Our MAE | Paper MAE | Delta | Coverage | Participants |
|------|---------|-----------|-------|----------|--------------|
| **Zero-shot** | **0.717** | 0.796 | **-0.079 (better)** | 56.9% | 40/41 |
| **Few-shot** | **0.860** | 0.619 | **+0.241 (worse)** | 71.6% | 40/41 |

**Key findings**:
- Our zero-shot baseline **beats the paper's zero-shot MAE** (0.717 vs 0.796).
- Our few-shot run has **much higher coverage** than the paper's few-shot (~50%), and **worse MAE**.
  This is a coverage/MAE tradeoff issue (selective prediction), not a simple apples-to-apples comparison.

---

## Validated Run: Zero-Shot (2025-12-26)

### Run Metadata

| Field | Value |
|-------|-------|
| Run ID | `92c7a6f2` |
| Git Commit | `5b8f588` |
| Git Dirty | `false` |
| Timestamp | 2025-12-26T18:57:47 |
| Output File | `data/outputs/zero_shot_paper_backfill-off_20251226_201946.json` |

### Configuration

| Setting | Value | Paper Reference |
|---------|-------|-----------------|
| Quantitative Model | `gemma3:27b-it-qat` | Gemma 3 27B (Section 2.2) |
| Embedding Model | `qwen3-embedding:8b` | Qwen 3 8B Embedding (Section 2.2) |
| LLM Backend | `ollama` | Local inference |
| Embedding Backend | `huggingface` | FP16 precision |
| Keyword Backfill | `OFF` | Paper parity (Section 3.2) |
| Split | `paper` | 41 test participants |

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Item MAE (by item)** | **0.717** | Primary metric for paper comparison |
| Item MAE (weighted) | 0.698 | Weighted by item coverage |
| Item MAE (by subject) | 0.640 | Averaged per-subject first |
| Coverage | 56.9% | Items with predictions (vs N/A) |
| Evaluated | 40/41 | 1 excluded (all N/A) |
| Runtime | 82 min | M1 Max 64GB |

### Per-Item Breakdown

| PHQ-8 Item | MAE | Coverage | N/A Count | Notes |
|------------|-----|----------|-----------|-------|
| NoInterest | 0.55 | 50% | 20 | Interest/anhedonia |
| Depressed | 0.52 | 82.5% | 7 | Feeling down/hopeless |
| **Sleep** | **0.61** | **95%** | 2 | Best coverage |
| Tired | 0.90 | 75% | 10 | Fatigue/low energy |
| Appetite | 0.75 | **20%** | 32 | Lowest coverage (paper confirms) |
| Failure | 0.78 | 67.5% | 13 | Self-blame/worthlessness |
| Concentrating | 0.94 | 40% | 24 | Difficulty focusing |
| Moving | 0.70 | **25%** | 30 | Psychomotor issues (low coverage) |

**Observations**:
- Sleep has the highest coverage (95%) - frequently discussed in interviews
- Appetite has low coverage (~20%) - consistent with the paper noting appetite evidence is rarely found during retrieval
- Concentrating has highest MAE (0.94) despite moderate coverage

### Excluded Participant

**Participant 339**: Excluded from evaluation (8/8 N/A items)
- Ground truth total: 11 (moderate depression)
- The model found no explicit symptom evidence in the transcript
- This is expected behavior per paper methodology - conservative N/A handling

---

## Validated Run: Few-Shot (2025-12-27)

### Run Metadata

| Field | Value |
|-------|-------|
| Run ID | `507f129e` |
| Git Commit | `f6d2653` |
| Git Dirty | `true` |
| Timestamp | 2025-12-26T21:52:50 |
| Output File | `data/outputs/few_shot_paper_backfill-off_20251227_000125.json` |

### Configuration

| Setting | Value | Paper Reference |
|---------|-------|-----------------|
| Quantitative Model | `gemma3:27b-it-qat` | Gemma 3 27B (Section 2.2) |
| Embedding Model | `qwen3-embedding:8b` | Qwen 3 8B Embedding (Section 2.2) |
| LLM Backend | `ollama` | Local inference |
| Embedding Backend | `huggingface` | FP16 precision |
| Keyword Backfill | `OFF` | Paper parity (Section 3.2) |
| Split | `paper` | 41 test participants |
| Embeddings Artifact | `data/embeddings/huggingface_qwen3_8b_paper_train.npz` | Paper-train reference set (58) |

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Item MAE (by item)** | **0.860** | Primary metric for paper comparison |
| Item MAE (weighted) | 0.852 | Weighted by item coverage |
| Item MAE (by subject) | 0.831 | Averaged per-subject first |
| Coverage | 71.6% | Items with predictions (vs N/A) |
| Evaluated | 40/41 | 1 failed (timeout) |
| Runtime | 128 min | M1 Max 64GB |

### Per-Item Breakdown

| PHQ-8 Item | MAE | Coverage | N/A Count |
|------------|-----|----------|-----------|
| NoInterest | 0.73 | 82.5% | 7 |
| Depressed | 0.56 | 90.0% | 4 |
| Sleep | 1.03 | 95.0% | 2 |
| Tired | 0.97 | 85.0% | 6 |
| Appetite | 1.18 | 42.5% | 23 |
| Failure | 0.83 | 72.5% | 11 |
| Concentrating | 1.17 | 57.5% | 17 |
| Moving | 0.42 | 47.5% | 21 |

---

## Model & Embedding Configuration

See `docs/models/model-registry.md` for full configuration details.

### Chat Model: Gemma 3 27B

| Aspect | Our Setup | Paper |
|--------|-----------|-------|
| Model | `gemma3:27b-it-qat` | Gemma 3 27B |
| Quantization | QAT 4-bit (17GB) | Likely BF16 (54GB) |
| Backend | Ollama | Unknown |

**Note**: QAT (Quantization-Aware Training) is Google's optimized 4-bit format, claiming near-BF16 quality at Q4 size.

### Embedding Model: Qwen 3 8B

| Aspect | Our Setup | Paper |
|--------|-----------|-------|
| Model | `qwen3-embedding:8b` | Qwen 3 8B Embedding |
| Precision | FP16 (HuggingFace) | Unknown |
| Dimension | 4096 | 4096 (Appendix D) |

---

## Comparison with Paper

### Zero-Shot Baseline

| Metric | Ours | Paper | Delta |
|--------|------|-------|-------|
| Item MAE | 0.717 | 0.796 | **-0.079 (10% better)** |
| Coverage | 56.9% | ~50% | +6.9% |

**Interpretation**: Our zero-shot MAE is **better than the paper's zero-shot baseline**. This could be due to:
1. QAT quantization performing well for this task
2. HuggingFace FP16 embeddings (better similarity precision)
3. Minor prompt/parsing differences

### Deviations from Paper

| Aspect | Ours | Paper | Impact |
|--------|------|-------|--------|
| Quantization | QAT 4-bit | Likely BF16 | Unknown (QAT claims BF16 parity) |
| Embedding precision | FP16 | Unknown | Likely positive |
| Runtime | 82 min | "~1 min" (Section 2.3.5) | Different hardware/batch |

---

## Aberrations & Known Issues

### PID 339: Excluded (8/8 N/A)

The model could not make any predictions for participant 339:
- Ground truth: 11 (moderate depression)
- All 8 items returned N/A
- The interview transcript may not contain explicit symptom discussions

**This is expected behavior** - the paper methodology is conservative and returns N/A when evidence is insufficient.

### BUG-025: Missing PHQ8_Sleep (Resolved)

Participant 319 was missing `PHQ8_Sleep` in the upstream AVEC2017 dataset. This was deterministically reconstructed to `2` based on the mathematical invariant `PHQ8_Score = sum(items)`.

See: `docs/bugs/bug-025-missing-phq8-ground-truth-paper-test.md`

---

## Reproduction Commands

### Prerequisites

```bash
# 1. Install dependencies
make dev

# 2. Pull required models
ollama pull gemma3:27b-it-qat
ollama pull qwen3-embedding:8b

# 3. Create paper splits (58/43/41)
uv run python scripts/create_paper_split.py --verify

# 4. Generate reference embeddings for few-shot (optional, ~65 min)
uv run python scripts/generate_embeddings.py --split paper-train
```

### Run Evaluation

```bash
# Zero-shot only
uv run python scripts/reproduce_results.py --split paper --zero-shot-only

# Few-shot only (requires embeddings)
uv run python scripts/reproduce_results.py --split paper --few-shot-only

# Both modes
uv run python scripts/reproduce_results.py --split paper
```

### Output Artifacts

| File | Description |
|------|-------------|
| `data/outputs/zero_shot_*.json` | Full results with per-participant predictions |
| `data/outputs/few_shot_*.json` | Full results with per-participant predictions |
| `data/experiments/registry.yaml` | Run metadata and summary metrics |

---

## Next Steps

1. [ ] Complete few-shot evaluation (in progress)
2. [ ] Compare few-shot MAE with paper's 0.619
3. [ ] Investigate PID 339 transcript to understand 8/8 N/A
4. [ ] Consider Q8_0 quantization for closer paper parity (if MAE differs significantly)

---

## References

- Paper: `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`
- Section 3.2: Quantitative Assessment results
- Appendix D: Hyperparameter optimization (embedding dimension, top-k)
- Appendix E: Retrieval statistics (confirms Appetite/Moving low coverage)
- Appendix F: MedGemma comparison (0.505 MAE, more N/A)
