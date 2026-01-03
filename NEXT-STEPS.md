# Next Steps

**Status**: RUN 9 COMPLETE - Spec 046 Evaluated
**Last Updated**: 2026-01-03

---

## Why We Abandoned "Paper-Parity"

The original paper (Greene et al.) has **severe methodological flaws** that make reproduction impossible. We have built a **robust, independent implementation** that fixes these issues.

### Documented Failures (see closed GitHub issues)

| Issue | Problem |
|-------|---------|
| [#81](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/81) | Participant-level PHQ-8 scores assigned to individual chunks (semantic mismatch) |
| [#69](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/69) | Few-shot retrieval attaches participant scores to arbitrary text chunks |
| [#66](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/66) | Paper uses invalid statistical comparison (MAE at different coverages) |
| [#47](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/47) | Paper does not specify model quantization |
| [#46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46) | Paper does not specify temperature/top_k/top_p sampling parameters |
| [#45](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/45) | Paper uses undocumented custom 58/43/41 split |

### Reference Code Quality

The paper's reference implementation (`_reference/ai_psychiatrist/`) demonstrates:
- No configuration system (hardcoded paths)
- No error handling
- No tests
- No typing
- Inconsistent naming (e.g., `qualitive_evaluator.py`)
- Single-file "server" with no separation of concerns

**Our implementation** provides: Pydantic configuration, comprehensive tests (80%+ coverage), strict typing, structured logging, modular architecture, and proper experiment tracking.

### Terminology Change

| Old (deprecated) | New (use this) |
|------------------|----------------|
| "paper-parity" | "baseline/conservative defaults" |
| "paper-optimal" | "validated configuration" |
| "reproduce the paper" | "evaluate PHQ-8 assessment" |

---

## 1. Run 9 Results (Spec 046 Confidence Signals) ✅ COMPLETE

**Run 9 completed**: 2026-01-03T02:58:43

**File**: `data/outputs/both_paper-test_backfill-off_20260102_215843.json`

### Results Summary

| Mode | MAE_item | AURC | AUGRC | Cmax |
|------|----------|------|-------|------|
| Zero-shot | 0.776 | 0.144 | 0.032 | 48.8% |
| Few-shot | 0.662 | 0.135 | 0.035 | 53.0% |

### Spec 046 Confidence Signal Ablation (few-shot)

| Confidence Signal | AURC | vs baseline |
|-------------------|------|-------------|
| `llm` (evidence count) | 0.135 | — |
| `retrieval_similarity_mean` | **0.128** | **-5.4%** |
| `retrieval_similarity_max` | **0.128** | **-5.4%** |
| `hybrid_evidence_similarity` | 0.135 | +0.2% |

### Key Findings

1. **Retrieval similarity improves AURC by 5.4%**: `retrieval_similarity_mean` provides better ranking
2. **AUGRC unchanged**: Still at ~0.031-0.035 (target was <0.020)
3. **Hybrid signal not helpful**: Multiplying evidence × similarity doesn't help
4. **GitHub Issue #86 hypothesis partially validated**: Retrieval signals help AURC but don't substantially move AUGRC

---

## 2. Configuration Summary

All features are **gated by `.env`**. Copy `.env.example` to `.env` before running.

### Enabled Features (in `.env.example`)

| Feature | Setting | Value |
|---------|---------|-------|
| Chunk-level scoring | `EMBEDDING_REFERENCE_SCORE_SOURCE` | `chunk` |
| Item-tag filtering | `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `true` |
| Similarity threshold | `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.3` |
| Context limit | `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `500` |
| Participant-only transcripts | `DATA_TRANSCRIPTS_DIR` | `data/transcripts_participant_only` |
| HuggingFace embeddings (FP16) | `EMBEDDING_BACKEND` | `huggingface` |

### Code Defaults (conservative baseline)

Code defaults exist for testing and fallback only. They are NOT recommended for evaluation runs.

---

## 3. Spec 046 Implementation Status

**Status**: ✅ IMPLEMENTED AND TESTED (2026-01-03)

**What was added**:
- New fields in `ItemAssessment`: `retrieval_reference_count`, `retrieval_similarity_mean`, `retrieval_similarity_max`
- New confidence variants in `evaluate_selective_prediction.py`: `retrieval_similarity_mean`, `retrieval_similarity_max`, `hybrid_evidence_similarity`

**Run 9 Results**:
- `retrieval_similarity_mean` improves AURC by 5.4% vs `llm` (evidence count only)
- AUGRC did not materially improve (0.034 vs 0.035)
- `hybrid_evidence_similarity` did not help

---

## 4. Future Work (If Pursuing AUGRC <0.020)

Per GitHub Issue #86, the following phases were proposed:

### Phase 2: Verbalized Confidence (Medium Effort)
- Modify LLM prompt to request confidence rating (1-5) alongside score
- Apply temperature scaling calibration
- Expected: 20-40% AUGRC reduction

### Phase 3: Multi-Signal Ensemble (Higher Effort)
- Train logistic regression calibrator on [evidence, similarity, verbalized] → correctness
- Use paper-train split for calibration
- Expected: 30-50% AUGRC reduction

**Current recommendation**: Phase 1 (retrieval similarity) is now complete. Phase 2/3 require prompt engineering and additional training data. Evaluate whether AUGRC improvement is worth the effort.

---

## 5. Definition of Done

| Milestone | Status |
|-----------|--------|
| Paper MAE_item parity | ✅ few-shot 0.609 vs paper 0.619 |
| Chunk-level scoring (Spec 35) | ✅ Implemented |
| Participant-only preprocessing | ✅ Implemented |
| Retrieval confidence signals (Spec 046) | ✅ Tested (+5.4% AURC) |
| AUGRC < 0.020 target | ❌ Current best: 0.031 |

---
