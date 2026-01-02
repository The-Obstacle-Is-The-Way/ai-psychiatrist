# Next Steps

**Status**: READY FOR RUN 9
**Last Updated**: 2026-01-02

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

## 1. Immediate Action: Run 9 (Spec 046 Confidence Signals)

**Why**: Spec 046 adds retrieval similarity signals to run outputs, enabling new confidence variants for AURC/AUGRC evaluation. Run 8 was produced BEFORE Spec 046 was implemented, so it does NOT contain these signals.

**Prerequisites verified** (all present):

| Artifact | Path | Status |
|----------|------|--------|
| Embeddings NPZ | `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz` | 68 MB |
| Text chunks | `data/embeddings/...participant_only.json` | 2.1 MB |
| Metadata | `data/embeddings/...participant_only.meta.json` | 432 B |
| Item tags (Spec 34) | `data/embeddings/...participant_only.tags.json` | 48 KB |
| Chunk scores (Spec 35) | `data/embeddings/...participant_only.chunk_scores.json` | 1.1 MB |
| Chunk scores meta | `data/embeddings/...participant_only.chunk_scores.meta.json` | 284 B |
| Transcripts | `data/transcripts_participant_only/` | 190 dirs |

**Command**:

```bash
# In tmux for long runs (~2-3 hours):
tmux new -s run9
uv run python scripts/reproduce_results.py \
  --split paper-test \
  2>&1 | tee data/outputs/run9_spec046_$(date +%Y%m%d_%H%M%S).log
```

**After Run 9, evaluate with new confidence variants**:

```bash
# Compare confidence signals for few-shot
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<RUN9_OUTPUT>.json \
  --mode few_shot \
  --confidence llm \
  --seed 42

uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<RUN9_OUTPUT>.json \
  --mode few_shot \
  --confidence retrieval_similarity_mean \
  --seed 42

uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<RUN9_OUTPUT>.json \
  --mode few_shot \
  --confidence hybrid_evidence_similarity \
  --seed 42
```

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

**Status**: IMPLEMENTED (2026-01-02)

**What was added**:
- New fields in `ItemAssessment`: `retrieval_reference_count`, `retrieval_similarity_mean`, `retrieval_similarity_max`
- New confidence variants in `evaluate_selective_prediction.py`: `retrieval_similarity_mean`, `retrieval_similarity_max`, `hybrid_evidence_similarity`

**What Run 9 will test**:
- Whether retrieval-grounded confidence signals improve AURC/AUGRC vs evidence-count-only
- Hypothesis: `hybrid_evidence_similarity` should be a better ranking signal than `llm` alone

---

## 4. Post-Run 9 Work

1. **Analyze results**: Compare AURC/AUGRC across confidence variants
2. **Document findings**: Update `docs/results/run-history.md` with Run 9
3. **If improvement**: Consider Phase 2 work (verbalized confidence, calibrator)
4. **If no improvement**: Investigate alternative signals

---

## 5. Senior Review Request

Before launching Run 9, request senior review of:
1. This NEXT-STEPS.md document
2. Configuration in `.env.example` vs code defaults
3. Spec 046 implementation completeness
4. All root documentation files (CLAUDE.md, AGENTS.md, GEMINI.md)

---

*Delete this file after Run 9 is complete and documented.*
