# Paper Reproduction Results (Current Status)

**Last Updated**: 2026-01-01

This page is a high-level, **current-state** summary. The canonical timeline + per-run statistics live in:
- `docs/results/run-history.md`

---

## Current Status

- **Few-shot gap closed to 9%** - CIs now overlap (statistically indistinguishable)
- **Few-shot MAE beats zero-shot** (0.639 vs 0.698) - first time in reproduction history
- **Spec 35 chunk-level scoring** improved few-shot AURC by 29% (Run 7 vs Run 5)
- **Remaining lever**: Participant-only transcript preprocessing to improve retrieval quality

---

## Current Best Retained Results (Paper-Test)

From `docs/results/run-history.md` (Run 3 zero-shot / Run 7 few-shot):

| Run | Change | Zero-shot AURC | Few-shot AURC | Notes |
|-----|--------|----------------|---------------|------|
| Run 3 | Spec 31/32 | **0.134** | 0.193 | Best zero-shot baseline |
| Run 5 | Spec 33+34 | 0.138 | 0.213 | Guardrails + tags made few-shot worse |
| Run 7 | Spec 35 | 0.138 | **0.151** | Chunk scoring: 29% improvement |

**Winner**: Zero-shot still best on AURC (0.134 vs 0.151), but **few-shot has better MAE** (0.639 vs 0.698).

**Interpretation**: Spec 35 chunk-level scoring fixed the core label problem. The remaining gap is within statistical noise.

---

## Why We Use AURC/AUGRC (Not MAE)

When abstention/coverage differs across modes, raw MAE comparisons are invalid.

See:
- `docs/statistics/statistical-methodology-aurc-augrc.md`
- `docs/statistics/metrics-and-evaluation.md`

---

## How To Run and Evaluate

### 1) Run reproduction

```bash
# Both modes
uv run python scripts/reproduce_results.py --split paper-test

# Single mode
uv run python scripts/reproduce_results.py --split paper-test --zero-shot-only
uv run python scripts/reproduce_results.py --split paper-test --few-shot-only
```

The runner prints the saved path:

```text
Results saved to: data/outputs/{mode}_{split}_backfill-{on,off}_{YYYYMMDD_HHMMSS}.json
```

### 2) Compute selective prediction metrics

```bash
uv run python scripts/evaluate_selective_prediction.py --input data/outputs/YOUR_OUTPUT.json
```

---

## Spec 35 Chunk-Level Scoring (Now Enabled)

Chunk-level scoring is enabled when `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk` (default: `participant`).

Generated artifacts:
- `data/embeddings/ollama_qwen3_8b_paper_train.chunk_scores.json`
- `data/embeddings/ollama_qwen3_8b_paper_train.chunk_scores.meta.json`

See:
- `docs/embeddings/chunk-scoring.md`
- `docs/statistics/crag-validation-guide.md` (Spec 36; optional validation layer)

---

## Related Docs

- Feature defaults: `docs/pipeline-internals/features.md`
- Configuration philosophy: `docs/configs/configuration-philosophy.md`
- Retrieval debugging: `docs/embeddings/debugging-retrieval-quality.md`
- Batch query embedding: `docs/embeddings/batch-query-embedding.md`
