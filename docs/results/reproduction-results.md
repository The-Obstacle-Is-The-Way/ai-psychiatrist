# Paper Reproduction Results (Current Status)

**Last Updated**: 2026-01-02

This page is a high-level, **current-state** summary. The canonical timeline + per-run statistics live in:
- `docs/results/run-history.md`

---

## Current Status

- **Participant-only transcript preprocessing evaluated** (Run 8)
- **Paper MAE_item parity achieved**: few-shot `0.609` vs paper `0.619`; zero-shot `0.776` vs paper `0.796`
- **Selective prediction**: AURC/AUGRC are very similar between modes in Run 8 (paired ΔAURC CI overlaps 0), suggesting confidence/abstention quality is not materially different
- **Tradeoff**: overall Cmax dropped to ~51% in Run 8 (vs ~66% in Run 7), indicating more abstention
- **Next lever**: improve confidence signals for AURC/AUGRC (Spec 046: `docs/_specs/spec-046-selective-prediction-confidence-signals.md`)
- **Run 9**: in progress (Spec 046 confidence variants; see `docs/results/run-history.md`)

---

## Current Best Retained Results (Paper-Test)

From `docs/results/run-history.md` (Run 3 / Run 7 / Run 8):

| Run | Change | Zero-shot AURC | Few-shot AURC | Notes |
|-----|--------|----------------|---------------|------|
| Run 3 | Spec 31/32 | **0.134** | 0.193 | Best zero-shot baseline |
| Run 5 | Spec 33+34 | 0.138 | 0.213 | Guardrails + tags made few-shot worse |
| Run 7 | Spec 35 | 0.138 | **0.151** | Chunk scoring: 29% improvement |
| Run 8 | Participant-only transcripts | 0.141 | **0.125** | Lower Cmax (~49% / ~51%) |

**Interpretation**: Run 8 closes the remaining “participant-only preprocessing” lever, but does so by substantially lowering coverage. AURC/Cmax must be interpreted together.

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

Chunk-level scoring is enabled when `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk` (code default: `participant`; `.env.example` uses `chunk` for the participant-only pipeline).

Generated artifacts:
- `data/embeddings/ollama_qwen3_8b_paper_train.chunk_scores.json`
- `data/embeddings/ollama_qwen3_8b_paper_train.chunk_scores.meta.json`
- `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json`
- `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json`

See:
- `docs/embeddings/chunk-scoring.md`
- `docs/statistics/crag-validation-guide.md` (Spec 36; optional validation layer)

---

## Related Docs

- Feature defaults: `docs/pipeline-internals/features.md`
- Configuration philosophy: `docs/configs/configuration-philosophy.md`
- Retrieval debugging: `docs/embeddings/debugging-retrieval-quality.md`
- Batch query embedding: `docs/embeddings/batch-query-embedding.md`
