# Paper Reproduction Results (Current Status)

**Last Updated**: 2026-01-03

This page is a high-level, **current-state** summary. The canonical timeline + per-run statistics live in:
- `docs/results/run-history.md`

---

## Current Status

- **Participant-only transcript preprocessing evaluated** (Run 8)
- **Paper MAE_item parity achieved**: few-shot `0.609` vs paper `0.619`; zero-shot `0.776` vs paper `0.796`
- **Selective prediction**: AURC/AUGRC are very similar between modes (paired ΔAURC CI overlaps 0)
- **Spec 046 evaluated** (Run 9): `retrieval_similarity_mean` improves AURC by 5.4% vs evidence-count-only
- **AUGRC target not reached**: Best AUGRC is 0.031 (target was <0.020 per Issue #86)
- **Tradeoff**: overall Cmax at ~51% (vs ~66% in Run 7), indicating more abstention

---

## Current Best Retained Results (Paper-Test)

From `docs/results/run-history.md` (Run 3 / Run 7 / Run 8 / Run 9):

| Run | Change | Zero-shot AURC | Few-shot AURC | Notes |
|-----|--------|----------------|---------------|------|
| Run 3 | Spec 31/32 | **0.134** | 0.193 | Best zero-shot baseline |
| Run 5 | Spec 33+34 | 0.138 | 0.213 | Guardrails + tags made few-shot worse |
| Run 7 | Spec 35 | 0.138 | **0.151** | Chunk scoring: 29% improvement |
| Run 8 | Participant-only transcripts | 0.141 | **0.125** | Lower Cmax (~49% / ~51%) |
| Run 9 | Spec 046 confidence signals | 0.144 | 0.135 (0.128 w/ similarity) | 5.4% AURC improvement with retrieval similarity |

**Interpretation**: Run 8 closes the remaining "participant-only preprocessing" lever, but does so by substantially lowering coverage. AURC/Cmax must be interpreted together.

**Interpretation**: Spec 35 chunk-level scoring fixed the core label problem. Spec 046 retrieval similarity provides modest additional improvement.

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
Results saved to: data/outputs/{mode}_{split}_{YYYYMMDD_HHMMSS}.json
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
- `docs/embeddings/crag-validation.md` (Spec 36; optional validation layer)

---

## Related Docs

- Feature defaults: `docs/pipeline-internals/features.md`
- Configuration philosophy: `docs/configs/configuration-philosophy.md`
- Retrieval debugging: `docs/embeddings/debugging-retrieval-quality.md`
- Batch query embedding: `docs/embeddings/batch-query-embedding.md`
