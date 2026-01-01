# Paper Reproduction Results (Current Status)

**Last Updated**: 2026-01-01

This page is a high-level, **current-state** summary. The canonical timeline + per-run statistics live in:
- `docs/results/run-history.md`

---

## Current Status

- **Few-shot still underperforms zero-shot** under coverage-aware metrics (AURC/AUGRC) on retained paper-test runs.
- **Spec 33+34 ablation (Run 5)** did not improve few-shot; it regressed it.
- **Spec 37** removed the “hardcoded 120s query embedding timeout” regression (Run 4b).
- **Spec 35** is implemented; chunk scoring preprocessing is the next gating step (chunk scores are being generated offline).

---

## Current Best Retained Results (Paper-Test)

From `docs/results/run-history.md` (see Run 3 / Run 5):

| Run | Change | Zero-shot AURC | Few-shot AURC | Notes |
|-----|--------|----------------|---------------|------|
| Run 3 | Spec 31/32 | **0.134** | 0.193 | Best retained paper-parity-format baseline |
| Run 5 | Spec 33+34 | 0.138 | **0.213** | Guardrails + tags made few-shot worse |

**Interpretation**: Retrieval filters alone cannot fix the core label problem: participant-level scores attached to chunks. That requires Spec 35 (chunk-level scoring).

---

## Why We Use AURC/AUGRC (Not MAE)

When abstention/coverage differs across modes, raw MAE comparisons are invalid.

See:
- `docs/reference/statistical-methodology-aurc-augrc.md`
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

## What’s Blocking a Meaningful Few-Shot “Fix”

Few-shot retrieval can only be meaningfully evaluated after chunk labels are corrected:

- **Spec 35**: generates `{emb}.chunk_scores.json` + `{emb}.chunk_scores.meta.json`
- Enable with: `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`

See:
- `docs/data/chunk-scoring.md`
- `docs/guides/crag-validation-guide.md` (Spec 36; optional after chunk scores exist)

---

## Related Docs

- Feature defaults: `docs/reference/features.md`
- Configuration philosophy: `docs/configs/configuration-philosophy.md`
- Retrieval debugging: `docs/guides/debugging-retrieval-quality.md`
- Batch query embedding: `docs/guides/batch-query-embedding.md`
