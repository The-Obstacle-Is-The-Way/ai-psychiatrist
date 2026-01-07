# Paper Reproduction Results (Current Status)

**Last Updated**: 2026-01-07

This page is a high-level, **current-state** summary. The canonical timeline + per-run statistics live in:

- `docs/results/run-history.md`

---

## ⚠️ Run Integrity Warnings

### Prompt Confound Bug (BUG-035) - Fixed 2026-01-06

A prompt confound was discovered and fixed on 2026-01-06 where few-shot mode produced different prompts than zero-shot **even when retrieval returned zero references**.

**Impact**: Runs 1–12 are confounded for zero-shot vs few-shot comparisons.

**Clean baseline**: Run 13 (`data/outputs/both_paper-test_20260107_134730.json`) is the first valid post-fix comparative run.

See: `docs/_archive/bugs/BUG-035_FEW_SHOT_PROMPT_CONFOUND.md` and `docs/results/run-history.md`.

### Silent Fallback Bug (ANALYSIS-026) - Fixed 2026-01-03

**A silent fallback bug was discovered on 2026-01-03** that could have caused few-shot mode to silently degrade to zero-shot if JSON parsing failed. Runs 1-9 may be affected. Run 10 started with pre-fix code.

**For publication-quality results, re-run with post-fix code.**

See: `docs/_archive/bugs/ANALYSIS-026_JSON_PARSING_ARCHITECTURE_AUDIT.md` and `docs/results/run-history.md` for details.

**Run 10 (confidence suite attempt) is not a valid comparison point**:
- Zero-shot evaluated 39/41 participants (2 hard failures).
- Few-shot evaluated 0/41 participants due to missing HuggingFace optional deps (`torch`).

Use `docs/results/run-history.md` as the SSOT. Run 11 is diagnostic-only (selection bias); Run 12 is a complete confidence-suite run (41/41 evaluated in both modes) but is pre-BUG-035; Run 13 is the first clean post-BUG-035 comparative run.

---

## Current Status

- **Participant-only transcript preprocessing evaluated** (Run 8)
- **Historical paper reference** (Run 8, pre-BUG-035): few-shot `0.609` vs paper `0.619`; zero-shot `0.776` vs paper `0.796`
- **Clean comparative baseline established (Run 13, post-BUG-035)**: zero-shot MAE_item `0.6079`; few-shot MAE_item `0.6571` (zero-shot wins)
- **Selective prediction**: AURC/AUGRC are very similar between modes (paired ΔAURC CI overlaps 0)
- **Spec 046 evaluated** (Run 9): `retrieval_similarity_mean` improves AURC by 5.4% vs evidence-count-only
- **Confidence Suite validated (Run 12)**: 41/41 evaluated in both modes; token-level CSFs improve AURC/AUGRC over `llm` (pre-BUG-035)
- **AUGRC target not reached (yet)**: Best artifact-free AUGRC is 0.0216 (`token_energy`, Run 12; target was <0.020 per Issue #86)
- **Tradeoff**: coverage ceiling is ~46–49% in Run 12 (vs ~66% in Run 7), indicating more abstention

### Few-Shot vs Zero-Shot (Run 13 Baseline)

In Run 13 (first clean post-BUG-035 comparative run), **zero-shot outperformed few-shot** (MAE_item 0.6079 vs 0.6571) at similar coverage. This is a valid research result explained by:

1. Evidence grounding (Spec 053) rejects ~50% of quotes, starving few-shot of reference data
2. Consistency sampling benefits zero-shot more than few-shot
3. PHQ-8 scoring is evidence-limited, not knowledge-limited

**Recommendation**: Zero-shot with consistency sampling is the recommended approach.

See: [Few-Shot Analysis](few-shot-analysis.md) for full details.

For the underlying construct-validity constraint (PHQ-8 frequency vs transcript evidence), see: `docs/clinical/task-validity.md`.

---

## Current Best Retained Results (Paper-Test)

From `docs/results/run-history.md` (default `confidence=llm` unless noted). Note: runs prior to Run 13 are pre-BUG-035 and should not be used for zero-shot vs few-shot comparative claims.

| Run | Change | Zero-shot AURC | Few-shot AURC | Notes |
|-----|--------|----------------|---------------|------|
| Run 3 | Spec 31/32 | **0.134** | 0.193 | Best zero-shot baseline |
| Run 5 | Spec 33+34 | 0.138 | 0.213 | Guardrails + tags made few-shot worse |
| Run 7 | Spec 35 | 0.138 | **0.151** | Chunk scoring: 29% improvement |
| Run 8 | Participant-only transcripts | 0.141 | **0.125** | Lower Cmax (~49% / ~51%) |
| Run 9 | Spec 046 confidence signals | 0.144 | 0.135 (0.128 w/ similarity) | 5.4% AURC improvement with retrieval similarity |
| Run 12 | Confidence suite (Specs 048–052) | **0.102** | 0.109 | Complete run; pre-BUG-035 |
| Run 13 | Post-BUG-035 baseline | 0.107 | 0.115 | First clean comparative run |

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

# Optional evaluation modes (Specs 061/062)
uv run python scripts/reproduce_results.py --split dev --prediction-mode total
uv run python scripts/reproduce_results.py --split dev --prediction-mode binary
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
- `docs/rag/chunk-scoring.md`
- `docs/rag/runtime-features.md` (includes Spec 36 CRAG validation)

---

## Related Docs

- Feature defaults: `docs/pipeline-internals/features.md`
- Configuration philosophy: `docs/configs/configuration-philosophy.md`
- RAG debugging: `docs/rag/debugging.md`
- RAG runtime features: `docs/rag/runtime-features.md`
