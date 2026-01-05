# Next Steps

**Status**: Run 12 complete (valid); ready for Run 13
**Last Updated**: 2026-01-05

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

## 2. Run 10 Postmortem (Confidence Suite Attempt) ⚠️ INVALID

**Log**: `data/outputs/run10_confidence_suite_20260103_111959.log`
**Output**: `data/outputs/both_paper-test_20260103_182316.json`
**Run ID**: `3186a50d` (`git_dirty=true`)

**What this run was trying to do**: emit the full confidence suite signals (Specs 048–051) and then evaluate AURC/AUGRC deltas.

**Why it is not a valid comparison point**:

- **Zero-shot evaluated 39/41 participants**: PIDs 383 and 427 failed with `Exceeded maximum retries (3) for output validation` due to invalid control characters in LLM JSON output. This was fixed in `tolerant_json_fixups()` by adding step 5 (control char escaping).
- **Few-shot evaluated 0/41 participants**: every participant failed with missing HuggingFace deps (`torch`), because the run used `EMBEDDING_BACKEND=huggingface` without installing `--extra hf`.

**Never-again prevention** (now enforced):
- `make dev` installs HuggingFace extras by default (FP16 embeddings).
- Few-shot runs fail fast with `MissingHuggingFaceDependenciesError` before wasting hours.
- JSON parsing uses `parse_llm_json()` with `json-repair` as a last-resort fallback (Spec 059) and `PYDANTIC_AI_RETRIES=5` (Spec 058).

**What we can still learn from it** (debugging only):
- The run artifact contains the new `item_signals` keys (`verbalized_confidence`, `token_*`, `consistency_*`), so the instrumentation path works.
- Use it only as a “signals present” smoke test; do not interpret AURC/MAE deltas from this run.

**Next**: Run 12 completed and is the current SSOT for confidence-suite results (see below). Run 13 should focus on pushing AUGRC below the target (<0.020) via calibrated confidence (Spec 049) and/or additional robustness improvements.

---

## 3. Run 12 Results (Confidence Suite) ✅ VALID (No Selection Bias)

**Log**: `data/outputs/run12_confidence_suite_20260104_115021.log`
**Output**: `data/outputs/both_paper-test_20260105_072303.json`
**Failures**: `data/outputs/failures_05621949.json` (8 non-fatal `evidence_hallucination`)
**Telemetry**: `data/outputs/telemetry_05621949.json` (`json_fixups_applied` × 9; 0 parse failures)
**Run ID**: `05621949` (`git_dirty=false`, `git_commit=c0d79c5`)

### Reproduction Summary

| Mode | N_eval | MAE_item | Coverage |
|------|--------|----------|----------|
| Zero-shot | 41/41 | 0.5715 | 48.5% |
| Few-shot | 41/41 | 0.6159 | 46.0% |

### Selective Prediction Summary (`loss=abs_norm`, 10,000 bootstrap resamples)

| Mode | Confidence | AURC | AUGRC | Cmax |
|------|------------|------|-------|------|
| Zero-shot | `llm` | 0.1019 | 0.0252 | 48.5% |
| Zero-shot | best AURC: `verbalized` | 0.0917 | 0.0257 | 48.5% |
| Zero-shot | best AUGRC: `token_pe` | 0.0932 | 0.0234 | 48.5% |
| Few-shot | `llm` | 0.1085 | 0.0242 | 46.0% |
| Few-shot | best AURC/AUGRC: `token_energy` | 0.0862 | 0.0216 | 46.0% |

**All variants**:
- Zero-shot: `data/outputs/selective_prediction_metrics_run12_zero_shot_all.json`
- Few-shot: `data/outputs/selective_prediction_metrics_run12_few_shot_all.json`
- Paired (few − zero, default): `data/outputs/selective_prediction_metrics_run12_paired_default.json`

---

## 4. Configuration Summary

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

## 5. Spec 046 Implementation Status

**Status**: ✅ IMPLEMENTED AND TESTED (2026-01-03)

**What was added**:
- New fields in `ItemAssessment`: `retrieval_reference_count`, `retrieval_similarity_mean`, `retrieval_similarity_max`
- New confidence variants in `evaluate_selective_prediction.py`: `retrieval_similarity_mean`, `retrieval_similarity_max`, `hybrid_evidence_similarity`

**Run 9 Results**:
- `retrieval_similarity_mean` improves AURC by 5.4% vs `llm` (evidence count only)
- AUGRC did not materially improve (0.034 vs 0.035)
- `hybrid_evidence_similarity` did not help

---

## 6. Few-Shot vs Zero-Shot Analysis (Run 12 Finding)

**Key Finding**: In Run 12, zero-shot **outperformed** few-shot (MAE 0.572 vs 0.616) at similar coverage. This is a valid research result, not a bug.

### Root Cause

Evidence grounding (Spec 053) rejects ~50% of extracted quotes, starving few-shot of reference data:

| Metric | Run 8 (few-shot won) | Run 12 (zero-shot won) |
|--------|---------------------|------------------------|
| Items with LLM evidence | 61.3% | 32.0% |
| Items with references | Higher | 15.2% |
| Consistency sampling | No | Yes |

### Why Few-Shot Can Be Neutral or Harmful

1. **Evidence-limited, not knowledge-limited**: PHQ-8 needs frequency; RAG can't add missing evidence
2. **Embedding similarity ≠ severity similarity**: Retrieved examples match topic, not severity
3. **Anchoring harm**: Reference scores can dominate test evidence
4. **Sparse references**: After guardrails, most items get 0-1 exemplars

See: `docs/results/few-shot-analysis.md` for full analysis.

### Recommendation

For DAIC-WOZ PHQ-8 scoring with strict evidence grounding, **zero-shot with consistency sampling is the recommended approach**.

---

## 7. Future Work (If Pursuing AUGRC <0.020)

Specs 048–051 are implemented and validated (Run 12). The next step is to push AUGRC further via:
- **Spec 049 calibrated confidence** (train on non-test data, then evaluate on paper-test)
- Optional: systematic tuning of consistency sampling settings (n, temperature) and re-run evaluation
  - Prefer paired comparisons + all-variant evaluation for claims

### What Run 13 should test

| Spec | Capability | Where it shows up |
|------|------------|-------------------|
| 048 | Verbalized confidence (1–5) | `item_signals[*]["verbalized_confidence"]` |
| 049 | Supervised calibrator | `scripts/evaluate_selective_prediction.py --confidence calibrated --calibration <artifact>` |
| 050 | Consistency-based confidence (multi-sample) | `item_signals[*]["consistency_*"]` (requires consistency enabled) |
| 051 | Token-level CSFs from logprobs | `item_signals[*]["token_msp|token_pe|token_energy"]` (backend-dependent) |

### Run 13 checklist (don’t skip)

1. Preflight: confirm the validated configuration is active
   - `cp .env.example .env` (if needed)
   - If `EMBEDDING_BACKEND=huggingface`: install deps + verify they load (this was the Run 10 failure mode):
     - `make dev`
     - `uv run python -c "import torch; print(torch.__version__)"`
   - Confirm transcripts exist: `ls -d data/transcripts_participant_only/*_P | wc -l`
   - `uv run python scripts/reproduce_results.py --split paper-test --dry-run`
   - Confirm the header shows:
     - `Embeddings Artifact: data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz`
     - `Tags Sidecar: ...tags.json (FOUND)`
     - `Chunk Scores Sidecar: ...chunk_scores.json (FOUND)`
     - `Reference Score Source: chunk`
     - `Item Tag Filter: True`
     - `Min Reference Similarity: 0.3`
     - `Max Reference Chars Per Item: 500`

   Token CSF readiness (Spec 051):
   - Token-level confidence signals require the LLM backend to return per-token logprobs via the OpenAI-compatible `/v1` API.
   - Verified locally on `ollama` `0.13.5` + `pydantic_ai` `1.39.0`: logprobs are returned and exposed as `result.response.provider_details["logprobs"]`.
   - The code requests logprobs automatically during scoring; if you are on an older Ollama version or a different backend, logprobs may be absent.
   - To remove uncertainty, run a 1-participant smoke test and check the output:

     ```bash
     uv run python scripts/reproduce_results.py --split paper-test --few-shot-only --limit 1
     # Then confirm token signals exist in the saved JSON:
     rg -n '\"token_msp\"|\"token_pe\"|\"token_energy\"' data/outputs/both_*.json | head
     ```

   - If the run artifact contains no `token_*` keys, skip token variants for this run (the evaluator will fail fast by design).

2. Enable consistency signals (Spec 050)
   - `.env.example` enables consistency by default for the confidence suite:
     - `CONSISTENCY_ENABLED=true`
     - `CONSISTENCY_N_SAMPLES=5`
     - `CONSISTENCY_TEMPERATURE=0.2`
   - Optional: override via CLI (explicit in the run log):
     - `--consistency-samples 10` (tighter agreement estimate)

3. Run in tmux

   ```bash
   tmux new -s run13
   uv run python scripts/reproduce_results.py \
     --split paper-test \
     2>&1 | tee data/outputs/run13_confidence_suite_$(date +%Y%m%d_%H%M%S).log
   ```

4. Evaluate selective prediction (compare confidence variants)

   Use the JSON output path printed by `reproduce_results.py` (the `both_*.json` file), then run:

   ```bash
   uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence llm
    uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence retrieval_similarity_mean
    uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence verbalized
    uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence hybrid_verbalized
    uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence consistency
    uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence hybrid_consistency
    ```

   Notes:
   - `token_msp|token_pe|token_energy` variants only work when the backend returns token logprobs; if the run artifact doesn’t include `item_signals[*][\"token_*\"]`, those variants will fail fast (by design).
   - If `token_*` keys are present, also evaluate:

     ```bash
     uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence token_msp
     uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence token_pe
     uv run python scripts/evaluate_selective_prediction.py --input <both_run.json> --mode few_shot --confidence token_energy
     ```

   - For multi-signal calibration (Spec 049), train a calibrator on a training output and then re-evaluate using `--confidence calibrated`.

---

## 8. Definition of Done

| Milestone | Status |
|-----------|--------|
| Paper MAE_item parity | ✅ few-shot 0.609 vs paper 0.619 (Run 8) |
| Chunk-level scoring (Spec 35) | ✅ Implemented |
| Participant-only preprocessing | ✅ Implemented |
| Retrieval confidence signals (Spec 046) | ✅ Tested (+5.4% AURC) |
| Confidence improvement suite (Specs 048–051) | ✅ Implemented + validated (Run 12) |
| Evidence grounding (Spec 053) | ✅ Implemented |
| AUGRC < 0.020 target | ❌ Current best: 0.0216 (`token_energy`, Run 12) |
| Few-shot analysis documented | ✅ See `docs/results/few-shot-analysis.md` |

---
