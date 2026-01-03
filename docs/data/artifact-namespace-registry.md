# Artifact Namespace Registry

**Purpose**: Single source of truth for all data artifacts, scripts, and naming conventions
**Created**: 2025-12-23
**Status**: DRAFT - Identified during reproduction run

---

## Naming Convention Summary

| Prefix | Meaning | Example |
|--------|---------|---------|
| (none) | AVEC2017 official splits | `reference_embeddings.npz` |
| `paper_` | Paper-style custom splits | `paper_reference_embeddings.npz` |
| `{backend}_...` | Embedding generator output | `huggingface_qwen3_8b_paper_train_participant_only.npz` |

**Note**: `scripts/generate_embeddings.py` defaults to `{backend}_{model_slug}_{split}.npz` naming and writes an optional `.meta.json`. For collision-free runs, include a transcript-variant suffix (e.g., `_participant_only`) in the output name.
Legacy filenames like `paper_reference_embeddings.npz` are still supported (use `--output` to regenerate with a specific name).

---

## Data Splits

### AVEC2017 Official Splits (Default)

| File | Participants | Per-Item PHQ-8 | Usage |
|------|-------------|----------------|-------|
| `data/train_split_Depression_AVEC2017.csv` | 107 | Yes | Training/few-shot knowledge base |
| `data/dev_split_Depression_AVEC2017.csv` | 35 | Yes | Default evaluation |
| `data/test_split_Depression_AVEC2017.csv` | 47 | No | Competition (unusable for item MAE) |
| `data/full_test_split.csv` | 47 | Total only | Competition alt |

### Paper Custom Splits

| File | Participants | Per-Item PHQ-8 | Usage |
|------|-------------|----------------|-------|
| `data/paper_splits/paper_split_train.csv` | 58 | Yes | Paper-style knowledge base |
| `data/paper_splits/paper_split_val.csv` | 43 | Yes | Hyperparameter tuning |
| `data/paper_splits/paper_split_test.csv` | 41 | Yes | Paper reproduction evaluation |
| `data/paper_splits/paper_split_metadata.json` | - | - | Split provenance |

**Important**: These splits are now generated using **ground truth participant IDs** reverse-engineered from the paper authors' output files (see `docs/data/paper-split-registry.md`). This ensures exact reproduction.

The script `scripts/create_paper_split.py` defaults to `--mode ground-truth`. The legacy seeded algorithmic generation (Appendix C) is available via `--mode algorithmic`.

---

## Transcript Artifacts

### Raw (Extraction Output)

`scripts/prepare_dataset.py` writes raw transcripts to:

- `data/transcripts/{id}_P/{id}_TRANSCRIPT.csv`

These are **not speaker-filtered** and may contain known DAIC-WOZ issues (interruptions, sync markers, missing Ellie transcripts).

### Preprocessed Variants (Recommended for Bias-Aware Retrieval)

`scripts/preprocess_daic_woz_transcripts.py` writes deterministic variants under:

- `data/transcripts_{variant}/{id}_P/{id}_TRANSCRIPT.csv`

Recommended variants:
- `participant_only` (bias-aware retrieval default)
- `both_speakers_clean` (clean baseline, keeps Ellie + Participant)
- `participant_qa` (participant + minimal question context)

Select a variant via:

```bash
DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
```

See: `docs/data/daic-woz-preprocessing.md`.

---

## Embeddings Artifacts

### Legacy (Backward Compatible)

| File | Source Split | Participants | Size | Notes |
|------|-------------|--------------|------|-------|
| `data/embeddings/paper_reference_embeddings.npz` | Paper-train | 58 | ~101 MB | NPZ embeddings |
| `data/embeddings/paper_reference_embeddings.json` | Paper-train | 58 | ~2.9 MB | Text chunks sidecar |

### Current Generator Output (Default)

`scripts/generate_embeddings.py` writes:
- `{output}.npz` (embeddings)
- `{output}.json` (text chunks)
- `{output}.meta.json` (provenance metadata)
- `{output}.tags.json` (optional, with `--write-item-tags` flag)
- `{output}.partial.json` (debug-only, with `--allow-partial`; Spec 40)

Additional optional sidecars (separate preprocessing steps):
- `{output}.chunk_scores.json` + `{output}.chunk_scores.meta.json` (Spec 35; from `scripts/score_reference_chunks.py`)

### Item Tags Sidecar (Spec 34)

When generated with `--write-item-tags`, the `.tags.json` sidecar contains per-chunk PHQ-8 item tags:

```json
{
  "303": [
    ["PHQ8_Sleep", "PHQ8_Tired"],
    [],
    ["PHQ8_Depressed"]
  ],
  "304": []
}
```

**Purpose**: Enables item-level filtering at retrieval time (`EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`).

**Validation**: `ReferenceStore` validates that:
- Participant IDs match the texts sidecar
- Per-participant list length equals chunk count
- Tag values are valid `PHQ8_*` strings

### Chunk Scores Sidecar (Spec 35)

Chunk scoring produces per-chunk estimated PHQ-8 item scores aligned with `{output}.json`:

- `{output}.chunk_scores.json`
- `{output}.chunk_scores.meta.json`

**Purpose**: Enables chunk-level labels when `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`.

**Validation**: `ReferenceStore` validates that:
- Participant IDs match the embeddings/text sidecars exactly
- Per-participant list length equals chunk count
- Keys are exactly the 8 `PHQ8_*` strings
- Values are `0..3` or `null`
- `prompt_hash` matches the current scorer prompt (unless explicitly overridden as unsafe)

See: `docs/data/chunk-scoring.md`.

### Partial Output Manifest (Spec 40)

If embeddings are generated with `--allow-partial`, the script writes `{output}.partial.json` when skips occur.

**Rule**: Partial artifacts are debug-only and must not be used for final evaluation.

### Embedding Auto-Selection Logic

Reference embeddings are selected via config, not `--split`:

1. If `DATA_EMBEDDINGS_PATH` is explicitly set, use it.
2. Otherwise use `EMBEDDING_EMBEDDINGS_FILE` resolved under `{DATA_BASE_DIR}/embeddings/`.

If `{artifact}.meta.json` exists, `ReferenceStore` validates metadata (backend, model, dimension, chunking, `min_evidence_chars`, split CSV hash) against config at load time.

---

## Scripts

### Split Creation

| Script | Output | Purpose |
|--------|--------|---------|
| `scripts/create_paper_split.py` | `data/paper_splits/paper_split_*.csv` | Create paper-style 58/43/41 split |

### Embedding Generation

| Script | Input Split | Output | Purpose |
|--------|-------------|--------|---------|
| `scripts/generate_embeddings.py --split avec-train` | `train_split_Depression_AVEC2017.csv` | `{backend}_{model_slug}_avec_train.*` | AVEC embeddings |
| `scripts/generate_embeddings.py --split paper-train` | `paper_split_train.csv` | `{backend}_{model_slug}_paper_train.*` | Paper embeddings |

### Reproduction

| Script | Eval Split | Embeddings Used | Purpose |
|--------|------------|-----------------|---------|
| `scripts/reproduce_results.py --split dev` | AVEC dev (35) | Configured reference artifact (`EMBEDDING_EMBEDDINGS_FILE` / `DATA_EMBEDDINGS_PATH`) | Default evaluation |
| `scripts/reproduce_results.py --split paper` | Paper test (41) | Configured reference artifact (`EMBEDDING_EMBEDDINGS_FILE` / `DATA_EMBEDDINGS_PATH`) | Paper reproduction |

---

## Configuration Parameters

### Paper-Specified (Appendix D)

| Parameter | Value | Config Path |
|-----------|-------|-------------|
| `chunk_size` | 8 | `embedding.chunk_size` |
| `chunk_step` | 2 | `embedding.chunk_step` |
| `top_k_references` | 2 | `embedding.top_k_references` |
| `dimension` | 4096 | `embedding.dimension` |
| `max_iterations` | 10 | `feedback.max_iterations` |
| `score_threshold` | 3 | `feedback.score_threshold` |

### Implementation Defaults (GAP-001)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Clinical AI best practice (Med-PaLM, medRxiv 2025) |
| `top_k` | — | Not set (irrelevant at temp=0) |
| `top_p` | — | Not set (best practice: use temp only) |

See [Agent Sampling Registry](../configs/agent-sampling-registry.md) for citations.

### Model Precision Options

| Backend | Chat Model | Embedding Model | Precision |
|---------|------------|-----------------|-----------|
| **Ollama (default)** | `gemma3:27b` | `qwen3-embedding:8b` | Q4_K_M (4-bit) |
| **HuggingFace (high-quality)** | `google/medgemma-27b-text-it` | `Qwen/Qwen3-Embedding-8B` | FP16 (16-bit) |

See [Model Registry](../models/model-registry.md#high-quality-setup-recommended-for-production) and [Issue #42](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/42) for details.

---

## Output Artifacts

| File Pattern | Purpose |
|--------------|---------|
| `data/outputs/{mode}_{split}_{YYYYMMDD_HHMMSS}.json` | Reproduction results with run + per-experiment provenance (from `scripts/reproduce_results.py`) |
| `data/outputs/selective_prediction_metrics_*.json` | AURC/AUGRC + bootstrap CIs (from `scripts/evaluate_selective_prediction.py`) |
| `data/outputs/RUN_LOG.md` | Human-maintained run history log (append-only) |
| `data/outputs/*.log` | Console log captures for long runs / tmux sessions (optional) |
| `data/experiments/registry.yaml` | Registry of run metadata + summary metrics (updated by `scripts/reproduce_results.py`) |

---

## Friction Points / Improvements Needed

### 1. Naming Consistency Review

**Current State**: Mostly consistent with `paper_` prefix convention.

**Potential Issues**:
- [ ] `--split paper` vs `--split paper-test` - both work but confusing
- [ ] No explicit `avec_` prefix for AVEC artifacts (implied by absence)

### 2. Documentation Gaps

- [ ] Add inline comments to `src/ai_psychiatrist/config.py` referencing this registry
- [ ] Update `scripts/reproduce_results.py` help text with artifact mapping

### 3. Potential Improvements

- [ ] Consider adding `avec_` prefix for symmetry
- [x] Fail-fast if embeddings are missing (implemented in `scripts/reproduce_results.py`)

---

## See Also

- [data-splits-overview.md](./data-splits-overview.md) - Detailed split documentation
- [Agent Sampling Registry](../configs/agent-sampling-registry.md) - Sampling parameters (paper leaves some unspecified)
