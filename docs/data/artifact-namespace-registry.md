# Artifact Namespace Registry

**Purpose**: Single source of truth for all data artifacts, scripts, and naming conventions
**Last Updated**: 2026-01-03

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

See [Data Splits Overview](./data-splits-overview.md) for the authoritative reference on AVEC2017 vs paper splits.

**Quick Reference**:
- AVEC2017: 107 train / 35 dev / 47 test (test has no per-item labels)
- Paper custom: 58 train / 43 val / 41 test (all have per-item labels)
- Ground truth IDs: [Data Splits Overview](./data-splits-overview.md#appendix-a-paper-split-participant-ids)

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

See: [DAIC-WOZ Preprocessing](./daic-woz-preprocessing.md).

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

See: [Chunk-level scoring](../rag/chunk-scoring.md).

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

## Output Artifacts

| File Pattern | Purpose |
|--------------|---------|
| `data/outputs/{mode}_{split}_{YYYYMMDD_HHMMSS}.json` | Reproduction results with run + per-experiment provenance (from `scripts/reproduce_results.py`) |
| `data/outputs/selective_prediction_metrics_*.json` | AURC/AUGRC + bootstrap CIs (from `scripts/evaluate_selective_prediction.py`) |
| `data/outputs/RUN_LOG.md` | Human-maintained run history log (append-only) |
| `data/outputs/*.log` | Console log captures for long runs / tmux sessions (optional) |
| `data/experiments/registry.yaml` | Registry of run metadata + summary metrics (updated by `scripts/reproduce_results.py`) |

---

## Related Documentation

- [Data Splits Overview](./data-splits-overview.md) - AVEC2017 vs paper splits
- [DAIC-WOZ Schema](./daic-woz-schema.md) - Dataset format and domain model
- [DAIC-WOZ Preprocessing](./daic-woz-preprocessing.md) - Transcript variant generation
- [RAG Artifact Generation](../rag/artifact-generation.md) - Embedding generation
- [Configuration](../configs/configuration.md) - Environment variables
- [Model Registry](../models/model-registry.md) - Model options and precision
