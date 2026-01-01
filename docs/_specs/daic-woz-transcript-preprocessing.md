# Spec: DAIC-WOZ Transcript Preprocessing (Bias-Aware, Deterministic Variants)

**Status**: Implemented
**Primary implementation**: `scripts/preprocess_daic_woz_transcripts.py`
**Integration points**: `src/ai_psychiatrist/config.py` (`DATA_TRANSCRIPTS_DIR`), `src/ai_psychiatrist/services/transcript.py`

## 0. Problem Statement

DAIC-WOZ transcripts contain:

1) **Interviewer prompt leakage**: Ellie’s prompts can leak protocol patterns into embedding-based retrieval, biasing few-shot selection before the LLM is prompted.
2) **Known “mechanical” transcript issues**: e.g., interruption windows and a small number of missing Ellie transcript sessions.
3) **Potential integrity issues in split CSVs** (depending on upstream copy): missing PHQ-8 item cells and known label inconsistencies (e.g., `PHQ8_Binary` mismatch).

We need a deterministic, reproducible preprocessing workflow that creates **collision-free transcript variants** without modifying raw data.

## 1. Goals / Non-Goals

### 1.1 Goals

- Produce **bias-aware transcript variants** (notably participant-only) for embeddings/retrieval.
- Apply deterministic cleanup for known transcript mechanical issues (sync markers, interruptions).
- Guarantee **raw vs processed** inputs never collide (no in-place overwrites).
- Maintain the **directory + filename convention** expected by the codebase.
- Provide a **machine-readable manifest** (counts + warnings; no transcript text) for auditability.

### 1.2 Non-Goals

- Audio preprocessing / audio-text alignment fixes (reference tool flags misaligned audio sessions; not required for text-only runs).
- “Classical ML” token stripping (e.g., removing `<laughter>` tokens) by default; this is an explicit ablation, not the default.
- Downloading/unzipping DAIC-WOZ data (handled by dataset prep tooling; this spec focuses on transcript variants once `data/transcripts/` exists).

## 2. Inputs (Raw, Untouched)

### 2.1 Canonical raw layout

Raw transcripts are expected in:

```text
data/
  transcripts/
    300_P/300_TRANSCRIPT.csv
    ...
```

The transcript file is tab-separated with required columns:

```text
start_time    stop_time    speaker    value
```

See: `docs/data/daic-woz-schema.md`.

### 2.2 Raw data must not be modified

- The preprocessing workflow **must never overwrite** anything under `data/transcripts/`.
- Processed variants must be written to a distinct directory root (see Section 3).

## 3. Outputs (Processed Variants)

### 3.1 Output directory convention

Processed transcripts are written to a new transcripts root that preserves the same on-disk convention:

```text
data/
  transcripts_preprocessed/
    <variant_name>/
      300_P/300_TRANSCRIPT.csv
      ...
```

### 3.2 Variant selection in runtime code

The runtime transcript loader is already configurable via `DATA_TRANSCRIPTS_DIR`:

- `src/ai_psychiatrist/config.py`: `DataSettings.transcripts_dir` (env prefix `DATA_`)
- `src/ai_psychiatrist/services/transcript.py`: `TranscriptService` reads `data_settings.transcripts_dir`

Example:

```bash
export DATA_TRANSCRIPTS_DIR=data/transcripts_preprocessed/participant_only
```

No code changes are required to select a variant: only configuration changes.

## 4. Dataset Facts (Reference Tool + Local Audit)

This implementation is aligned with the widely used Bailey/Plumbley DAIC-WOZ preprocessing tool mirrored under `_reference/daic_woz_process/`.

### 4.1 Known transcript mechanical issues (reference tool config)

From `_reference/daic_woz_process/config_files/config_process.py`:

- **Interruption windows**:
  - `373`: `[395, 428]` seconds
  - `444`: `[286, 387]` seconds
- **Missing Ellie transcripts** (participant-only transcripts):
  - `451`, `458`, `480`
- **Audio-only misalignment** (text can still be used):
  - `318`, `321`, `341`, `362`
- **Known label issue**:
  - `wrong_labels = {409: 1}` (`PHQ8_Binary` mismatch for score ≥ 10)
- **Absent sessions**:
  - `excluded_sessions = [342, 394, 398, 460]`

### 4.2 Local raw transcript audit expectations

On a complete DAIC-WOZ transcript dump under `data/transcripts/`, the following checks should hold:

- Transcript file count: `189`
- Speakers present: only `Ellie` and `Participant`
- Missing Ellie sessions: `451`, `458`, `480` contain no Ellie rows
- Interruption-window overlap counts (rows removed if applying interruption rule):
  - `373`: `5` rows overlap `[395, 428]`
  - `444`: `37` rows overlap `[286, 387]`

These are **data-validation expectations**, not hard-coded invariants: the preprocessing should handle deviations by warning/fail-fast depending on severity (see Section 6).

## 5. Variant Definitions

All variants apply the same **deterministic cleaning rules** (Section 6) first, then apply a variant-specific speaker selection rule.

### 5.1 `both_speakers_clean`

- Keep all cleaned rows for both speakers.
- Intended for “paper-parity-ish” text runs where you want noise removal without removing Ellie entirely.

### 5.2 `participant_only` (recommended for embeddings/retrieval)

- Keep only rows where `speaker == "Participant"` after cleaning.
- Rationale: minimizes interviewer protocol leakage in embedding generation and retrieval.

### 5.3 `participant_qa` (minimal question context)

- Keep all participant rows, plus the most recent prior Ellie prompt **once** per contiguous participant block.

Deterministic rule:
- When a participant row is kept, include the most recent prior Ellie row (if any) **exactly once** until another Ellie row appears.

## 6. Deterministic Cleaning Rules (Applied to All Variants)

### 6.1 Parse + schema validation (fail-fast)

For each `{pid}_TRANSCRIPT.csv`:

- Must contain columns: `start_time`, `stop_time`, `speaker`, `value`
- If required columns are missing: **fail** preprocessing for that transcript (do not silently continue)
- Drop rows where `speaker` or `value` is missing/NaN

### 6.2 Speaker normalization + validation (fail-fast)

Normalize `speaker` values by trimming and case-folding:

- `"ellie"` → `"Ellie"`
- `"participant"` → `"Participant"`

After normalization:
- If any speaker value is not in `{Ellie, Participant}`: **fail** preprocessing for that transcript.

### 6.3 Pre-interview removal (drop “preamble”)

If Ellie is present:
- Find the first row where `speaker == "Ellie"`.
- Drop all rows before it.

If Ellie is not present:
- This is expected only for sessions `{451, 458, 480}`.
- Drop leading sync markers / empty rows until the first non-empty, non-sync row.
- If Ellie is absent for a session not in the known list: do not fail; emit a **warning** (to avoid hard-coding assumptions that may vary across dataset copies).

### 6.4 Sync marker removal

Drop rows whose `value` is a sync marker.

Must match these canonical markers (case-insensitive, whitespace-trimmed), tolerating minor punctuation:

- `<sync>`, `<synch>`
- `[sync]`, `[synch]`, `[syncing]`, `[synching]`
- plus any value whose normalized form starts with `<sync` or `[sync`

### 6.5 Interruption window removal (text-safe)

Drop rows overlapping known interruption windows:

- `373`: `[395, 428]`
- `444`: `[286, 387]`

Row/window overlap definition:

```text
row_start < window_end AND row_end > window_start
```

### 6.6 Preserve nonverbal annotations (default)

Do not delete tokens like `<laughter>` / `<sigh>` by default, because these can carry affective signal for LLM reasoning.

Note on reference parity:
- The Bailey/Plumbley tool strips placeholder/unknown tokens (e.g., `xxx`/`xxxx`) and removes tokens containing `< > [ ]` as part of a Word2Vec/classical feature pipeline.
- In this repo, stripping is an explicit ablation; the default preprocessing preserves nonverbal tags for LLM use.

## 7. Preprocessing CLI Contract

### 7.1 Script entrypoint

Provide a deterministic CLI at:

- `scripts/preprocess_daic_woz_transcripts.py`

### 7.2 Required flags / behavior

- `--input-dir` (default `data/transcripts`)
- `--output-dir` (required)
- `--variant` in `{both_speakers_clean, participant_only, participant_qa}` (default `participant_only`)
- `--overwrite` to delete an existing output dir (explicit opt-in)
- `--dry-run` to validate and compute stats without writing outputs

Safety constraints:
- Refuse to run if `--output-dir` resolves to the same path as `--input-dir`.

Atomicity:
- Write outputs to a staging directory (e.g., `output_dir.tmp`) and rename to `output_dir` only on success.
- On failure, remove staging output to avoid partial/corrupt processed datasets.

Audit output:
- When writing outputs (non-dry-run), write `preprocess_manifest.json` containing:
  - counts (rows in/out)
  - per-file removals by category
  - warnings
  - no transcript text

## 8. Ground Truth Integrity (PHQ-8 CSVs)

These are deterministic repairs and should be treated as integrity fixes, not statistical imputation.

### 8.1 Missing PHQ-8 item cell repair (when applicable)

If exactly one PHQ-8 item is missing and `PHQ8_Score` is present:

```text
missing_item = PHQ8_Score - sum(known_items)
```

Tooling:

- `uv run python scripts/patch_missing_phq8_values.py --dry-run`
- `uv run python scripts/patch_missing_phq8_values.py --apply`

Doc:
- `docs/data/patch-missing-phq8-values.md`

### 8.2 `PHQ8_Binary` consistency rule

Treat:

```text
PHQ8_Binary = 1 iff PHQ8_Score >= 10
```

Known upstream issue to account for:
- Participant `409` has been observed with `PHQ8_Score=10` but `PHQ8_Binary=0` in some copies.

## 9. Collision-Free Artifact Workflow (Embeddings, Tags, Chunk Scores)

To avoid mixing artifacts from different transcript variants:

1) Keep raw transcripts in `data/transcripts/`
2) Generate a processed variant in `data/transcripts_preprocessed/<variant>/`
3) Set `DATA_TRANSCRIPTS_DIR` to that variant
4) Generate embeddings with a variant-stamped artifact name
5) Ensure `.tags.json` and `.chunk_scores.json` correspond to the same embeddings base name

See:
- `docs/data/artifact-namespace-registry.md`
- `docs/embeddings/embedding-generation.md`

## 10. Acceptance Criteria / Validation

### 10.1 Preprocessing correctness

- Preprocessing completes without error across all transcripts in `data/transcripts/`.
- Output transcript files preserve the `{pid}_P/{pid}_TRANSCRIPT.csv` convention.
- Every output transcript contains at least one participant utterance.
- Sessions `451/458/480` are processed without failure (Ellie absent).
- Sessions `373/444` have rows removed overlapping the specified windows.

### 10.2 Reproducibility and auditability

- Output directory includes `preprocess_manifest.json` (no transcript text).
- Re-running with identical inputs and settings produces identical outputs.

### 10.3 Downstream compatibility

- Setting `DATA_TRANSCRIPTS_DIR` to the output directory results in successful transcript loads via `TranscriptService`.
- Embeddings generation succeeds against the processed transcripts directory when configured.

## 11. Related Docs

- User-facing guide (overview + rationale): `docs/data/daic-woz-preprocessing.md`
- Local audit notes: `docs/_brainstorming/daic-woz-preprocessing.md`
- DAIC-WOZ schema: `docs/data/daic-woz-schema.md`
- Reference preprocessing repo mirror: `_reference/daic_woz_process/`
