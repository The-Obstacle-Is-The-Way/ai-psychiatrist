# DAIC-WOZ Transcript Preprocessing (Bias-Aware, Deterministic)

**Purpose**: Produce **clean, reproducible** DAIC-WOZ transcript variants (especially **participant-only**) without overwriting raw data, so that:

- Few-shot retrieval embeddings are not biased by interviewer prompt patterns
- Known dataset issues (interruptions, missing Ellie transcripts) are handled deterministically
- Inputs/outputs **never collide** (raw vs processed vs derived artifacts)

This document is written to be **implementation-ready**: it specifies file layouts, edge cases, and exact rules.

---

## Why This Exists

### 1) Interviewer prompt leakage (retrieval bias)

Recent analysis shows models can exploit **Ellie’s follow-up prompts** as a shortcut signal for depression classification, inflating performance in ways that may not generalize.

This repo’s few-shot pipeline can inherit this bias **before** the LLM sees the prompt:
- Evidence extraction → query embedding → similarity search
- If interviewer prompts are embedded, retrieval can return interviewer-driven “shortcuts”

See the internal analysis in `docs/_brainstorming/daic-woz-preprocessing.md` for background.

### 2) DAIC-WOZ has known “mechanical” issues

The Bailey/Plumbley preprocessing tool (mirrored in `_reference/daic_woz_process/`) documents known transcript problems:
- Session interruptions (e.g., 373, 444)
- Missing Ellie transcripts (451, 458, 480)
- Audio timing misalignment (318, 321, 341, 362) — only relevant if using audio
- A known `PHQ8_Binary` label bug (409)

### 3) AVEC2017 split CSVs can contain deterministic integrity bugs

This repo already includes deterministic repairs (see `data/DATA_PROVENANCE.md`), e.g.:
- Missing PHQ-8 item cell reconstructable from `PHQ8_Score` (participant 319)
- `PHQ8_Binary` inconsistency (participant 409)

---

## Inputs (Raw, Untouched)

Canonical raw layout (see `docs/data/daic-woz-schema.md`):

```
data/
  transcripts/
    300_P/300_TRANSCRIPT.csv
    ...
```

Raw transcript files are tab-separated with columns:

```text
start_time    stop_time    speaker    value
```

**Raw inputs must never be overwritten.** Preprocessing always writes to a new directory.

---

## Outputs (Processed Variants)

Preprocessing produces a **new transcripts root** that still matches the expected folder/file conventions:

```
data/
  transcripts_preprocessed/
    participant_only/
      300_P/300_TRANSCRIPT.csv
      ...
    both_speakers_clean/
      300_P/300_TRANSCRIPT.csv
      ...
    participant_qa/
      300_P/300_TRANSCRIPT.csv
      ...
```

Each variant is selectable via configuration:

- `DATA_TRANSCRIPTS_DIR=data/transcripts_preprocessed/participant_only`

No code changes are required: `TranscriptService` already accepts a configurable `transcripts_dir`.

---

## Variant Definitions (What “Participant-Only” Means)

All variants apply the **same deterministic cleaning rules** first (next section). Then:

### Variant A: `both_speakers_clean`

- Keep both speakers after cleaning.
- Use for “paper-parity-ish” runs where you want to minimize non-clinical noise without changing speaker content.

### Variant B: `participant_only`

- Drop all Ellie rows after cleaning.
- Use for **embedding generation + retrieval** to reduce interviewer-protocol leakage.

### Variant C: `participant_qa`

- Keep participant rows, plus **only the immediately preceding Ellie prompt** (one Ellie row) for each block of participant responses.
- Intended as a compromise to preserve minimal question context while avoiding “Ellie-only region” leakage.

Rule (deterministic):
- When a participant row is kept, include the most recent prior Ellie row **once** (do not repeat it before every consecutive participant line).

---

## Deterministic Cleaning Rules (Applied to All Variants)

These rules are designed to be **loss-minimizing** for LLM use while removing clearly non-interview artifacts.

### 1) Parse + basic validation

For each `{pid}_TRANSCRIPT.csv`:
- Must contain `start_time`, `stop_time`, `speaker`, `value`
- Drop rows where `speaker` or `value` is missing/NaN

Fail-fast if required columns are missing.

### 2) Remove “pre-interview chatter”

Goal: drop the preamble interaction that occurs before the interview begins.

Rule:
- If the file contains any `speaker == "Ellie"` row, find the **first** such row and drop all rows **before** it.

Missing Ellie sessions:
- Sessions 451/458/480 are known to contain only participant rows.
- For “no Ellie present” files: drop leading sync markers (see next rule) and keep the remaining rows.

### 3) Remove sync markers (where present)

Drop rows whose `value` is a sync marker, e.g.:

```text
<sync>, <synch>, [sync], [synching], ...
```

Implementation rule:
- Normalize with `strip().lower()` and tolerate trailing punctuation (e.g., `<sync.`).

### 4) Remove known interruption windows (text-safe)

Drop rows whose time range overlaps the interruption window:

- `373`: `[395, 428]` seconds
- `444`: `[286, 387]` seconds

Overlap definition:

```text
row_start < window_end AND row_end > window_start
```

Rationale: these spans contain non-interview events (“person enters room”, alarms, etc.) and are explicitly treated as noise by the preprocessing reference tool.

### 5) Preserve nonverbal annotations (default)

By default, do **not** delete nonverbal tags like `<laughter>` / `<sigh>` because they can carry affective signal for LLM reasoning.

If you want a “classical ML” style cleanup, make it an explicit variant/flag and ablate it.

---

## Ground Truth Integrity (PHQ-8 CSVs)

These are deterministic fixes; they are not “imputation”.

### A) Missing PHQ-8 item cells

If exactly one PHQ-8 item is missing and `PHQ8_Score` is present, the missing value is uniquely determined by:

```text
missing_item = PHQ8_Score - sum(known_items)
```

Tooling:
- `uv run python scripts/patch_missing_phq8_values.py --dry-run`
- `uv run python scripts/patch_missing_phq8_values.py --apply`

Doc:
- `docs/data/patch-missing-phq8-values.md`

### B) `PHQ8_Binary` consistency

This repo treats:

```text
PHQ8_Binary = 1 iff PHQ8_Score >= 10
```

Known upstream issue:
- Participant 409 had `PHQ8_Score=10` but `PHQ8_Binary=0` (now corrected; see `data/DATA_PROVENANCE.md`).

---

## Collision-Free Artifact Workflow (Recommended)

To avoid mixing artifacts from different transcript variants:

1) Keep raw transcripts in `data/transcripts/`
2) Generate a processed variant in `data/transcripts_preprocessed/<variant>/`
3) Point config to it:
   - `DATA_TRANSCRIPTS_DIR=data/transcripts_preprocessed/<variant>`
4) Generate embeddings with an explicit, variant-stamped name:
   - `uv run python scripts/generate_embeddings.py --split paper-train --output data/embeddings/<backend>_<model>_paper_train_<variant>.npz`
5) Set:
   - `EMBEDDING_EMBEDDINGS_FILE=<backend>_<model>_paper_train_<variant>`

Also ensure any `.tags.json` / `.chunk_scores.json` sidecars are generated from the **same** embeddings base name.

See:
- `docs/data/artifact-namespace-registry.md`
- `docs/embeddings/embedding-generation.md`

---

## Preprocessing CLI (Implemented)

Script:
- `scripts/preprocess_daic_woz_transcripts.py`

Examples:

```bash
# 1) Bias-aware variant for retrieval (recommended default)
uv run python scripts/preprocess_daic_woz_transcripts.py \
  --variant participant_only \
  --output-dir data/transcripts_preprocessed/participant_only \
  --overwrite

# 2) Keep both speakers, but remove mechanical noise
uv run python scripts/preprocess_daic_woz_transcripts.py \
  --variant both_speakers_clean \
  --output-dir data/transcripts_preprocessed/both_speakers_clean \
  --overwrite

# 3) Minimal Q/A context
uv run python scripts/preprocess_daic_woz_transcripts.py \
  --variant participant_qa \
  --output-dir data/transcripts_preprocessed/participant_qa \
  --overwrite
```

The script:
- Refuses to run if `--output-dir` equals `--input-dir`
- Writes a machine-readable manifest at `preprocess_manifest.json` (counts only; no transcript text)

---

## Validation Checklist (What “Done” Means)

### Dataset integrity

- `uv run python scripts/patch_missing_phq8_values.py --dry-run` reports no missing item cells
- `PHQ8_Binary` matches `PHQ8_Score >= 10` for train+dev

### Transcript preprocessing

- Output directory contains `*_P/` folders and `*_TRANSCRIPT.csv` files
- No output transcript is empty after preprocessing
- Sessions 451/458/480 are handled without failure (no Ellie speaker present)
- Sessions 373/444 have rows removed in the specified time windows

### Downstream consistency

- Regenerate embeddings using the processed transcripts dir
- Run `mkdocs build --strict` and ensure no new warnings are introduced by doc changes
