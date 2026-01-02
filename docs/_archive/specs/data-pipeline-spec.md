# Data Pipeline Specification

**Date**: 2026-01-02
**Status**: ACTIVE (Executed 2026-01-02)
**Purpose**: Define the complete data processing chain from raw transcripts to evaluation

---

## Table of Contents

1. [Current State (Verified via ls -la)](#1-current-state-verified-via-ls--la)
2. [Target State (What We Need)](#2-target-state-what-we-need)
3. [The Full Pipeline](#3-the-full-pipeline)
4. [Archive Decisions](#4-archive-decisions)
5. [Execution Plan](#5-execution-plan)
6. [Open Questions](#6-open-questions)

---

## 1. Current State (Verified via ls -la)

### 1.1 Complete Data Directory Structure

```
data/
├── DATA_PROVENANCE.md              # Documents patched values (319, 409)
│
├── transcripts/                    # Raw DAIC-WOZ transcripts (NEVER MODIFIED)
│   ├── 300_P/300_TRANSCRIPT.csv    # 189 participant folders total
│   ├── 301_P/301_TRANSCRIPT.csv
│   └── ...
│
├── transcripts_participant_only/   # Preprocessed participant-only transcripts (bias-aware retrieval)
│   ├── 300_P/300_TRANSCRIPT.csv
│   ├── 301_P/301_TRANSCRIPT.csv
│   ├── ...
│   └── preprocess_manifest.json    # Audit trail (no transcript text)
│
├── paper_splits/                   # Paper-defined train/val/test (from authors' outputs)
│   ├── paper_split_train.csv       # 58 participants (reference pool for embeddings)
│   ├── paper_split_val.csv         # 43 participants (tuning)
│   ├── paper_split_test.csv        # 41 participants (evaluation)
│   └── paper_split_metadata.json   # Provenance: how splits were derived
│
├── embeddings/                     # Pre-computed embeddings for few-shot
│   ├── _archive/
│   │   └── pre-preprocessing-*/    # Archived raw-transcript embeddings + older runs
│   │       ├── README.md
│   │
│   ├── huggingface_qwen3_8b_paper_train_participant_only.npz
│   ├── huggingface_qwen3_8b_paper_train_participant_only.json
│   ├── huggingface_qwen3_8b_paper_train_participant_only.meta.json
│   ├── huggingface_qwen3_8b_paper_train_participant_only.tags.json
│   ├── huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json
│   └── huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json
│
├── outputs/                        # Reproduction run outputs
│   ├── RUN_LOG.md                  # Run history documentation
│   ├── both_paper-test_*.json      # Raw reproduction outputs
│   ├── selective_prediction_*.json # Evaluation metrics
│   └── *.log                       # Execution logs
│
├── experiments/                    # Experiment tracking
│   └── registry.yaml               # All runs with git commits and metrics
│
├── train_split_Depression_AVEC2017.csv  # Original AVEC train (107 participants)
├── dev_split_Depression_AVEC2017.csv    # Original AVEC dev (35 participants)
├── test_split_Depression_AVEC2017.csv   # Original AVEC test (47 participants, no PHQ-8)
└── full_test_split.csv                  # AVEC test WITH labels (47 participants)
```

### 1.2 Key Counts (Verified)

| Artifact | Count | Notes |
|----------|-------|-------|
| Raw transcripts | 189 folders | Never modified |
| Paper train split | 58 participants | Used for embeddings |
| Paper val split | 43 participants | Tuning |
| Paper test split | 41 participants | Evaluation |
| AVEC train | 107 participants | Original dataset |
| AVEC dev | 35 participants | Original dataset |
| AVEC test | 47 participants | Has `full_test_split.csv` with labels |

### 1.3 Critical Issue: Raw Transcripts Leak Protocol Patterns

Raw transcripts (`data/transcripts/`) contain:

- **Ellie's prompts** → Interviewer protocol leakage in retrieval
- **Sync markers** → `<sync>`, `[synch]`, etc.
- **Pre-interview preamble** → Noise before interview starts
- **Interruption windows** → 373: 395-428s, 444: 286-387s

Mitigation: participant-only preprocessing is now executed and produces `data/transcripts_participant_only/`, which is used for embedding generation and reproduction runs via:

```bash
DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
```

### 1.4 Provenance Documentation

`DATA_PROVENANCE.md` documents two deterministic corrections:

1. **Participant 319**: Reconstructed `PHQ8_Sleep=2` from total score invariant
2. **Participant 409**: Corrected `PHQ8_Binary=0→1` per score≥10 rule

Both are mathematically verifiable, not imputations.

---

## 2. Target State (What We Need)

### 2.1 New Directory: Preprocessed Transcripts

```
data/
├── transcripts/                         # Raw (NEVER MODIFIED)
└── transcripts_participant_only/         # Preprocessed participant-only variant
    ├── 300_P/300_TRANSCRIPT.csv
    ├── 301_P/301_TRANSCRIPT.csv
    ├── ...
    └── preprocess_manifest.json         # Audit trail (no transcript text)
```

### 2.2 New Embeddings (From Preprocessed Transcripts)

**Naming convention**: `{backend}_{model}_{split}_participant_only.*`

```
data/embeddings/
├── _archive/
│   ├── pre-spec-34-baseline/            # Already exists
│   └── pre-preprocessing-v1/            # NEW: Archive current embeddings
│
└── huggingface_qwen3_8b_paper_train_participant_only.npz
    huggingface_qwen3_8b_paper_train_participant_only.json
    huggingface_qwen3_8b_paper_train_participant_only.meta.json
    huggingface_qwen3_8b_paper_train_participant_only.tags.json
    huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json
    huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json
```

### 2.3 Archived Outputs

```
data/outputs/
├── _archive/
│   └── pre-preprocessing-v1/            # All current outputs
│       ├── both_paper-test_*.json
│       ├── selective_prediction_*.json
│       └── *.log
│
├── RUN_LOG.md                           # Keep, add V2 section
└── (new outputs after preprocessing)
```

---

## 3. The Full Pipeline

### 3.1 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                     │
│  │ Raw Transcripts │  data/transcripts/ (189 sessions)                   │
│  │ (NEVER MODIFY)  │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: Preprocessing                                               │ │
│  │ Script: scripts/preprocess_daic_woz_transcripts.py                  │ │
│  │ Input:  data/transcripts/                                           │ │
│  │ Output: data/transcripts_participant_only/                          │ │
│  │                                                                     │ │
│  │ Removes:                                                            │ │
│  │   ✗ Ellie utterances (interviewer prompt leakage)                   │ │
│  │   ✗ Sync markers (<sync>, [synch], etc.)                            │ │
│  │   ✗ Pre-interview preamble                                          │ │
│  │   ✗ Interruption windows (373: 395-428s, 444: 286-387s)             │ │
│  │                                                                     │ │
│  │ Preserves:                                                          │ │
│  │   ✓ Original case (no .lower())                                     │ │
│  │   ✓ Nonverbal tokens (<laughter>, <sigh>)                           │ │
│  │   ✓ All participant utterances                                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: Set Configuration                                           │ │
│  │                                                                     │ │
│  │ .env:                                                               │ │
│  │   DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only            │ │
│  │                                                                     │ │
│  │ This affects ALL downstream code:                                   │ │
│  │   - TranscriptService.load_transcript()                             │ │
│  │   - Embedding generation                                            │ │
│  │   - Reproduction script                                             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: Generate Embeddings                                         │ │
│  │ Script: scripts/generate_embeddings.py --write-item-tags            │ │
│  │ Input:  Preprocessed transcripts + paper_split_train.csv (58 pids)  │ │
│  │ Output: data/embeddings/{name}.{npz,json,meta.json,tags.json}       │ │
│  │                                                                     │ │
│  │ Note: ONLY train split is embedded (avoids data leakage)            │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 4: Generate Chunk Scores (Spec 35)                             │ │
│  │ Script: scripts/score_reference_chunks.py                           │ │
│  │ Input:  Embeddings file                                             │ │
│  │ Output: {name}.chunk_scores.json, {name}.chunk_scores.meta.json     │ │
│  │                                                                     │ │
│  │ ⚠️ SLOW: 4-8 hours for 58 participants (~6800 chunks)               │ │
│  │ Run in tmux/screen                                                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 5: Update Configuration for Chunk Scoring                      │ │
│  │                                                                     │ │
│  │ .env:                                                               │ │
│  │   EMBEDDING_EMBEDDINGS_FILE={name}                                  │ │
│  │   EMBEDDING_REFERENCE_SCORE_SOURCE=chunk                            │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 6: Run Reproduction                                            │ │
│  │ Script: scripts/reproduce_results.py --split paper-test             │ │
│  │ Output: data/outputs/{mode}_{split}_{timestamp}.json                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 7: Evaluate                                                    │ │
│  │ Script: scripts/evaluate_selective_prediction.py --input <file>     │ │
│  │ Output: AURC, AUGRC, MAE, coverage metrics                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Zero-Shot vs Few-Shot Data Flow

| Mode | Uses Embeddings? | Uses Preprocessed Transcripts? |
|------|------------------|--------------------------------|
| **Zero-shot** | No | Yes (query transcript only) |
| **Few-shot** | Yes (reference retrieval) | Yes (query + references must match) |

**Critical**: Both modes read from `DATA_TRANSCRIPTS_DIR`. Embeddings must be generated from the SAME preprocessed transcripts to ensure text alignment.

### 3.3 Script Inventory

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `preprocess_daic_woz_transcripts.py` | Create preprocessed transcripts | `data/transcripts/` | `data/transcripts_{variant}/` |
| `generate_embeddings.py` | Create embeddings + tags | Preprocessed transcripts | `*.npz`, `*.json`, `*.meta.json`, `*.tags.json` |
| `score_reference_chunks.py` | Score each chunk via LLM | Embeddings | `*.chunk_scores.json`, `*.chunk_scores.meta.json` |
| `reproduce_results.py` | Run PHQ-8 predictions | Transcripts + Embeddings | `both_paper-test_*.json` |
| `evaluate_selective_prediction.py` | Compute AURC/MAE metrics | Reproduction output | Metrics JSON |

---

## 4. Archive Decisions

### 4.1 What to Archive

| Artifact | Archive Path | Reason |
|----------|--------------|--------|
| `ollama_qwen3_8b_paper_train.*` | `_archive/pre-preprocessing-v1/` | Built from raw transcripts |
| `huggingface_qwen3_8b_paper_train.*` | `_archive/pre-preprocessing-v1/` | Built from raw transcripts |
| `data/outputs/both_*.json` | `_archive/pre-preprocessing-v1/` | All runs pre-preprocessing |
| `data/outputs/selective_prediction_*.json` | `_archive/pre-preprocessing-v1/` | Pre-preprocessing metrics |
| `data/outputs/*.log` | `_archive/pre-preprocessing-v1/` | Historical logs |

### 4.2 What to Keep (NOT Archive)

| Artifact | Reason |
|----------|--------|
| `data/transcripts/` | Raw data, never modified |
| `data/paper_splits/` | Paper split definitions |
| `data/*_AVEC2017.csv` | Original AVEC splits |
| `data/full_test_split.csv` | AVEC test with labels |
| `data/DATA_PROVENANCE.md` | Patch documentation |
| `data/outputs/RUN_LOG.md` | Keep, add V2 section |
| `data/experiments/registry.yaml` | Experiment tracking |
| `_archive/pre-spec-34-baseline/` | Already archived baseline |

### 4.3 Archive Commands

```bash
# Create archive directories
mkdir -p data/embeddings/_archive/pre-preprocessing-v1
mkdir -p data/outputs/_archive/pre-preprocessing-v1

# Archive embeddings (keep _archive intact)
mv data/embeddings/ollama_qwen3_8b_paper_train.* data/embeddings/_archive/pre-preprocessing-v1/
mv data/embeddings/huggingface_qwen3_8b_paper_train.* data/embeddings/_archive/pre-preprocessing-v1/

# Archive outputs (except RUN_LOG.md and registry)
mv data/outputs/both_*.json data/outputs/_archive/pre-preprocessing-v1/
mv data/outputs/selective_prediction_*.json data/outputs/_archive/pre-preprocessing-v1/
mv data/outputs/*.log data/outputs/_archive/pre-preprocessing-v1/
mv data/outputs/paper_test_full_run_*.json data/outputs/_archive/pre-preprocessing-v1/
mv data/outputs/few_shot_*.json data/outputs/_archive/pre-preprocessing-v1/
```

---

## 5. Execution Plan

### Phase 1: Preprocessing (~5 minutes)

```bash
uv run python scripts/preprocess_daic_woz_transcripts.py \
  --input-dir data/transcripts \
  --output-dir data/transcripts_participant_only \
  --variant participant_only

# Verify
cat data/transcripts_participant_only/preprocess_manifest.json | jq '.transcript_count'
# Expected: 189
```

### Phase 2: Archive Old Artifacts (~1 minute)

```bash
# Run archive commands from Section 4.3
```

### Phase 3: Generate HuggingFace Embeddings (~30-60 minutes)

```bash
export DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only

uv run python scripts/generate_embeddings.py \
  --backend huggingface \
  --split paper-train \
  --output data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz \
  --write-item-tags \
  2>&1 | tee data/outputs/embedding_gen_$(date +%Y%m%d_%H%M%S).log
```

### Phase 4: Generate Chunk Scores (~4-8 hours)

```bash
tmux new-session -d -s chunk_scores "
  export DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
  uv run python scripts/score_reference_chunks.py \
    --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
    --scorer-backend ollama \
    --scorer-model gemma3:27b-it-qat \
    --allow-same-model \
    2>&1 | tee data/outputs/chunk_scoring_$(date +%Y%m%d_%H%M%S).log
"
```

### Phase 5: Update .env

```bash
# .env (production config after preprocessing)
DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train_participant_only
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
EMBEDDING_BACKEND=huggingface
```

### Phase 6: Run Reproduction (~2-4 hours)

```bash
tmux new-session -d -s repro "
  uv run python scripts/reproduce_results.py --split paper-test \
    2>&1 | tee data/outputs/repro_v2_$(date +%Y%m%d_%H%M%S).log
"
```

### Phase 7: Evaluate

```bash
OUTPUT=$(ls -t data/outputs/both_paper-test_*.json | head -1)
uv run python scripts/evaluate_selective_prediction.py --input "$OUTPUT"
```

---

## 6. Open Questions

### 6.1 Naming Convention

**Options**:
1. `_v2` suffix: `huggingface_qwen3_8b_paper_train_v2`
2. `_participant_only` suffix: `huggingface_qwen3_8b_paper_train_participant_only`
3. Separate directory: `embeddings_v2/huggingface_qwen3_8b_paper_train`

**Recommendation**: Use `_participant_only` for clarity about the preprocessing variant.

### 6.2 Ollama Embeddings

**Question**: Should we regenerate Ollama embeddings from preprocessed transcripts?

**Recommendation**: No. Focus on HuggingFace only:
- HuggingFace FP16 > Ollama Q4_K_M precision
- Ollama was prototyping only
- One embedding set is sufficient

### 6.3 Experiments Registry

**Question**: Reset or continue `experiments/registry.yaml`?

**Recommendation**: Continue. Add a comment marking the preprocessing boundary:

```yaml
# === V2: Post-preprocessing runs (2026-01-XX onwards) ===
```

---

## 7. Related Documents

| Document | Purpose |
|----------|---------|
| `docs/_specs/daic-woz-transcript-preprocessing.md` | Preprocessing spec (full detail) |
| `docs/data/daic-woz-preprocessing.md` | User-facing guide |
| `data/DATA_PROVENANCE.md` | Patched values documentation |
| `docs/configs/configuration-philosophy.md` | Config design principles |
| `docs/_archive/configs/post-ablation-defaults.md` | Post-ablation config changes (archived) |
| `NEXT-STEPS.md` | (Superseded by this document) |

---

## 8. Approval Checklist

- [ ] Senior review of pipeline design
- [ ] Agreement on naming convention (`_participant_only` recommended)
- [ ] Agreement on archive scope
- [ ] Confirmation Ollama embeddings NOT regenerated
- [ ] Sign-off to execute

---

*Document Author*: Claude Code
*Date*: 2026-01-02
*Status*: ACTIVE (Executed 2026-01-02)
