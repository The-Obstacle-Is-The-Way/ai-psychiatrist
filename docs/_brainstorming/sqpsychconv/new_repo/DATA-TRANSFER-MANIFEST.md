# Data Transfer Manifest: ai-psychiatrist → vibe-check

**Date**: 2026-01-02
**Purpose**: Define exactly what data files need to transfer between repos

---

## TL;DR: vibe-check needs ZERO data from ai-psychiatrist

**The vibe-check repo is designed to be completely independent of DAIC-WOZ.**

---

## Why No Data Transfer?

Per SPEC-vibe-check.md Section 3.2 (Data Governance):

> **Policy (non-negotiable for this spec)**:
> - **Do not send DAIC-WOZ transcripts to third-party APIs** (OpenAI/Anthropic/Google).
> - The `vibe-check` **core labeling pipeline must not require DAIC-WOZ**.
> - DAIC-WOZ data and derived artifacts must never be checked into the repo.

---

## Data Classification

| Data in ai-psychiatrist | Stays in ai-psychiatrist | Goes to vibe-check | Reason |
|-------------------------|--------------------------|--------------------| -------|
| `transcripts/` | ✅ YES | ❌ NO | DAIC-WOZ restricted data |
| `transcripts_participant_only/` | ✅ YES | ❌ NO | DAIC-WOZ derived |
| `embeddings/` | ✅ YES | ❌ NO | DAIC-WOZ derived (non-redistributable) |
| `paper_splits/` | ✅ YES | ❌ NO | DAIC-WOZ participant IDs |
| `*_AVEC2017.csv` | ✅ YES | ❌ NO | DAIC-WOZ ground truth labels |
| `full_test_split.csv` | ✅ YES | ❌ NO | DAIC-WOZ split |
| `outputs/` | ✅ YES | ❌ NO | ai-psychiatrist run artifacts |
| `experiments/` | ✅ YES | ❌ NO | ai-psychiatrist experiments |
| `DATA_PROVENANCE.md` | ✅ YES | ❌ NO | References DAIC-WOZ |
| `sqpsychconv/` | ❌ DELETE | ❌ NO | Should not be here! |

---

## What vibe-check DOES Need (Fresh Sources)

vibe-check downloads its data directly from HuggingFace:

```python
# In vibe-check repo (NOT copied from ai-psychiatrist)
from datasets import load_dataset

# Download fresh from HuggingFace
sqpsychconv = load_dataset("AIMH/SQPsychConv_qwq")
```

**Why fresh download?**
1. Ensures vibe-check is self-contained
2. No DAIC-WOZ contamination risk
3. Clear provenance chain
4. HuggingFace handles versioning

---

## Architecture: Two Independent Repos

```
┌─────────────────────────────────────────────────────────────────────┐
│                          vibe-check                                  │
│  (scores synthetic data with frontier APIs)                         │
│                                                                      │
│  Input:  HuggingFace → SQPsychConv (fresh download)                 │
│  Output: scored_sqpsychconv.jsonl + embeddings/*.npz                │
│                                                                      │
│  ⚠️  NO DAIC-WOZ DATA IN THIS REPO                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Transfer embeddings (one-way)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ai-psychiatrist                               │
│  (local-only evaluation against DAIC-WOZ)                           │
│                                                                      │
│  Input:  vibe-check embeddings + DAIC-WOZ transcripts (LOCAL ONLY) │
│  Output: transfer_eval.json (sim-to-real metrics)                   │
│                                                                      │
│  ⚠️  DAIC-WOZ NEVER LEAVES THIS REPO                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Action Items

### In ai-psychiatrist (this repo):

```bash
# DELETE the sqpsychconv folder - it shouldn't be here
rm -rf data/sqpsychconv/

# The following STAY in ai-psychiatrist:
# - data/transcripts/
# - data/transcripts_participant_only/
# - data/embeddings/
# - data/paper_splits/
# - data/*_AVEC2017.csv
# - data/outputs/
```

### In vibe-check (new repo):

```bash
# Start fresh - no data copied from ai-psychiatrist
mkdir -p data/raw/
mkdir -p data/checkpoints/
mkdir -p data/outputs/

# SQPsychConv is downloaded via HuggingFace datasets library
# See scripts/score_corpus.py
```

---

## Phase 3 Evaluation Flow

When evaluating sim-to-real transfer (SPEC-vibe-check Section 12.4):

1. **vibe-check** generates: `data/outputs/embeddings/*.npz`
2. **Copy embeddings** to ai-psychiatrist: `cp vibe-check/data/outputs/embeddings/* ai-psychiatrist/data/vibe_check_embeddings/`
3. **ai-psychiatrist** runs local-only evaluation:
   ```bash
   # In ai-psychiatrist (NOT vibe-check)
   uv run python scripts/evaluate_transfer.py \
       --synthetic data/vibe_check_embeddings/ \
       --real-test paper-test \
       --k 5 \
       --output data/outputs/transfer_eval.json
   ```

This keeps DAIC-WOZ data strictly within ai-psychiatrist.

---

## Summary

| Question | Answer |
|----------|--------|
| Does vibe-check need transcripts? | NO |
| Does vibe-check need embeddings? | NO (generates its own) |
| Does vibe-check need paper_splits? | NO |
| Does vibe-check need AVEC2017 labels? | NO |
| Does vibe-check need ANY data from ai-psychiatrist? | **NO** |
| Should sqpsychconv be in ai-psychiatrist? | **NO - DELETE IT** |

---

## Verification Checklist

Before starting vibe-check development:

- [ ] Deleted `data/sqpsychconv/` from ai-psychiatrist
- [ ] vibe-check repo initialized with empty `data/` structure
- [ ] HuggingFace dataset access verified: `load_dataset("AIMH/SQPsychConv_qwq")`
- [ ] No DAIC-WOZ files present in vibe-check repo
- [ ] `.gitignore` in vibe-check excludes any DAIC-WOZ patterns

---

*This manifest ensures clean separation between the two repos and prevents accidental DAIC-WOZ leakage.*
