# CRITICAL: Data Split Mismatch (FIXED)

> **Status**: FIXED
> **Impact**: High (Invalidates previous reproduction attempts)
> **Date Discovered**: 2024-12-24
> **Date Fixed**: 2025-12-25

## Resolution

We have adopted the **ground truth** participant IDs reverse-engineered from the paper authors' output files. See `docs/data/DATA_SPLIT_REGISTRY.md` for the authoritative list.

The `scripts/create_paper_split.py` script now defaults to `--mode ground-truth`, which uses these hardcoded IDs. The legacy algorithmic generation is preserved under `--mode algorithmic`.

## Problem Description (Historical)

The paper describes a stratified split of 142 participants into 58 Train / 43 Val / 41 Test.
However, it does **not** provide the list of participant IDs.

We implemented the stratification algorithm described in Appendix C (seed=42).
**Crucially, our generated splits do NOT match the paper's splits.**

### Evidence

Comparison of our `paper_split_test.csv` vs the paper's `DIM_TEST_analysis_output`:

| Split | Paper Count | Our Count | Overlap | Mismatch |
|-------|-------------|-----------|---------|----------|
| Test  | 41          | 41        | 15      | **26**   |

**26 out of 41 test participants are different.**
This means we are testing on participants the paper used for training or validation.

### Impact

1.  **Metric Comparability**: We cannot compare our MAE/F1 scores to the paper because the test sets are effectively disjoint (only ~36% overlap).
2.  **Few-Shot Retrieval**: Our `paper_reference_embeddings.npz` (knowledge base) is built from our wrong `paper_split_train.csv`.
    - If the paper used Participant X in training, they are in the KB.
    - If we put Participant X in test, we are retrieving their own transcript from the KB (data leakage) OR missing them entirely.

## Root Cause

The stratification algorithm (Appendix C) is under-specified. It depends on:
1.  Random seed (unknown).
2.  Exact order of operations for "randomly selecting" participants to move between buckets.

We assumed `seed=42` would be "close enough" or that the stratification constraints were tight enough to force a unique solution. **They are not.** The solution space is large.

## Fix Implementation

1.  Reverse-engineered the *exact* participant IDs from the paper's raw analysis output files (which contain filenames like `302_TRANSCRIPT.csv`).
2.  Documented these IDs in `docs/data/DATA_SPLIT_REGISTRY.md`.
3.  Updated `scripts/create_paper_split.py` to use these IDs by default.
4.  Regenerated `data/paper_splits/` and `data/embeddings/paper_reference_embeddings.npz`.

See `SPEC-DATA-SPLIT-FIX.md` for details.
