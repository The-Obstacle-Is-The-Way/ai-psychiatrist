# Paper Split Methodology Analysis

**Status**: VERIFIED
**Date**: 2025-12-25
**Related**: `paper-split-registry.md`, `critical-data-split-mismatch.md`, `randomization-methodology-discovered.md`

---

## Executive Summary

We **reconstructed the paper's exact TRAIN/VAL/TEST participant IDs** by extracting them from the paper authors' published output files in `_reference/analysis_output/`. These IDs are codified in `paper-split-registry.md` and are the authoritative ground truth for reproducing the paper's results.

We also identified how the paper authors implemented their split randomization in code (see `randomization-methodology-discovered.md`). The implementation uses NumPy RNG (`np.random.seed` + `np.random.shuffle`) with a per-stratum reseed pattern and a post-processing override step for PHQ-8 scores with exactly two participants.

Even with the implementation details and seed discovered, **the safest reproducibility strategy remains using the hardcoded ground truth IDs** from `paper-split-registry.md`.

---

## 1. Data Source

### 1.1 What the Paper Used

The paper combined AVEC2017 train and dev sets (142 participants total):

| AVEC Split | Participants | Per-Item PHQ-8 | Used by Paper |
|------------|--------------|----------------|---------------|
| train | 107 | Yes | Yes |
| dev | 35 | Yes | Yes |
| test | 47 | **No** | **Excluded** |

**Key insight**: The AVEC2017 test split does not include per-item PHQ-8 labels. In this repo, `data/test_split_Depression_AVEC2017.csv` contains identifiers (and gender) only, and `data/full_test_split.csv` (if present) contains total PHQ scores only. The paper's quantitative assessment agent requires per-item PHQ-8.

### 1.2 Paper's Re-Split

The paper re-split 142 participants into:
- TRAIN: 58 (41%)
- VAL: 43 (30%)
- TEST: 41 (29%)

---

## 2. How We Reconstructed the Splits

### 2.1 The Problem

The paper describes their methodology (Appendix C) but does NOT provide:
- The exact participant IDs for each split
- The random seed used for stratification
- The specific library/function used

### 2.2 Our Solution: Reconstruction from Output Artifacts

We extracted the exact participant IDs directly from the paper authors' published output files:

| Output File | What It Contains | IDs Extracted |
|-------------|------------------|---------------|
| `_reference/analysis_output/quan_gemma_zero_shot.jsonl` | All 142 participants | ALL |
| `_reference/analysis_output/quan_gemma_few_shot/VAL_analysis_output/*.jsonl` | VAL set evaluations | 43 VAL IDs (union across files) |
| `_reference/analysis_output/quan_gemma_few_shot/TEST_analysis_output/*.jsonl` | TEST set evaluations | 41 TEST IDs |
| Computed: ALL - VAL - TEST | Remainder | 58 TRAIN IDs |

**This reconstruction is authoritative** because these are the actual IDs the paper used for their experiments.

### 2.3 Verification

| Check | Result |
|-------|--------|
| Total participants | 142 |
| TRAIN + VAL + TEST | 58 + 43 + 41 = 142 |
| TRAIN ∩ VAL | 0 (no overlap) |
| TRAIN ∩ TEST | 0 (no overlap) |
| VAL ∩ TEST | 0 (no overlap) |
| All IDs exist in AVEC train+dev | Yes |
| Consistent across output files | Yes (split membership is consistent; a minority of individual VAL analysis files omit 1 ID, so use union / `paper-split-registry.md`) |

---

## 3. Paper's Stated Methodology (Appendix C)

### 3.1 What the Paper Says

> "We stratified the data according to PHQ-8 total scores and gender. [...] For PHQ-8 total scores with two participants, we put one in the validation set and one in the test set. For PHQ-8 total scores with one participant, we put that one participant in the training set."

### 3.2 Verified Heuristics

We verified these heuristics against the reconstructed splits:

#### Rule 1: Single-Participant Scores → TRAIN

| PHQ-8 Score | Participant | Gender | Split | Status |
|-------------|-------------|--------|-------|--------|
| 17 | 388 | F | TRAIN | Verified |
| 23 | 346 | M | TRAIN | Verified |

**Result**: 2/2 follow the rule (100%)

#### Rule 2: Two-Participant Scores → One VAL, One TEST

| PHQ-8 Score | Participants | Genders | Split Distribution | Status |
|-------------|--------------|---------|-------------------|--------|
| 13 | 319, 372 | F, M | VAL + TEST | Verified |
| 14 | 351, 389 | M, F | VAL + TEST | Verified |
| 18 | 441, 448 | M, F | VAL + TEST | Verified |
| 19 | 367, 440 | F, M | VAL + TEST | Verified |

**Result**: 4/4 follow the rule (100%)

#### Rule 3: Three-Participant Strata (Observed Pattern)

For (score, gender) groups with exactly 3 participants, sorted by Participant_ID:
- 1st ID → TRAIN
- 2nd ID → VAL
- 3rd ID → TEST

| Score | Gender | Participant IDs | Splits | Pattern Match |
|-------|--------|-----------------|--------|---------------|
| 3 | M | 415, 419, 472 | TRAIN, VAL, TEST | Yes |
| 6 | M | 304, 425, 456 | TRAIN, VAL, TEST | Yes |
| 8 | F | 317, 331, 385 | TRAIN, VAL, TEST | Yes |
| 9 | M | 391, 401, 484 | TRAIN, VAL, TEST | Yes |
| 9 | F | 343, 371, 390 | TRAIN, VAL, TEST | Yes |
| 12 | M | 335, 376, 422 | TRAIN, VAL, TEST | Yes |
| 16 | F | 347, 381, 459 | TRAIN, VAL, TEST | Yes |
| 20 | M | 321, 348, 362 | TRAIN, VAL, TEST | Yes |

**Result**: 8/8 follow the pattern (100%)

### 3.3 Groups with 4+ Participants

For larger (score, gender) strata, the paper used randomized stratification to achieve the global 58/43/41 split. The paper's implementation details are summarized in `randomization-methodology-discovered.md`.

---

## 4. Randomization Implementation (Discovered from Code)

This section summarizes what we learned by reviewing the paper authors' code (see `randomization-methodology-discovered.md`). This is distinct from the ground truth split membership (which is derived from output files and recorded in `paper-split-registry.md`).

### 4.1 Key Findings

- The split is implemented using NumPy RNG (`np.random.seed(42)` + `np.random.shuffle`), not `sklearn.model_selection.train_test_split`.
- The seed `42` is reset inside the per-stratum loop for strata with ≥3 participants.
- Strata with exactly 2 participants (by `Gender + '_' + PHQ8_Score`) are shuffled without reseeding and initially assigned 1 TRAIN + 1 TEST.
- A post-processing override reassigns PHQ-8 scores with exactly 2 participants (regardless of gender) to 1 VAL + 1 TEST.

### 4.2 High-Level Algorithm (as Implemented)

```text
Input: AVEC train+dev participants (142) with PHQ8_Score + Gender

1) Create primary strata: strat_var = Gender + '_' + PHQ8_Score
2) Process by stratum size:
   - size ≥ 3: reseed(42), shuffle, then split per-stratum (target ~40/30/30 with rounding)
   - size == 2: shuffle (no reseed), then 1 TRAIN + 1 TEST
   - size == 1: 1 TRAIN
3) Post-processing override:
   - For PHQ8_Score values with exactly 2 participants total (regardless of gender):
     remove from current assignments; shuffle (no reseed); then 1 VAL + 1 TEST
```

### 4.3 Practical Reproducibility Notes

- The split membership is reproducible by definition via `paper-split-registry.md` (output-derived ground truth).
- Algorithmic reproduction is more fragile due to ordering and mixed reseeded/unseeded shuffles (details in `randomization-methodology-discovered.md`).
- With identical input CSVs, NumPy/pandas versions, and category iteration order, the paper’s splitting code is deterministic (seeding makes pseudo-random operations reproducible); the fragility is that those factors are not guaranteed across environments.

---

## 5. Summary

### What We Know (Verified)

1. **Exact participant IDs**: Reconstructed from `_reference/analysis_output/`
2. **Stated heuristics followed**: Rules for 1-2 participant scores are 100% followed
3. **Stratification variables**: PHQ-8 score and Gender (jointly)
4. **Global split counts**: 58 TRAIN / 43 VAL / 41 TEST (≈41% / 30% / 29%)
5. **Implementation detail**: NumPy RNG splitting with per-stratum reseed + post-processing override (see `randomization-methodology-discovered.md`)

### What We Don't Know

1. **Exact processing order**: category iteration order affects mixed reseeded/unseeded shuffle state
2. **Exact input files and versions**: the paper's code reads CSVs outside this repo
3. **Potential manual adjustments**: not evidenced in the output artifacts

### Our Approach

Since the output artifacts already encode the truth, we:
1. **Extracted exact IDs from the paper's output artifacts** (authoritative source)
2. **Documented in `paper-split-registry.md`** (single source of truth)
3. **Hardcoded these IDs in `scripts/create_paper_split.py` (default behavior)** for reproducible local artifact generation

---

## 6. Reproducibility Note

### What the Paper Did Right

- Published their output files (which allowed us to reconstruct the splits)
- Documented their stratification methodology in Appendix C
- Used standard, freely available tools

### Recommendations for Future Papers

For even easier reproducibility, papers should also publish:

- Exact participant IDs for each split, OR
- Random seed AND exact library version used

In this case, we were able to reconstruct the splits from output artifacts - but this may not always be possible for other papers.

---

## 7. Related Documents

| Document                          | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `paper-split-registry.md`         | Ground truth participant IDs (the authoritative source) |
| `critical-data-split-mismatch.md` | How we discovered our algorithmic splits were wrong     |
| `spec-data-split-fix.md`          | Implementation spec for fixing our artifacts            |
| `randomization-methodology-discovered.md` | Discovered implementation details from paper authors' code |
| `artifact-namespace-registry.md`  | Naming conventions for split-related files              |

---

## 8. Artifact Provenance

```text
Source: _reference/analysis_output/ (paper authors' published outputs)
    │
    ├── quan_gemma_zero_shot.jsonl ──────────────► ALL 142 IDs
    ├── quan_gemma_few_shot/VAL_analysis_output/ ─► VAL 43 IDs
    └── quan_gemma_few_shot/TEST_analysis_output/ ► TEST 41 IDs
                                                        │
                                                        ▼
                                              TRAIN = ALL - VAL - TEST
                                                   (58 IDs)
                                                        │
                                                        ▼
                                        docs/data/paper-split-registry.md
                                           (Single Source of Truth)
                                                        │
                                                        ▼
                                        data/paper_splits/paper_split_*.csv
                                             (Generated Artifacts)
```

---

*Analysis performed: 2025-12-25*
*Reconstructed from: `_reference/analysis_output/`*
*Verified by: Cross-referencing multiple output files for consistency*
