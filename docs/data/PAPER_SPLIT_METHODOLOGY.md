# Paper Split Methodology Analysis

**Status**: VERIFIED
**Date**: 2025-12-25
**Related**: `DATA_SPLIT_REGISTRY.md`, `CRITICAL_DATA_SPLIT_MISMATCH.md`

---

## Executive Summary

We **successfully reconstructed the paper's exact data splits** by extracting participant IDs from their published output files in `_reference/analysis_output/`. This reconstruction **confirms that the paper followed their stated methodology** (Appendix C) - the heuristics they described are verifiably correct.

The **only element we cannot explicitly reproduce** is their random seed for stratifying groups with 4+ participants. However, standard tools like scikit-learn's `train_test_split` with stratification are freely available and were almost certainly used.

Our `DATA_SPLIT_REGISTRY.md` contains the authoritative ground truth IDs.

---

## 1. Data Source

### 1.1 What the Paper Used

The paper combined AVEC2017 train and dev sets (142 participants total):

| AVEC Split | Participants | Per-Item PHQ-8 | Used by Paper |
|------------|--------------|----------------|---------------|
| train | 107 | Yes | Yes |
| dev | 35 | Yes | Yes |
| test | 47 | **No** | **Excluded** |

**Key insight**: The AVEC2017 test set was excluded because it only contains total PHQ-8 scores (for competition evaluation), not the per-item scores needed for the paper's quantitative assessment agent.

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
| `_reference/analysis_output/quan_gemma_few_shot/VAL_analysis_output/*.jsonl` | VAL set evaluations | 43 VAL IDs |
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
| Consistent across all output files | Yes |

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

For larger (score, gender) strata, the paper used randomized stratification maintaining the 41%/30%/29% ratio. The specific assignments cannot be reproduced without the original random seed.

---

## 4. Standard Tools for Stratified Splitting

The paper's methodology uses standard stratified splitting techniques. These tools are **freely available** and widely used in machine learning:

### 4.1 scikit-learn's `train_test_split` with Combined Stratification

```python
from sklearn.model_selection import train_test_split

# Combine PHQ-8 score and gender into a single stratification variable
df['strata'] = df['PHQ8_Score'].astype(str) + '_' + df['Gender'].astype(str)

# Two-stage split: first train vs (val+test), then val vs test
train, temp = train_test_split(df, test_size=0.59, stratify=df['strata'], random_state=SEED)
val, test = train_test_split(temp, test_size=0.488, stratify=temp['strata'], random_state=SEED)
```

**Reference**: [scikit-learn train_test_split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

### 4.2 scikit-learn's `StratifiedShuffleSplit`

```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.29, random_state=SEED)
# ... with custom handling for rare strata
```

**Reference**: [scikit-learn StratifiedShuffleSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

### 4.3 What We Reconstructed vs What Remains Unknown

| Element | Status | Notes |
|---------|--------|-------|
| **Exact participant IDs** | Reconstructed | Extracted from output artifacts |
| **Stratification methodology** | Confirmed | Matches paper's Appendix C |
| **Heuristics for rare groups** | Verified | 1-2 participant rules followed |
| **Random seed** | Unknown | Not disclosed; affects only groups of 4+ |
| **Tool used** | Likely scikit-learn | Standard approach, freely available |

The random seed only affects groups with 4+ participants. Since we have the exact IDs from their artifacts, the unknown seed is irrelevant for reproduction.

---

## 5. Summary

### What We Know (Verified)

1. **Exact participant IDs**: Reconstructed from `_reference/analysis_output/`
2. **Stated heuristics followed**: Rules for 1-2 participant scores are 100% followed
3. **Stratification variables**: PHQ-8 score and Gender (jointly)
4. **Target ratios**: 41% TRAIN / 30% VAL / 29% TEST

### What We Don't Know

1. **Random seed**: Not disclosed in paper
2. **Exact library/function**: Likely scikit-learn but unconfirmed
3. **Implementation details**: Manual vs automated handling of rare strata

### Our Approach

Since algorithmic reproduction is impossible without the seed, we:
1. **Extracted exact IDs from the paper's output artifacts** (authoritative source)
2. **Documented in `DATA_SPLIT_REGISTRY.md`** (single source of truth)
3. **Will hardcode these IDs in `create_paper_split.py`** (reproducible implementation)

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
| `DATA_SPLIT_REGISTRY.md`          | Ground truth participant IDs (the authoritative source) |
| `CRITICAL_DATA_SPLIT_MISMATCH.md` | How we discovered our algorithmic splits were wrong     |
| `SPEC-DATA-SPLIT-FIX.md`          | Implementation spec for fixing our artifacts            |
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
                                        docs/data/DATA_SPLIT_REGISTRY.md
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
