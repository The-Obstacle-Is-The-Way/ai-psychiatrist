# RANDOMIZATION METHODOLOGY DISCOVERED

## Executive Summary

We successfully identified the **exact randomization methodology** used by the paper authors by examining their source code in `quantitative_assessment/embedding_batch_script.py` and `quantitative_assessment/embedding_quantitative_analysis.ipynb`.

**Key Discovery**: The split is implemented with NumPy RNG (`np.random.seed` + `np.random.shuffle`), not sklearn `train_test_split`. The seed `42` is reset **inside the per-stratum loop for strata with ≥3 participants**, while other shuffles (2-participant strata and the PHQ8-score override) run **without resetting the seed**.

---

## Source Code Evidence

### File: `quantitative_assessment/embedding_batch_script.py`

**Lines 850-856** (randomization for ≥3 participant groups):
```python
# Randomly assign participants
np.random.seed(42)
shuffled = category_participants.copy()
np.random.shuffle(shuffled)

train_ids.extend(shuffled[:actual_train])
val_ids.extend(shuffled[actual_train:actual_train+actual_val])
test_ids.extend(shuffled[actual_train+actual_val:])
```

**Lines 862-865** (randomization for exactly-2 participant strata; no seed reset here):
```python
# Randomly assign 1 to train and 1 to test
np.random.shuffle(category_participants)
train_ids.append(category_participants[0])
test_ids.append(category_participants[1])
```

**Lines 879-900** (post-processing override for 2-participant PHQ8 scores):
```python
# Post-Prossessing Override: Handle cases where PHQ8_Score has exactly 2 participants
# (regardless of gender) after previous stratification takes place
phq8_score_counts = phq8_ground_truths['PHQ8_Score'].value_counts()
scores_with_2 = phq8_score_counts[phq8_score_counts == 2].index.tolist()

for score in scores_with_2:
    score_participants = phq8_ground_truths[phq8_ground_truths['PHQ8_Score'] == score]['Participant_ID'].tolist()

    # Remove these participants from their current assignments
    for pid in score_participants:
        if pid in train_ids:
            train_ids.remove(pid)
        if pid in val_ids:
            val_ids.remove(pid)
        if pid in test_ids:
            test_ids.remove(pid)

    # Reassign: 1 to validation, 1 to test
    np.random.shuffle(score_participants)
    val_ids.append(score_participants[0])
    test_ids.append(score_participants[1])
```

---

## Complete Algorithm (Reconstructed from Code)

```
Input: 142 participants from AVEC train+dev (with per-item PHQ-8 scores)
Output: TRAIN (58), VAL (43), TEST (41)

Step 1: Create stratification variable
    strat_var = Gender + '_' + PHQ8_Score

Step 2: Categorize by strata size
    - categories_gte3: strata with ≥3 participants
    - categories_eq2: strata with exactly 2 participants
    - categories_eq1: strata with exactly 1 participant

Step 3: Process each category

    For categories with ≥3 participants:
        np.random.seed(42)  # Reset seed for EACH stratum
        shuffled = category_participants.copy()
        np.random.shuffle(shuffled)  # in-place

        # Per-stratum target split: 40/30/30 (with rounding)
        actual_train = round(n * 0.40)
        actual_val = round(n * 0.30)
        actual_test = n - actual_train - actual_val
        if actual_test < 0:
            actual_train = max(1, actual_train - 1)
            actual_test = n - actual_train - actual_val

        train_ids += shuffled[:actual_train]
        val_ids += shuffled[actual_train:actual_train+actual_val]
        test_ids += shuffled[actual_train+actual_val:]

    For categories with exactly 2 participants:
        # Note: no seed reset here; uses current NumPy RNG state
        np.random.shuffle(category_participants)
        train_ids += [first]
        test_ids += [second]

    For categories with exactly 1 participant:
        train_ids += [participant]

Step 4: POST-PROCESSING OVERRIDE
    For PHQ8_Score values with exactly 2 participants (REGARDLESS of gender):
        Remove from current assignments
        # Note: no seed reset here; uses current NumPy RNG state
        np.random.shuffle(score_participants)
        val_ids += [first]
        test_ids += [second]
```

---

## Critical Implementation Details

### 1. Seed Reset Per Category
The seed `42` is reset inside the loop for each category with ≥3 participants. This means:
- The first category always gets the same shuffle
- But subsequent categories also reset to seed 42
- This is NOT the same as setting seed once globally

**Important nuance**: the seed reset only happens in the ≥3-participant loop (`quantitative_assessment/embedding_batch_script.py:850`). The shuffles for 2-participant strata (`quantitative_assessment/embedding_batch_script.py:863`) and the PHQ8-score override (`quantitative_assessment/embedding_batch_script.py:896`) do **not** reset the seed.

### 1.1 Bug in Original Code (`quantitative_assessment/embedding_batch_script.py:867`)
The original code has contradictory log messages:
```python
log_message(f"  {len(categories_eq2)} categories: 1 to train, 1 to test each")
# ...
log_message(f"  {len(categories_eq2)} categories: 1 to train, 1 to validation each")
```
First says "1 to test", then says "1 to validation". The **actual behavior** is 1 to train, 1 to test (see `quantitative_assessment/embedding_batch_script.py:864`). This is sloppy code by the original author and may have caused confusion.

### 2. Post-Processing Override
The paper's Appendix C mentions: "For PHQ-8 total scores with two participants, we put one in the validation set and one in the test set."

This is implemented as a POST-PROCESSING step (lines 879-900) that:
1. Finds PHQ8_Score values with exactly 2 participants total
2. Removes those participants from their current assignments
3. Reassigns: 1 to VAL, 1 to TEST

**Note**: This is different from the initial handling of 2-participant strata (which assigns 1 to train, 1 to test).

### 3. No sklearn Usage for Splitting
Despite importing `train_test_split` (`quantitative_assessment/embedding_batch_script.py:14`) and a misleading comment (`quantitative_assessment/embedding_batch_script.py:827`, `# Process categories with >= 3 subjects using sklearn`), the actual splitting uses:
- `np.random.seed(42)`
- `np.random.shuffle()`
- Manual slicing of shuffled lists

sklearn's `train_test_split` is never called in the splitting logic.

---

## Why We Still Can't Reproduce Exactly

With a fixed seed, pseudo-random operations are deterministic and reproducible: if you run the exact same code with the exact same input CSVs, NumPy/pandas versions, and category iteration order, you will get the same split.

This specific implementation is fragile across environments, so exact reproduction may still fail due to:

1. **Category processing order matters**: `categories_gte3` is derived from `value_counts()` and iterated in that resulting order (not explicitly sorted), so tie-breaking/ordering details can affect which stratum is processed last.

2. **Mixed seeded + unseeded shuffles**: the ≥3-stratum shuffles reset to seed 42 each iteration, but the 2-stratum shuffle(s) and the PHQ8-score override shuffle do not reset the seed, so they depend on the RNG state left by the last ≥3-stratum shuffle (and any prior unseeded shuffles).

3. **Input CSV versioning**: the code reads AVEC train+dev CSVs from `/data/.../daic_woz_dataset/*.csv`, which are not included in this repo; any differences in those source files will change the strata and outcomes.

4. **Library/version details**: pandas `value_counts()` ordering and NumPy RNG behavior are typically stable, but are not guaranteed across all versions/environments.

---

## Verification Against Output Files

We have the **ground truth IDs** from the paper's output files (see `DATA_SPLIT_REGISTRY.md`):

| Split | Count | Source |
|-------|-------|--------|
| TRAIN | 58 | Derived from `_reference/analysis_output/quan_gemma_zero_shot.jsonl` minus TEST minus VAL |
| VAL | 43 | Union of `_reference/analysis_output/quan_gemma_few_shot/VAL_analysis_output/*.jsonl` |
| TEST | 41 | `_reference/analysis_output/quan_gemma_few_shot/TEST_analysis_output/*.jsonl` |

These are authoritative regardless of whether we can reproduce the algorithm.

---

## Conclusion

### What We Now Know (Confirmed from Code)

| Element | Value | Evidence |
|---------|-------|----------|
| Random seed | 42 | Line 850: `np.random.seed(42)` |
| Library | NumPy | `np.random.shuffle()` not sklearn |
| Per-stratum target split | 40/30/30 (rounded) | Lines 835-842 |
| Stratification | `Gender + '_' + PHQ8_Score` | Lines 800-802 |
| Post-processing override | 2-participant PHQ8 scores → 1 VAL, 1 TEST | Lines 879-900 |

### What Remains Uncertain

- Exact NumPy version used
- Exact processing order of categories
- Whether any manual adjustments were made

### Recommendation

**Use the hardcoded IDs from `DATA_SPLIT_REGISTRY.md`** as the authoritative source. The reconstructed IDs from output files are guaranteed correct, while algorithmic reproduction has edge cases that may differ.

---

## Files Referenced

| File | Lines | Content |
|------|-------|---------|
| `quantitative_assessment/embedding_batch_script.py` | 820-926 | Complete splitting algorithm |
| `quantitative_assessment/embedding_quantitative_analysis.ipynb` | Cell 7995b586 | Same algorithm (notebook version) |
| `DATA_SPLIT_REGISTRY.md` | - | Authoritative ground truth IDs |

---

*Analysis performed: 2025-12-25*
*Source: trendscenter/ai-psychiatrist repository*
