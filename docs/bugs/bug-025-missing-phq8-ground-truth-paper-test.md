# BUG-025: Missing PHQ-8 Item-Level Ground Truth in Paper Test Split

> **STATUS: RESOLVED**
>
> **Discovered**: 2025-12-26
>
> **Resolved**: 2025-12-26
>
> **Severity**: Blocker (prevented paper reproduction run on `--split paper`)
>
> **Affects**: `scripts/reproduce_results.py --split paper`
>
> **Resolution**: Mathematical reconstruction applied via `scripts/patch_missing_phq8_values.py`

---

## Summary

The paper test split (`data/paper_splits/paper_split_test.csv`) contains 1 participant (ID 319) with missing PHQ-8 item-level ground truth (PHQ8_Sleep is NaN). This causes `reproduce_results.py` to crash with `ValueError: cannot convert float NaN to integer`.

---

## Complete Data Provenance Chain

### Level 1: AVEC2017 Raw Dataset (Upstream)

**File**: `data/train_split_Depression_AVEC2017.csv`

```csv
Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest,PHQ8_Depressed,PHQ8_Sleep,PHQ8_Tired,PHQ8_Appetite,PHQ8_Failure,PHQ8_Concentrating,PHQ8_Moving
319,1,13,1,2,1,,1,1,2,3,1
```

**Observation**: PHQ8_Sleep is empty (missing) in the original AVEC2017 dataset. This is a data collection issue from the DAIC-WOZ study, not introduced by our code.

### Level 2: Paper Authors' Ground Truth IDs

**File**: `docs/data/paper-split-registry.md`
**Source**: Reverse-engineered from `_reference/analysis_output/quan_gemma_few_shot/TEST_analysis_output/*.jsonl`

```text
TEST (41 participants):
316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389,
390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451,
455, 456, 459, 468, 472, 484, 485, 487, 489
```

**Observation**: Participant 319 IS in the paper authors' test set. This means either:
1. The paper authors had complete data we don't have
2. The paper authors handled missing data somehow (imputation? exclusion from item-level metrics?)
3. The paper's evaluation code has similar handling we need to replicate

### Level 3: create_paper_split.py Processing

**File**: `scripts/create_paper_split.py`

```python
# Line 163-165: Hardcoded ground truth test IDs
_GROUND_TRUTH_TEST_IDS = [
    316,
    319,  # <-- Participant with missing PHQ8_Sleep
    ...
]

# Line 258-263: PHQ8_Total computation
item_cols = [c for c in combined.columns if c.startswith("PHQ8_")
             and c not in {"PHQ8_Binary", "PHQ8_Score"}]
combined["PHQ8_Total"] = combined[item_cols].sum(axis=1).astype(int)
# NOTE: pandas.sum() treats NaN as 0, so PHQ8_Total = 11 (not 13)
```

**Observation**: The script copies raw AVEC data to paper splits without validating for complete per-item data.

### Level 4: Paper Split Output

**File**: `data/paper_splits/paper_split_test.csv`

```csv
319,1,13,1,2,1,,1,1,2,3,1,11
```

| Column | Value | Notes |
|--------|-------|-------|
| Participant_ID | 319 | |
| PHQ8_Binary | 1 | Depressed classification |
| PHQ8_Score | 13 | **Ground truth total from AVEC** |
| Gender | 1 | Female |
| PHQ8_NoInterest | 2 | |
| PHQ8_Depressed | 1 | |
| **PHQ8_Sleep** | **EMPTY** | **MISSING VALUE** |
| PHQ8_Tired | 1 | |
| PHQ8_Appetite | 1 | |
| PHQ8_Failure | 2 | |
| PHQ8_Concentrating | 3 | |
| PHQ8_Moving | 1 | |
| PHQ8_Total | 11 | **Computed (treats NaN as 0)** |

**Key Discrepancy**:
- `PHQ8_Score` (AVEC ground truth) = 13
- `PHQ8_Total` (computed from items) = 11
- Difference = 2 → This is the missing PHQ8_Sleep value

### Level 5: reproduce_results.py Failure

**File**: `scripts/reproduce_results.py`

```python
# Line 199-201
for item in PHQ8Item.all_items():
    col = f"PHQ8_{item.value}"
    scores[item] = int(row[col])  # CRASH: cannot convert NaN to int
```

**Error**:
```
ValueError: cannot convert float NaN to integer
  File "scripts/reproduce_results.py", line 201
```

---

## Scope Analysis

| Split | Total | Missing Items | Percentage | Affected IDs |
|-------|-------|---------------|------------|--------------|
| paper-train | 58 | 0 | 0.0% | None |
| paper-val | 43 | 0 | 0.0% | None |
| paper-test | 41 | 1 | 2.4% | **319** |
| AVEC train | 107 | 1 | 0.9% | 319 |
| AVEC dev | 35 | 0 | 0.0% | None |

Only participant 319 is affected across all splits.

---

## Transcript & Mathematical Analysis

### The Missing Value is Mathematically Deterministic

The AVEC2017 dataset provides `PHQ8_Score` (the authoritative total) separately from per-item scores. This allows us to **reconstruct** the missing value with certainty:

```
PHQ8_Score (ground truth total from AVEC) = 13
Known items: PHQ8_NoInterest=2, PHQ8_Depressed=1, PHQ8_Tired=1,
             PHQ8_Appetite=1, PHQ8_Failure=2, PHQ8_Concentrating=3, PHQ8_Moving=1
Sum of known items = 2+1+1+1+2+3+1 = 11
Missing: PHQ8_Sleep = 13 - 11 = 2
```

**This is NOT imputation or guessing** - it is mathematical reconstruction from authoritative ground truth. The `PHQ8_Score=13` is the questionnaire total (patient-reported via PHQ-8 self-assessment); we are simply recovering the missing component.

### Transcript Evidence (Corroborating)

The transcript for participant 319 (`data/transcripts/319_P/319_TRANSCRIPT.csv`) provides supporting clinical evidence:

**Direct Sleep Question:**
```
Ellie: "what are you like when you don't sleep well"
Participant: "irritable"
Participant: "cranky"
Ellie: "that sounds really hard"
Participant: "yeah it is"
```

**Depression Symptoms (self-described):**
```
Participant: "always tired and"
Participant: "not excited about things anymore and"
Participant: "kinda lethargic you know laying around and"
Participant: "just not feeling myself"
```

**Current State:**
```
Ellie: "how have you been feeling lately"
Participant: "mm about the same"
```

### PHQ-8 Sleep Item Scoring Reference

PHQ-8 Item 3 (Sleep): "Trouble falling or staying asleep, or sleeping too much"
- 0 = Not at all
- 1 = Several days
- 2 = More than half the days
- 3 = Nearly every day

The participant describes being "irritable" and "cranky" when they don't sleep well, confirms "yeah it is" hard, and reports ongoing symptoms ("about the same"). A score of **2 ("More than half the days")** is consistent with this clinical presentation.

### Conclusion

The missing `PHQ8_Sleep` value can be **deterministically reconstructed as 2** based on:
1. **Mathematical proof**: `PHQ8_Score(13) - sum(other items)(11) = 2`
2. **Transcript corroboration**: Clinical presentation consistent with moderate sleep disturbance

This is a data entry/export error in the upstream AVEC2017 dataset (a missing cell), NOT missing clinical information.

---

## Open Questions for Senior Review

### Q1: How did the paper authors handle participant 319?

The paper authors included 319 in their TEST output files. Possibilities:
- They had complete data from a different source
- They imputed PHQ8_Sleep = 2 (which makes sum = 13 = PHQ8_Score)
- Their evaluation code skipped this participant for item-level MAE
- Their code had similar NaN handling we need to replicate

### Q2: How should we fix the missing value?

**Option A: Reconstruct the value in upstream AVEC CSV (RECOMMENDED)**
- Fix `data/train_split_Depression_AVEC2017.csv`: change `319,1,13,1,2,1,,1,1,2,3,1` to `319,1,13,1,2,1,2,1,1,2,3,1`
- Regenerate paper splits with `python scripts/create_paper_split.py`
- **Justification**: This is NOT imputation - it's mathematical reconstruction. The value 2 is deterministically derivable from `PHQ8_Score=13` minus sum of other items.
- Preserves N=41 test participants
- Matches what paper authors likely had

**Option B: Exclude participant 319**
- Test set becomes 40 participants instead of 41
- Overly conservative given we can prove the value mathematically
- May not match paper methodology

**Option C: Fix only in reproduce_results.py**
- Skip participants with missing ground truth, log warning
- Defensive coding pattern
- Still results in N=40 for paper-test
- Doesn't fix root cause

### Q3: Recommended Fix Location

| Location | Action | Rationale |
|----------|--------|-----------|
| `data/train_split_Depression_AVEC2017.csv` | Add `2` for PHQ8_Sleep | Fix at source; value is mathematically certain |
| `scripts/create_paper_split.py` | Regenerate splits | Propagate fix to paper splits |
| `scripts/reproduce_results.py` | Add defensive NaN check | Belt-and-suspenders for future issues |

---

## Data Propagation Chain (Fix Locations)

Scanning the entire data tree shows the missing value exists in exactly 2 files:

```
data/train_split_Depression_AVEC2017.csv    <-- SOURCE (fix here)
    ↓ (via create_paper_split.py)
data/paper_splits/paper_split_test.csv      <-- DERIVED (regenerate)
```

**Full scan confirmed**: No other files contain participant 319 with missing data. No other participants have missing values (`,,` pattern).

### Recommended Fix Steps

1. **Fix source**: Edit `data/train_split_Depression_AVEC2017.csv` line for participant 319:
   - FROM: `319,1,13,1,2,1,,1,1,2,3,1`
   - TO: `319,1,13,1,2,1,2,1,1,2,3,1`

2. **Regenerate paper splits**:
   ```bash
   uv run python scripts/create_paper_split.py --verify
   ```

3. **Add validation** (belt-and-suspenders): Add NaN check to `reproduce_results.py` to fail loudly if any future missing values occur.

---

## Recommended Defensive Validation

After fixing the data, add validation to catch any future issues. The system should **fail loudly** on NaN rather than silently produce wrong results.

**In `reproduce_results.py`** - after loading ground truth:
```python
# Validate no NaN in ground truth
for pid, scores in ground_truth.items():
    for item, score in scores.items():
        if pd.isna(score):
            raise ValueError(f"Missing ground truth for {pid} {item.value}")
```

**In `create_paper_split.py`** - before saving:
```python
# Validate no NaN in PHQ-8 items
phq_cols = [c for c in df.columns if c.startswith("PHQ8_")
            and c not in {"PHQ8_Binary", "PHQ8_Score", "PHQ8_Total"}]
missing = df[df[phq_cols].isna().any(axis=1)]
if not missing.empty:
    raise ValueError(f"Missing PHQ-8 values for participants: {missing['Participant_ID'].tolist()}")
```

---

## Temporary Workaround

Until resolved, use AVEC dev split for testing:
```bash
uv run python scripts/reproduce_results.py --split dev --zero-shot-only
```

AVEC dev has 35 participants with complete ground truth (no missing values).

---

## Files Involved

| File | Role | Issue |
|------|------|-------|
| `data/train_split_Depression_AVEC2017.csv` | Upstream data | Contains original missing value |
| `docs/data/paper-split-registry.md` | Ground truth IDs | Lists 319 in TEST |
| `scripts/create_paper_split.py` | Split creation | No validation for complete data |
| `data/paper_splits/paper_split_test.csv` | Output | Contains propagated missing value |
| `scripts/reproduce_results.py` | Evaluation | Crashes on NaN |

---

## Related Bugs

- **BUG-003**: Participant 487 data corruption (macOS resource fork issue)
  - Different root cause (extraction issue vs. upstream missing data)
  - Similar category (data integrity)

---

## Appendix: Verification Commands

```bash
# Check which participants have missing data
uv run python -c "
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/paper_splits/paper_split_{split}.csv')
    phq_cols = [c for c in df.columns if c.startswith('PHQ8_')
                and c not in {'PHQ8_Binary', 'PHQ8_Score', 'PHQ8_Total'}]
    missing = df[df[phq_cols].isna().any(axis=1)]['Participant_ID'].tolist()
    print(f'paper-{split}: missing={missing}')
"

# Trace participant 319
grep "^319," data/train_split_Depression_AVEC2017.csv
grep "^319," data/paper_splits/paper_split_test.csv
```

---

## Resolution Applied (2025-12-26)

### Actions Taken

1. **Created deterministic patch script**: `scripts/patch_missing_phq8_values.py`
   - Validates preconditions (exactly one missing item, reconstructed value in [0,3])
   - Applies mathematical reconstruction with full audit trail
   - Verifies invariant `PHQ8_Score == sum(items)` for all complete rows

2. **Applied patch to upstream CSV**: `data/train_split_Depression_AVEC2017.csv`
   - Changed: `319,1,13,1,2,1,,1,1,2,3,1` → `319,1,13,1,2,1,2,1,1,2,3,1`
   - Preserves N=41 test participants (matches paper)

3. **Regenerated paper splits**: `uv run python scripts/create_paper_split.py --verify`
   - `paper_split_test.csv` now has `PHQ8_Sleep=2` for participant 319
   - `PHQ8_Total=13` matches `PHQ8_Score=13`

4. **Added fail-loud validation** to both scripts:
   - `scripts/reproduce_results.py`: Raises `ValueError` with fix instructions on NaN
   - `scripts/create_paper_split.py`: Raises `ValueError` if any PHQ-8 items are missing

5. **Created data provenance note**: `data/DATA_PROVENANCE.md`
   - Documents the patch with mathematical proof
   - Provides verification commands

### Verification

```bash
# Confirm no missing values
uv run python scripts/patch_missing_phq8_values.py --dry-run
# Output: "OK: No missing values" for both files

# Confirm participant 319 is correct
grep "^319," data/train_split_Depression_AVEC2017.csv
# Output: 319,1,13,1,2,1,2,1,1,2,3,1

grep "^319," data/paper_splits/paper_split_test.csv
# Output: 319,1,13,1,2,1,2,1,1,2,3,1,13
```

---

*Discovered during preflight check for paper reproduction run, 2025-12-26*
*Resolved same day after senior review approval*
