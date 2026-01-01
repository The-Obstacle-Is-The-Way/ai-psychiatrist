# Data Splits Overview

**Purpose**: Definitive reference for all data split configurations
**Last Updated**: 2025-12-26
**Tracked by**: [GitHub Issue #45](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/45)

This document explains the relationship between AVEC2017 competition splits, the paper's custom splits, and our implementation.

---

## The Core Problem

The AVEC2017 test set does NOT have per-item PHQ-8 scores. You cannot compute item-level MAE without per-item ground truth.

| Split | Per-Item PHQ-8 | Total Score | Can Compute Item MAE? |
|-------|---------------|-------------|----------------------|
| Train | ✅ Yes | ✅ Yes | ✅ Yes |
| Dev | ✅ Yes | ✅ Yes | ✅ Yes |
| Test | ❌ No | ✅ Yes | ❌ **NO** |

**This is why the paper created their own split.**

---

## Part 1: AVEC2017 Official Splits

### What is AVEC2017?

- **AVEC** = Audio/Visual Emotion Challenge (annual competition)
- **2017** = The year this depression detection challenge ran
- **DAIC-WOZ** = The dataset used (Distress Analysis Interview Corpus - Wizard of Oz)

AVEC2017 defined official train/dev/test splits of the DAIC-WOZ dataset.

### Official Split Counts

From the original challenge (Ringeval et al., 2019):

| Split | Participants | Per-Item PHQ-8 | Purpose |
|-------|-------------|----------------|---------|
| Train | 107 | ✅ Available | Model training |
| Dev | 35 | ✅ Available | Hyperparameter tuning |
| Test | 47 | ❌ **Not available** | Competition evaluation |
| **Total** | **189** | | |

### Our Local Data

```text
data/train_split_Depression_AVEC2017.csv  → 107 participants
data/dev_split_Depression_AVEC2017.csv    → 35 participants
data/test_split_Depression_AVEC2017.csv   → 47 participants (matches)
data/full_test_split.csv                  → 47 participants (total score only)
```

Note: `data/` is gitignored due to DAIC-WOZ licensing. These files exist locally after
running the dataset preparation step (`scripts/prepare_dataset.py`), but are not
committed to the repository.

### Test Set Schema Comparison

**Train/Dev CSVs** (have everything):
```csv
Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest,PHQ8_Depressed,PHQ8_Sleep,PHQ8_Tired,PHQ8_Appetite,PHQ8_Failure,PHQ8_Concentrating,PHQ8_Moving
```

**Test Split CSV** (NO PHQ-8 at all):
```csv
participant_ID,Gender
```

**Full Test CSV** (total score only, NO per-item):
```csv
Participant_ID,PHQ_Binary,PHQ_Score,Gender
```

---

## Part 2: What the Paper Did

### The Problem

The paper wanted to report **item-level MAE** (error on each of the 8 PHQ-8 items).

But the AVEC2017 test set doesn't have per-item labels. So they couldn't use it.

### Their Solution

From Paper Section 2.4.1:

> "We split 142 subjects with eight-item PHQ-8 scores from the DAIC-WOZ database into training, validation, and test sets."

They:
1. Combined train + dev = 142 participants (all have per-item labels)
2. Created a NEW 58/43/41 stratified split
3. Used their 58 for the few-shot knowledge base
4. Reported MAE on their 41-participant custom test set

### Paper's Custom Split

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| Paper Train | 58 | 41% | Few-shot embedding knowledge base |
| Paper Val | 43 | 30% | Hyperparameter tuning (Appendix D) |
| Paper Test | 41 | 29% | Final MAE evaluation (Table 1, Figure 4/5) |

### Stratification Algorithm (Appendix C)

> "We stratified 142 subjects [...] based on PHQ-8 total scores and gender information."
> "For PHQ-8 total scores with two participants, we put one in the validation set and one in the test set. For PHQ-8 total scores with one participant, we put that one participant in the training set."

This ensures balanced distribution of severity levels across splits.

---

## Part 3: Our Implementation Options

### Option A: Use AVEC2017 Dev (Current Approach)

```bash
uv run python scripts/reproduce_results.py --split dev
```

| Aspect | Value |
|--------|-------|
| Evaluation set | AVEC2017 dev split (35 participants) |
| Knowledge base | AVEC2017 train split (107 participants) |
| Comparable to paper? | ⚠️ Different participants |
| Valid methodology? | ✅ Yes |

**Pros**: Simpler, no data leakage concerns, uses official splits
**Cons**: Results not directly comparable to paper's 0.619 MAE

### Option B: Use Paper Ground Truth Split (Recommended for Reproduction)

```bash
uv run python scripts/create_paper_split.py  # uses --mode ground-truth by default
uv run python scripts/generate_embeddings.py --split paper-train
uv run python scripts/reproduce_results.py --split paper
```

| Aspect | Value |
|--------|-------|
| Evaluation set | Custom test split (41 participants) |
| Knowledge base | Custom train split (58 participants) |
| Comparable to paper? | ✅ **Exact match** (IDs reverse-engineered from output files) |
| Valid methodology? | ✅ Yes |

**Pros**: Matches paper participants exactly.
**Cons**: Requires generating paper split + paper embeddings artifact locally (not committed).

### Option C: Full Train+Dev Evaluation

```bash
uv run python scripts/reproduce_results.py --split train+dev
```

| Aspect | Value |
|--------|-------|
| Evaluation set | All 142 participants |
| Knowledge base | Same participants (⚠️ data leakage!) |
| Comparable to paper? | ❌ No |
| Valid methodology? | ⚠️ Only for debugging |

**Warning**: This has data leakage because you're testing on the same participants used for few-shot retrieval.

---

## Part 4: Data Leakage Considerations

### What is Data Leakage?

If you use participant X's transcript to build the few-shot knowledge base, then test on participant X, the model has "seen" the answer. This artificially inflates performance.

### How to Avoid It

| Scenario | Knowledge Base | Test Set | Leakage? |
|----------|---------------|----------|----------|
| AVEC2017 approach | Train only (107) | Dev only (35) | ✅ No leakage |
| Paper approach | Paper train (58) | Paper test (41) | ✅ No leakage |
| Our current | Train only (107) | Dev (35) | ✅ No leakage |
| Dangerous | Train+Dev | Train+Dev | ❌ **LEAKAGE** |

### Our `scripts/generate_embeddings.py` Uses Train Only

From `scripts/generate_embeddings.py`:
```python
# Uses ONLY training IDs to avoid leakage:
# - AVEC2017: data/train_split_Depression_AVEC2017.csv
# - Paper: data/paper_splits/paper_split_train.csv
```

This is correct. We never include dev participants in the knowledge base.

---

## Part 5: Recommended Approach

### For Paper Reproduction

1. **Generate paper ground truth splits**:
   ```bash
   uv run python scripts/create_paper_split.py
   ```
   This uses the exact participant IDs reverse-engineered from the paper authors' output files.

2. **Regenerate embeddings using paper train split only**:
   ```bash
   uv run python scripts/generate_embeddings.py --split paper-train
   ```

3. **Evaluate on paper test split**:
   ```bash
   uv run python scripts/reproduce_results.py --split paper --few-shot-only
   ```

### For General Use

Use AVEC2017 official splits. They're simpler and avoid any ambiguity:
```bash
uv run python scripts/reproduce_results.py --split dev
```

---

## Part 6: Summary Table

| Split Source | Train | Val/Dev | Test | Total | Per-Item Labels |
|-------------|-------|---------|------|-------|-----------------|
| AVEC2017 Official | 107 | 35 | 47 | 189 | Train+Dev only |
| Paper Custom | 58 | 43 | 41 | 142 | All (from train+dev) |
| Our Local Data | 107 | 35 | 47 | 189 | Train+Dev only |

---

## Files Reference

| File | Source | Count | Has Per-Item PHQ-8 |
|------|--------|-------|-------------------|
| `train_split_Depression_AVEC2017.csv` | AVEC2017 | 107 | ✅ Yes |
| `dev_split_Depression_AVEC2017.csv` | AVEC2017 | 35 | ✅ Yes |
| `test_split_Depression_AVEC2017.csv` | AVEC2017 | 47 | ❌ No |
| `full_test_split.csv` | AVEC2017 | 47 | ❌ Total only |
| `paper_splits/paper_split_train.csv` | Our impl | 58 | ✅ Yes |
| `paper_splits/paper_split_val.csv` | Our impl | 43 | ✅ Yes |
| `paper_splits/paper_split_test.csv` | Our impl | 41 | ✅ Yes |

---

## See Also

- [daic-woz-schema.md](./daic-woz-schema.md) - Full dataset schema
- [agent-sampling-registry.md](../reference/agent-sampling-registry.md) - Sampling parameters (paper leaves some unspecified)
- [reproduction-results.md](../results/reproduction-results.md) - Reproduction status
