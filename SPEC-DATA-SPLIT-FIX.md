# SPEC: Data Split Correction and Artifact Regeneration

> **Status**: APPROVED FOR IMPLEMENTATION
> **Branch**: `fix/paper-data-splits`
> **Priority**: HIGH - Blocks paper reproduction parity
> **Related**: GH Issue #45, `CRITICAL_DATA_SPLIT_MISMATCH.md`
> **Date**: 2025-12-25

---

## Executive Summary

Our `data/paper_splits/` files and `data/embeddings/paper_reference_embeddings.*` were generated algorithmically with seed=42, producing participant IDs that **do not match** the paper's actual splits. The ground truth was reverse-engineered from the paper authors' output files and is documented in `docs/data/DATA_SPLIT_REGISTRY.md`.

**This spec defines the complete fix**: delete wrong artifacts, update scripts to use ground truth IDs, and regenerate all dependent artifacts.

---

## 1. Current State (WRONG)

### 1.1 Wrong Paper Splits

| File | Status | Issue |
|------|--------|-------|
| `data/paper_splits/paper_split_train.csv` | ❌ WRONG | 27 wrong participant IDs |
| `data/paper_splits/paper_split_val.csv` | ❌ WRONG | 25 wrong participant IDs |
| `data/paper_splits/paper_split_test.csv` | ❌ WRONG | 26 wrong participant IDs |
| `data/paper_splits/paper_split_metadata.json` | ❌ WRONG | Contains wrong IDs |

**Evidence**: Comparison in `CRITICAL_DATA_SPLIT_MISMATCH.md` shows ~50% participant mismatch per split.

### 1.2 Wrong Embeddings

| File | Status | Issue |
|------|--------|-------|
| `data/embeddings/paper_reference_embeddings.npz` | ❌ WRONG | Generated from wrong TRAIN (58 wrong IDs) |
| `data/embeddings/paper_reference_embeddings.json` | ❌ WRONG | Text chunks from wrong participants |

**Impact**: Few-shot retrieval uses embeddings from wrong participants → different similarity matches → different predictions → cannot reproduce paper results.

### 1.3 Current Script Behavior

`scripts/create_paper_split.py` implements the paper's **stratification algorithm** (Appendix C) but:
- Uses `random.Random(seed=42)` to assign participants
- The paper authors never disclosed their seed
- Result: Correct sizes (58/43/41) but wrong participant assignments

---

## 2. Ground Truth (CORRECT)

### 2.1 Source of Truth

**File**: `docs/data/DATA_SPLIT_REGISTRY.md`

**Provenance**: Reverse-engineered from paper authors' actual output files in `_reference/analysis_output/`:
- TRAIN: Derived from `quan_gemma_zero_shot.jsonl` minus TEST minus VAL
- VAL: From `quan_gemma_few_shot/VAL_analysis_output/*.jsonl`
- TEST: From `quan_gemma_few_shot/TEST_analysis_output/*.jsonl`

### 2.2 Correct Participant IDs

```python
# GROUND TRUTH - From docs/data/DATA_SPLIT_REGISTRY.md
TRAIN_IDS = [
    303, 304, 305, 310, 312, 313, 315, 317, 318, 321, 324, 327, 335, 338, 340, 343,
    344, 346, 347, 350, 352, 356, 363, 368, 369, 388, 391, 395, 397, 400, 402, 404,
    406, 412, 414, 415, 416, 418, 426, 429, 433, 434, 437, 439, 444, 458, 463, 464,
    473, 474, 475, 476, 477, 478, 483, 486, 488, 491
]  # 58 participants

VAL_IDS = [
    302, 307, 320, 322, 325, 326, 328, 331, 333, 336, 341, 348, 351, 353, 355, 358,
    360, 364, 366, 371, 372, 374, 376, 380, 381, 382, 392, 401, 403, 419, 420, 425,
    440, 443, 446, 448, 454, 457, 471, 479, 482, 490, 492
]  # 43 participants

TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389,
    390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451,
    455, 456, 459, 468, 472, 484, 485, 487, 489
]  # 41 participants
```

### 2.3 Verification

| Check | Expected | Source |
|-------|----------|--------|
| TRAIN + VAL + TEST | 142 | Paper Section 2.4.1 |
| TRAIN ∩ VAL | 0 | No overlap |
| TRAIN ∩ TEST | 0 | No overlap |
| VAL ∩ TEST | 0 | No overlap |
| All IDs have PHQ-8 labels | Yes | Combined AVEC train+dev |

---

## 3. Files to DELETE

### 3.1 Paper Splits (Replace with correct IDs)

```bash
# DELETE these files (will be regenerated)
rm data/paper_splits/paper_split_train.csv
rm data/paper_splits/paper_split_val.csv
rm data/paper_splits/paper_split_test.csv
rm data/paper_splits/paper_split_metadata.json
```

### 3.2 Embeddings (Regenerate from correct TRAIN)

```bash
# DELETE these files (will be regenerated)
rm data/embeddings/paper_reference_embeddings.npz
rm data/embeddings/paper_reference_embeddings.json
```

**Note**: There is no `.meta.json` for the legacy embeddings (they predate the metadata system).

---

## 4. Scripts to MODIFY

### 4.1 `scripts/create_paper_split.py`

**Current behavior**: Algorithmic stratification with seed=42
**New behavior**: Use ground truth IDs from `DATA_SPLIT_REGISTRY.md`

#### Design Decision: Hardcode vs External File

| Option | Pros | Cons |
|--------|------|------|
| **Hardcode IDs in script** | Single source of truth in code, no external dependency | IDs duplicated (also in DATA_SPLIT_REGISTRY.md) |
| **Read from DATA_SPLIT_REGISTRY.md** | Single source of truth, DRY | Fragile parsing of markdown |
| **New JSON file for IDs** | Clean separation, easy to validate | Yet another file |

**Decision**: **Hardcode IDs in script**. The IDs are static and will never change. The DATA_SPLIT_REGISTRY.md serves as documentation; the script is the implementation.

#### Changes Required

1. Add constant lists: `_GROUND_TRUTH_TRAIN_IDS`, `_GROUND_TRUTH_VAL_IDS`, `_GROUND_TRUTH_TEST_IDS`
2. Replace `stratified_split()` function with `load_ground_truth_split()` that returns the hardcoded IDs
3. Remove algorithmic stratification code (keep for historical reference in comments)
4. Update `--seed` argument to be removed (no longer applicable)
5. Add `--verify` flag to validate IDs against AVEC data (all exist, no overlap)
6. Update docstrings and help text

### 4.2 `scripts/generate_embeddings.py`

**No changes required** - This script reads from `paper_split_train.csv` which will be regenerated with correct IDs.

### 4.3 `scripts/reproduce_results.py`

**No changes required** - This script reads from `paper_split_*.csv` files which will be regenerated with correct IDs.

---

## 5. Namespace and Folder Conventions

### 5.1 Current Naming (Keep)

The current naming convention is sound:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `paper_` prefix | `paper_reference_embeddings.npz` | Paper-style artifacts |
| `{backend}_{model}_{split}` | `huggingface_qwen3_8b_paper_train.npz` | New generator output |

### 5.2 Folder Structure (Keep)

```
data/
├── paper_splits/                    # Paper-style custom splits
│   ├── paper_split_train.csv        # 58 participants (CORRECT IDs)
│   ├── paper_split_val.csv          # 43 participants (CORRECT IDs)
│   ├── paper_split_test.csv         # 41 participants (CORRECT IDs)
│   └── paper_split_metadata.json    # Provenance (ground truth source)
├── embeddings/
│   ├── paper_reference_embeddings.npz   # Legacy name (for backward compat)
│   └── paper_reference_embeddings.json
├── train_split_Depression_AVEC2017.csv  # Original AVEC (107)
├── dev_split_Depression_AVEC2017.csv    # Original AVEC (35)
└── transcripts/                         # Per-participant transcripts
```

### 5.3 Metadata Provenance

The new `paper_split_metadata.json` should indicate ground truth source:

```json
{
  "description": "Paper ground truth splits (reverse-engineered from output files)",
  "source": "docs/data/DATA_SPLIT_REGISTRY.md",
  "methodology": {
    "derivation": "Extracted participant IDs from paper authors' output files",
    "train_source": "quan_gemma_zero_shot.jsonl minus TEST minus VAL",
    "val_source": "quan_gemma_few_shot/VAL_analysis_output/*.jsonl",
    "test_source": "quan_gemma_few_shot/TEST_analysis_output/*.jsonl"
  },
  "actual_sizes": {
    "train": 58,
    "val": 43,
    "test": 41
  },
  "participant_ids": {
    "train": [...],
    "val": [...],
    "test": [...]
  }
}
```

---

## 6. Implementation Steps

### Phase 1: Script Update

1. **Create branch**: `git checkout -b fix/paper-data-splits`

2. **Update `scripts/create_paper_split.py`**:
   - Add ground truth ID constants
   - Replace algorithmic split with hardcoded IDs
   - Add `--verify` flag for validation
   - Update help text and docstrings

3. **Verify the script works**:
   ```bash
   python scripts/create_paper_split.py --dry-run --verify
   ```

### Phase 2: Delete Wrong Artifacts

4. **Delete wrong paper splits**:
   ```bash
   rm data/paper_splits/paper_split_*.csv
   rm data/paper_splits/paper_split_metadata.json
   ```

5. **Delete wrong embeddings**:
   ```bash
   rm data/embeddings/paper_reference_embeddings.npz
   rm data/embeddings/paper_reference_embeddings.json
   ```

### Phase 3: Regenerate Correct Artifacts

6. **Generate correct splits**:
   ```bash
   python scripts/create_paper_split.py
   ```

7. **Verify split correctness**:
   ```bash
   # Should show 58/43/41 with correct IDs
   python -c "import pandas as pd; print(pd.read_csv('data/paper_splits/paper_split_train.csv')['Participant_ID'].tolist())"
   ```

8. **Generate embeddings** (requires Ollama or HuggingFace):
   ```bash
   # Option A: Using HuggingFace (higher precision)
   EMBEDDING_BACKEND=huggingface python scripts/generate_embeddings.py \
     --split paper-train \
     --output data/embeddings/paper_reference_embeddings.npz

   # Option B: Using Ollama (if HuggingFace unavailable)
   EMBEDDING_BACKEND=ollama python scripts/generate_embeddings.py \
     --split paper-train \
     --output data/embeddings/paper_reference_embeddings.npz
   ```

### Phase 4: Verification

9. **Verify embeddings loaded correctly**:
   ```bash
   python -c "
   import numpy as np
   data = np.load('data/embeddings/paper_reference_embeddings.npz', allow_pickle=False)
   print(f'Participants: {len([k for k in data.keys() if k.startswith(\"emb_\")])}')
   "
   ```
   Expected output: `Participants: 58`

10. **Run tests**:
    ```bash
    make test-unit
    ```

11. **Optional: Run reproduction** (requires Ollama running):
    ```bash
    python scripts/reproduce_results.py --split paper --limit 5
    ```

### Phase 5: Commit and PR

12. **Commit changes**:
    ```bash
    git add scripts/create_paper_split.py
    git add data/paper_splits/
    # Note: data/embeddings/ is typically gitignored (large files)
    git commit -m "fix: use ground truth paper split IDs (GH-45)"
    ```

13. **Create PR**:
    ```bash
    gh pr create --title "fix: correct paper split participant IDs" \
      --body "Fixes #45. Uses ground truth IDs from DATA_SPLIT_REGISTRY.md instead of algorithmic generation."
    ```

---

## 7. Post-Fix Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| TRAIN count | `wc -l data/paper_splits/paper_split_train.csv` | 59 (58 + header) |
| VAL count | `wc -l data/paper_splits/paper_split_val.csv` | 44 (43 + header) |
| TEST count | `wc -l data/paper_splits/paper_split_test.csv` | 42 (41 + header) |
| First TRAIN ID | `head -2 data/paper_splits/paper_split_train.csv` | 303 |
| First VAL ID | `head -2 data/paper_splits/paper_split_val.csv` | 302 |
| First TEST ID | `head -2 data/paper_splits/paper_split_test.csv` | 316 |
| Embedding participants | `python -c "..."` (see above) | 58 |
| Tests pass | `make test-unit` | All green |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong IDs in DATA_SPLIT_REGISTRY.md | Double-check extraction logic; IDs verified against 7+ output files |
| Embeddings not gitignored | Verify `.gitignore` excludes `data/embeddings/*.npz` |
| Breaking existing workflows | Legacy `paper_reference_embeddings.*` filename preserved |
| Circular dependency (script needs data, data needs script) | AVEC CSVs are independent; script only filters them |

---

## 9. Files Modified Summary

| File | Action | Reason |
|------|--------|--------|
| `scripts/create_paper_split.py` | MODIFY | Use ground truth IDs |
| `data/paper_splits/paper_split_train.csv` | DELETE + REGENERATE | Wrong IDs |
| `data/paper_splits/paper_split_val.csv` | DELETE + REGENERATE | Wrong IDs |
| `data/paper_splits/paper_split_test.csv` | DELETE + REGENERATE | Wrong IDs |
| `data/paper_splits/paper_split_metadata.json` | DELETE + REGENERATE | Wrong IDs |
| `data/embeddings/paper_reference_embeddings.npz` | DELETE + REGENERATE | Generated from wrong TRAIN |
| `data/embeddings/paper_reference_embeddings.json` | DELETE + REGENERATE | Generated from wrong TRAIN |

---

## 10. Acceptance Criteria

- [ ] `scripts/create_paper_split.py` uses hardcoded ground truth IDs
- [ ] `data/paper_splits/paper_split_train.csv` contains exactly 58 correct participant IDs
- [ ] `data/paper_splits/paper_split_val.csv` contains exactly 43 correct participant IDs
- [ ] `data/paper_splits/paper_split_test.csv` contains exactly 41 correct participant IDs
- [ ] All IDs match those in `docs/data/DATA_SPLIT_REGISTRY.md`
- [ ] Embeddings regenerated from correct TRAIN split (58 participants)
- [ ] `CRITICAL_DATA_SPLIT_MISMATCH.md` updated to status FIXED
- [ ] All unit tests pass
- [ ] GitHub Issue #45 closed

---

## 11. Related Documentation

| Document | Purpose |
|----------|---------|
| `docs/data/DATA_SPLIT_REGISTRY.md` | Ground truth split IDs |
| `CRITICAL_DATA_SPLIT_MISMATCH.md` | Problem discovery and evidence |
| `docs/data/artifact-namespace-registry.md` | Naming conventions |
| `docs/data/daic-woz-schema.md` | Dataset schema reference |

---

*Spec Author: Claude Code*
*Date: 2025-12-25*
*Approved: Pending user review*
