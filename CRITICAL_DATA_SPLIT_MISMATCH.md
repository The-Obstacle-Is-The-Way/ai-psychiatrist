# CRITICAL FINDING: Data Split Mismatch

**Date**: 2024-12-24
**Status**: CONFIRMED - NOT FIXED YET
**Severity**: HIGH - Affects reproducibility of paper results

---

## Summary

**Our current `data/paper_splits/` files contain WRONG participant IDs.**

The split **sizes** are correct (58/43/41), but the **participant assignments** are completely different from what the paper actually used.

---

## Evidence

### Ground Truth (from `docs/data/DATA_SPLIT_REGISTRY.md`)

This was reverse-engineered from the paper authors' actual output files in `_reference/analysis_output/`:

| Split | Count | Source |
|-------|-------|--------|
| TRAIN | 58 | Derived from `quan_gemma_zero_shot.jsonl` minus TEST minus VAL |
| VAL | 43 | `quan_gemma_few_shot/VAL_analysis_output/*.jsonl` |
| TEST | 41 | `quan_gemma_few_shot/TEST_analysis_output/*.jsonl` |

### Our Current Splits (from `data/paper_splits/paper_split_metadata.json`)

Generated algorithmically with seed=42, following the paper's **methodology** but not their **exact IDs**.

### Comparison Results

```
=== COMPARISON: Our splits vs Paper ground truth ===

TRAIN: Ours=58, Registry=58
  Match: False
  In ours but not registry: [302, 319, 322, 326, 370, 371, 372, 374, 380, 385, 393, 422, 423, 436, 443, 446, 451, 454, 455, 456, 457, 459, 468, 479, 484, 487, 489]
  In registry but not ours: [303, 310, 312, 315, 317, 318, 324, 327, 343, 352, 363, 391, 395, 402, 406, 412, 414, 415, 429, 437, 439, 444, 463, 464, 474, 475, 483]

VAL: Ours=43, Registry=43
  Match: False
  In ours but not registry: [310, 315, 316, 318, 330, 352, 362, 363, 377, 379, 389, 391, 395, 406, 412, 413, 414, 417, 427, 439, 447, 463, 464, 472, 483]
  In registry but not ours: [302, 307, 322, 325, 326, 341, 348, 351, 366, 371, 372, 374, 376, 380, 381, 382, 401, 419, 443, 446, 454, 457, 479, 482, 492]

TEST: Ours=41, Registry=41
  Match: False
  In ours but not registry: [303, 307, 312, 317, 324, 325, 327, 341, 343, 348, 351, 366, 376, 381, 382, 401, 402, 415, 419, 429, 437, 444, 474, 475, 482, 492]
  In registry but not ours: [316, 319, 330, 362, 370, 377, 379, 385, 389, 393, 413, 417, 422, 423, 427, 436, 447, 451, 455, 456, 459, 468, 472, 484, 487, 489]
```

**27 participants in TRAIN are wrong.**
**25 participants in VAL are wrong.**
**26 participants in TEST are wrong.**

---

## Impact

1. **Embeddings are wrong**: `generate_embeddings.py --split paper-train` uses the wrong 58 participants
2. **Few-shot retrieval is wrong**: The reference embeddings were generated from wrong participants
3. **Reproduction will not match paper**: Any evaluation on our TEST split evaluates different people
4. **Metrics will differ**: Even with identical code, results won't match the paper

---

## Root Cause

Our `scripts/create_paper_split.py` implements the paper's **stratification algorithm** (Appendix C) with seed=42, but:

1. The paper authors never disclosed their random seed
2. The paper authors never disclosed exact participant IDs
3. We had to reverse-engineer the truth from their output files

The script even has this warning in its docstring:
> "IMPORTANT: The paper does NOT provide exact participant IDs, so our splits will differ from the paper's."

---

## Files Affected

| File | Status |
|------|--------|
| `data/paper_splits/paper_split_train.csv` | WRONG IDs |
| `data/paper_splits/paper_split_val.csv` | WRONG IDs |
| `data/paper_splits/paper_split_test.csv` | WRONG IDs |
| `data/paper_splits/paper_split_metadata.json` | WRONG IDs |
| `data/embeddings/paper_reference_embeddings.npz` | Generated from WRONG training set |
| `data/embeddings/paper_reference_embeddings.json` | Generated from WRONG training set |

---

## Required Fix

1. Regenerate `data/paper_splits/*.csv` using exact IDs from `docs/data/DATA_SPLIT_REGISTRY.md`
2. Regenerate embeddings using the corrected training split
3. Update `scripts/create_paper_split.py` to use ground truth IDs (not algorithmic generation)

---

## Related Issues

- GitHub Issue #45: "Research: Paper uses custom 58/43/41 stratified split (not AVEC2017 splits)"

---

*DO NOT CLOSE THIS ISSUE UNTIL SPLITS ARE VERIFIED CORRECT*
