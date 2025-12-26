# Paper Split Registry

## Authoritative Splits (Reconstructed from Output Files)

These splits were reconstructed by extracting participant IDs from the paper authors' output files in `_reference/analysis_output/`. This is the ground truth for reproducing the paper's results.

---

## TRAIN (58 participants)

**Source**: Derived from `_reference/analysis_output/quan_gemma_zero_shot.jsonl` minus TEST minus VAL

```text
303, 304, 305, 310, 312, 313, 315, 317, 318, 321, 324, 327, 335, 338, 340, 343,
344, 346, 347, 350, 352, 356, 363, 368, 369, 388, 391, 395, 397, 400, 402, 404,
406, 412, 414, 415, 416, 418, 426, 429, 433, 434, 437, 439, 444, 458, 463, 464,
473, 474, 475, 476, 477, 478, 483, 486, 488, 491
```

---

## VAL (43 participants)

**Source**: `_reference/analysis_output/quan_gemma_few_shot/VAL_analysis_output/*.jsonl`

```text
302, 307, 320, 322, 325, 326, 328, 331, 333, 336, 341, 348, 351, 353, 355, 358,
360, 364, 366, 371, 372, 374, 376, 380, 381, 382, 392, 401, 403, 419, 420, 425,
440, 443, 446, 448, 454, 457, 471, 479, 482, 490, 492
```

---

## TEST (41 participants)

**Source**: `_reference/analysis_output/quan_gemma_few_shot/TEST_analysis_output/*.jsonl`

```text
316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389,
390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451,
455, 456, 459, 468, 472, 484, 485, 487, 489
```

---

## Verification

| Check | Result |
|-------|--------|
| TRAIN + VAL + TEST | 58 + 43 + 41 = 142 ✓ |
| TRAIN ∩ VAL | 0 ✓ |
| TRAIN ∩ TEST | 0 ✓ |
| VAL ∩ TEST | 0 ✓ |
| TEST == metareview IDs | ✓ |
| TEST == medgemma IDs | ✓ |
| TEST == DIM_TEST IDs | ✓ |

---

## Output File Consistency

All output files use these same splits. Paths below are relative to `_reference/analysis_output/`:

| File | Split Used | Count | Consistent |
|------|------------|-------|------------|
| `quan_gemma_zero_shot.jsonl` | ALL | 142 | ✓ |
| `quan_gemma_few_shot/TEST_analysis_output/*.jsonl` | TEST | 41 | ✓ |
| `quan_gemma_few_shot/VAL_analysis_output/*.jsonl` | VAL | 43 | ✓ |
| `quan_gemma_few_shot/DIM_TEST_analysis_output/*.jsonl` | TEST | 41 | ✓ |
| `quan_medgemma_few_shot.jsonl` | TEST | 41 | ✓ |
| `quan_medgemma_zero_shot.jsonl` | TEST | 41 | ✓ |
| `metareview_gemma_few_shot.csv` | TEST | 41 | ✓ |
| `qual_gemma.csv` | ALL (142 unique IDs; one duplicated row) | 142 | ✓ |

---

## AVEC2017 Reference Splits (for comparison)

These are the original AVEC2017 competition splits, NOT what the paper used:

| Split | Count | Notes |
|-------|-------|-------|
| AVEC Train | 107 | `data/train_split_Depression_AVEC2017.csv` |
| AVEC Dev | 35 | `data/dev_split_Depression_AVEC2017.csv` |
| AVEC Test | 47 | `data/test_split_Depression_AVEC2017.csv` (no PHQ-8 labels) |

The paper combined AVEC Train + Dev (142 total) and re-split into 58/43/41.

---

## How to Use These Splits

To reproduce the paper's results, use these exact participant IDs:

```python
TRAIN_IDS = [303, 304, 305, 310, 312, 313, 315, 317, 318, 321, 324, 327, 335, 338, 340, 343, 344, 346, 347, 350, 352, 356, 363, 368, 369, 388, 391, 395, 397, 400, 402, 404, 406, 412, 414, 415, 416, 418, 426, 429, 433, 434, 437, 439, 444, 458, 463, 464, 473, 474, 475, 476, 477, 478, 483, 486, 488, 491]

VAL_IDS = [302, 307, 320, 322, 325, 326, 328, 331, 333, 336, 341, 348, 351, 353, 355, 358, 360, 364, 366, 371, 372, 374, 376, 380, 381, 382, 392, 401, 403, 419, 420, 425, 440, 443, 446, 448, 454, 457, 471, 479, 482, 490, 492]

TEST_IDS = [316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389, 390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451, 455, 456, 459, 468, 472, 484, 485, 487, 489]
```

---

*Last verified: 2025-12-25*
*Reconstructed from: `_reference/analysis_output/` (snapshot of paper authors' published outputs; upstream: `trendscenter/ai-psychiatrist`)*
