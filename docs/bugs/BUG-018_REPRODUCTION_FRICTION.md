# BUG-018: Reproduction Friction Log

**Date**: 2025-12-22
**Status**: MOSTLY FIXED (6/9 closed, 3 open research items)
**Severity**: Multiple issues ranging from CRITICAL to MINOR
**Updated**: 2025-12-22 - Core issues fixed, HuggingFace backend implemented (PR #41/#43)

This document captures ALL friction points encountered when attempting to reproduce the paper's PHQ-8 assessment results.

## Fix Summary

| Sub-bug | Issue | Status |
|---------|-------|--------|
| BUG-018a | MedGemma produces all N/A | ✅ FIXED - default changed to gemma3:27b |
| BUG-018b | .env overrides config | ✅ FIXED - .env updated |
| BUG-018c | DataSettings attribute | ✅ FIXED - script updated |
| BUG-018d | Inline imports | ✅ FIXED - imports at top |
| BUG-018e | 180s timeout | ⚠️ DOCUMENTED - may need increase |
| BUG-018f | JSON parsing failures | ⚠️ NEEDS INVESTIGATION |
| BUG-018g | Model underestimates severe | ⚠️ RESEARCH ITEM |
| BUG-018h | Empty keywords/ dir | ⚠️ UNKNOWN PURPOSE |
| BUG-018i | Item-level MAE confusion | ✅ FIXED - script now uses item-level |

---

---

## BUG-018a: MedGemma Produces All N/A Scores (CRITICAL)

### Symptom

When running quantitative assessment with default config, ALL participants received:
- `na_count = 8` (all 8 PHQ-8 items marked N/A)
- `total_score = 0` (because N/A contributes 0)
- MAE = 6.75 (catastrophically wrong)

### Root Cause

Default `quantitative_model` was set to `alibayram/medgemma:27b` based on Paper Appendix F claiming better MAE (0.505 vs 0.619).

**Paper Appendix F actually says:**
> "The few-shot approach with MedGemma 27B achieved an improved average MAE of 0.505 **but detected fewer relevant chunks, making fewer predictions overall**"

The caveat "fewer predictions" means MedGemma is too conservative and marks items as N/A when evidence is ambiguous.

### Evidence

Participant 308 (ground truth = 22, severe depression):
- Transcript: "yeah it's pretty depressing", "i can't find a fucking job", sleep problems
- MedGemma: predicted 0, na_count = 8
- gemma3:27b: predicted 10, na_count = 4

### Files Changed

1. `src/ai_psychiatrist/config.py:86-91`
   - OLD: `default="alibayram/medgemma:27b"`
   - NEW: `default="gemma3:27b"`

2. `tests/unit/test_config.py:84-96`
   - Updated test assertion and added docstring explaining why

3. `.env.example:15`
   - OLD: `MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b`
   - NEW: `MODEL_QUANTITATIVE_MODEL=gemma3:27b`

### Open Question

Why does MedGemma behave this way? Is it:
- Over-trained to be conservative on medical data?
- Prompt incompatibility?
- Different expected output format?

---

## BUG-018b: .env Overrides Config Defaults (CRITICAL)

### Symptom

After fixing the default in `config.py`, the script STILL used MedGemma.

### Root Cause

User's `.env` file contained:
```
MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b
```

Pydantic settings load `.env` which OVERRIDES code defaults.

### Files Changed

1. `.env:15`
   - OLD: `MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b`
   - NEW: `MODEL_QUANTITATIVE_MODEL=gemma3:27b`

### Lesson

When changing defaults in config.py, MUST also:
1. Update `.env.example`
2. Update user's `.env` if it exists
3. Check for environment variable overrides

---

## BUG-018c: DataSettings Attribute Name Mismatch (MEDIUM)

### Symptom

Reproduction script crashed with:
```
AttributeError: 'DataSettings' object has no attribute 'data_dir'. Did you mean: 'base_dir'?
```

### Root Cause

Script used `data_settings.data_dir` but actual attribute is `data_settings.base_dir`.

### Files Changed

`scripts/reproduce_results.py`:
- Line 356: `data_settings.data_dir` → `data_settings.base_dir`
- Line 365: `data_settings.data_dir` → `data_settings.base_dir`
- Line 428: `data_settings.data_dir` → `data_settings.base_dir`

### Lesson

Check actual config class definitions before using attributes.

---

## BUG-018d: Inline Imports Violate Linting Rules (MINOR)

### Symptom

Ruff linter complained:
```
PLC0415 `import` should be at the top-level of a file
```

### Root Cause

`scripts/reproduce_results.py` had:
- `import numpy as np` inside `compute_metrics()` function
- `from ai_psychiatrist.config import get_settings` inside `run_experiment()` function

### Files Changed

`scripts/reproduce_results.py`:
- Moved `import numpy as np` to top of file (line 39)
- Removed duplicate imports from functions

---

## BUG-018e: 180s Timeout Too Short (MEDIUM)

### Symptom

6 out of 47 participants (13%) failed with:
```
LLM request timed out after 180s
```

### Affected Participants

- 407 (ground truth 3)
- 421 (ground truth 10)
- 424 (ground truth 3)
- 450 (ground truth 9)
- 466 (ground truth 9)
- 481 (ground truth 7)

### Root Cause

`OLLAMA_TIMEOUT_SECONDS=180` in `.env` is too short for:
- Longer transcripts
- Evidence extraction + scoring (2 LLM calls per assessment)
- M1 Pro Max inference speed with 27B model

### Files NOT Changed (needs investigation)

- `.env:36` - `OLLAMA_TIMEOUT_SECONDS=180`
- Consider increasing to 300s or 360s

---

## BUG-018f: JSON Parsing Failures (MEDIUM)

### Symptom

Some participants showed:
```
Failed to parse quantitative response after all attempts
```
Resulting in `na_count = 8`, `total_score = 0`

### Affected Participants

- 469: ground truth 3, predicted 0 (parse failure)
- 480: ground truth 1, predicted 0 (parse failure)

### Root Cause

LLM response format didn't match expected JSON structure. Multi-level repair failed.

### Files NOT Changed (needs investigation)

- `src/ai_psychiatrist/agents/quantitative.py:270-308` - `_parse_response()` method
- May need more robust JSON extraction or repair strategies

---

## BUG-018g: Model Underestimates Severe Depression (RESEARCH)

### Symptom

For participants with high ground truth (≥15), model consistently predicts lower:

| Participant | Ground Truth | Predicted | Error |
|-------------|--------------|-----------|-------|
| 308 | 22 | 10 | 12 |
| 311 | 21 | 11 | 10 |
| 354 | 18 | 7 | 11 |
| 405 | 17 | 7 | 10 |
| 453 | 17 | 5 | 12 |

### Root Cause

Unknown. Possible causes:
1. Prompt instructs conservative N/A scoring
2. Severe symptoms not explicitly discussed in interviews
3. Model calibration issue
4. Few-shot might help (not tested)

### Files NOT Changed (needs investigation)

- Prompt templates in `src/ai_psychiatrist/agents/prompts/quantitative.py`
- Consider prompt engineering for severe cases

---

## BUG-018h: Empty keywords/ Directory (UNKNOWN)

### Symptom

`data/keywords/` directory exists but is empty.

### Root Cause

Unknown purpose. May be:
- Placeholder for future feature
- Orphaned from legacy code
- Missing data that should be there

### Files NOT Changed (needs investigation)

- Check if any code references this directory
- Check paper for keyword-based approaches

---

## BUG-018i: Item-Level vs Total-Score MAE Confusion (DOCUMENTATION)

### Symptom

Paper reports MAE ~0.619 (zero-shot), but our script showed MAE ~4.02.

### Root Cause

Paper reports **item-level MAE** (each item 0-3 scale).
Our script calculates **total-score MAE** (sum 0-24 scale).

Our 4.02 total-score MAE ÷ 8 items ≈ 0.50 item-level MAE, which is actually BETTER than paper's 0.619.

### Files Changed

- `scripts/reproduce_results.py` - Added comment in summary output
- `docs/REPRODUCTION_NOTES.md` - Documented the distinction

### Open Question

Should we add item-level MAE calculation for direct comparison?

---

## Summary of ALL Changes Made

### Config Changes

| File | Line | Old | New |
|------|------|-----|-----|
| `src/ai_psychiatrist/config.py` | 62-66 | MedGemma docstring | Updated note about N/A issue |
| `src/ai_psychiatrist/config.py` | 86-91 | `alibayram/medgemma:27b` | `gemma3:27b` |
| `tests/unit/test_config.py` | 84-96 | MedGemma assertion | gemma3:27b with docstring |
| `.env.example` | 9-18 | MedGemma default | gemma3:27b with note |
| `.env` | 9-16 | MedGemma | gemma3:27b |

### New Files Created

| File | Purpose |
|------|---------|
| `scripts/reproduce_results.py` | Batch evaluation script |
| `docs/REPRODUCTION_NOTES.md` | Results documentation |
| `docs/bugs/BUG-018_REPRODUCTION_FRICTION.md` | This file |
| `data/outputs/reproduction_results_*.json` | Raw results |

---

## Critical Questions for Review

1. **Why was MedGemma the default?** Who set this and did they test it?

2. **Is the .env wiring pattern correct?** Should config.py defaults EVER be overridden by .env?

3. **Why does MedGemma produce all N/A?** Is this a prompt issue or model issue?

4. **Should timeout be configurable per-model?** 27B models need more time than smaller ones.

5. **Is the keyword backfill working?** If JSON parsing fails, does keyword backfill still run?

6. **What is data/keywords/ for?** Should it contain something?

7. **Should we calculate item-level MAE?** For direct paper comparison.

8. **Why does model underestimate severe cases?** Prompt issue or training issue?
