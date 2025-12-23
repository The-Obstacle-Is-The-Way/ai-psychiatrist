# BUG-018: Reproduction Friction Log

**Date**: 2025-12-22
**Status**: INVESTIGATED - paper-parity workflow runs end-to-end; MAE parity not yet achieved
**Severity**: HIGH - initially blocked paper-parity evaluation
**Updated**: 2025-12-23 - paper-style split + embeddings + evaluation executed; metrics still diverge (see results)

This document captures ALL friction points encountered when attempting to reproduce the paper's PHQ-8 assessment results.

Note: `.env` is gitignored; any `.env` references below describe **local developer configuration**
changes made during reproduction attempts.

---

## ⚠️ CRITICAL (Fixed): MAE Methodology Was Fundamentally Wrong

### The Core Problem

We were computing a **completely different metric** than the paper:

| Our Implementation | Paper's Implementation |
|-------------------|----------------------|
| Total-score MAE (0-24 scale) | Item-level MAE (0-3 scale) |
| `\|predicted_total - gt_total\|` | `mean(\|pred_item - gt_item\|)` per item |
| N/A items = 0 in sum | N/A items excluded from MAE |
| Reported MAE ~4.02 | Paper reports MAE ~0.619 |

### Timeline of Events (Historical)

1. **04:01 AM Dec 22**: Ran reproduction with OLD script → MAE 4.02 (WRONG)
2. **Later Dec 22**: Rewrote `scripts/reproduce_results.py` to match paper methodology
3. **2025-12-23**: Paper-parity workflow executed end-to-end; see `docs/results/reproduction-notes.md` (MAE still above paper)

### The OLD Code (Wrong)

```python
# scripts/reproduce_results.py (BEFORE fix)
predicted = assessment.total_score  # Sum of 8 items (0-24)
absolute_error = abs(predicted - ground_truth)  # Total vs total
```

### The NEW Code (Correct; used in paper-parity run)

```python
# scripts/reproduce_results.py (AFTER fix, in main)
predicted_items = {item: assessment.items[item].score for item in PHQ8Item.all_items()}
errors: list[int] = []
for item in PHQ8Item.all_items():
    pred = predicted_items[item]
    if pred is None:  # Skip N/A - paper excludes these!
        continue
    errors.append(abs(pred - ground_truth_items[item]))  # Per-item comparison
mae_available = float(np.mean(errors))  # Average of per-item errors
```

### Why "÷8 ≈ 0.50" Was Wrong

The handwave `4.02 ÷ 8 ≈ 0.50` is **not** equivalent to the paper's methodology because:
1. N/A handling differs (we sum, paper excludes)
2. Coverage differs (paper reports % of predictions made)
3. Aggregation differs (paper has multiple views: weighted, by-item, by-subject)

### Action (Completed: methodology parity; results still diverge)

✅ Paper-parity reproduction was re-run end-to-end. Commands executed:
```bash
# Paper-parity workflow (paper-style split + paper embeddings + evaluation on paper test)
uv run python scripts/create_paper_split.py --seed 42
uv run python scripts/generate_embeddings.py --split paper-train
uv run python scripts/reproduce_results.py --split paper --few-shot-only

# Quick sanity check (AVEC dev split; has per-item labels)
uv run python scripts/reproduce_results.py --split dev
```

The output file `data/outputs/reproduction_results_20251222_040100.json` is **INVALID** and should be ignored.
See `docs/results/reproduction-notes.md` for the current example run metrics and divergence hypotheses.

---

## Fix Summary

| Sub-bug | Issue | Status |
|---------|-------|--------|
| BUG-018a | Misconfigured quantitative model default (MedGemma) | ✅ FIXED - paper-parity defaults use gemma3:27b |
| BUG-018b | `.env` overrides code defaults | ✅ DOCUMENTED - expected Pydantic behavior; templates updated |
| BUG-018c | DataSettings attribute | ✅ FIXED - script updated |
| BUG-018d | Inline imports | ✅ FIXED - imports at top |
| BUG-018e | Timeouts on long transcripts | ✅ INVESTIGATED - configurable via `OLLAMA_TIMEOUT_SECONDS` |
| BUG-018f | Evidence-extraction JSON malformation | ⚠️ MITIGATED - tolerant fixups + keyword backfill; still observed intermittently |
| BUG-018g | Model underestimates severe | ⚠️ RESEARCH ITEM |
| BUG-018h | Orphaned `data/keywords/` directory | ✅ DOCUMENTED - historical/local-only; not used by code |
| BUG-018i | Item-level MAE methodology | ✅ FIXED + RE-RUN (paper-style workflow executed) |

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

### Evidence (Historical; community Ollama conversion)

Participant 308 (ground truth = 22, severe depression):
- Transcript: "yeah it's pretty depressing", "i can't find a fucking job", sleep problems
- MedGemma: predicted 0, na_count = 8
- gemma3:27b: predicted 10, na_count = 4

### Files Changed

1. `src/ai_psychiatrist/config.py` (`ModelSettings.quantitative_model`)
   - OLD: `default="alibayram/medgemma:27b"` (historical)
   - NEW: `default="gemma3:27b"`

2. `tests/unit/test_config.py` (`TestModelSettings.test_paper_optimal_defaults`)
   - Updated test assertion and added docstring explaining why

3. `.env.example`
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

After fixing the default in `src/ai_psychiatrist/config.py`, the script STILL used MedGemma.

### Root Cause

User's `.env` file contained:
```
MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b
```

Pydantic settings load `.env` which OVERRIDES code defaults.

### Files Changed

1. `.env` (local, gitignored)
   - OLD: `MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b`
   - NEW: `MODEL_QUANTITATIVE_MODEL=gemma3:27b`

### Lesson

When changing defaults in `src/ai_psychiatrist/config.py`, MUST also:
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

## BUG-018e: Timeout Too Short for Long Transcripts (MEDIUM) - INVESTIGATED

### Symptom

6 out of 47 participants (13%) failed with:
```
LLM request timed out (configured timeout reached)
```

### Affected Participants

- 407 (ground truth 3) - **28K transcript**
- 421 (ground truth 10) - **16K transcript**
- 424 (ground truth 3) - **24K transcript**
- 450 (ground truth 9) - **24K transcript**
- 466 (ground truth 9) - **28K transcript**
- 481 (ground truth 7) - **24K transcript**

### Investigation Results (2025-12-22)

**Transcript size correlation confirmed:**
- Timed-out participants average ~24K transcript size
- Successful participants average ~16K transcript size
- Largest timeouts (407, 466) are 28K - double the successful average

**Contributing factors:**
1. 27B model on M1 Pro Max with concurrent workloads (Arc mesh training)
2. Two LLM calls per participant (evidence extraction + scoring)
3. Timeout applies per call (evidence extraction + scoring), so each call can timeout independently
4. No timeout issues correlated with ground truth severity

### Root Cause

Timeout is appropriate for most transcripts (87% success rate). Failures occur on:
1. Transcripts larger than ~20K
2. When GPU/CPU is shared with other workloads

### Recommendation

For users with concurrent GPU workloads or larger transcripts:
```bash
OLLAMA_TIMEOUT_SECONDS=300  # or 360 for very large transcripts
```

Timeout behavior is environmental and workload-dependent. The current default is 300s, but very
long transcripts may still require a higher value.

---

## BUG-018f: JSON Parsing Failures (MEDIUM) - MITIGATED

### Symptom

In some runs, the quantitative scoring step can fail to parse the LLM's JSON output, e.g.:
```
Failed to parse quantitative response after all attempts
```
Resulting in `na_count = 8`, `total_score = 0`

### Root Cause

LLM output is not guaranteed to be strict JSON. Common failure modes include:
- Markdown fences around JSON
- Smart quotes / non-ASCII punctuation
- Trailing commas
- Partial / truncated objects

### Current Mitigation (in production code)

The quantitative agent uses a multi-level repair cascade:
1. `_strip_json_block()` - Tag/code-fence stripping
2. `_tolerant_fixups()` - Syntax repair (smart quotes, trailing commas)
3. `json.loads(...)` - Parse attempt
4. `_llm_repair()` - LLM-based JSON repair (best-effort)
5. Fallback skeleton - Ensures an assessment object is still returned

### Conclusion

In the current paper-parity workflow, JSON parsing is generally reliable; we did not observe JSON
parse failures in the example run documented in `docs/results/reproduction-notes.md`.

However, because malformed JSON can reappear due to model variability and long generations, treat
this as **mitigated**, not permanently “resolved”.

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

## BUG-018h: Orphaned `data/keywords/` Directory (LOW) - DOCUMENTED

### Symptom

Some earlier local runs referenced an orphan `data/keywords/` directory. This directory may exist
locally (and is gitignored due to DAIC-WOZ licensing), but **no production code depends on it**.

### Investigation Results (2025-12-22)

**Code search for "keywords" found:**
- Actual keywords are in `src/ai_psychiatrist/resources/phq8_keywords.yaml`
- No code references `data/keywords/` anywhere
- Git history shows no commits ever added files to this directory

**The real keyword system:**
```python
# src/ai_psychiatrist/agents/prompts/quantitative.py (_KEYWORDS_RESOURCE_PATH)
_KEYWORDS_RESOURCE_PATH = "resources/phq8_keywords.yaml"
```

Keywords are bundled with the package as a resource file, NOT in `data/`.

### Root Cause

**Orphaned directory** - likely created manually during development but never used.
No code references it. The actual keywords are correctly located in `src/ai_psychiatrist/resources/`.

### Recommendation

This is a local cleanup item (the repo gitignores `data/`).

**If present in your local environment, safe to delete:**
```bash
rm -rf data/keywords/
```

This is a cleanup item, not a bug. The actual keyword-based backfill system works correctly.

---

## BUG-018i: Item-Level vs Total-Score MAE (CRITICAL METHODOLOGY ERROR)

### Symptom

Paper reports MAE ~0.619 (few-shot) / 0.796 (zero-shot), but our script showed MAE ~4.02.

### Root Cause (Deep Analysis 2025-12-22)

**We were computing a fundamentally different metric:**

| Aspect | OLD Script (Wrong) | Paper's Method | NEW Script (Correct) |
|--------|-------------------|----------------|---------------------|
| Scale | 0-24 (total score) | 0-3 (per item) | 0-3 (per item) |
| N/A handling | N/A = 0 in sum | N/A excluded | N/A excluded |
| Calculation | `\|Σpred - Σgt\|` | `mean(\|pred_i - gt_i\|)` | `mean(\|pred_i - gt_i\|)` |
| Ground truth | Total score only | Per-item scores | Per-item scores |
| Data split | Test (no item labels!) | Train/dev (has labels) | Train/dev |

### The "÷8" Handwave Was Invalid

Claiming `4.02 ÷ 8 ≈ 0.50` does NOT equal the paper's methodology because:
1. Division by 8 assumes all items predicted - but some were N/A
2. Paper EXCLUDES N/A from both numerator AND denominator
3. Paper reports coverage (% of items with predictions) separately
4. Total-score errors don't distribute evenly across items

### Files Changed (Paper-Parity Fix)

Complete rewrite of `scripts/reproduce_results.py`:
- Changed from total-score to item-level MAE
- Added per-item ground truth loading from AVEC2017 CSVs
- Changed from test split (no item labels) to train/dev splits (has item labels)
- Added coverage metrics and multiple MAE aggregation views
- Added N/A exclusion matching paper methodology

### Current Status

The paper-parity evaluation workflow exists (item-level MAE + paper-style split support), but a full
run reproducing the paper’s reported MAE values has not been completed yet.

The file `data/outputs/reproduction_results_20251222_040100.json` contains results from the OLD (wrong) methodology and should be ignored.

### Action Required

Run paper-parity evaluation:
```bash
uv run python scripts/create_paper_split.py --seed 42
uv run python scripts/generate_embeddings.py --split paper-train
uv run python scripts/reproduce_results.py --split paper --few-shot-only
```

---

## Summary of ALL Changes Made

### Config Changes

| File | Line | Old | New |
|------|------|-----|-----|
| `src/ai_psychiatrist/config.py` | 62-66 | MedGemma docstring | Updated note about N/A issue |
| `src/ai_psychiatrist/config.py` | 86-91 | `alibayram/medgemma:27b` | `gemma3:27b` |
| `tests/unit/test_config.py` | 84-96 | MedGemma assertion | gemma3:27b with docstring |
| `.env.example` | 9-18 | MedGemma default | gemma3:27b with note |
| `.env` | (local) | MedGemma override | gemma3:27b (recommended for parity) |

### New Files Created

| File | Purpose |
|------|---------|
| `scripts/reproduce_results.py` | Batch evaluation script |
| `docs/results/reproduction-notes.md` | Results documentation |
| `docs/bugs/bug-018-reproduction-friction.md` | This file |
| `data/outputs/reproduction_results_*.json` | Raw results |

---

## Critical Questions - ANSWERED

### 1. Why was MedGemma the default? Who set this and did they test it?

**Answer:** Appendix F shows MedGemma as an ALTERNATIVE evaluation, not the primary model. The caveat
"fewer predictions overall" is easy to miss. Also, Ollama does not publish an official MedGemma
library model; any MedGemma tag in Ollama is a community conversion.

**Fixed:** Default changed to `gemma3:27b` to match Section 2.2 paper baseline.

### 2. Is the .env wiring pattern correct?

**Answer:** Yes, but documentation was missing. Pydantic correctly prioritizes: env vars > .env > code defaults. This is expected behavior but requires updating .env when code defaults change.

**Fixed:** Both `.env` and `.env.example` now have consistent gemma3:27b default.

### 3. Why does MedGemma produce all N/A?

**Answer:** The paper acknowledges this behavior in Appendix F: MedGemma produces fewer predictions
overall (more N/A / lower coverage). Additionally, different community conversions/quantizations may
vary in behavior. This needs controlled evaluation using the paper’s MAE + coverage definitions.

**Resolution:** Use `gemma3:27b` for paper-parity baseline; treat MedGemma as an optional alternative
for the quantitative agent only.

### 4. Should timeout be configurable per-model?

**Answer:** Current per-request timeout (via `timeout_seconds` parameter) is configurable. Users
with large transcripts or concurrent GPU workloads can increase via `OLLAMA_TIMEOUT_SECONDS`.

**No change needed.** Environmental issue, not a code deficiency.

### 5. Is the keyword backfill working?

**Answer:** Yes! Keyword backfill runs AFTER evidence extraction, not affected by JSON parsing. The cascade is:
1. LLM extracts evidence
2. Parse JSON (with repair cascade)
3. Keyword backfill enriches missing items

**Verified working** - no code changes needed.

### 6. What is data/keywords/ for?

**Answer:** It's a historical/local-only orphan directory (not present in this repo tree). Real
keywords are in `src/ai_psychiatrist/resources/phq8_keywords.yaml`.

**Resolution:** Safe to delete. Added to cleanup list.

### 7. Should we calculate item-level MAE?

**Answer:** YES - and the current `scripts/reproduce_results.py` computes:
- Load per-item ground truth from AVEC2017 CSVs
- Calculate item-level MAE with N/A exclusion
- Report multiple aggregation views (weighted, by-item, by-subject)
- Track prediction coverage

**Code is correct. Reproduction NOT re-run.** The old output file uses wrong methodology.

### 8. Why does model underestimate severe cases?

**Answer:** RESEARCH ITEM - requires investigation. Possible causes:
1. Severe symptoms may not be explicitly discussed in interviews
2. Model may be calibrated toward moderate predictions
3. Few-shot might help (needs testing)
4. Prompt may need adjustment for severe cases

**Status:** Open research question for future improvement.

---

## Cleanup Actions

- [x] Delete `data/keywords/` if present locally (historical orphan)
- [ ] Consider increasing default timeout for HPC environments
- [ ] Research severe depression underestimation (BUG-018g)

---

## NEXT STEPS (Required Before Any Further Work)

1. **Run paper-parity reproduction**:
   ```bash
   uv run python scripts/create_paper_split.py --seed 42
   uv run python scripts/generate_embeddings.py --split paper-train
   uv run python scripts/reproduce_results.py --split paper --few-shot-only
   ```

2. **Compare results** to paper's reported values:
   - Paper few-shot MAE: 0.619
   - Paper zero-shot MAE: 0.796
   - Paper MedGemma MAE: 0.505 (fewer predictions)

3. **Update documentation** with correct results once obtained

**DO NOT proceed with other work until reproduction is validated with correct methodology.**

---

## Reproduction Run Log (2025-12-23)

### Pre-flight Status

| Item | Status | Notes |
|------|--------|-------|
| Paper splits | ✅ Exist | 58/43/41 counts match paper; membership not published (see `docs/bugs/gap-001-paper-unspecified-parameters.md`) |
| AVEC embeddings | ✅ Exist | `reference_embeddings.npz` (107 participants) |
| Paper embeddings | ⚠️ Missing | Had to generate with `--split paper-train` |
| Transcripts | ✅ Exist | 189 participant folders (plus directory entries) |

### Embedding Generation (paper-train)

- **Duration**: ~65 minutes for 58 participants
- **Output**: `paper_reference_embeddings.npz` (101.44 MB), 6998 total chunks
- **Status**: ✅ Completed successfully

### Test Run (3 participants)

| Metric | Value | Note |
|--------|-------|------|
| MAE_item | 0.233 | Not statistically meaningful (n=3) |
| Coverage | 50% | 4/8 items N/A per participant |
| Time | 16 min | ~5 min per participant |

### Full Run (41 participants)

- **Start**: 2025-12-23 02:50 UTC
- **Estimated Duration**: ~3.5 hours
- **Status**: Completed (see `docs/results/reproduction-notes.md`)

### Friction Points Encountered

#### F-001: Paper Embeddings Required Separate Generation

**Issue**: Paper embeddings (`paper_reference_embeddings.npz`) did not exist, requiring a 65-minute generation step before reproduction could begin.

**Impact**: Adds significant time to first-time reproduction.

**Recommendation**: Document this requirement prominently; consider adding pre-generated embeddings to releases.

#### F-002: JSON Parsing Warning in Evidence Extraction

**Symptom**:
```text
Failed to parse evidence JSON, using empty evidence
```

**Location**: `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._extract_evidence`)

**Root Cause Analysis** (2025-12-23):

The LLM (e.g., `gemma3:27b`) can occasionally produce malformed JSON during evidence extraction
(unescaped quotes, truncated arrays/objects, or other formatting errors). When this happens, the
parser falls back to an empty evidence dict and relies on keyword backfill.

Example shape (illustrative):

```json
{
  "PHQ8_NoInterest": [],
  "PHQ8_Depressed": ["i had been having a lot of deaths around me...
```

The exact malformed pattern varies by model/backend and is surfaced in logs via the
`response_preview` field.

**Parsing Pipeline**:
1. `_strip_json_block()` - Handles markdown code fences ✅
2. `_tolerant_fixups()` - Handles smart quotes, trailing commas (best effort)
3. Falls through to empty evidence fallback on parse failure

**Impact**:
- Evidence extraction fails → empty evidence dict returned
- `_keyword_backfill()` then adds keyword-matched sentences
- Result: Only keyword-matched evidence (2 items in test case) instead of LLM-extracted evidence
- May contribute to N/A predictions for items without keyword matches

**Possible Fixes (for future consideration)**:
1. Add `""` → `"` fixup in `_tolerant_fixups()`
2. Add LLM repair for evidence extraction (currently only used for scoring response)
3. Prompt engineering to prevent the malformed output

**Status**: DOCUMENTED - keyword backfill provides partial mitigation; evidence JSON malformation still occurs intermittently.

#### F-003: High N/A Rate (50% in test run)

**Observation**: Test run showed 4/8 items as N/A per participant (50% coverage).

**Paper Context**: Paper Section 3.2 mentions excluding N/A from MAE calculation but doesn't report overall coverage percentage.

**Status**: Monitoring in full run - if consistent, may explain differences from paper results.

---

## Namespace/Artifact Registry Created

During this run, created `docs/data/artifact-namespace-registry.md` to document:
- Split naming conventions (AVEC vs paper)
- Embedding file naming
- Script input/output mapping
- Configuration parameters
