# BUG-015: Legacy Scripts Hardcode Absolute HPC Paths

**Severity**: MEDIUM (P2)  
**Status**: OPEN  
**Date Identified**: 2025-12-19  
**Spec Reference**: `docs/specs/04A_DATA_ORGANIZATION.md`, `docs/specs/12.5_FINAL_CLEANUP_LEGACY_REMOVAL.md`

---

## Executive Summary

Multiple legacy scripts embed absolute, machine-specific file paths (e.g., `/data/users4/...`, `/home/users...`). These scripts fail out-of-the-box on any other system and cannot be used to regenerate embeddings or reproduce results without manual edits. This blocks Spec 09.5 workflows when relying on legacy scripts for artifacts (e.g., reference embeddings).

---

## Evidence

Examples of hardcoded paths:

- `quantitative_assessment/embedding_batch_script.py:774-803`  
  - `OUTPUT_DIR = "/data/users2/user/ai-psychiatrist/analysis_output"`  
  - `pd.read_csv("/data/users4/user/ai-psychiatrist/datasets/daic_woz_dataset/...")`
- `quantitative_assessment/quantitative_analysis.py:20-21,255-267`
- `qualitative_assessment/qual_assessment.py:15-18,318-357`
- `qualitative_assessment/feedback_loop.py:45-118`
- `meta_review/meta_review.py:45`

---

## Impact

- Non-portable: scripts crash on local machines and CI.
- Embedding artifacts required by Spec 09.5 cannot be generated without manual path surgery.
- Encourages ad-hoc, one-off fixes rather than reproducible pipelines.

---

## Recommended Fix

1. Replace hardcoded paths with `DataSettings`-driven paths or CLI arguments.
2. If legacy scripts are no longer intended for use, archive or remove them per Spec 12.5.
3. Document a supported, modern embedding-generation path in `scripts/` that uses project settings.

---

## Files Involved

- `quantitative_assessment/embedding_batch_script.py`
- `quantitative_assessment/quantitative_analysis.py`
- `qualitative_assessment/qual_assessment.py`
- `qualitative_assessment/feedback_loop.py`
- `meta_review/meta_review.py`
