# BUG-013: Legacy Codebase Pollution

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-20
**Spec Reference**: `docs/specs/12.5_FINAL_CLEANUP_LEGACY_REMOVAL.md`, `docs/specs/00_OVERVIEW.md`

---

## Resolution

All legacy directories have been archived to `_legacy/`:

- `agents/` → `_legacy/agents/`
- `meta_review/` → `_legacy/meta_review/`
- `qualitative_assessment/` → `_legacy/qualitative_assessment/`
- `quantitative_assessment/` → `_legacy/quantitative_assessment/`
- `slurm/` → `_legacy/slurm/`
- `assets/` → `_legacy/assets/`
- `visualization/` → `_legacy/visualization/`
- `analysis_output/` → `_legacy/analysis_output/`

**Import audit confirmed**: No active code imports from legacy directories.
**pyproject.toml updated**: Ruff exclude simplified to `_legacy/` directory.
**Tests verified**: 603 tests pass at 96.52% coverage.

---

## Executive Summary

The project root contains a legacy `agents/` directory and other deprecated files (`qualitative_assessment/`, `quantitative_assessment/`, `meta_review/`) that duplicate functionality found in `src/ai_psychiatrist/`. This "Split Brain" structure causes confusion (see BUG-012), complicates refactoring, and poses a risk of developers editing the wrong files.

---

## Evidence

- Root directory contains:
  - `agents/` (contains `qualitative_assessor_f.py`, etc.)
  - `quantitative_assessment/` (contains notebooks and scripts)
  - `qualitative_assessment/`
  - `meta_review/`
- The modern codebase is strictly within `src/ai_psychiatrist/`.
- `server.py` and potentially other scripts still reference these legacy paths.

---

## Impact

- **Developer Confusion**: Unclear which file is the source of truth.
- **Architectural Drift**: Legacy files don't follow the new Domain/Infrastructure separation.
- **Maintenance Burden**: Duplicated logic implies double maintenance or rot.

---

## Scope & Disposition

- **Code Path**: Legacy-only directories outside `src/`.
- **Fix Category**: Cleanup / source-of-truth clarity.
- **Recommended Action**: Fix via archiving/removal per Spec 12.5; if preserving functionality, port into `src/` or `scripts/` first (do not patch legacy code).

---

## Recommended Fix

1.  **Archive Legacy Code**: Move `agents/`, `quantitative_assessment/`, `qualitative_assessment/`, and `meta_review/` into a `_archive/` or `_legacy/` directory.
2.  **Audit Imports**: Grep for any imports from `agents.` or `quantitative_assessment.` and update them to `src.ai_psychiatrist.`.
3.  **Delete** the legacy directories once verified unused.

---

## Files Involved

- `agents/`
- `quantitative_assessment/`
- `qualitative_assessment/`
- `meta_review/`
