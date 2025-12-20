# BUG-013: Legacy Codebase Pollution

**Severity**: MEDIUM (P2)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/12.5_FINAL_CLEANUP_LEGACY_REMOVAL.md`, `docs/specs/00_OVERVIEW.md`

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
