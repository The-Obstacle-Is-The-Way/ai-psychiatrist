# BUG-037: Non-Archive Doc Link Drift (Broken References)

**Date**: 2026-01-07
**Status**: FIXED
**Severity**: P3 (Docs correctness; research workflow friction)
**Affects**: MkDocs navigation + internal research docs
**Discovered By**: Senior agent audit (MkDocs `--strict` link warnings)

---

## Executive Summary

Several non-archive docs referenced bug/audit documents using stale paths (e.g., `docs/_bugs/...`)
after those documents were moved/renamed into `docs/_archive/bugs/`. Some `_research/` docs also
used incorrect `docs/...`-prefixed relative links, producing broken-link warnings in MkDocs and
making cross-references unreliable.

---

## Root Causes

1. Bug/audit docs were archived and/or renamed (kebab-case → uppercase snake-case), but references
   were not updated.
2. `_research/` docs used repo-root style links (e.g., `docs/results/...`) which are incorrect when
   rendered from within `docs/`.

---

## Fix

Updated links to point to the correct MkDocs-resolvable targets:

- `docs/results/run-history.md` → BUG-035 link now points to `docs/_archive/bugs/BUG-035_FEW_SHOT_PROMPT_CONFOUND.md`
- `docs/results/few-shot-analysis.md` → BUG-035 link now points to the archived bug doc
- `docs/pipeline-internals/evidence-extraction.md` and `docs/_specs/index.md` → ANALYSIS-026 links
  now point to `docs/_archive/bugs/ANALYSIS-026_JSON_PARSING_ARCHITECTURE_AUDIT.md`
- `docs/_research/*.md` → fixed `docs/...` relative links to proper `../...` paths
- `data/outputs/RUN_LOG.md` → updated BUG-035 reference to the archived path

---

## Verification

- `uv run mkdocs build --strict` no longer reports broken links for these non-archive references
  (remaining INFO warnings are confined to intentionally frozen `_archive/` content).
