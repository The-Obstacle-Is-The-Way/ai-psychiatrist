# Spec Documentation Gaps Analysis

**Date**: 2026-01-01
**Purpose**: Identify gaps between archived specs and canonical docs to enable archive deletion.

---

## Executive Summary

Audited archived specs against canonical documentation. Goal: make `docs/` self-sufficient so `docs/archive/` could be deleted without losing any *active* documentation (history/provenance may still be desirable).

| Spec | Status | Primary Gap | Can Delete? |
|------|--------|-------------|-------------|
| **25** | Covered | Metrics definitions + schema now live in `docs/reference/metrics-and-evaluation.md` | **YES** (if unreferenced) |
| **31** | Covered | Prompt format spec now lives in `docs/concepts/few-shot-prompt-format.md` | **YES** (if unreferenced) |
| **32** | Covered | Diagnostics workflow now lives in `docs/guides/debugging-retrieval-quality.md` | **YES** (if unreferenced) |
| **33** | Covered | Guardrails documented in `docs/reference/features.md` + retrieval debugging guide | **YES** (if unreferenced) |
| **34** | Covered | Tagging workflow + schema now lives in `docs/guides/item-tagging-setup.md` | **YES** (if unreferenced) |
| **35** | Covered | Chunk scoring setup + schema now lives in `docs/reference/chunk-scoring.md` | **YES** (if unreferenced) |
| **36** | Covered | CRAG guide now lives in `docs/guides/crag-validation-guide.md` | **YES** (if unreferenced) |
| **37** | Covered | Batch query embedding documented in `docs/reference/features.md` | **YES** (if unreferenced) |
| **38** | Covered | Fail-fast semantics documented in `docs/concepts/error-handling.md` | **YES** (if unreferenced) |
| **39** | Covered | Exception taxonomy documented in `docs/reference/exceptions.md` | **YES** (if unreferenced) |
| **40** | Covered | Embedding generation guide now lives in `docs/guides/embedding-generation.md` | **YES** (if unreferenced) |

**Remaining work**: remove *all* links/references from active docs to `docs/archive/**`, and ensure each feature is documented in the appropriate Diátaxis category (concept / guide / reference) under `docs/`.

---

## Migration Map (Specs → Canonical Docs)

The following table documents where the *active* (non-archive) SSOT now lives.

| Spec | Canonical Docs | Notes |
|------|----------------|-------|
| 25 (AURC/AUGRC) | `docs/reference/metrics-and-evaluation.md` | Definitions + schema; see also `docs/reference/statistical-methodology-aurc-augrc.md` |
| 31 (Reference Examples format) | `docs/concepts/few-shot-prompt-format.md` | Current code uses `</Reference Examples>` (Spec 33 XML fix) |
| 32 (Retrieval audit) | `docs/guides/debugging-retrieval-quality.md` | How to interpret logs and diagnose retrieval |
| 33 (Guardrails) | `docs/reference/features.md` | Threshold + budget behavior; see also retrieval debugging guide |
| 34 (Item tags) | `docs/guides/item-tagging-setup.md` | Schema + fail-fast semantics (Spec 38) |
| 35 (Chunk scoring) | `docs/reference/chunk-scoring.md` | Schema + workflow; scorer is configurable |
| 36 (CRAG validation) | `docs/guides/crag-validation-guide.md` | Fail-fast semantics (Spec 38) |
| 37 (Batch query embedding) | `docs/reference/features.md` | Performance/stability; default ON in code |
| 38 (Conditional feature loading) | `docs/concepts/error-handling.md` | “skip-if-disabled, crash-if-broken” |
| 39 (Preserve exception types) | `docs/concepts/error-handling.md`, `docs/reference/exceptions.md` | Do not wrap exceptions; preserve type |
| 40 (Fail-fast embedding gen) | `docs/guides/embedding-generation.md` | Strict by default; partial is debug-only |

---

## Remaining Work (Archive-Deletion Readiness)

1. **Link hygiene**: remove all links from active docs to `docs/archive/**`.
2. **Index completeness**: ensure `docs/index.md` and `docs/reference/features.md` are sufficient entry points.
3. **MkDocs strict**: keep `uv run mkdocs build --strict` green on case-sensitive filesystems (CI).

### Verification Commands

```bash
# Build docs with link validation
uv run mkdocs build --strict

# Ensure active docs do not depend on archive pages
rg "archive/specs|archive/bugs|\\.{2}/archive/" docs --glob '!docs/archive/**'
```

---

## Status (2026-01-01)

The canonical docs listed in this file exist under `docs/` and active documentation no longer depends on `docs/archive/**`.

Next steps:
- Keep root-level helper docs (e.g., `FEATURES.md`, `CONFIGURATION-PHILOSOPHY.md`) consistent with `docs/reference/features.md` and `docs/reference/configuration-philosophy.md`.
- Re-run the verification commands before merges.
