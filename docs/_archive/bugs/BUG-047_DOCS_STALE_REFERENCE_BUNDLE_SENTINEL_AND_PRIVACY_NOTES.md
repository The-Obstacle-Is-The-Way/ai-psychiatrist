# BUG-047: Docs Drift After BUG-035 / Spec 064 (Stale Sentinel + Privacy Notes)

**Date**: 2026-01-07
**Status**: FIXED
**Severity**: P3 (docs can mislead research/debugging)
**Affects**: `docs/rag/*`, `docs/_research/*`

---

## Summary

Several non-archive documentation pages still describe pre-fix behavior and/or outdated risk notes:

- They claim empty retrieval inserts:
  ```text
  <Reference Examples>
  No valid evidence found
  </Reference Examples>
  ```
  but BUG-035 changed empty bundles to emit `""` (no wrapper).
- They claim retrieval audit logging emits `chunk_preview` from reference text, but Spec 064 removed this (now `chunk_hash` + `chunk_chars`).

---

## Impact

- Researchers may misinterpret runs and chase the wrong failure mode (“why do I see the sentinel?”).
- Debugging docs can contradict the current SSOT implementation.
- Risk notes can be mis-aimed (pointing at already-fixed code paths while missing the true ones).

---

## Affected Locations (Non-Archive)

- `docs/rag/debugging.md` (Step 4)
- `docs/rag/runtime-features.md` (Reference bundle format)
- `docs/_research/hypotheses-for-improvement.md` (few-shot confound description; retrieval audit risk)

---

## Fix

- Update the above docs to:
  - Treat the “No valid evidence found” wrapper as **historical** (BUG-035), not current runtime behavior.
  - Describe current behavior: empty reference bundle → omitted entirely.
  - Update audit logging description to match current fields (`chunk_hash`, `chunk_chars`).

---

## Verification

- `uv run mkdocs build --strict`
