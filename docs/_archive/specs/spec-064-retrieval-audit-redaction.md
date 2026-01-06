# Spec 064: Retrieval Audit Redaction (No Transcript Text in Logs)

**Status**: IMPLEMENTED
**Created**: 2026-01-06
**Implemented**: 2026-01-06
**Priority**: P1 (privacy/compliance + shareable artifacts)

---

## Problem

When retrieval audit logging is enabled (`EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true`), the pipeline
currently logs a `chunk_preview` field derived from the reference chunk text. If those references
come from DAIC-WOZ, this can leak restricted transcript content into logs and run artifacts.

This is an observability feature, but it must be **privacy-safe by construction**.

---

## Requirements

1. **No raw transcript text in retrieval audit logs**
   - Remove `chunk_preview` (or any equivalent preview) from the `retrieved_reference` log event.
   - Never emit any field containing raw chunk text.

2. **Keep audit usefulness via safe identifiers**
   - Log `chunk_hash` (stable short SHA-256 prefix of the chunk text).
   - Keep `chunk_chars` (length only).
   - Keep existing metadata: `participant_id`, `item`, `rank`, `similarity`, `reference_score`.

3. **Backwards compatibility**
   - The log event name (`retrieved_reference`) stays the same.
   - Downstream tooling/docs updated to reference the new fields.

4. **Deterministic and idempotent**
   - Hashing must be stable across runs and machines (same text → same hash).

---

## Implementation Plan (TDD)

### Step 1: Unit test (RED)

Update `tests/unit/services/test_embedding.py`:

- `TestEmbeddingService::test_build_reference_bundle_logs_audit_when_enabled`
- Assert `chunk_hash`/`chunk_chars` are present and `chunk_preview` is absent.
- Assert raw chunk text does not appear in structured log fields.

### Step 2: Code change (GREEN)

In `src/ai_psychiatrist/services/embedding.py`:

- Replace `chunk_preview=match.chunk.text[:160]` with:
  - `chunk_hash=stable_text_hash(match.chunk.text)`
  - Keep `chunk_chars=len(match.chunk.text)`

### Step 3: Doc updates

Update any non-archive docs that mention `chunk_preview` to match the new safe fields:

- `docs/rag/debugging.md`
- `docs/configs/configuration-philosophy.md` (if it enumerates audit fields)

### Step 4: Verification

- `make ci`
- `uv run mkdocs build --strict`

---

## Definition of Done

- Retrieval audit logs contain no raw chunk text.
- `chunk_hash` is present and stable (SHA-256 prefix via `stable_text_hash`).
- All tests pass; MkDocs strict build produces no new warnings in non-archive docs.
