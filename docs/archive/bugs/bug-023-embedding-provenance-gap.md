# BUG-023: Embedding Provenance Gap

**Date**: 2025-12-23
**Status**: RESOLVED - Provenance recorded in outputs
**Severity**: HIGH (historical runs unverifiable)
**Last Updated**: 2025-12-24

---

## Summary

Historically, reproduction outputs did not record **which reference embeddings artifact** was used
(AVEC vs paper-train). This made it impossible to verify whether past runs used the correct
knowledge base and therefore whether the reported metrics were comparable.

This is now fixed:
- `scripts/reproduce_results.py` writes a `provenance` object into `data/outputs/reproduction_results_*.json`.
- The provenance includes `split`, `embeddings_path`, `embedding_*` hyperparameters, `quantitative_model`,
  and `participants_evaluated` (the list of evaluated participant IDs).

---

## What Was Wrong (Historical)

1. **Embeddings artifact ambiguity**
   - Paper reproduction requires `data/embeddings/paper_reference_embeddings.npz` (paper-train knowledge base).
   - Earlier workflows could accidentally use AVEC embeddings if configuration was not explicit.

2. **No persisted provenance**
   - Even when the script selected the correct embeddings path, it was not persisted to output JSON.
   - Without provenance, old output files cannot be audited after the fact.

---

## Current SSOT (Verified)

### Reproduction Script Behavior

- For paper splits (`--split paper*`), `scripts/reproduce_results.py` uses:
  - `data/embeddings/paper_reference_embeddings.npz` (enforced via `get_effective_embeddings_path(...)`).
- For non-paper splits, the script uses:
  - `DATA_EMBEDDINGS_PATH` (i.e., `DataSettings.embeddings_path`).

### Output Provenance

Example (local-only; `data/` is gitignored due to DAIC-WOZ licensing):
- `data/outputs/reproduction_results_20251224_003441.json` includes:

```json
{
  "provenance": {
    "split": "paper",
    "embeddings_path": "data/embeddings/paper_reference_embeddings.npz",
    "quantitative_model": "gemma3:27b",
    "embedding_model": "qwen3-embedding:8b",
    "embedding_dimension": 4096,
    "embedding_chunk_size": 8,
    "embedding_chunk_step": 2,
    "embedding_top_k": 2,
    "enable_keyword_backfill": false
  }
}
```

---

## Verification Checklist (For Any New Run)

1. Ensure the embeddings artifact exists:
   - `data/embeddings/<name>.npz` and `data/embeddings/<name>.json` sidecar.
2. Run reproduction.
3. Verify the output JSON contains:
   - `.provenance.split`
   - `.provenance.embeddings_path` matches the intended split
   - `.provenance.embedding_dimension/chunk_size/chunk_step/top_k` match your intended hyperparameters
4. Treat any older outputs without `.provenance` as **unverified**.

---

## Related

- `docs/guides/preflight-checklist-few-shot.md`
- `docs/guides/preflight-checklist-zero-shot.md`
- `docs/bugs/bug-018-reproduction-friction.md`
- `docs/bugs/investigation-026-reproduction-mae-divergence.md`
