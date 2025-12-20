# BUG-006: Missing Reference Embeddings Artifact

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/08_EMBEDDING_SERVICE.md`, `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

The reference embeddings file expected by the few-shot pipeline is missing:
`data/embeddings/participant_embedded_transcripts.pkl`. Without it, `ReferenceStore` loads an empty dataset and `EmbeddingService` returns no reference examples. This degrades **few-shot mode** into effectively **zero-shot**, blocking the Spec 09.5 requirement to validate retrieval-based prompting and paper MAE targets.

---

## Evidence

- `data/embeddings/` is empty.
- `ReferenceStore._load_embeddings()` logs "Embeddings file not found" and returns `{}`.
- `EmbeddingService` warns "No reference embeddings available" and returns no matches.

---

## Impact

- Few-shot retrieval is non-functional in real runs.
- Paper reproduction (MAE 0.619) cannot be verified.
- Spec 09.5 Gate 2 fails (embeddings not computed).

---

## Resolution Plan

1. Generate reference embeddings from **training** transcripts only (avoid data leakage).
2. Use paper-optimal hyperparameters: `chunk_size=8`, `step_size=2`, `Nexample=2`, `dimension=4096`.
3. Save as `data/embeddings/participant_embedded_transcripts.pkl` (gitignored).
4. Re-run checkpoint verification.

Suggested generator (legacy research script):
- `quantitative_assessment/embedding_batch_script.py`

---

## Verification

```bash
ls -la data/embeddings/participant_embedded_transcripts.pkl
python - <<'PY'
from ai_psychiatrist.config import DataSettings, EmbeddingSettings
from ai_psychiatrist.services.reference_store import ReferenceStore

data = DataSettings()
embed = EmbeddingSettings()
store = ReferenceStore(data, embed)
print(f"Participants: {store.participant_count}")
PY
```

---

## Notes

This is a data artifact gap, not a code bug. It must be resolved to complete Spec 09.5.
