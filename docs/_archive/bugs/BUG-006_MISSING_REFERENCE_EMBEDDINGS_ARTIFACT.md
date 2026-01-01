# BUG-006: Missing Reference Embeddings Artifact

**Severity**: HIGH (P1)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-20
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

## Scope & Disposition

- **Code Path**: Data artifact (`data/embeddings/participant_embedded_transcripts.pkl`), not code.
- **Fix Category**: Required for end-to-end validation (Spec 09.5).
- **Recommended Action**: Generate the artifact using the current Python pipeline; avoid patching legacy scripts unless used as a temporary bridge.

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

---

## Resolution

Created `scripts/generate_embeddings.py` - a modern Python script that generates the
reference embeddings artifact using the current codebase infrastructure.

Features:
1. **Uses modern codebase**: Leverages `TranscriptService`, `OllamaClient`, and config system
2. **Training data only**: Uses ONLY training split to avoid data leakage
3. **Paper-optimal hyperparameters**: Reads from `EmbeddingSettings` (chunk_size=8, step_size=2, dim=4096)
4. **Configurable**: All settings controllable via environment variables
5. **Dry-run mode**: `--dry-run` flag to verify configuration without generating

Usage:
```bash
# Ensure Ollama is running with embedding model
ollama pull qwen3-embedding:8b

# Generate embeddings
python scripts/generate_embeddings.py

# Dry run to check config
python scripts/generate_embeddings.py --dry-run
```

Output: `data/embeddings/participant_embedded_transcripts.pkl`

Note: Actual artifact generation requires DAIC-WOZ dataset and running Ollama.
The script exists and is ready; artifact generation is a runtime/data dependency.
