# BUG-004: Spec 09.5 Checkpoint Drift vs Paper & Repo

**Severity**: MEDIUM (P2)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

Spec 09.5 documented **incorrect paper hyperparameters** (chunk size 5, N examples 3, k=3) and referenced **non-existent commands/files** (CLI evaluate/assess, `test_embedding_service.py`, `reference_embeddings.npy`). The paper (Appendix D) and current implementation use **chunk size 8**, **N examples 2**, **dimension 4096**, and the repo uses a **pickle** reference store. These mismatches could send developers down a false path and block accurate checkpoint verification.

---

## Evidence (Pre-Fix)

- Spec 09.5 listed:
  - `N examples = 3`, `chunk size = 5`, `k=3` (paper Appendix D states `Nchunk=8`, `Nexample=2`).
  - `pytest tests/unit/services/test_embedding_service.py` (file does not exist).
  - `python -m ai_psychiatrist.cli evaluate/assess` (CLI not implemented yet).
  - `data/embeddings/reference_embeddings.npy` (repo uses `participant_embedded_transcripts.pkl`).

---

## Impact

- Misleads integration checkpoint execution and paper reproduction.
- Blocks verification steps by pointing to missing commands/files.
- Risks incorrect hyperparameter configuration.

---

## Scope & Disposition

- **Code Path**: Documentation only (`docs/specs`).
- **Fix Category**: Spec drift (no runtime impact).
- **Recommended Action**: Resolved; treat Spec 09.5 as SSOT and keep aligned going forward.

---

## Resolution

Updated Spec 09.5 to align with the paper and repo:

1. Corrected hyperparameters: `chunk size = 8`, `N examples = 2`, `k=2`.
2. Replaced invalid commands with real tests and pickle-based checks.
3. Annotated CLI/evaluation steps as future work (Spec 11+).

---

## Verification

```bash
rg -n "chunk size|N examples|k-neighbors" docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md
rg -n "test_embedding.py|participant_embedded_transcripts.pkl" docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md
```

---

## Files Changed

- `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`
