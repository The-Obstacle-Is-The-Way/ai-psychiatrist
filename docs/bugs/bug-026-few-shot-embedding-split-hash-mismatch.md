# BUG-026: Few-Shot Paper Run Fails Due to Embedding Split Hash Mismatch

> **STATUS: RESOLVED**
>
> **Discovered**: 2025-12-26
>
> **Resolved**: 2025-12-27
>
> **Severity**: Blocker (prevents paper reproduction in `few_shot` mode)
>
> **Affects**: `scripts/reproduce_results.py --split paper` (few-shot path only)
>
> **Root Cause**: Reference embeddings metadata (`split_csv_hash`) was stale relative to the current `data/paper_splits/paper_split_train.csv`.
>
> **Resolution**: Updated hash in `.meta.json` files after verifying participant IDs are identical (58/58 exact match). See [Issue #64](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/64) for design improvement to prevent recurrence.

---

## Summary

A recent paper reproduction run succeeded in **zero-shot** mode, but **failed for all 41 paper-test participants in few-shot mode** with:

```text
Embedding artifact validation failed:
  - split_csv_hash mismatch: artifact='e7ff0bbd11b6', current='789c5f289023' (split='paper-train')
Regenerate embeddings or update config to match.
```

Few-shot requires loading the reference embeddings artifact (paper-train), and `ReferenceStore` correctly fails validation when the `paper_split_train.csv` content hash differs from the one recorded in the embeddings `.meta.json`.

### Important Clarification: This Is Not an Embedding Backend Wiring Bug

This failure can look like “the wrong embeddings backend/artifact was used”, but the run provenance shows the opposite:

- The few-shot run was configured to use **HuggingFace embeddings** (`embedding_backend=huggingface`)
- The embeddings artifact path was the **HuggingFace paper-train artifact**:
  `data/embeddings/huggingface_qwen3_8b_paper_train.npz`

The failure happened *after* selecting the correct backend/artifact: validation rejected the artifact because its stored
`split_csv_hash` no longer matches the current `data/paper_splits/paper_split_train.csv`.

---

## Evidence (Observed Outputs)

- Few-shot log: `data/outputs/few_shot_paper_20251226_203417.log`
- Few-shot results: `data/outputs/few_shot_paper_backfill-off_20251226_210503.json`
  - `total_subjects=41`, `successful_subjects=0`, `failed_subjects=41`
  - Each failed record contains the same `split_csv_hash mismatch` error

Zero-shot log (for contrast): `data/outputs/zero_shot_paper_20251226_185746.log` (ran normally because zero-shot does not require reference embeddings).

---

## Root Cause (First Principles)

### What is being validated?

`src/ai_psychiatrist/services/reference_store.py` validates embedding artifact metadata and checks:

- backend/model/dimension/chunking config match
- **`split_csv_hash` matches the current split file content**

For `split="paper-train"`, validation hashes:

`data/paper_splits/paper_split_train.csv`

### Why did it fail now?

The embeddings artifacts were generated earlier and recorded:

- `split_csv_hash = e7ff0bbd11b6` in:
  - `data/embeddings/huggingface_qwen3_8b_paper_train.meta.json`
  - `data/embeddings/paper_reference_embeddings.meta.json`

But the current `data/paper_splits/paper_split_train.csv` content hash is now:

- `split_csv_hash = 789c5f289023`

This typically happens after regenerating paper splits (or otherwise rewriting the CSV) **after** embeddings were generated.

### Why didn’t switching artifacts/backends help?

Both of the currently-present “paper-train” embedding artifacts were generated against the same older split CSV hash:

- `data/embeddings/huggingface_qwen3_8b_paper_train.meta.json` → `split_csv_hash=e7ff0bbd11b6`
- `data/embeddings/paper_reference_embeddings.meta.json` → `split_csv_hash=e7ff0bbd11b6`

So **either** artifact would fail once `data/paper_splits/paper_split_train.csv` changed to a different content hash.

### Important nuance

Even if the **participant ID set** is unchanged, the current implementation treats **any** change to the CSV bytes (column changes, ordering, formatting, line endings) as requiring re-generation (strict provenance).

This is the direct reason the run can fail even when a “participant ID set” alignment check passes.

---

## How To Confirm Locally

```bash
# Current split CSV hash (paper-train)
python - <<'PY'
import hashlib
from pathlib import Path
p = Path("data/paper_splits/paper_split_train.csv")
print(hashlib.sha256(p.read_bytes()).hexdigest()[:12])
PY

# Stored split hash in the embeddings metadata (HF example)
python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("data/embeddings/huggingface_qwen3_8b_paper_train.meta.json").read_text())
print(meta["split"], meta["split_csv_hash"])
PY
```

---

## Alternative Fix: Local Data Regeneration (Not Used)

This incident was resolved by updating the `.meta.json` hash after verifying the embeddings were generated from the same
58 paper-train participant IDs. Regenerating embeddings is still a valid alternative if you prefer “clean-room” metadata
that reflects the exact split CSV bytes at generation time.

Regenerate embeddings for the current paper-train split so the `.meta.json` records the current `split_csv_hash`:

```bash
# Best-quality backend (FP16) – overwrites the default HF artifact path
EMBEDDING_BACKEND=huggingface uv run python scripts/generate_embeddings.py --split paper-train
```

If you are intentionally running few-shot with the legacy Ollama embeddings:

```bash
EMBEDDING_BACKEND=ollama uv run python scripts/generate_embeddings.py --split paper-train \\
  --output data/embeddings/paper_reference_embeddings.npz
```

Then re-run:

```bash
uv run python scripts/reproduce_results.py --split paper
```

---

## Preventing Repeat Incidents (Recommended Follow-Up)

1. **Fail fast for few-shot**: add a startup/preflight check in `scripts/reproduce_results.py` that loads `ReferenceStore` once and aborts immediately on `EmbeddingArtifactMismatchError` (instead of spending time evaluating 41 participants that will all fail).
2. **Document the coupling**: update the few-shot preflight checklist to explicitly require re-generating embeddings after any change to `data/paper_splits/paper_split_train.csv`.

### Design Improvement Option (Reduces Unnecessary Regeneration)

Right now, `split_csv_hash` is a hash of the entire CSV file bytes. For embeddings provenance, the *semantically relevant*
piece of the split CSV is typically just the **Participant_ID membership set** (since embeddings are generated from
transcripts, not from PHQ labels/columns).

If we want to avoid forcing a full re-embed when only non-ID columns/formatting change, we can:

- Change `split_csv_hash` to hash the canonicalized **sorted Participant_ID list** (stable across CSV formatting/extra columns), or
- Add a new metadata field (e.g., `split_ids_hash`) and validate that instead of raw file bytes.

This would make few-shot runs robust to harmless split CSV rewrites while still detecting true "wrong split membership".

---

## Resolution Applied (2025-12-27)

### Root Cause Confirmed

The hash mismatch was a **false positive**:
- CSV was regenerated on 2025-12-26 (BUG-025 fix for participant 319)
- Participant 319 is in **TEST**, not TRAIN
- The 58 TRAIN participant IDs were **unchanged**
- Only the CSV bytes changed (triggering hash mismatch)

### Verification Performed

```bash
# Confirmed participant IDs are identical
python3 - <<'PY'
import json, pandas as pd
from pathlib import Path

csv_ids = set(pd.read_csv("data/paper_splits/paper_split_train.csv")["Participant_ID"])
emb_ids = set(int(k) for k in json.loads(
    Path("data/embeddings/huggingface_qwen3_8b_paper_train.json").read_text()
).keys())

print(f"CSV IDs: {len(csv_ids)}, Embeddings IDs: {len(emb_ids)}")
print(f"Exact match: {csv_ids == emb_ids}")
PY
# Output: CSV IDs: 58, Embeddings IDs: 58, Exact match: True
```

### Fix Applied

Updated `split_csv_hash` in both `.meta.json` files:
- `data/embeddings/huggingface_qwen3_8b_paper_train.meta.json`
- `data/embeddings/paper_reference_embeddings.meta.json`

Changed: `e7ff0bbd11b6` → `789c5f289023`

Added audit fields documenting the post-hoc update:
```json
{
  "split_csv_hash": "789c5f289023",
  "split_csv_hash_updated": "2025-12-27T02:50:00Z",
  "split_csv_hash_update_reason": "BUG-026: CSV regenerated after BUG-025 fix; participant IDs unchanged (verified)"
}
```

### Design Improvement Tracked

[Issue #64](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/64): Add semantic ID hash (`split_ids_hash`) for validation, keeping content hash for audit. This prevents false failures from harmless CSV rewrites.

---

*Discovered during few-shot paper reproduction run, 2025-12-26*
*Resolved after first-principles analysis, 2025-12-27*
