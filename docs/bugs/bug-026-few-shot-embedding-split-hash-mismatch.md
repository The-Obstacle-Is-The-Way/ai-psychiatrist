# BUG-026: Few-Shot Paper Run Fails Due to Embedding Split Hash Mismatch

> **STATUS: OPEN**
>
> **Discovered**: 2025-12-26
>
> **Severity**: Blocker (prevents paper reproduction in `few_shot` mode)
>
> **Affects**: `scripts/reproduce_results.py --split paper` (few-shot path only)
>
> **Root Cause**: Reference embeddings metadata (`split_csv_hash`) is stale relative to the current `data/paper_splits/paper_split_train.csv`.

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

## Fix (Local Data Regeneration)

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

This would make few-shot runs robust to harmless split CSV rewrites while still detecting true “wrong split membership”.
