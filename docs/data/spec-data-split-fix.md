# SPEC: Data Split Correction and Artifact Regeneration

> **Status**: IMPLEMENTED
> **Branch**: `fix/paper-data-splits`
> **Priority**: HIGH - Blocks paper reproduction parity
> **Related**: GH Issue #45, `critical-data-split-mismatch.md`
> **Date**: 2025-12-25

---

## Executive Summary

Our `data/paper_splits/` files were generated algorithmically with seed=42, producing participant IDs that **do not match** the paper's actual splits. Our legacy paper embeddings artifact (`data/embeddings/paper_reference_embeddings.*`) was generated from `data/paper_splits/paper_split_train.csv`, so it inherits the same wrong membership.

The ground truth was reverse-engineered from the paper authors' output files and is documented in `docs/data/paper-split-registry.md`.

**This spec defines the complete fix**: delete wrong artifacts, update scripts to use ground truth IDs, and regenerate all dependent artifacts.

**Important repo constraint**: `data/*` is gitignored due to DAIC-WOZ licensing. This work must **not** commit dataset-derived CSVs or embeddings; it commits only code + docs changes and regenerates `data/...` artifacts locally.

---

## 1. Current State (WRONG)

Note: This section describes the pre-fix state that motivated the work; it is no longer the current behavior.

### 1.1 Wrong Paper Splits

| File | Status | Issue |
|------|--------|-------|
| `data/paper_splits/paper_split_train.csv` | ❌ WRONG | 27 wrong participant IDs |
| `data/paper_splits/paper_split_val.csv` | ❌ WRONG | 25 wrong participant IDs |
| `data/paper_splits/paper_split_test.csv` | ❌ WRONG | 26 wrong participant IDs |
| `data/paper_splits/paper_split_metadata.json` | ❌ WRONG | Contains wrong IDs |

**Evidence**: Comparison in `critical-data-split-mismatch.md` shows ~50% participant mismatch per split.

### 1.2 Wrong Embeddings

| File | Status | Issue |
|------|--------|-------|
| `data/embeddings/paper_reference_embeddings.npz` | ❌ WRONG | Generated from the **algorithmic** TRAIN split (31 overlap with ground truth; 27 extra + 27 missing) |
| `data/embeddings/paper_reference_embeddings.json` | ❌ WRONG | Text chunks keyed by those same algorithmic TRAIN participant IDs |

**Impact**: Few-shot retrieval uses embeddings from wrong participants → different similarity matches → different predictions → cannot reproduce paper results.

### 1.3 Current Script Behavior

`scripts/create_paper_split.py` implements the paper's **stratification algorithm** (Appendix C) but:
- Uses `random.Random(seed=42)` to assign participants
- The paper authors never disclosed their seed
- Result: Correct sizes (58/43/41) but wrong participant assignments

### 1.4 Blast Radius (What Breaks If These Files Are Missing/Wrong)

**Embeddings artifact (`paper_reference_embeddings.*`)**

- `src/ai_psychiatrist/config.py` defaults few-shot to `huggingface_qwen3_8b_paper_train` (via `EmbeddingSettings.embeddings_file` and `DataSettings.embeddings_path`). The legacy/compat basename `paper_reference_embeddings` is still supported but must be explicitly selected.
- `src/ai_psychiatrist/services/reference_store.py` expects `{embeddings}.npz` + `{embeddings}.json` sidecar; without them it loads an empty store (few-shot degraded) and `scripts/reproduce_results.py` fails fast for few-shot mode.
- `scripts/reproduce_results.py` requires a precomputed embeddings artifact for few-shot evaluation (`--few-shot-only` or default combined run).

**Paper split CSVs (`paper_split_*.csv`)**

- `scripts/generate_embeddings.py --split paper-train` reads `data/paper_splits/paper_split_train.csv` to choose the 58 knowledge-base participants.
- `scripts/reproduce_results.py --split paper*` reads `data/paper_splits/paper_split_{train,val,test}.csv` for paper-split evaluation.

**AVEC embeddings artifact (`reference_embeddings.*`)**

- `data/embeddings/reference_embeddings.npz` / `.json` (107 participants) is an AVEC-train knowledge base artifact (optional; not present in this repo snapshot). It is unaffected by this spec and should **not** be deleted if it exists locally.

**Tests**

- Unit tests do not rely on local `data/*` artifacts (they are gitignored); deleting embeddings/splits does not break tests.
- The one exception is that `tests/unit/scripts/test_create_paper_split.py` must be updated to cover the new ground-truth mode (and should keep the existing algorithmic `stratified_split` tests).

---

## 2. Ground Truth (CORRECT)

### 2.1 Source of Truth

**File**: `docs/data/paper-split-registry.md`

**Provenance**: Reverse-engineered from paper authors' actual output files in `_reference/analysis_output/`:
- TRAIN: Derived from `quan_gemma_zero_shot.jsonl` minus TEST minus VAL
- VAL: From `quan_gemma_few_shot/VAL_analysis_output/*.jsonl`
- TEST: From `quan_gemma_few_shot/TEST_analysis_output/*.jsonl`

### 2.2 Correct Participant IDs

```python
# GROUND TRUTH - From docs/data/paper-split-registry.md
TRAIN_IDS = [
    303, 304, 305, 310, 312, 313, 315, 317, 318, 321, 324, 327, 335, 338, 340, 343,
    344, 346, 347, 350, 352, 356, 363, 368, 369, 388, 391, 395, 397, 400, 402, 404,
    406, 412, 414, 415, 416, 418, 426, 429, 433, 434, 437, 439, 444, 458, 463, 464,
    473, 474, 475, 476, 477, 478, 483, 486, 488, 491
]  # 58 participants

VAL_IDS = [
    302, 307, 320, 322, 325, 326, 328, 331, 333, 336, 341, 348, 351, 353, 355, 358,
    360, 364, 366, 371, 372, 374, 376, 380, 381, 382, 392, 401, 403, 419, 420, 425,
    440, 443, 446, 448, 454, 457, 471, 479, 482, 490, 492
]  # 43 participants

TEST_IDS = [
    316, 319, 330, 339, 345, 357, 362, 367, 370, 375, 377, 379, 383, 385, 386, 389,
    390, 393, 409, 413, 417, 422, 423, 427, 428, 430, 436, 441, 445, 447, 449, 451,
    455, 456, 459, 468, 472, 484, 485, 487, 489
]  # 41 participants
```

### 2.3 Verification

| Check | Expected | Source |
|-------|----------|--------|
| TRAIN + VAL + TEST | 142 | Paper Section 2.4.1 |
| TRAIN ∩ VAL | 0 | No overlap |
| TRAIN ∩ TEST | 0 | No overlap |
| VAL ∩ TEST | 0 | No overlap |
| All IDs have PHQ-8 labels | Yes | Combined AVEC train+dev |

### 2.4 Verified Against This Repo Snapshot (2025-12-25)

The ground-truth registry is internally consistent and matches the local dataset files:

- 58/43/41 IDs in `docs/data/paper-split-registry.md`, total 142, no overlaps
- All 142 IDs exist in `data/train_split_Depression_AVEC2017.csv` ∪ `data/dev_split_Depression_AVEC2017.csv`
- All 142 IDs exist as transcript directories under `data/transcripts/{ID}_P/`
- No registry IDs overlap `data/test_split_Depression_AVEC2017.csv` (unlabeled AVEC test)

---

## 3. Files to DELETE

### 3.1 Paper Splits (Replace with correct IDs)

```bash
# DELETE (local-only; `data/*` is gitignored and not committed)
rm data/paper_splits/paper_split_train.csv
rm data/paper_splits/paper_split_val.csv
rm data/paper_splits/paper_split_test.csv
rm data/paper_splits/paper_split_metadata.json
```

### 3.2 Embeddings (Regenerate from correct TRAIN)

```bash
# DELETE (local-only; `data/*` is gitignored and not committed)
rm data/embeddings/paper_reference_embeddings.npz
rm data/embeddings/paper_reference_embeddings.json
rm -f data/embeddings/paper_reference_embeddings.meta.json
```

**Do not delete**: `data/embeddings/reference_embeddings.*` (AVEC-train knowledge base), if present.

**Note**: The current legacy embeddings have no `.meta.json`, but regeneration via `scripts/generate_embeddings.py` will create one.

---

## 4. Scripts to MODIFY

### 4.1 `scripts/create_paper_split.py`

**Current behavior**: Algorithmic stratification with seed=42
**New behavior**: Default to **paper ground truth** IDs from `docs/data/paper-split-registry.md`, while retaining the existing algorithmic implementation as an explicit opt-in mode.

#### Design Decision: Hardcode vs External File

| Option | Pros | Cons |
|--------|------|------|
| **Hardcode IDs in script** | Single source of truth in code, no external dependency | IDs duplicated (also in paper-split-registry.md) |
| **Read from paper-split-registry.md** | Single source of truth, DRY | Fragile parsing of markdown |
| **New JSON file for IDs** | Clean separation, easy to validate | Yet another file |

**Decision**: **Hardcode IDs in script**, but add an automated check that the hardcoded lists exactly match `docs/data/paper-split-registry.md`. This keeps the markdown registry as the human-readable source of truth while guaranteeing the script cannot silently drift.

#### Changes Required

1. Add constant lists: `_GROUND_TRUTH_TRAIN_IDS`, `_GROUND_TRUTH_VAL_IDS`, `_GROUND_TRUTH_TEST_IDS` (must match `docs/data/paper-split-registry.md`)
2. Keep `stratified_split()` for algorithmic “paper-style” generation (do not delete; tests cover this behavior)
3. Add a selection mode:
   - `--mode ground-truth` (default): use the hardcoded ground truth IDs
   - `--mode algorithmic`: run the existing stratified algorithm
4. Keep `--seed` but scope it to `--mode algorithmic` (error or warn if provided with ground-truth mode)
5. Add `--verify` flag that checks:
   - split sizes are 58/43/41 and total is 142
   - no overlaps between splits
   - all IDs exist in AVEC train+dev CSVs
   - all IDs have transcript directories under `data/transcripts/{ID}_P/`
6. Update `paper_split_metadata.json` schema to reflect mode:
   - ground-truth mode: provenance points to `docs/data/paper-split-registry.md` (no seed)
   - algorithmic mode: preserve current seed/methodology fields (as today)
7. Update docstrings/help text to clarify “paper ground truth” vs “paper-style algorithmic”

### 4.2 `tests/unit/scripts/test_create_paper_split.py`

This test suite currently validates the algorithmic `stratified_split()` behavior and the seed-bearing metadata output. If `scripts/create_paper_split.py` defaults to ground truth, add tests to cover the new mode without removing the existing algorithmic tests:

- Assert `--mode ground-truth` (or an exported helper) returns the exact ID sets from `docs/data/paper-split-registry.md`
- Assert `--verify` fails loudly on overlap/missing IDs/missing transcripts
- Keep existing tests for `stratified_split(df, seed=...)` and `save_splits(..., seed=...)`

### 4.3 `scripts/generate_embeddings.py`

**No changes required** - This script reads from `paper_split_train.csv` which will be regenerated with correct IDs.

### 4.4 `scripts/reproduce_results.py`

**No changes required** - This script reads from `paper_split_*.csv` files which will be regenerated with correct IDs.

---

## 5. Documentation to UPDATE (Required)

Fixing `paper_split_*.csv` from “paper-style seeded” → “paper ground truth” will make existing docs stale. Update these tracked docs to match the new reality:

- `docs/data/artifact-namespace-registry.md` (remove “paper does not publish IDs” phrasing; document the two modes)
- `docs/data/data-splits-overview.md` (remove seed=42 guidance for paper reproduction; point to ground truth mode)
- `docs/data/daic-woz-schema.md` (paper_splits are no longer described as “optional paper-style seeded” only)
- `docs/data/critical-data-split-mismatch.md` (status → FIXED and note the ground truth adoption)

---

## 6. Namespace and Folder Conventions

### 6.1 Current Naming (Keep)

The current naming convention is sound:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `paper_` prefix | `paper_reference_embeddings.npz` | Paper-derived artifacts (legacy/compat) |
| `{backend}_{model}_{split}` | `huggingface_qwen3_8b_paper_train.npz` | New generator output |

### 6.2 Folder Structure (Keep)

```
data/
├── paper_splits/                    # Paper ground truth split (58/43/41)
│   ├── paper_split_train.csv        # 58 participants (ground truth IDs)
│   ├── paper_split_val.csv          # 43 participants (ground truth IDs)
│   ├── paper_split_test.csv         # 41 participants (ground truth IDs)
│   └── paper_split_metadata.json    # Provenance (ground truth source + IDs)
├── embeddings/
│   ├── paper_reference_embeddings.npz   # Legacy name (for backward compat)
│   ├── paper_reference_embeddings.json
│   └── paper_reference_embeddings.meta.json  # Provenance metadata (generated)
├── train_split_Depression_AVEC2017.csv  # Original AVEC (107)
├── dev_split_Depression_AVEC2017.csv    # Original AVEC (35)
└── transcripts/                         # Per-participant transcripts
```

### 6.3 Metadata Provenance

The new `paper_split_metadata.json` should indicate ground truth source:

```json
{
  "description": "Paper ground truth splits (reverse-engineered from output files)",
  "source": "docs/data/paper-split-registry.md",
  "methodology": {
    "derivation": "Extracted participant IDs from paper authors' output files",
    "train_source": "quan_gemma_zero_shot.jsonl minus TEST minus VAL",
    "val_source": "quan_gemma_few_shot/VAL_analysis_output/*.jsonl",
    "test_source": "quan_gemma_few_shot/TEST_analysis_output/*.jsonl"
  },
  "actual_sizes": {
    "train": 58,
    "val": 43,
    "test": 41
  },
  "participant_ids": {
    "train": [...],
    "val": [...],
    "test": [...]
  }
}
```

---

## 7. Implementation Steps

### Phase 1: Script Update

1. **Create branch**: `git checkout -b fix/paper-data-splits`

2. **Update `scripts/create_paper_split.py`**:
   - Add ground truth ID constants
   - Add `--mode` (default ground truth, optional algorithmic)
   - Add `--verify` flag for validation
   - Keep algorithmic implementation for optional use + tests
   - Update metadata schema + help text/docstrings

3. **Verify the script works**:
   ```bash
   uv run python scripts/create_paper_split.py --mode ground-truth --dry-run --verify
   ```

### Phase 2: Delete Wrong Artifacts

4. **Delete wrong paper splits**:
   ```bash
   rm data/paper_splits/paper_split_*.csv
   rm data/paper_splits/paper_split_metadata.json
   ```

5. **Delete wrong embeddings**:
   ```bash
   rm data/embeddings/paper_reference_embeddings.npz
   rm data/embeddings/paper_reference_embeddings.json
   rm -f data/embeddings/paper_reference_embeddings.meta.json
   ```

### Phase 3: Regenerate Correct Artifacts

6. **Generate correct splits**:
   ```bash
   uv run python scripts/create_paper_split.py --mode ground-truth --verify
   ```

7. **Verify split correctness**:

```bash
# Fail-fast: compare generated CSVs to docs/data/paper-split-registry.md
uv run python - <<'PY'
import hashlib
import re
from pathlib import Path

import pandas as pd

# Parse registry IDs from the first ```text blocks.
text = Path("docs/data/paper-split-registry.md").read_text().splitlines()
ids_by_section = {}
section = None
in_code = False
buf = []
for line in text:
    if line.startswith("## "):
        if line.startswith("## TRAIN"):
            section = "train"
        elif line.startswith("## VAL"):
            section = "val"
        elif line.startswith("## TEST"):
            section = "test"
        else:
            section = None
    if section and line.strip().startswith("```text"):
        in_code = True
        buf = []
        continue
    if in_code and line.strip().startswith("```"):
        ids_by_section[section] = [int(x) for x in re.findall(r"\b\d+\b", "\n".join(buf))]
        in_code = False
        continue
    if in_code:
        buf.append(line)

registry = {k: set(v) for k, v in ids_by_section.items()}

required_cols = {
    "Participant_ID",
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
}

current = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/paper_splits/paper_split_{split}.csv")
    missing_cols = sorted(required_cols - set(df.columns))
    assert not missing_cols, f"paper_split_{split}.csv missing columns: {missing_cols}"
    current[split] = set(df["Participant_ID"].astype(int))

# Basic invariants
assert len(registry["train"]) == 58
assert len(registry["val"]) == 43
assert len(registry["test"]) == 41
all_registry = registry["train"] | registry["val"] | registry["test"]
assert len(all_registry) == 142
assert not (registry["train"] & registry["val"])
assert not (registry["train"] & registry["test"])
assert not (registry["val"] & registry["test"])

# Exact match
for split in ["train", "val", "test"]:
    ours = current[split]
    reg = registry[split]
    if ours != reg:
        raise SystemExit(
            f"Split mismatch: {split}\\n"
            f"  ours-not-reg: {sorted(ours - reg)[:20]}\\n"
            f"  reg-not-ours: {sorted(reg - ours)[:20]}"
        )

# Cross-check against AVEC2017 train+dev (must be exactly the same 142 participants).
avec_train = set(
    pd.read_csv("data/train_split_Depression_AVEC2017.csv")["Participant_ID"].astype(int)
)
avec_dev = set(pd.read_csv("data/dev_split_Depression_AVEC2017.csv")["Participant_ID"].astype(int))
assert not (avec_train & avec_dev)
avec_labeled = avec_train | avec_dev
assert all_registry == avec_labeled, "Registry IDs must equal AVEC labeled (train+dev) IDs"

# Ensure registry does not overlap unlabeled AVEC test split.
test_df = pd.read_csv("data/test_split_Depression_AVEC2017.csv")
test_col = "participant_ID" if "participant_ID" in test_df.columns else "Participant_ID"
avec_test = set(test_df[test_col].astype(int))
assert not (all_registry & avec_test)

# Transcript presence for all 142 IDs
missing = []
for pid in sorted(all_registry):
    if not (Path("data/transcripts") / f"{pid}_P").exists():
        missing.append(pid)
assert not missing, f"Missing transcript dirs for: {missing}"

# Hash the paper-train CSV (used by embeddings metadata)
train_bytes = Path("data/paper_splits/paper_split_train.csv").read_bytes()
print("paper-train CSV sha256[:12]=", hashlib.sha256(train_bytes).hexdigest()[:12])
print("OK: CSVs match registry + AVEC + transcripts present")
PY
```

8. **Generate embeddings** (requires Ollama or HuggingFace):
   ```bash
   # Option A: Using HuggingFace (higher precision)
   EMBEDDING_BACKEND=huggingface uv run python scripts/generate_embeddings.py \
     --split paper-train \
     --output data/embeddings/paper_reference_embeddings.npz

   # Option B: Using Ollama (if HuggingFace unavailable)
   EMBEDDING_BACKEND=ollama uv run python scripts/generate_embeddings.py \
     --split paper-train \
     --output data/embeddings/paper_reference_embeddings.npz
   ```

### Phase 4: Verification

9. **Verify embeddings artifact matches paper-train IDs (not just count)**:

```bash
uv run python - <<'PY'
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

train_ids = set(pd.read_csv("data/paper_splits/paper_split_train.csv")["Participant_ID"].astype(int))
npz_path = Path("data/embeddings/paper_reference_embeddings.npz")
meta_path = npz_path.with_suffix(".meta.json")

npz = np.load(npz_path, allow_pickle=False)
try:
    pids = {
        int(m.group(1))
        for k in npz.files
        if (m := re.match(r"emb_(\d+)$", k))
    }
finally:
    npz.close()

missing = sorted(train_ids - pids)
extra = sorted(pids - train_ids)
print("participants_in_npz", len(pids))
print("missing_from_npz", missing)
print("extra_in_npz", extra)
assert not missing and not extra, "NPZ participants must exactly match paper_split_train.csv"

if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    split_hash = meta.get("split_csv_hash")
    current_hash = hashlib.sha256(
        Path("data/paper_splits/paper_split_train.csv").read_bytes()
    ).hexdigest()[:12]
    print("meta.split", meta.get("split"))
    print("meta.split_csv_hash", split_hash)
    print("current_split_csv_hash", current_hash)
    assert split_hash == current_hash, "Metadata split hash must match current paper_split_train.csv"

print("OK: embeddings match paper-train IDs (+ metadata hash if present)")
PY
```

10. **Run tests**:
    ```bash
    make test-unit
    ```

11. **Optional: Run reproduction** (requires Ollama running):
    ```bash
    uv run python scripts/reproduce_results.py --split paper --limit 5
    ```

### Phase 5: Commit and PR

12. **Commit changes**:
    ```bash
    # DO NOT commit `data/*` (gitignored due to licensing).
    git add scripts/create_paper_split.py
    git add tests/unit/scripts/test_create_paper_split.py
    git add docs/data/paper-split-registry.md
    git add docs/data/artifact-namespace-registry.md
    git add docs/data/data-splits-overview.md
    git add docs/data/daic-woz-schema.md
    git add docs/data/critical-data-split-mismatch.md
    git add docs/data/spec-data-split-fix.md
    git commit -m "fix(data-splits): adopt paper ground truth IDs (GH-45)"
    ```

13. **Create PR**:
    ```bash
    gh pr create --title "fix: correct paper split participant IDs" \
      --body "Fixes #45. Uses ground truth IDs from paper-split-registry.md instead of algorithmic generation."
    ```

---

## 8. Post-Fix Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| TRAIN count | `wc -l data/paper_splits/paper_split_train.csv` | 59 (58 + header) |
| VAL count | `wc -l data/paper_splits/paper_split_val.csv` | 44 (43 + header) |
| TEST count | `wc -l data/paper_splits/paper_split_test.csv` | 42 (41 + header) |
| CSVs match registry | `uv run python - <<'PY' ... PY` (see Phase 3.7) | Exact match |
| Transcripts present | Included in Phase 3.7 script | 0 missing |
| Embeddings participants | `uv run python - <<'PY' ... PY` (see Phase 4.9) | Exact match |
| Tests pass | `make test-unit` | All green |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong IDs in paper-split-registry.md | Double-check extraction logic; IDs verified against 7+ output files |
| Accidentally committing DAIC-WOZ-derived data | Keep `data/*` gitignored; do not add exceptions; commit only code/docs |
| Embeddings not gitignored | Verify `.gitignore` excludes `data/*` (includes embeddings + split CSVs) |
| Breaking existing workflows | Legacy `paper_reference_embeddings.*` filename preserved via `--output` |
| Circular dependency (script needs data, data needs script) | AVEC CSVs are independent; script only filters them |
| Silent mismatch: new splits, old embeddings | Require Phase 4.9 equality check (IDs + split hash); regenerate embeddings after split fix |
| Docs drift (“paper-style seeded” language becomes false) | Update docs in Section 5 (required) and add acceptance criteria |

---

## 10. Files Modified Summary

| File | Action | Reason |
|------|--------|--------|
| `scripts/create_paper_split.py` | MODIFY | Use ground truth IDs |
| `tests/unit/scripts/test_create_paper_split.py` | MODIFY | Cover ground truth mode; keep algorithmic tests |
| `docs/data/artifact-namespace-registry.md` | MODIFY | Remove stale “paper-style seeded” claims |
| `docs/data/data-splits-overview.md` | MODIFY | Align reproduction guidance with ground truth mode |
| `docs/data/daic-woz-schema.md` | MODIFY | Align dataset docs with ground truth paper split |
| `docs/data/critical-data-split-mismatch.md` | MODIFY | Update status to FIXED after implementation |
| `data/paper_splits/paper_split_*.csv` | DELETE + REGENERATE (local-only) | Wrong IDs (gitignored) |
| `data/paper_splits/paper_split_metadata.json` | DELETE + REGENERATE (local-only) | Wrong IDs (gitignored) |
| `data/embeddings/paper_reference_embeddings.*` | DELETE + REGENERATE (local-only) | Generated from wrong TRAIN split (gitignored) |

---

## 11. Acceptance Criteria

- [ ] `scripts/create_paper_split.py` defaults to ground truth IDs and supports algorithmic mode explicitly
- [ ] Hardcoded ID lists are verified (test) to match `docs/data/paper-split-registry.md`
- [ ] `data/paper_splits/paper_split_train.csv` contains exactly 58 correct participant IDs
- [ ] `data/paper_splits/paper_split_val.csv` contains exactly 43 correct participant IDs
- [ ] `data/paper_splits/paper_split_test.csv` contains exactly 41 correct participant IDs
- [ ] All IDs match those in `docs/data/paper-split-registry.md`
- [ ] All 142 IDs have transcript directories under `data/transcripts/{ID}_P/`
- [ ] Embeddings regenerated from correct TRAIN split and validated against `paper_split_train.csv` (exact IDs, not just count)
- [ ] `docs/data/critical-data-split-mismatch.md` updated to status FIXED
- [ ] Docs in Section 5 updated to remove “paper-style seeded” ambiguity
- [ ] All unit tests pass
- [ ] GitHub Issue #45 closed

---

## 12. Related Documentation

| Document | Purpose |
|----------|---------|
| `docs/data/paper-split-registry.md` | Ground truth split IDs |
| `docs/data/critical-data-split-mismatch.md` | Problem discovery and evidence |
| `docs/data/artifact-namespace-registry.md` | Naming conventions |
| `docs/data/daic-woz-schema.md` | Dataset schema reference |

---

*Spec Author: Claude Code*
*Date: 2025-12-25*
*Reviewed and approved: 2025-12-25*
