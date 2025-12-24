# BUG-023: Embedding Provenance Gap - Critical Reproducibility Issue

**Date**: 2025-12-23
**Status**: OPEN - Critical reproducibility gap identified
**Severity**: HIGH - Affects all historical reproduction runs; no provenance tracking
**Author**: Investigation triggered by user observation

---

## Executive Summary

There is **NO provenance tracking** for which embeddings artifact is used during reproduction runs. The output JSON files do not record:
- Which embeddings file was loaded
- Which participants were in the knowledge base
- Whether the embeddings matched the evaluation split

This makes it **impossible to verify** whether past runs used correct embeddings.

---

## The Core Problem

### 1. Config Default Points to Wrong Embeddings

```python
# src/ai_psychiatrist/config.py:291-292
embeddings_path: Path = Field(
    default=Path("data/embeddings/reference_embeddings.npz"),  # ← AVEC-train (107 participants)
    ...
)
```

The default config points to `reference_embeddings.npz` (AVEC-train, 107 participants), but paper reproduction requires `paper_reference_embeddings.npz` (paper-train, 58 participants).

### 2. Script Overrides Are Not Logged

The `reproduce_results.py` script correctly overrides the embeddings path for paper splits:

```python
# scripts/reproduce_results.py:578-579
if args.split.startswith("paper"):
    paper_path = data_settings.base_dir / "embeddings" / "paper_reference_embeddings.npz"
```

But this override is:
- Only printed to stdout (ephemeral)
- **NOT saved to the output JSON**
- **NOT verifiable after the fact**

### 3. Output Files Have No Provenance

All output files lack critical metadata:

```json
// data/outputs/reproduction_results_20251223_014119.json
{
  "timestamp": "2025-12-23T01:41:19.525226",
  "experiments": [...]
  // NO embeddings_path field
  // NO knowledge_base_participants field
  // NO config snapshot
}
```

---

## Timeline Analysis

| Timestamp | File | Participants | Embeddings Available | Status |
|-----------|------|--------------|---------------------|--------|
| Dec 21 15:43 | reference_embeddings.npz | 107 | Created | AVEC-train |
| Dec 22 00:29 | reproduction_20251222_002941.json | N/A | Only AVEC | **UNRELIABLE** |
| Dec 22 00:48 | reproduction_20251222_004802.json | N/A | Only AVEC | **UNRELIABLE** |
| Dec 22 00:59 | reproduction_20251222_005953.json | N/A | Only AVEC | **UNRELIABLE** |
| Dec 22 04:01 | reproduction_20251222_040100.json | N/A | Only AVEC | **INVALID** (BUG-018i) |
| Dec 22 19:00 | reproduction_20251222_190008.json | 1 | Only AVEC | **UNRELIABLE** |
| Dec 22 21:32 | paper_reference_embeddings.npz | 58 | Created | Paper-train |
| Dec 22 21:49 | reproduction_20251222_214944.json | 3 | Both exist | **LIKELY CORRECT** (post-creation) |
| Dec 23 01:41 | reproduction_20251223_014119.json | 41 | Both exist | **LIKELY CORRECT** (matches paper test) |

### Critical Observations

1. **All runs before Dec 22 21:32 could only have used AVEC embeddings** - paper embeddings didn't exist yet

2. **The 41-participant run (Dec 23) is LIKELY correct** because:
   - It ran AFTER paper_reference_embeddings.npz was created
   - It used `--split paper` which triggers the override logic
   - 41 participants matches paper test split exactly

3. **We cannot prove any run used correct embeddings** - no provenance recorded

---

## Impact Assessment

### What This Affects

| Component | Impact |
|-----------|--------|
| Historical reproduction runs | **UNVERIFIABLE** - no way to know which embeddings were used |
| Current reproduction runs | **PROBABLY CORRECT** - script overrides work, but not logged |
| Server API (direct use) | **USES WRONG DEFAULT** - config points to AVEC embeddings |
| Documentation claims | **UNVERIFIABLE** - can't prove results in docs used correct embeddings |

### Specific Concerns

1. **The MAE 0.757 result in reproduction-notes.md**: Likely correct (ran after paper embeddings existed, used paper split), but **cannot be proven**

2. **Any claims about few-shot vs zero-shot improvement**: May be comparing against wrong knowledge base

3. **Future runs via server API**: Will use AVEC embeddings by default, not paper embeddings

---

## Root Cause Analysis

### Why This Happened

1. **Evolutionary development**: AVEC embeddings were created first for initial testing, paper split came later

2. **Script vs Config disconnect**: `reproduce_results.py` has correct logic, but `config.py` default wasn't updated

3. **No provenance requirements**: Output schema was designed before the split distinction was critical

4. **Silent fallback**: If paper embeddings don't exist, system uses AVEC embeddings without warning

---

## Related Issues Discovered

### ISSUE-023a: Config Default Mismatch

The `DataSettings.embeddings_path` default points to AVEC embeddings, but paper reproduction is the primary use case.

**Location**: `src/ai_psychiatrist/config.py:292`

### ISSUE-023b: No Embeddings Validation

The system doesn't validate that:
- Embedding participants match the expected knowledge base
- Embedding participants are disjoint from evaluation participants
- Embedding dimension matches config

**Location**: `src/ai_psychiatrist/services/reference_store.py` (load time)

### ISSUE-023c: Script Override Not Persisted

The `--split paper` override in `reproduce_results.py` is correct but not saved to output.

**Location**: `scripts/reproduce_results.py:578-586`

### ISSUE-023d: Server API Uses Wrong Default (CONFIRMED)

If someone uses the server API directly for few-shot assessment, it uses AVEC embeddings.

**Location**: `server.py:52`

```python
# server.py:52 - Uses settings.data which has AVEC default
reference_store = ReferenceStore(settings.data, settings.embedding)
```

The `ReferenceStore` is initialized with `settings.data.embeddings_path`, which defaults to `reference_embeddings.npz` (AVEC-train, 107 participants).

**Impact**: Any API requests with `mode=few_shot` will use AVEC embeddings, NOT paper embeddings. This is a **silent misconfiguration** for anyone trying to use the API for paper-parity evaluation.

### ISSUE-023e: Stale Outputs in data/outputs/

Multiple output files exist from invalid runs:
- Pre-Dec-22-21:32 runs used AVEC embeddings (if any)
- Some files are empty or have N/A participants
- Difficult to distinguish valid from invalid

**Location**: `data/outputs/reproduction_results_*.json`

---

## Recommended Fixes

### FIX-1: Add Provenance to Output (HIGH PRIORITY)

```python
# In reproduce_results.py, add to output:
output = {
    "timestamp": ...,
    "provenance": {
        "embeddings_path": str(embeddings_path),
        "embeddings_participants": list(sorted(knowledge_base_pids)),
        "evaluation_split": split,
        "evaluation_participants": list(sorted(eval_pids)),
        "config_snapshot": {
            "quantitative_model": model_settings.quantitative_model,
            "embedding_dimension": embedding_settings.dimension,
            "enable_keyword_backfill": quantitative_settings.enable_keyword_backfill,
            # ... all relevant settings
        }
    },
    "experiments": ...
}
```

### FIX-2: Add Embeddings Validation

```python
# In reference_store.py or reproduce_results.py
def validate_embeddings(embeddings_pids: set[int], eval_pids: set[int], expected_train_pids: set[int]) -> None:
    # Check embeddings match expected train split
    if embeddings_pids != expected_train_pids:
        raise ValueError(f"Embeddings mismatch: got {len(embeddings_pids)}, expected {len(expected_train_pids)}")

    # Check no leakage (eval participants in knowledge base)
    overlap = embeddings_pids & eval_pids
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} eval participants in knowledge base: {overlap}")
```

### FIX-3: Update Config Default

Consider changing the default, or making it more explicit:

```python
# Option A: Change default to paper embeddings (breaking change)
embeddings_path: Path = Field(
    default=Path("data/embeddings/paper_reference_embeddings.npz"),
    description="Path to reference embeddings. Use paper_reference_embeddings.npz for paper reproduction.",
)

# Option B: Require explicit setting (safer)
embeddings_path: Path | None = Field(
    default=None,
    description="Path to reference embeddings. Must be set explicitly for few-shot mode.",
)
```

### FIX-4: Clean Up Stale Outputs

Archive or delete the pre-paper-embeddings outputs that cannot be verified:

```bash
# Move unreliable outputs to archive
mkdir -p data/outputs/archive_unverified
mv data/outputs/reproduction_results_2025122{2_002941,2_004802,2_005953,2_040100,2_190008}.json data/outputs/archive_unverified/
```

### FIX-5: Update Preflight Checklists

Add verification step to confirm correct embeddings will be used before running.

---

## Verification: Is the Dec 23 Run Valid?

### Evidence FOR Validity

1. **Timing**: Ran 4 hours AFTER paper_reference_embeddings.npz was created
2. **Participant count**: 41 = paper test split (not 35 AVEC dev, not 47 AVEC test)
3. **Script logic**: `--split paper` triggers correct override at line 578-579
4. **Console output**: Would have printed "Embeddings Artifact: .../paper_reference_embeddings.npz"

### Evidence AGAINST Certainty

1. **No provenance in output**: Cannot prove which file was loaded
2. **No console logs preserved**: Only output JSON exists
3. **Config default is wrong**: If override failed, would silently use AVEC

### Conclusion

The Dec 23 run is **HIGHLY LIKELY CORRECT** but **CANNOT BE PROVEN**. This is exactly why provenance tracking is critical.

---

## Action Items

- [x] **CRITICAL**: Add provenance to `reproduce_results.py` output (FIX-1) ✅ DONE
- [ ] **HIGH**: Add embeddings validation (FIX-2) - Future work
- [x] **MEDIUM**: Update preflight checklists with explicit embeddings verification ✅ DONE
- [x] **LOW**: Update config default to paper embeddings (FIX-3) ✅ DONE
- [x] **LOW**: Delete stale outputs (FIX-4) ✅ DONE - Deleted pre-paper-embeddings outputs
- [x] **CLEANUP**: Delete old AVEC embeddings (reference_embeddings.npz) ✅ DONE
- [ ] **DOCUMENTATION**: Update reproduction-notes.md with caveat about provenance gap

---

## Related Documentation

- [BUG-018](./bug-018-reproduction-friction.md) - Reproduction friction log (mentions invalid 04:01 run)
- [artifact-namespace-registry.md](../data/artifact-namespace-registry.md) - Documents both embedding files
- [preflight-checklist-few-shot.md](../guides/preflight-checklist-few-shot.md) - Should verify embeddings

---

## Appendix: File Inventory

### Embeddings Files

| File | Participants | Size | Created | Purpose |
|------|-------------|------|---------|---------|
| `reference_embeddings.npz` | 107 | 189 MB | Dec 21 15:43 | AVEC-train knowledge base |
| `reference_embeddings.json` | 107 | ~5 MB | Dec 21 15:43 | AVEC-train text chunks |
| `paper_reference_embeddings.npz` | 58 | 106 MB | Dec 22 21:32 | Paper-train knowledge base |
| `paper_reference_embeddings.json` | 58 | ~3 MB | Dec 22 21:32 | Paper-train text chunks |

### Output Files

| File | Participants | Likely Valid | Notes |
|------|-------------|--------------|-------|
| `reproduction_results_20251222_002941.json` | N/A | NO | Pre-paper-embeddings, empty |
| `reproduction_results_20251222_004802.json` | N/A | NO | Pre-paper-embeddings, empty |
| `reproduction_results_20251222_005953.json` | N/A | NO | Pre-paper-embeddings, empty |
| `reproduction_results_20251222_040100.json` | N/A | NO | Pre-paper-embeddings, documented invalid in BUG-018i |
| `reproduction_results_20251222_190008.json` | 1 | NO | Pre-paper-embeddings, partial |
| `reproduction_results_20251222_214944.json` | 3 | MAYBE | Post-paper-embeddings, test run |
| `reproduction_results_20251223_014119.json` | 41 | LIKELY | Post-paper-embeddings, full run |

### Recommended Cleanup

```bash
# Archive unreliable outputs (do not delete - historical record)
mkdir -p data/outputs/archive_pre_paper_embeddings
mv data/outputs/reproduction_results_2025122{2_002941,2_004802,2_005953,2_040100,2_190008}.json \
   data/outputs/archive_pre_paper_embeddings/

# Keep potentially valid outputs
# - reproduction_results_20251222_214944.json (3 participants, test run)
# - reproduction_results_20251223_014119.json (41 participants, main run)
```
