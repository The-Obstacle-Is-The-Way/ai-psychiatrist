# BUG-024: Preflight Checklist Friction Points

**Status**: Resolved
**Severity**: Medium (documentation inconsistency, not runtime bug)
**Found**: 2025-12-23 (during preflight for few-shot reproduction)
**Related**: preflight-checklist-few-shot.md, preflight-checklist-zero-shot.md

---

## Summary

During preflight checks for paper reproduction, we found several documentation inconsistencies between the checklists and actual system state. These don't prevent reproduction but cause confusion and friction.

---

## Friction Points Found

### 1. Make Target `lint-check` Doesn't Exist

**Location**: Both preflight checklists, Phase 9.1/7.1

**Checklist says**:
```bash
make lint-check
```

**Actual Makefile targets**:
```text
lint        - Run linter (ruff)
format-check - Check formatting without changes
```

**Fix**: Change `make lint-check` → `make lint` in checklists

---

### 2. Labels Path `data/labels/` Doesn't Exist

**Location**: Both preflight checklists, Phase 6.3/4.3

**Checklist says**:
```bash
ls data/labels/
# Should show: train_split.csv, dev_split.csv (minimum)
```

**Actual location**: Labels are in `data/` root:
```text
data/dev_split_Depression_AVEC2017.csv
data/train_split_Depression_AVEC2017.csv
data/test_split_Depression_AVEC2017.csv
```

**Fix**: Update path and filenames in checklists

---

### 3. NPZ Structure Doesn't Match Checklist Expectations

**Location**: preflight-checklist-few-shot.md, Phase 4.2

**Checklist expects**:
```python
data = np.load(str(p))
emb = data['embeddings']       # Single combined array
pids = data['participant_ids']  # Separate array
```

**Actual NPZ structure** (per-participant keys):
```python
# Keys: ['emb_302', 'emb_304', 'emb_305', ...]
# Each key: shape=(N_chunks, 4096), dtype=float32
```

**Fix**: Update the verification script to handle per-participant format:
```python
data = np.load(p)
pids = [int(k.split('_')[1]) for k in data.keys()]
total_chunks = sum(data[k].shape[0] for k in data.keys())
dim = next(iter(data.values())).shape[1]
print(f'Participants: {len(pids)}, Total chunks: {total_chunks}, Dimension: {dim}')
```

---

### 4. CSV Column Name Case Mismatch

**Location**: preflight-checklist-few-shot.md, Phase 8.3

**Checklist uses**:
```python
train_pids = {int(row['participant_id']) for row in csv.DictReader(f)}
```

**Actual CSV column name**:
```text
Participant_ID  (capital P, capital I, capital D)
```

**Fix**: Use `row['Participant_ID']` or make case-insensitive

---

### 5. User .env Missing New Settings

**Location**: User's `.env` file

**Issue**: If `.env` was created before SPEC-003 merged, it lacks:
```bash
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
QUANTITATIVE_TRACK_NA_REASONS=true
```

The `.env.example` is correct (has these settings), but existing `.env` files won't have them.

**Fix**: Add note to checklists: "If .env predates SPEC-003, re-copy from .env.example or manually add missing settings"

---

## Action Items

- [x] Fix `make lint-check` → `make lint` in both checklists
- [x] Fix `data/labels/` → `data/` with correct filenames in both checklists
- [x] Update NPZ verification script in few-shot checklist for per-participant format
- [x] Fix `participant_id` → `Participant_ID` in few-shot checklist
- [x] Add .env staleness warning to both checklists

**All fixes applied**: 2025-12-23

---

## Root Cause

Checklists were written based on expected/planned structure rather than verified against actual implementation. This is a gap in our doc-code parity process.

## Prevention

After any checklist creation/update:
1. Run ALL commands in the checklist manually
2. Verify output matches expected output
3. Test on fresh clone if possible
