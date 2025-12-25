# Preflight Checklist: Few-Shot Reproduction

**Purpose**: Comprehensive pre-run verification for few-shot paper reproduction
**Last Updated**: 2025-12-23
**Related**: [Zero-Shot Checklist](./preflight-checklist-zero-shot.md) | [Paper Parity Guide](./paper-parity-guide.md) | [BUG-018](../bugs/bug-018-reproduction-friction.md)

---

## Overview

This checklist prevents reproduction failures by verifying ALL known gotchas before running. Use this **every time** you start a few-shot reproduction run.

Few-shot mode uses reference embeddings to retrieve similar transcript chunks as examples for the LLM. This requires:
1. Pre-computed reference embeddings
2. Matching embedding dimensions
3. Correct embedding model

**Paper Target**: MAE = 0.619 (few-shot) vs 0.796 (zero-shot)

---

## Phase 1: Environment Setup

### 1.1 Dependencies

- [ ] **Install all dependencies**: `make dev` (NOT `uv sync --dev`)
  - **Gotcha (BUG-021)**: `uv sync --dev` does NOT install `[project.optional-dependencies].dev`

- [ ] **Verify installation**: `uv run pytest --co -q | head -5` (should show test count)

### 1.2 Configuration File

- [ ] **Copy template**: `cp .env.example .env`
  - **Gotcha (BUG-018b)**: `.env` OVERRIDES code defaults! Always start fresh.
  - **Gotcha (BUG-024)**: If `.env` predates SPEC-003, it may lack `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false`. Re-copy from `.env.example` or add missing settings.

- [ ] **Review .env file manually** - open it and verify:
  ```bash
  cat .env | grep -E "^[^#]" | sort
  ```

### 1.3 Ollama Status

- [ ] **Ollama running**: `curl -s http://localhost:11434/api/tags | head`
  - Should return JSON with model list

- [ ] **Required models pulled**:
  ```bash
  ollama list | grep -E "gemma3:27b|qwen3-embedding"
  ```
  If missing:
  ```bash
  ollama pull gemma3:27b
  ollama pull qwen3-embedding:8b
  ```

---

## Phase 2: Model Configuration (CRITICAL)

### 2.1 Quantitative Model Selection

**Reference**: Paper Section 2.2, BUG-018a

- [ ] **Verify quantitative model is Gemma3** (NOT MedGemma):
  ```bash
  grep "MODEL_QUANTITATIVE_MODEL" .env
  # MUST show: MODEL_QUANTITATIVE_MODEL=gemma3:27b
  ```

  **Gotcha (BUG-018a)**: MedGemma produces ALL N/A scores due to being too conservative. Appendix F says it "detected fewer relevant chunks, making fewer predictions overall."

- [ ] **Check for MedGemma contamination**:
  ```bash
  grep -i "medgemma" .env
  # Should return NOTHING or only commented lines
  ```

### 2.2 Embedding Model Selection

**Reference**: Paper Section 2.2, Appendix D

- [ ] **Verify embedding model**:
  ```bash
  grep "MODEL_EMBEDDING_MODEL" .env
  # Should show: MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
  ```

- [ ] **Verify embedding backend is paper-parity (Ollama)**:
  ```bash
  grep "EMBEDDING_BACKEND" .env
  # MUST show: EMBEDDING_BACKEND=ollama
  ```

### 2.3 Sampling Parameters

**Reference**: GAP-001b/c, [Agent Sampling Registry](../reference/agent-sampling-registry.md)

- [ ] **Temperature is zero** (clinical AI best practice):
  ```bash
  grep "MODEL_TEMPERATURE" .env
  # Should show: MODEL_TEMPERATURE=0.0
  ```

  **Note**: We use temp=0 for all agents. top_k/top_p are not set (irrelevant at temp=0).

---

## Phase 3: Embedding Hyperparameters (CRITICAL)

### 3.1 Paper-Optimal Values

**Reference**: Paper Appendix D

- [ ] **Chunk size = 8** (Nchunk):
  ```bash
  grep "EMBEDDING_CHUNK_SIZE" .env
  # MUST show: EMBEDDING_CHUNK_SIZE=8
  ```

- [ ] **Chunk step = 2** (overlap):
  ```bash
  grep "EMBEDDING_CHUNK_STEP" .env
  # MUST show: EMBEDDING_CHUNK_STEP=2
  ```

- [ ] **Dimension = 4096** (Ndimension):
  ```bash
  grep "EMBEDDING_DIMENSION" .env
  # MUST show: EMBEDDING_DIMENSION=4096
  ```

- [ ] **Top-k references = 2** (Nexample):
  ```bash
  grep "EMBEDDING_TOP_K_REFERENCES" .env
  # MUST show: EMBEDDING_TOP_K_REFERENCES=2
  ```

### 3.2 Dimension Mismatch Check

**Reference**: BUG-009

**Gotcha**: Dimension mismatches can result in skipped chunks. If *all* chunks are mismatched, the system fails loudly.
If only some chunks are mismatched, retrieval quality degrades. Always validate dimensions pre-run.

- [ ] **Verify dimension consistency**:
  ```bash
  # If embeddings exist, check their dimension
  uv run python -c "
  import numpy as np
  from pathlib import Path
  p = Path('data/embeddings/paper_reference_embeddings.npz')
  if p.exists():
      data = np.load(str(p))
      # NPZ uses per-participant keys: emb_302, emb_304, etc.
      dim = next(iter(data.values())).shape[1]
      print(f'Embedding dimension: {dim}')
      print(f'Config expects: 4096')
      assert dim == 4096, 'DIMENSION MISMATCH!'
      print('OK - dimensions match')
  else:
      print('Embeddings not found - will need to generate')
  "
  ```

---

## Phase 4: Reference Embeddings (FEW-SHOT SPECIFIC)

### 4.1 Embeddings Exist

**Reference**: BUG-006

- [ ] **Check for embedding file**:
  ```bash
  ls -lh data/embeddings/*.npz
  ```

  If missing, generate (takes ~65 min for 58 participants):
  ```bash
  uv run python scripts/generate_embeddings.py --split paper-train --backend ollama --output data/embeddings/paper_reference_embeddings.npz
  ```

### 4.2 Verify Embedding Integrity

- [ ] **Embedding file is valid**:
  ```bash
  uv run python -c "
  import numpy as np
  from pathlib import Path

  # Try paper embeddings first, then AVEC
  for name in ['paper_reference_embeddings.npz', 'reference_embeddings.npz']:
      p = Path('data/embeddings') / name
      if p.exists():
          data = np.load(str(p))
          # NPZ uses per-participant keys: emb_302, emb_304, etc.
          pids = [int(k.split('_')[1]) for k in data.keys()]
          total_chunks = sum(data[k].shape[0] for k in data.keys())
          dim = next(iter(data.values())).shape[1]
          print(f'File: {name}')
          print(f'  Participants: {len(pids)}')
          print(f'  Total chunks: {total_chunks}')
          print(f'  Dimension: {dim}')
          break
  else:
      print('ERROR: No embedding file found!')
      print('Run: uv run python scripts/generate_embeddings.py --split paper-train')
  "
  ```

  Expected output for paper-train:
  ```text
  File: paper_reference_embeddings.npz
    Participants: 58
    Total chunks: ~7000
    Dimension: 4096
  ```

### 4.3 Sidecar File Check

- [ ] **JSON sidecar exists** (for chunk text):
  ```bash
  ls -lh data/embeddings/*embeddings.json
  # Should show matching JSON file for NPZ
  ```

---

## Phase 5: Quantitative Settings (SPEC-003)

### 5.1 Keyword Backfill Toggle

**Reference**: SPEC-003, Coverage Investigation

- [ ] **Backfill is DISABLED** (paper parity = ~50% coverage):
  ```bash
  grep "QUANTITATIVE_ENABLE_KEYWORD_BACKFILL" .env
  # MUST show: QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
  ```

  **Gotcha**: Backfill ON = ~74% coverage, which diverges from paper's ~50%.

### 5.2 N/A Reason Tracking

- [ ] **N/A tracking enabled** (for debugging):
  ```bash
  grep "QUANTITATIVE_TRACK_NA_REASONS" .env
  # Should show: QUANTITATIVE_TRACK_NA_REASONS=true
  ```

---

## Phase 6: Data Integrity

### 6.1 Transcripts Present

- [ ] **Transcripts directory exists**:
  ```bash
  ls data/transcripts/ | wc -l
  # Should show ~189 (or your participant count)
  ```

### 6.2 Participant 487 Validation

**Reference**: BUG-003, BUG-022

- [ ] **Participant 487 is NOT corrupted**:
  ```bash
  file data/transcripts/487_P/487_TRANSCRIPT.csv
  # MUST show: ASCII text, or UTF-8 Unicode text
  # NOT: AppleDouble encoded, or binary
  ```

- [ ] **Correct file size** (~20KB, not 4KB):
  ```bash
  ls -lh data/transcripts/487_P/487_TRANSCRIPT.csv
  # Should be ~18-25KB, NOT 4KB
  ```

  **Gotcha (BUG-003)**: macOS ZIP extraction can extract AppleDouble resource forks instead of real files.

### 6.3 Ground Truth Labels

- [ ] **AVEC2017 labels exist** (for item-level MAE):
  ```bash
  ls data/*_split_Depression_AVEC2017.csv
  # Should show: dev_split, train_split, test_split files
  ```

---

## Phase 7: Timeout Configuration

### 7.1 Timeout Setting

**Reference**: BUG-018e

- [ ] **Adequate timeout** for transcript size:
  ```bash
  grep "OLLAMA_TIMEOUT_SECONDS" .env
  # Should show: OLLAMA_TIMEOUT_SECONDS=300 (minimum)
  ```

  **Gotcha**: 6/47 participants (13%) timed out on first run. Large transcripts (~24KB+) may need 360+ seconds, especially with concurrent GPU workloads.

### 7.2 Check for Long Transcripts

- [ ] **Identify large transcripts** that may timeout:
  ```bash
  find data/transcripts -name "*TRANSCRIPT.csv" -exec wc -c {} + | sort -n | tail -10
  # Note any files > 25KB - these may need extra timeout
  ```

---

## Phase 8: Paper Split

### 8.1 Create or Verify Splits

- [ ] **Paper splits exist OR will be created**:
  ```bash
  ls data/paper_splits/
  # If empty or missing:
  uv run python scripts/create_paper_split.py --seed 42
  ```

### 8.2 Verify Split Sizes

**Reference**: Paper Section 2.4.1

- [ ] **Split sizes match paper** (58/43/41):
  ```bash
  wc -l data/paper_splits/paper_split_*.csv
  # Should show: 59 train (58+header), 44 val, 42 test
  ```

### 8.3 Embedding-Split Alignment

**CRITICAL for few-shot**: Embeddings must be from TRAINING set only!

- [ ] **Verify embeddings are from paper-train**:
  ```bash
  uv run python -c "
  import numpy as np
  import csv

  # Load embedding participant IDs from NPZ keys (emb_302, emb_304, etc.)
  emb = np.load('data/embeddings/paper_reference_embeddings.npz')
  emb_pids = {int(k.split('_')[1]) for k in emb.keys()}
  print(f'Embedding participants: {len(emb_pids)}')

  # Load paper train split (column is Participant_ID)
  with open('data/paper_splits/paper_split_train.csv') as f:
      train_pids = {int(row['Participant_ID']) for row in csv.DictReader(f)}
  print(f'Paper train participants: {len(train_pids)}')

  # Check alignment
  if emb_pids == train_pids:
      print('OK - embeddings match paper train split')
  else:
      print('WARNING: Embeddings do not match train split!')
      print(f'  In emb but not train: {sorted(emb_pids - train_pids)}')
      print(f'  In train but not emb: {sorted(train_pids - emb_pids)}')
  "
  ```

---

## Phase 9: Pre-Run Verification

### 9.1 Quick Sanity Check

- [ ] **Run linter**: `make lint`
- [ ] **Run type checker**: `make typecheck`
- [ ] **Run unit tests**: `make test-unit`

### 9.2 Configuration Summary Check

Run this to dump your effective configuration:
```bash
uv run python -c "
from ai_psychiatrist.config import get_settings
s = get_settings()
print('=== CRITICAL SETTINGS ===')
print(f'Quantitative Model: {s.model.quantitative_model}')
print(f'Embedding Model: {s.model.embedding_model}')
print(f'Temperature: {s.model.temperature}')
print(f'Keyword Backfill: {s.quantitative.enable_keyword_backfill}')
print(f'Timeout: {s.ollama.timeout_seconds}s')
print()
print('=== EMBEDDING SETTINGS (Appendix D) ===')
print(f'Dimension: {s.embedding.dimension} (paper: 4096)')
print(f'Chunk Size: {s.embedding.chunk_size} (paper: 8)')
print(f'Chunk Step: {s.embedding.chunk_step} (paper: 2)')
print(f'Top-K References: {s.embedding.top_k_references} (paper: 2)')
"
```

Expected output:
```text
=== CRITICAL SETTINGS ===
Quantitative Model: gemma3:27b
Embedding Model: qwen3-embedding:8b
Temperature: 0.0
Keyword Backfill: False
Timeout: 300s

=== EMBEDDING SETTINGS (Appendix D) ===
Dimension: 4096 (paper: 4096)
Chunk Size: 8 (paper: 8)
Chunk Step: 2 (paper: 2)
Top-K References: 2 (paper: 2)
```

### 9.3 Few-Shot Readiness Final Check

- [ ] **Embeddings match config dimension** (from Phase 3.2)
- [ ] **Embeddings are from train split** (from Phase 8.3)
- [ ] **Config matches paper Appendix D** (from Phase 9.2)

---

## Phase 10: Execute Few-Shot Run

### 10.1 Use tmux for Long-Running Processes

**CRITICAL**: Reproduction runs take ~5-6 min/participant. Use tmux to prevent losing progress if your terminal disconnects.

- [ ] **Start or attach to tmux session**:
  ```bash
  # Start new session
  tmux new -s reproduction

  # Or attach to existing session
  tmux attach -t reproduction
  ```

- [ ] **Verify you're inside tmux**: Look for green status bar at bottom, or run:
  ```bash
  echo $TMUX
  # Should show something like: /private/tmp/tmux-501/default,12345,0
  # If empty, you're NOT in tmux!
  ```

### 10.2 Run Command

```bash
# Few-shot on paper test split (primary reproduction)
uv run python scripts/reproduce_results.py --split paper --few-shot-only

# Few-shot on AVEC dev split (sanity check)
uv run python scripts/reproduce_results.py --split dev --few-shot-only
```

**Note**: `--few-shot-only` ensures few-shot mode. Without it, uses config `ENABLE_FEW_SHOT` setting.

### 10.3 Monitor for Issues

Watch for these log patterns:

| Log Pattern | Issue | Action |
|-------------|-------|--------|
| `LLM request timed out` | Transcript too long | Increase `OLLAMA_TIMEOUT_SECONDS` |
| `Failed to parse evidence JSON` | LLM output malformed | Keyword backfill mitigates; check model |
| `na_count = 8` for all | MedGemma contamination | Check model setting is `gemma3:27b` |
| `No reference embeddings found` | Missing/wrong embeddings | Generate: `scripts/generate_embeddings.py` |
| `Embedding dimension mismatch` | Dimension inconsistency | Regenerate embeddings with correct dimension |
| `0 similar chunks found` | Silent dimension mismatch | Check `EMBEDDING_DIMENSION` matches NPZ |

---

## Phase 11: Post-Run Validation

### 11.1 Output File Created

- [ ] **Results file exists**:
  ```bash
  ls -lt data/outputs/reproduction_results_*.json | head -1
  ```

### 11.2 Metrics Sanity Check

- [ ] **Coverage is ~50%** (paper parity with backfill OFF):
  ```bash
  # Check the output JSON for coverage metrics
  python -c "
  import json
  from pathlib import Path
  f = sorted(Path('data/outputs').glob('reproduction_results_*.json'))[-1]
  data = json.loads(f.read_text())
  print(f'Coverage: {data.get(\"aggregate\", {}).get(\"coverage_pct\", \"N/A\")}%')
  print(f'MAE (weighted): {data.get(\"aggregate\", {}).get(\"mae_weighted\", \"N/A\")}')
  print(f'MAE (by-item): {data.get(\"aggregate\", {}).get(\"mae_by_item\", \"N/A\")}')
  "
  ```

  Expected (few-shot, paper parity):
  - Coverage: ~50% (with backfill OFF)
  - MAE: ~0.619 (paper reports 0.619 for few-shot)
  - Our actual: ~0.757 (higher coverage explains difference)

### 11.3 Compare to Paper Targets

| Metric | Paper | Acceptable Range | Your Result |
|--------|-------|-----------------|-------------|
| MAE (few-shot) | 0.619 | 0.52 - 0.72 | ______ |
| MAE (zero-shot) | 0.796 | 0.70 - 0.90 | ______ |
| Coverage | ~50% | 40% - 60% | ______ |

**Note**: Paper acknowledges stochasticity. Results within ±0.1 MAE and ±10% coverage are consistent.

---

## Common Failure Modes Quick Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| All items N/A | MedGemma model | Change to `gemma3:27b` |
| 74% coverage | Backfill ON | Set `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` |
| Timeouts on 13% | Long transcripts | Increase `OLLAMA_TIMEOUT_SECONDS=360` |
| Participant 487 fails | macOS resource fork | Re-extract with `unzip -x '._*'` |
| Config not applying | .env override | Start fresh: `cp .env.example .env` |
| MAE ~4.0 (wrong scale) | Old script | Use current `scripts/reproduce_results.py` |
| No few-shot effect | Missing embeddings | Generate: `scripts/generate_embeddings.py` |
| Silent zero-shot | Dimension mismatch | Check `EMBEDDING_DIMENSION=4096` matches NPZ |
| Wrong participants | AVEC vs paper split | Use `--split paper` for paper methodology |

---

## Checklist Complete?

If ALL items are checked:
1. You're ready to run few-shot reproduction
2. Expected runtime: ~5-6 min/participant (varies by hardware)
3. Embedding generation: ~65 min for 58 participants (one-time)
4. Paper few-shot MAE target: 0.619

**Remember**: The paper acknowledges stochasticity - results within ±0.1 MAE are considered consistent.

---

## Complete Reproduction Workflow

```bash
# 1. Setup (first time only)
make dev
cp .env.example .env

# 2. Create paper-style split
uv run python scripts/create_paper_split.py --seed 42

# 3. Generate embeddings from paper-train (takes ~65 min)
uv run python scripts/generate_embeddings.py --split paper-train

# 4. Run few-shot reproduction
uv run python scripts/reproduce_results.py --split paper --few-shot-only

# 5. (Optional) Compare with zero-shot
uv run python scripts/reproduce_results.py --split paper
```

---

## Related Documentation

- [Zero-Shot Preflight](./preflight-checklist-zero-shot.md) - Simpler, no embeddings
- [Paper Parity Guide](./paper-parity-guide.md) - Full paper methodology
- [BUG-018](../bugs/bug-018-reproduction-friction.md) - Reproduction friction log (includes BUG-018a-i sub-issues)
- [GAP-001](../bugs/gap-001-paper-unspecified-parameters.md) - Unspecified parameters
- [Coverage Investigation](../bugs/coverage-investigation.md) - 74% vs 50% explained
- [SPEC-003](../specs/SPEC-003-backfill-toggle.md) - Backfill toggle specification
