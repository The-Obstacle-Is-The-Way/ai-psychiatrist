# Preflight Checklist: Few-Shot Reproduction

**Purpose**: Comprehensive pre-run verification for few-shot paper reproduction
**Last Updated**: 2025-12-27
**Related**: [Zero-Shot Checklist](./preflight-checklist-zero-shot.md) | [Configuration Reference](../configs/configuration.md)

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
  # Production-recommended (QAT-quantized, faster):
  ollama pull gemma3:27b-it-qat
  # Standard Ollama tag (GGUF Q4_K_M):
  ollama pull gemma3:27b
  # Embedding model:
  ollama pull qwen3-embedding:8b
  ```

---

## Phase 2: Model Configuration (CRITICAL)

### 2.1 Quantitative Model Selection

**Reference**: Paper Section 2.2, BUG-018a

- [ ] **Verify quantitative model is Gemma3** (NOT MedGemma):
  ```bash
  grep "MODEL_QUANTITATIVE_MODEL" .env
  # Acceptable values:
  #   MODEL_QUANTITATIVE_MODEL=gemma3:27b-it-qat  (QAT-optimized, faster inference)
  #   MODEL_QUANTITATIVE_MODEL=gemma3:27b         (standard Ollama quantization)
  ```

  **Note on quantization**: The paper authors likely used full-precision BF16 weights. Ollama's `gemma3:27b` uses Q4_K_M quantization; `-it-qat` adds QAT optimization for faster inference. Both are acceptable for reproduction (neither is true BF16).

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

- [ ] **Verify embedding backend** (HF recommended for higher quality):
  ```bash
  grep "EMBEDDING_BACKEND" .env
  # Recommended (default): EMBEDDING_BACKEND=huggingface (FP16, higher quality)
  # Alternative: EMBEDDING_BACKEND=ollama (Q4_K_M, paper-parity)
  ```

  **Note**: HuggingFace backend requires `make dev-hf` to install dependencies.

### 2.3 Sampling Parameters

**Reference**: GAP-001b/c, [Agent Sampling Registry](../configs/agent-sampling-registry.md)

- [ ] **Temperature is zero** (clinical AI best practice):
  ```bash
  grep "MODEL_TEMPERATURE" .env
  # Should show: MODEL_TEMPERATURE=0.0
  ```

  **Note**: We use temp=0 for all agents. top_k/top_p are not set (irrelevant at temp=0).

### 2.4 Pydantic AI (Structured Validation)

**Reference**: Spec 13 - Enabled by default since 2025-12-26

- [ ] **Pydantic AI is enabled** (recommended for structured output validation):
  ```bash
  grep "PYDANTIC_AI_ENABLED" .env
  # Should show: PYDANTIC_AI_ENABLED=true (or be absent, as true is the default)
  ```

  **What it does**: Adds structured validation + automatic retries (up to 3x) for quantitative scoring, judge metrics, and meta-review. Falls back to legacy parsing on failure.

- [ ] **Verify in config summary**:
  ```bash
  uv run python -c "
  from ai_psychiatrist.config import get_settings
  s = get_settings()
  print(f'Pydantic AI Enabled: {s.pydantic_ai.enabled}')
  print(f'Pydantic AI Retries: {s.pydantic_ai.retries}')
  "
  # Expected: Enabled=True, Retries=3
  ```

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
  from ai_psychiatrist.config import get_settings, resolve_reference_embeddings_path
  s = get_settings()
  p = resolve_reference_embeddings_path(s.data, s.embedding)
  if p.exists():
      data = np.load(str(p))
      # NPZ uses per-participant keys: emb_302, emb_304, etc.
      dim = data[data.files[0]].shape[1]
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

- [ ] **Check for embedding file**:
  ```bash
  ls -lh data/embeddings/*.npz
  ```

  **Default embedding artifact**: `huggingface_qwen3_8b_paper_train_participant_only.npz` (FP16, participant-only transcripts; recommended)
  **Alternative**: `paper_reference_embeddings.npz` (Ollama Q4_K_M, paper-parity)

	  If missing, generate (takes ~65 min for 58 participants):
	  ```bash
	  # Generate HuggingFace FP16 embeddings (recommended, collision-free naming)
	  DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only \
	  uv run python scripts/generate_embeddings.py \
	    --backend huggingface \
	    --split paper-train \
	    --output data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz
	  # Optional (Spec 34): also write per-chunk PHQ-8 item tags sidecar
	  # (recommended for retrieval): add --write-item-tags

	  # Or generate Ollama embeddings (paper-parity)
	  EMBEDDING_BACKEND=ollama uv run python scripts/generate_embeddings.py --split paper-train
	  # Output: data/embeddings/ollama_qwen3_8b_paper_train.npz
	  ```

### 4.2 Verify Embedding Integrity

- [ ] **Embedding file is valid**:
  ```bash
  uv run python -c "
  import numpy as np
  from pathlib import Path
  from ai_psychiatrist.config import get_settings, resolve_reference_embeddings_path

  s = get_settings()
  candidates = [
      resolve_reference_embeddings_path(s.data, s.embedding),
      Path('data/embeddings/reference_embeddings.npz'),
  ]
  for p in candidates:
      if p.exists():
          data = np.load(str(p))
          # NPZ uses per-participant keys: emb_302, emb_304, etc.
          pids = [int(k.split('_')[1]) for k in data.keys()]
          total_chunks = sum(data[k].shape[0] for k in data.keys())
          dim = data[data.files[0]].shape[1]
          print(f'File: {p.name}')
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
  File: <your configured embeddings artifact>
    Participants: 58
    Total chunks: ~7000
    Dimension: 4096
  ```

### 4.3 Sidecar File Check

- [ ] **JSON sidecar exists** (for chunk text) and (optional) tags sidecar:
  ```bash
  uv run python -c "
  from ai_psychiatrist.config import get_settings, resolve_reference_embeddings_path

  s = get_settings()
  npz = resolve_reference_embeddings_path(s.data, s.embedding)
  paths = [
      ('json', npz.with_suffix('.json')),
      ('meta', npz.with_suffix('.meta.json')),
      ('tags', npz.with_suffix('.tags.json')),
  ]
  print(f'NPZ: {npz}')
  for name, path in paths:
      status = 'OK' if path.exists() else 'MISSING'
      print(f'{name}: {path.name} ({status})')
  "
  ```

- `.tags.json` is only required if you set `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`; otherwise it is ignored.

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

**Reference**: BUG-018e, BUG-027

- [ ] **Set generous timeout** for GPU-safe operation:
  ```bash
  grep -E "^(OLLAMA_TIMEOUT_SECONDS|PYDANTIC_AI_TIMEOUT_SECONDS)=" .env
  # Recommended: 600  (10 min, safe default)
  # For slow GPU: 3600 (1 hour, research runs)
  ```

  **Gotcha (BUG-027)**: Pydantic AI timeout is configurable via `PYDANTIC_AI_TIMEOUT_SECONDS`.
  If you set only one of `{OLLAMA_TIMEOUT_SECONDS, PYDANTIC_AI_TIMEOUT_SECONDS}`, Settings syncs the other; if you set both, keep them equal to avoid fallback timeouts.

  **Gotcha**: 6/47 participants (13%) timed out on first run with 300s. Large transcripts (~24KB+) need 600s+ (often 3600s on slow GPUs).

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
  uv run python scripts/create_paper_split.py --verify
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
  from ai_psychiatrist.config import get_settings, resolve_reference_embeddings_path

  s = get_settings()
  npz_path = resolve_reference_embeddings_path(s.data, s.embedding)

  # Load embedding participant IDs from NPZ keys (emb_302, emb_304, etc.)
  emb = np.load(str(npz_path))
  emb_pids = {int(k.split('_')[1]) for k in emb.keys()}
  print(f'Embedding participants: {len(emb_pids)}')
  print(f'Embeddings file: {npz_path}')

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
print(f'Pydantic AI Enabled: {s.pydantic_ai.enabled}')
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
Quantitative Model: gemma3:27b-it-qat  (or gemma3:27b for paper-parity)
Embedding Model: qwen3-embedding:8b
Temperature: 0.0
Keyword Backfill: False
Timeout: 600s  (or higher for research runs)
Pydantic AI Enabled: True

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
| `na_count = 8` for all | MedGemma contamination | Ensure model is Gemma3 (`gemma3:27b-it-qat` or `gemma3:27b`), not MedGemma |
| `No reference embeddings found` | Missing/wrong embeddings | Generate: `scripts/generate_embeddings.py` |
| `Embedding dimension mismatch` | Dimension inconsistency | Regenerate embeddings with correct dimension |
| `0 similar chunks found` | Silent dimension mismatch | Check `EMBEDDING_DIMENSION` matches NPZ |

---

## Phase 11: Post-Run Validation

### 11.1 Output File Created

- [ ] **Results file exists**:
  ```bash
  ls -lt data/outputs/*.json | head -1
  ```

### 11.2 Verify `item_signals` Present (Required for AURC/AUGRC)

**Reference**: Spec 25 - Required for selective prediction evaluation

- [ ] **Output includes `item_signals`** for each participant:
  ```bash
  python3 -c "
  import json
  from pathlib import Path
  f = sorted(Path('data/outputs').glob('*.json'))[-1]
  data = json.loads(f.read_text())
  results = data['experiments'][0]['results']['results']
  success = next((r for r in results if r.get('success')), None)
  if success:
      has_signals = 'item_signals' in success
      print(f'Has item_signals: {has_signals}')
      if has_signals:
          print(f'Signal keys: {list(success[\"item_signals\"].keys())[:3]}...')
      assert has_signals, 'FAIL: Missing item_signals! Re-run with latest code.'
  else:
      print('WARNING: No successful results found')
  "
  ```

  **Gotcha**: Outputs created before 2025-12-27 lack `item_signals`. The AURC/AUGRC
  evaluation script requires this field. Re-run if missing.

### 11.3 Metrics Sanity Check

- [ ] **Coverage is ~50-70%** (few-shot typically higher than zero-shot):
  ```bash
  python3 -c "
  import json
  from pathlib import Path
  f = sorted(Path('data/outputs').glob('*.json'))[-1]
  data = json.loads(f.read_text())
  exp = data['experiments'][0]['results']
  print(f'Mode: {exp.get(\"mode\", \"unknown\")}')
  print(f'Success: {sum(1 for r in exp[\"results\"] if r.get(\"success\"))}')
  print(f'Failed: {sum(1 for r in exp[\"results\"] if not r.get(\"success\"))}')
  "
  ```

  Expected (few-shot):
  - Coverage: ~50-72% (higher than zero-shot due to few-shot examples)
  - MAE: ~0.62-0.90 (varies with coverage - see BUG-029)

### 11.4 Compare to Paper Targets

| Metric | Paper | Our Actual | Notes |
|--------|-------|------------|-------|
| MAE (few-shot) | 0.619 | ~0.86 | Higher coverage = higher MAE (expected) |
| Coverage (few-shot) | ~50% | ~72% | Our system predicts more items |

**Note**: Paper compares MAE at different coverages (invalid per Spec 25). Use AURC/AUGRC for fair comparison.

### 11.5 Run AURC/AUGRC Evaluation (Recommended)

**Reference**: Spec 25 - Proper selective prediction evaluation

- [ ] **Evaluate with risk-coverage metrics**:
  ```bash
  uv run python scripts/evaluate_selective_prediction.py \
    --input data/outputs/<your_output>.json \
    --mode few_shot \
    --seed 42
  ```

- [ ] **Compare zero-shot vs few-shot** (paired analysis):
  ```bash
  uv run python scripts/evaluate_selective_prediction.py \
    --input data/outputs/<zero_shot>.json \
    --input data/outputs/<few_shot>.json \
    --seed 42
  ```

  This computes AURC, AUGRC, and paired Δ with bootstrap CIs - the statistically
  valid way to compare selective prediction systems.

---

## Common Failure Modes Quick Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| All items N/A | MedGemma model | Change to `gemma3:27b` |
| 74% coverage | Backfill ON | Set `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` |
| Timeouts on 13% | Long transcripts | Increase `OLLAMA_TIMEOUT_SECONDS=600` or higher |
| Participant 487 fails | macOS resource fork | Re-extract with `unzip -x '._*'` |
| Config not applying | .env override | Start fresh: `cp .env.example .env` |
| MAE ~4.0 (wrong scale) | Old script | Use current `scripts/reproduce_results.py` |
| No few-shot effect | Missing embeddings | Generate: `scripts/generate_embeddings.py` |
| Silent zero-shot | Dimension mismatch | Check `EMBEDDING_DIMENSION=4096` matches NPZ |
| Wrong participants | AVEC vs paper split | Use `--split paper` for paper methodology |
| Missing `item_signals` | Old output file | Re-run with code from 2025-12-27+ |
| AURC eval fails | No `item_signals` | Re-run reproduction to generate new outputs |
| Embedding hash mismatch | Wrong split used | Regenerate embeddings for paper-train split |

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
make dev-hf     # Install with HuggingFace deps (recommended)
cp .env.example .env

# 2. Pull required Ollama models
# Production-recommended (QAT-quantized, faster):
ollama pull gemma3:27b-it-qat
# Standard Ollama tag (GGUF Q4_K_M):
ollama pull gemma3:27b
# Embedding model:
ollama pull qwen3-embedding:8b

# 3. Create paper ground truth split
uv run python scripts/create_paper_split.py --verify

# 4. Generate embeddings from paper-train (takes ~65 min)
# Default uses HuggingFace FP16 (higher quality)
uv run python scripts/generate_embeddings.py --split paper-train

# 5. Run few-shot reproduction (Pydantic AI enabled by default)
uv run python scripts/reproduce_results.py --split paper --few-shot-only

# 6. (Optional) Compare with zero-shot
uv run python scripts/reproduce_results.py --split paper
```

**Note**: Pydantic AI is enabled by default, providing structured validation with automatic retries. No additional configuration needed.

---

## Related Documentation

- [Zero-Shot Preflight](./preflight-checklist-zero-shot.md) - Simpler, no embeddings
- [Configuration Philosophy](../configs/configuration-philosophy.md) - Why we've moved beyond paper parity
- [Model Registry](../models/model-registry.md) - Model configuration and backend options
