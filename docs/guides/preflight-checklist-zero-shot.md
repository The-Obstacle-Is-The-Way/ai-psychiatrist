# Preflight Checklist: Zero-Shot Reproduction

**Purpose**: Comprehensive pre-run verification for zero-shot paper reproduction
**Last Updated**: 2025-12-27
**Related**: [Few-Shot Checklist](./preflight-checklist-few-shot.md) | [Paper Parity Guide](./paper-parity-guide.md)

---

## Overview

This checklist prevents reproduction failures by verifying ALL known gotchas before running. Use this **every time** you start a zero-shot reproduction run.

Zero-shot mode uses NO reference embeddings - the model scores symptoms from transcript alone.

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

- [ ] **Required model pulled**: `ollama list | grep -E "gemma3:27b|gemma3:27b-it-qat"`
  - If missing:
    ```bash
    # Production-recommended (QAT-quantized, faster):
    ollama pull gemma3:27b-it-qat
    # Standard Ollama tag (GGUF Q4_K_M):
    ollama pull gemma3:27b
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

### 2.2 Sampling Parameters

**Reference**: GAP-001b/c, [Agent Sampling Registry](../reference/agent-sampling-registry.md)

- [ ] **Temperature is zero** (clinical AI best practice):
  ```bash
  grep "MODEL_TEMPERATURE" .env
  # Should show: MODEL_TEMPERATURE=0.0
  ```

  **Note**: We use temp=0 for all agents. top_k/top_p are not set (irrelevant at temp=0).

### 2.3 Pydantic AI (Structured Validation)

**Reference**: Spec 13 - Enabled by default since 2025-12-26

- [ ] **Pydantic AI is enabled** (recommended for structured output validation):
  ```bash
  grep "PYDANTIC_AI_ENABLED" .env
  # Should show: PYDANTIC_AI_ENABLED=true (or be absent, as true is the default)
  ```

  **What it does**: Adds structured validation + automatic retries (up to 3x) for quantitative scoring, judge metrics, and meta-review. Falls back to legacy parsing on failure.

---

## Phase 3: Quantitative Settings (SPEC-003)

### 3.1 Keyword Backfill Toggle

**Reference**: SPEC-003, Coverage Investigation

- [ ] **Backfill is DISABLED** (paper parity = ~50% coverage):
  ```bash
  grep "QUANTITATIVE_ENABLE_KEYWORD_BACKFILL" .env
  # MUST show: QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
  ```

  **Gotcha**: Backfill ON = ~74% coverage, which diverges from paper's ~50%.

### 3.2 N/A Reason Tracking

- [ ] **N/A tracking enabled** (for debugging):
  ```bash
  grep "QUANTITATIVE_TRACK_NA_REASONS" .env
  # Should show: QUANTITATIVE_TRACK_NA_REASONS=true
  ```

---

## Phase 4: Data Integrity

### 4.1 Transcripts Present

- [ ] **Transcripts directory exists**:
  ```bash
  ls data/transcripts/ | wc -l
  # Should show ~189 (or your participant count)
  ```

### 4.2 Participant 487 Validation

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

### 4.3 Ground Truth Labels

- [ ] **AVEC2017 labels exist** (for item-level MAE):
  ```bash
  ls data/*_split_Depression_AVEC2017.csv
  # Should show: dev_split, train_split, test_split files
  ```

---

## Phase 5: Timeout Configuration

### 5.1 Timeout Setting

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

### 5.2 Check for Long Transcripts

- [ ] **Identify large transcripts** that may timeout:
  ```bash
  find data/transcripts -name "*TRANSCRIPT.csv" -exec wc -c {} + | sort -n | tail -10
  # Note any files > 25KB - these may need extra timeout
  ```

---

## Phase 6: Paper Split (If Using Paper Methodology)

### 6.1 Create or Verify Splits

- [ ] **Paper splits exist OR will be created**:
  ```bash
  ls data/paper_splits/
  # If empty or missing:
  uv run python scripts/create_paper_split.py --verify
  ```

### 6.2 Verify Split Sizes

**Reference**: Paper Section 2.4.1

- [ ] **Split sizes match paper** (58/43/41):
  ```bash
  wc -l data/paper_splits/paper_split_*.csv
  # Should show: 59 train (58+header), 44 val, 42 test
  ```

---

## Phase 7: Pre-Run Verification

### 7.1 Quick Sanity Check

- [ ] **Run linter**: `make lint`
- [ ] **Run type checker**: `make typecheck`
- [ ] **Run unit tests**: `make test-unit`

### 7.2 Configuration Summary Check

Run this to dump your effective configuration:
```bash
uv run python -c "
from ai_psychiatrist.config import get_settings
s = get_settings()
print('=== CRITICAL SETTINGS ===')
print(f'Quantitative Model: {s.model.quantitative_model}')
print(f'Temperature: {s.model.temperature}')
print(f'Keyword Backfill: {s.quantitative.enable_keyword_backfill}')
print(f'Timeout: {s.ollama.timeout_seconds}s')
print(f'Pydantic AI Enabled: {s.pydantic_ai.enabled}')
print(f'Embedding Dimension: {s.embedding.dimension}')
"
```

Expected output:
```text
=== CRITICAL SETTINGS ===
Quantitative Model: gemma3:27b-it-qat  (or gemma3:27b)
Temperature: 0.0
Keyword Backfill: False
Timeout: 600s  (or higher for research runs)
Pydantic AI Enabled: True
Embedding Dimension: 4096
```

---

## Phase 8: Execute Zero-Shot Run

### 8.1 Use tmux for Long-Running Processes

**CRITICAL**: Reproduction runs take ~5 min/participant. Use tmux to prevent losing progress if your terminal disconnects.

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

### 8.2 Run Command

```bash
# Zero-shot on AVEC dev split (has per-item labels)
uv run python scripts/reproduce_results.py --split dev

# Zero-shot on paper test split
uv run python scripts/reproduce_results.py --split paper
```

### 8.3 Monitor for Issues

Watch for these log patterns:

| Log Pattern | Issue | Action |
|-------------|-------|--------|
| `LLM request timed out` | Transcript too long | Increase `OLLAMA_TIMEOUT_SECONDS` |
| `Failed to parse evidence JSON` | LLM output malformed | Keyword backfill mitigates; check model |
| `na_count = 8` for all | MedGemma contamination | Ensure model is Gemma3 (`gemma3:27b-it-qat` or `gemma3:27b`), not MedGemma |

---

## Phase 9: Post-Run Validation

### 9.1 Output File Created

- [ ] **Results file exists**:
  ```bash
  ls -lt data/outputs/*.json | head -1
  ```

### 9.2 Verify `item_signals` Present (Required for AURC/AUGRC)

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

### 9.3 Metrics Sanity Check

- [ ] **Coverage is ~50-60%** (paper parity with backfill OFF):
  ```bash
  # Check the output JSON for coverage metrics
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

  Expected (zero-shot, paper parity):
  - Coverage: ~50-60% (with backfill OFF)
  - MAE: ~0.72-0.80 (paper reports 0.796)

### 9.4 Run AURC/AUGRC Evaluation (Optional)

**Reference**: Spec 25 - Proper selective prediction evaluation

- [ ] **Evaluate with risk-coverage metrics**:
  ```bash
  uv run python scripts/evaluate_selective_prediction.py \
    --input data/outputs/<your_output>.json \
    --mode zero_shot \
    --seed 42
  ```

  This computes AURC, AUGRC, MAE@coverage with bootstrap CIs - the statistically
  valid way to evaluate selective prediction systems.

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
| Missing `item_signals` | Old output file | Re-run with code from 2025-12-27+ |
| AURC eval fails | No `item_signals` | Re-run reproduction to generate new outputs |

---

## Checklist Complete?

If ALL items are checked:
1. You're ready to run zero-shot reproduction
2. Expected runtime: ~5 min/participant (varies by hardware)
3. Paper zero-shot MAE target: 0.796

**Remember**: The paper acknowledges stochasticity - results within ±0.1 MAE are considered consistent.

---

## Related Documentation

- [Few-Shot Preflight](./preflight-checklist-few-shot.md) - For few-shot runs (includes embedding setup)
- [Paper Parity Guide](./paper-parity-guide.md) - Full paper methodology
- [Model Registry](../models/model-registry.md) - Model configuration and backend options
