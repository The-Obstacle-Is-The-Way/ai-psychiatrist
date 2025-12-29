# Complete Run History & Statistical Analysis

**Purpose**: Comprehensive record of all reproduction runs, code changes, and statistical analyses for posterity.

**Last Updated**: 2025-12-29

---

## Quick Reference: Current Best Results

| Mode | AURC | AUGRC | Cmax | MAE_w |
|------|------|-------|------|-------|
| **Zero-shot** | **0.134** [0.094-0.176] | **0.037** [0.024-0.053] | 55.5% | 0.698 |
| Few-shot | 0.193 [0.142-0.244] | 0.065 [0.043-0.091] | 70.1% | 0.774 |

**Winner**: Zero-shot (paired bootstrap delta CI excludes 0)

**Note**: `Cmax` is the max coverage in the risk–coverage curve (counts participants with 8/8 N/A as 0 coverage). `MAE_w` is computed over evaluated subjects only.

---

## Why AURC/AUGRC Instead of MAE?

**MAE comparisons are not coverage-adjusted when coverages differ.**

- Zero-shot `Cmax`: 55.5%
- Few-shot `Cmax`: 70.1%

Few-shot predicts on ~15% more items. Those additional items are inherently harder cases that zero-shot abstained from. Comparing raw MAE is like comparing a surgeon who only takes easy cases vs one who takes hard cases.

**AURC/AUGRC integrate over the entire risk-coverage curve**, providing a fair comparison regardless of coverage differences.

See: `docs/reference/statistical-methodology-aurc-augrc.md`

---

## Run Timeline (Chronological)

### Run 1: Dec 26, 2025 - Initial Validated Runs

**Files**:
- `zero_shot_paper_backfill-off_20251226_201946.json`
- `few_shot_paper_backfill-off_20251227_000125.json`

**Git Commits**: Various (`5b8f588`, `f6d2653`)

**Code State**:
- Pre-Spec 31/32 (old reference format)
- 8 separate `<Reference Examples>` blocks per PHQ-8 item
- Per-item headers like `[Sleep]`
- XML-style closing tags `</Reference Examples>`
- Empty items showed "No valid evidence found"

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w |
|------|------|-------|------|-------|
| Zero-shot | ~0.134 | ~0.037 | 55.5% | 0.698 |
| Few-shot | ~0.21 | ~0.07 | 71.6% | 0.860 |

**Notes**: Initial baseline. Few-shot significantly worse than zero-shot.

---

### Run 2: Dec 27, 2025 - Pre-Spec 31/32 Full Run

**File**: `paper_test_full_run_20251228.json` (filename misleading - actually Dec 27)

**Git Commit**: `0a98662`

**Timestamp**: 2025-12-27T23:10:45

**Code State**: Same as Run 1 (pre-Spec 31/32)

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | MAE_item |
|------|------|-------|------|-------|----------|
| Zero-shot | 0.134 | 0.037 | 55.5% | 0.698 | 0.717 |
| Few-shot | 0.214 | 0.074 | 71.9% | 0.804 | N/A |

**Statistical Analysis**: AURC computed via `scripts/evaluate_selective_prediction.py`

---

### Run 3: Dec 29, 2025 - Post-Spec 31/32 (Paper-Parity Format)

**File**: `both_paper-test_backfill-off_20251229_003543.json`

**Git Commit**: `7d54d98`

**Timestamp**: 2025-12-28T21:39:32

**Code Changes (Spec 31/32)**:
- Single unified `<Reference Examples>` block
- Inline labels: `(PHQ8_Sleep Score: 2)` instead of `(Score: 2)`
- Empty items skipped entirely (no per-item blocks)
- Same opening/closing tag: `<Reference Examples>` (not XML-style)

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | MAE_item | MAE_subj |
|------|------|-------|------|-------|----------|----------|
| Zero-shot | 0.134 | 0.037 | 55.5% | 0.698 | 0.717 | 0.640 |
| Few-shot | 0.193 | 0.065 | 70.1% | 0.774 | 0.762 | 0.712 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.094, 0.176] | [0.024, 0.053] | [0.473, 0.640] |
| Few-shot | [0.142, 0.244] | [0.043, 0.091] | [0.604, 0.799] |

**Statistical Analysis**:
- Computed 2025-12-29 via `scripts/evaluate_selective_prediction.py --seed 42`
- Metrics files: `selective_prediction_metrics_20251229T164344Z.json` (zero-shot), `selective_prediction_metrics_20251229T164403Z.json` (few-shot)
- Paired comparison: `selective_prediction_metrics_20251229T1644_paired.json` (ΔAURC = +0.058 [0.016, 0.107], few-shot − zero-shot)

---

### Run 4: Dec 29, 2025 - Spec 33 Development Snapshot (Pre-merge)

**File**: `both_paper-test_backfill-off_20251229_173727.json`

**Git Commit**: `5e62455` (pre-merge dev commit; not on `main`)

**Timestamp**: 2025-12-29T14:41:44

**Code Changes (Spec 33)**:
- Retrieval quality guardrails (similarity threshold + per-item reference budget)
- XML-style closing tag: `</Reference Examples>` (deviates from notebook tag mirroring)

**Results** (single-run metrics; note different included-N due to one zero-shot failure):

| Mode | AURC | AUGRC | Cmax | MAE_w | N_included (AURC) |
|------|------|-------|------|-------|-------------------|
| Zero-shot | 0.138 | 0.039 | 56.9% | 0.698 | 40 |
| Few-shot | 0.192 | 0.058 | 65.5% | 0.777 | 41 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI | N_included (AURC) |
|------|---------|----------|---------|-------------------|
| Zero-shot | [0.097, 0.180] | [0.025, 0.055] | [0.491, 0.650] | 40 |
| Few-shot | [0.144, 0.243] | [0.039, 0.081] | [0.555, 0.753] | 41 |

**Statistical Analysis**:
- Computed 2025-12-29 via `scripts/evaluate_selective_prediction.py --seed 42`
- Metrics files: `selective_prediction_metrics_20251229T231237Z.json` (zero-shot), `selective_prediction_metrics_20251229T231302Z.json` (few-shot)
- Paired comparison (overlap N=40 due to one zero-shot failure): `selective_prediction_metrics_20251229T233314Z.json` (ΔAURC = +0.058 [0.010, 0.109], few-shot − zero-shot)

**Note**: This was a pre-merge development snapshot. The fully merged Spec 33 + Spec 34 codebase has not been rerun yet.

---

## Spec 31/32 Impact Analysis

### What Changed

| Aspect | Before (Old Format) | After (Spec 31/32) |
|--------|---------------------|---------------------|
| Block structure | 8 separate blocks | 1 unified block |
| Item labels | `[Sleep]` header | `(PHQ8_Sleep Score: X)` inline |
| Empty items | "No valid evidence found" | Omitted entirely |
| Closing tag | `</Reference Examples>` | `<Reference Examples>` |

### Impact on Metrics

| Metric | Pre-Spec 31 | Post-Spec 31 | Delta | % Change |
|--------|-------------|--------------|-------|----------|
| **Zero-shot AURC** | 0.134 | 0.134 | 0 | 0% |
| **Zero-shot AUGRC** | 0.037 | 0.037 | 0 | 0% |
| **Few-shot AURC** | 0.214 | 0.193 | -0.021 | **-10%** |
| **Few-shot AUGRC** | 0.074 | 0.065 | -0.009 | **-12%** |
| Few-shot MAE_w | 0.804 | 0.774 | -0.030 | -3.7% |
| Few-shot Cmax | 71.9% | 70.1% | -1.8% | -2.5% |

### Interpretation

1. **Zero-shot unchanged**: Expected - doesn't use reference examples
2. **Few-shot improved 10-12%**: Paper-parity format helps
3. **Gap remains ~30%**: Zero-shot still significantly better (0.134 vs 0.193)
4. **Paired bootstrap delta excludes 0**: Statistically significant difference at α=0.05

---

## Key Findings

### 1. Zero-Shot Beats Few-Shot (Counterintuitive)

The paper claims few-shot beats zero-shot. Our reproduction shows the opposite:

| Metric | Paper Claim | Our Result |
|--------|-------------|------------|
| Better mode | Few-shot | **Zero-shot** |
| Few-shot MAE | 0.619 | 0.774 |
| Zero-shot MAE | 0.796 | 0.698 |

**Possible explanations** (to be tested with Specs 33-36):
1. Reference example quality issues
2. Embedding similarity matches topic, not severity
3. Low-similarity references inject noise
4. Model overconfidence with few-shot

### 2. Paper's MAE Comparison Was Not Coverage-Adjusted

The paper compared MAE at different coverages without analyzing the risk–coverage tradeoff. MAE alone does not establish dominance when abstention rates differ.

### 3. Formatting Matters But Isn't Everything

Spec 31/32 improved few-shot by ~10%, proving formatting matters. But the gap to zero-shot remains, suggesting retrieval quality is the bigger issue.

---

## Pending Work

### Specs 33-36: Retrieval Quality Fixes

| Spec | Description | Status | Hypothesis |
|------|-------------|--------|------------|
| 33 | Similarity threshold + context budget | ✅ Implemented (merged) | Filter low-quality references |
| 34 | Item-tagged reference embeddings | ✅ Implemented (not yet rerun) | Improve semantic matching |
| 35 | Offline chunk-level PHQ-8 scoring | PLANNED | Better ground truth for retrieval |
| 36 | CRAG reference validation | PLANNED | LLM-based relevance filtering |

**Next action**: Run a fresh ablation with the current merged codebase (Spec 33 + Spec 34) and update this run history.

---

## Reproduction Commands

### Run Evaluation

```bash
# Full reproduction (both modes)
uv run python scripts/reproduce_results.py --split paper-test

# Zero-shot only
uv run python scripts/reproduce_results.py --split paper-test --zero-shot-only

# Few-shot only (requires embeddings)
uv run python scripts/reproduce_results.py --split paper-test --few-shot-only
```

### Compute AURC/AUGRC

```bash
# Single mode (writes `data/outputs/selective_prediction_metrics_*.json`)
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/YOUR_OUTPUT.json \
  --mode zero_shot \
  --seed 42

# Paired comparison (recommended): pass the same run file twice with different modes
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/YOUR_OUTPUT.json \
  --mode zero_shot \
  --input data/outputs/YOUR_OUTPUT.json \
  --mode few_shot \
  --seed 42

# Or run separately for each mode
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/YOUR_OUTPUT.json \
  --mode few_shot \
  --seed 42
```

### Generate Embeddings (for few-shot)

```bash
uv run python scripts/generate_embeddings.py --split paper-train
```

---

## File Locations

| Type | Path |
|------|------|
| Run outputs | `data/outputs/*.json` |
| AURC metrics | `data/outputs/selective_prediction_metrics_*.json` |
| Run log (gitignored) | `data/outputs/RUN_LOG.md` |
| Embeddings | `data/embeddings/*.npz` |
| Experiment registry | `data/experiments/registry.yaml` |

---

## References

- Statistical methodology: `docs/reference/statistical-methodology-aurc-augrc.md`
- Spec 31 (format): `docs/archive/specs/31-paper-parity-reference-examples-format.md`
- Spec 32 (audit): `docs/archive/specs/32-few-shot-retrieval-diagnostics.md`
- Spec 33 (guardrails): `docs/archive/specs/33-retrieval-quality-guardrails.md`
- Spec 34 (item tagging): `docs/specs/34-item-tagged-reference-embeddings.md`
- Paper analysis: `docs/paper-reproduction-analysis.md`
