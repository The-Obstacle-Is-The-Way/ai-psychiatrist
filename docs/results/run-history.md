# Complete Run History & Statistical Analysis

**Purpose**: Comprehensive record of all reproduction runs, code changes, and statistical analyses for posterity.

**Last Updated**: 2026-01-04

---

## ⚠️ CRITICAL: Run Integrity Warning (2026-01-03)

### Silent Fallback Bug (ANALYSIS-026)

A critical bug was discovered and fixed on 2026-01-03 where `_extract_evidence()` would **silently return `{}` on JSON parse failure** instead of raising an exception.

**Impact on Mode Isolation**:

- Few-shot mode with empty evidence `{}` → no reference bundle → effectively zero-shot
- This violated the independence of zero-shot and few-shot as research methodologies
- Published results claiming "few-shot" could have been partially zero-shot

**Status by Run**:

| Run | Code Version | Affected? | Notes |
|-----|--------------|-----------|-------|
| Run 1-9 | Pre-fix | **Unknown** | Bug was SILENT - no way to know without re-running |
| Run 10 | Pre-fix (git dirty) | **Yes** | Completed but invalid (zero-shot partial, few-shot failed entirely) |
| Future runs | Post-fix | No | Will fail loudly if JSON parsing fails |

**Why we can't be certain about Run 1-9**:

- The bug only triggers if LLM returns malformed JSON
- If LLM always returned valid JSON, bug never triggered
- Results look plausible (few-shot beats zero-shot on MAE with chunk scoring)
- But we have NO PROOF the bug never triggered

**Fix Applied**: Commit on 2026-01-03

- `_extract_evidence()` now raises `json.JSONDecodeError` on failure
- Uses `format="json"` for grammar-level JSON constraint
- All parsers use canonical `parse_llm_json()` function

**Recommendation**: For publication-quality results, consider re-running with post-fix code.

See: `docs/_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md`

---

## Quick Reference: Current Best Results

| Mode | AURC | AUGRC | Cmax | Run |
|------|------|-------|------|-----|
| **Zero-shot** | **0.134** [0.094-0.176] | **0.031** [0.022-0.043] | 48.8% | Run 8 |
| **Few-shot** | **0.125** [0.099-0.151] | **0.031** [0.022-0.041] | 50.9% | Run 8 |

**Winner (lowest AURC)**: Few-shot (Run 8), but with substantially lower Cmax than earlier runs.

**Spec 046 Finding (Run 9)**: Using `retrieval_similarity_mean` as confidence signal improves AURC by 5.4% (0.1351 → 0.1278) compared to evidence-count-only.

**AUGRC Improvement Suite (Specs 048–052, implemented 2026-01-03)**: New confidence variants and calibration utilities are available (verbalized confidence, supervised calibrator, token-level CSFs, consistency mode, and excess metrics). Baseline re-evaluation (few-shot, `confidence=llm`): `data/outputs/selective_prediction_metrics_20260103T070506Z.json` (AURC 0.1351, AUGRC 0.0346, Cmax 0.5305).

**Note**: Run 8 has much lower coverage ceiling (`Cmax` ~51%) than Run 7 (`Cmax` ~66%). Interpret AURC alongside Cmax.

**Note**: `Cmax` is the max coverage in the risk–coverage curve (counts participants with 8/8 N/A as 0 coverage). `MAE_w` is computed over evaluated subjects only.

---

## Why AURC/AUGRC Instead of MAE?

**MAE comparisons are not coverage-adjusted when coverages differ.**

- Run 7 `Cmax`: zero-shot 56.9%, few-shot 65.9%
- Run 8 `Cmax`: zero-shot 48.8%, few-shot 50.9%

When one system predicts on more items, those additional items are inherently harder cases that another system abstained from. Comparing raw MAE without a coverage-adjusted metric is like comparing a surgeon who only takes easy cases vs one who takes hard cases.

**AURC/AUGRC integrate over the entire risk-coverage curve**, providing a fair comparison regardless of coverage differences.

See: `docs/statistics/statistical-methodology-aurc-augrc.md`

---

## Run Timeline (Chronological)

### Run 1: Dec 26, 2025 - Initial Validated Runs

**Artifacts**: Not retained in this repo snapshot (early outputs used different naming and were not committed). Treat this run as historical context only; later runs include stored JSON artifacts under `data/outputs/`.

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

### Run 3: Dec 29, 2025 - Post-Spec 31/32 (Legacy Prompt Format)

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

**Note on comparability**: The paired comparison recomputes both modes on the overlap only (N=40). On that overlap, few-shot is slightly worse than the single-mode table above (AURC ≈ 0.196, AUGRC ≈ 0.060) because the dropped participant only affects the paired analysis, not the standalone few-shot evaluation.

**Note**: This was a pre-merge development snapshot. See Run 5 for the clean, post-merge Spec 33+34 ablation run.

---

### Run 4b: Dec 30, 2025 - Post-Spec 34 Regression (Query Embedding Timeouts)

**File**: `both_paper_backfill-off_20251230_053108.json`

**Git Commit**: `be35e35` (dirty)

**Timestamp**: 2025-12-29T23:34:42

**What went wrong**:
- Few-shot had **9/41 failures (22%)**, all `"LLM request timed out after 120s"`.
- Runtime roughly doubled vs the expected ~95 minutes.

**Root cause (since fixed)**:
- Spec 37 was required (batch query embedding + configurable query embedding timeout).

**Results** (includes failures; do not treat as a valid baseline):

| Mode | AURC | AUGRC | Cmax | MAE_w | N_included (AURC) | Failed |
|------|------|-------|------|-------|-------------------|--------|
| Zero-shot | 0.138 | 0.039 | 56.9% | 0.698 | 40 | 1 |
| Few-shot | 0.163 | 0.037 | 53.5% | 0.745 | 32 | 9 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.097, 0.180] | [0.025, 0.055] | [0.491, 0.650] |
| Few-shot | [0.098, 0.217] | [0.020, 0.060] | [0.426, 0.648] |

**Paired comparison** (overlap N=31 due to failures): ΔAURC = +0.037 [-0.028, +0.087] (few-shot − zero-shot).

---

### Run 5: Dec 30, 2025 - Post-Spec 33+34 (Full Ablation)

**File**: `both_paper-test_backfill-off_20251230_230349.json`

**Git Commit**: `36995f0` (clean)

**Timestamp**: 2025-12-30T20:27:38

**Code Changes (Spec 33+34)**:
- Spec 33: Retrieval quality guardrails (min_similarity=0.3, max_chars_per_item=500)
- Spec 34: Item-tag filtering (only retrieve domain-matched chunks)
- Spec 35/36: NOT enabled (chunk scores file doesn't exist)

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | N_included |
|------|------|-------|------|-------|------------|
| Zero-shot | 0.138 | 0.039 | 56.9% | 0.698 | 40 |
| Few-shot | 0.213 | 0.073 | 71.0% | 0.807 | 41 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.097, 0.180] | [0.025, 0.055] | [0.491, 0.650] |
| Few-shot | [0.153, 0.276] | [0.047, 0.103] | [0.610, 0.805] |

**Statistical Analysis**:
- Computed 2025-12-30 via `scripts/evaluate_selective_prediction.py --seed 42`
- Metrics files: `selective_prediction_metrics_run5_zero_shot.json`, `selective_prediction_metrics_run5_few_shot.json`

**Comparison vs Run 3 (Spec 31/32 baseline)**:

| Metric | Run 3 | Run 5 | Delta | % Change |
|--------|-------|-------|-------|----------|
| few_shot AURC | 0.193 | 0.213 | +0.020 | **+10% (worse)** |
| few_shot AUGRC | 0.065 | 0.073 | +0.008 | **+12% (worse)** |
| zero_shot AURC | 0.134 | 0.138 | +0.004 | +3% (noise) |

**Key Finding**: Spec 33+34 did NOT improve few-shot. Performance regressed ~10%.

**Interpretation**: Domain filtering (Spec 34) and quality guardrails (Spec 33) cannot fix the fundamental chunk-scoring problem documented in `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md`. Chunks still have participant-level scores, not chunk-specific scores. Filtering by domain helps retrieval precision but doesn't fix the misleading score labels.

**Conclusion**: Spec 35 (chunk-level scoring) is required before further ablations are meaningful.

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
2. **Few-shot improved 10-12%**: Legacy prompt format helps
3. **Gap remains ~30%**: Zero-shot still significantly better (0.134 vs 0.193)
4. **Paired bootstrap delta excludes 0**: Statistically significant difference at α=0.05

---

## Key Findings

### 1. Few-Shot vs Zero-Shot (Paper Claim)

The paper claims few-shot beats zero-shot (by item-level MAE).

**Update (Run 8)**: With participant-only transcript preprocessing + chunk scoring enabled, few-shot matches the paper’s reported MAE_item and slightly beats it:

| Metric | Paper (reported) | Run 8 (participant-only) |
|--------|------------------|--------------------------|
| Better mode (by MAE_item) | Few-shot | **Few-shot** |
| Few-shot MAE_item | 0.619 | 0.609 |
| Zero-shot MAE_item | 0.796 | 0.776 |

**Note**: Earlier runs (Run 3 / Run 7) still showed zero-shot as better on AURC due to the confidence/coverage tradeoff; Run 8 changes the retrieval setting but lowers Cmax substantially.

**Possible explanations** (partially addressed by Specs 33-35 and transcript preprocessing):
1. Reference example quality issues
2. Embedding similarity matches topic, not severity
3. Low-similarity references inject noise
4. Model overconfidence with few-shot

### 2. Paper's MAE Comparison Was Not Coverage-Adjusted

The paper compared MAE at different coverages without analyzing the risk–coverage tradeoff. MAE alone does not establish dominance when abstention rates differ.

### 3. Formatting Matters But Isn't Everything

Spec 31/32 improved few-shot by ~10%, proving formatting matters. Retrieval quality still dominates: chunk scoring (Spec 35) and participant-only transcripts (Run 8) substantially change outcomes, but coverage/confidence tradeoffs remain.

---

## Pending Work

### Specs 33-36: Retrieval Quality Fixes

| Spec | Description | Status | Result |
|------|-------------|--------|--------|
| 33 | Similarity threshold + context budget | ✅ Implemented + tested | No improvement (Run 5) |
| 34 | Item-tagged reference embeddings | ✅ Implemented + tested | No improvement (Run 5) |
| 35 | Offline chunk-level PHQ-8 scoring | ✅ Implemented + tested | **29% improvement (Run 7)** |
| 36 | CRAG reference validation | ✅ Implemented (optional) | Pending ablation (runtime cost) |

**Run 5 Conclusion**: Spec 33+34 alone did not improve few-shot.

**Run 7 Conclusion**: Spec 35 chunk-level scoring improved few-shot AURC by 29% (0.213 → 0.151). Gap to zero-shot closed to 9% (CIs overlap).

**Run 8 Conclusion**: Participant-only transcript preprocessing reaches paper MAE_item parity, but reduces `Cmax` substantially; next work is improving confidence signals for AURC/AUGRC (Spec 046: `docs/_specs/spec-046-selective-prediction-confidence-signals.md`) and then revisiting coverage.

---

### Run 6: Dec 31, 2025 - Spec 35 Chunk Scoring Preprocessing

**Log File**: `data/outputs/run6_spec35_20251231_122458.log`

**Purpose**: Generate chunk-level PHQ-8 scores (Spec 35 preprocessing step)

**Configuration**:
- Embeddings: `ollama_qwen3_8b_paper_train.npz`
- Scorer model: `gemma3:27b-it-qat`
- Backend: Ollama
- Temperature: 0.0

**Output**: `data/embeddings/ollama_qwen3_8b_paper_train.chunk_scores.json`

**Notes**: This was a preprocessing run to generate chunk scores, not an evaluation run. See Run 7 for the subsequent evaluation.

---

### Run 7: Jan 1, 2026 - Post-Spec 35 Chunk Scoring (Full Run)

**File**: `both_paper-test_backfill-off_20260101_111354.json`

**Git Commit**: Current `dev` branch

**Timestamp**: 2026-01-01T11:13:54

**Code State**:
- Spec 33: Retrieval quality guardrails ✅
- Spec 34: Item-tag filtering ✅
- Spec 35: Chunk-level scoring ✅ (`EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`)
- Spec 37: Batch query embedding ✅

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | MAE_item | MAE_subj | N_included | Failed |
|------|------|-------|------|-------|----------|----------|------------|--------|
| Zero-shot | 0.138 | 0.039 | 56.9% | 0.698 | 0.717 | 0.640 | 40 | 1 |
| Few-shot | 0.151 | 0.048 | 65.9% | 0.639 | 0.636 | 0.606 | 41 | 0 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.097, 0.180] | [0.025, 0.055] | [0.491, 0.650] |
| Few-shot | [0.109, 0.194] | [0.033, 0.065] | [0.570, 0.747] |

**Statistical Analysis**:
- Computed 2026-01-01 via `scripts/evaluate_selective_prediction.py --seed 42`
- Metrics files: `selective_prediction_metrics_20260101T165303Z.json` (zero-shot), `selective_prediction_metrics_20260101T165328Z.json` (few-shot)

**Known Issue**: Participant 339 failed in zero-shot mode due to JSON parsing error (missing comma). See [GitHub Issue #84](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/84).

**Comparison vs Run 5**:

| Metric | Run 5 | Run 7 | Delta | % Change |
|--------|-------|-------|-------|----------|
| few_shot AURC | 0.213 | 0.151 | -0.062 | **-29% (better)** |
| few_shot AUGRC | 0.073 | 0.048 | -0.025 | **-34% (better)** |
| zero_shot AURC | 0.138 | 0.138 | 0.000 | 0% (unchanged) |

**Key Finding**: With Spec 35 chunk-level scoring enabled, few-shot improved 29% on AURC vs Run 5. Few-shot now has **better MAE** (0.639 vs 0.698) but AURC is still slightly worse due to confidence calibration.

**Interpretation**: Spec 35 significantly improved few-shot performance. The remaining gap is now within statistical noise (CIs overlap). The next lever was participant-only transcript preprocessing (implemented in Run 8).

---

### Run 8: Jan 2, 2026 - Participant-Only Transcript Preprocessing (Full Run)

**File**: `both_paper-test_backfill-off_20260102_065249.json`

**Log**: `repro_post_preprocessing_20260101_183533.log`

**Run ID**: `19b42478`

**Git Commit**: `1b48d7a` (dirty)

**Timestamp**: 2026-01-02T04:22:43

**Code State**:
- Spec 33: Retrieval quality guardrails ✅
- Spec 34: Item-tag filtering ✅
- Spec 35: Chunk-level scoring ✅ (`EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`)
- Spec 37: Batch query embedding ✅
- Transcript preprocessing: participant-only turns ✅ (`data/transcripts_participant_only/`)

**Reference Artifacts**:
- Few-shot embeddings: `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz`
- Chunk scores sidecar: `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json` (loaded; train participants=58)

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | MAE_item | MAE_subj | N_included | Failed |
|------|------|-------|------|-------|----------|----------|------------|--------|
| Zero-shot | 0.141 | 0.031 | 48.8% | 0.744 | 0.776 | 0.736 | 41 | 0 |
| Few-shot | 0.125 | 0.031 | 50.9% | 0.706 | 0.609 | 0.688 | 40 | 1 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.108, 0.174] | [0.022, 0.043] | [0.412, 0.567] |
| Few-shot | [0.099, 0.151] | [0.022, 0.041] | [0.447, 0.575] |

**Statistical Analysis**:
- Computed 2026-01-02 via `scripts/evaluate_selective_prediction.py --loss abs_norm --seed 42`
- Metrics files: `selective_prediction_metrics_20260102T132843Z.json` (zero-shot), `selective_prediction_metrics_20260102T132902Z.json` (few-shot)
- Paired comparison (overlap N=40; `--intersection-only`): `selective_prediction_metrics_20260102T132930Z_paired.json` (ΔAURC = -0.020 [-0.053, +0.014], few-shot − zero-shot)

**Paper MAE comparison (MAE_item)**:
- Zero-shot: `0.776` vs paper `0.796` (better)
- Few-shot: `0.609` vs paper `0.619` (better)

**Interpretation (first principles)**:
- **Accuracy vs abstention**: In Run 8, both modes abstain at similar rates (`Cmax` ~49% vs ~51%), so the large MAE_item gap (0.776 → 0.609) is less likely to be an artifact of one mode simply “skipping harder items”.
- **Calibration unchanged**: AURC/AUGRC CIs overlap, and the paired ΔAURC CI includes 0. This suggests few-shot improves *scores on predicted items* but does not materially improve the model’s *ranking of confidence / abstention decisions*.
- **Practical takeaway**: If the goal is “predict more items correctly”, retrieval helps; if the goal is “know when not to predict”, focus on evidence availability + confidence signals (e.g., evaluate `participant_qa`, tune thresholds, improve confidence estimation).

**Known Issues**:
- Few-shot had 1/41 participant failure (PID 383): `Exceeded maximum retries (3) for output validation`.
- Zero-shot excluded 1/41 participant from MAE aggregation due to 8/8 N/A (counted as 0 coverage for Cmax).

---

### Run 9: Jan 2-3, 2026 - Spec 046 Confidence Signals Ablation

**File**: `both_paper-test_backfill-off_20260102_215843.json`

**Log**: `data/outputs/run9_spec046_20260102_181114.log`

**Git Commit**: Post Spec 046 + 047 (retrieval signals + keyword backfill removal)

**Timestamp**: 2026-01-03T02:58:43

**Code State**:
- Spec 33-35: Full retrieval stack ✅
- Spec 37: Batch query embedding ✅
- Spec 046: Retrieval similarity fields ✅
- Spec 047: Keyword backfill removal ✅

**Results**:

| Mode | AURC | AUGRC | Cmax | MAE_w | MAE_item | N_included |
|------|------|-------|------|-------|----------|------------|
| Zero-shot | 0.144 | 0.032 | 48.8% | 0.744 | 0.776 | 40 |
| Few-shot | 0.135 | 0.035 | 53.0% | 0.718 | 0.662 | 41 |

**95% Bootstrap CIs** (10,000 resamples, participant-level):

| Mode | AURC CI | AUGRC CI | Cmax CI |
|------|---------|----------|---------|
| Zero-shot | [0.110, 0.178] | [0.022, 0.045] | [0.412, 0.567] |
| Few-shot | [0.107, 0.165] | [0.025, 0.047] | [0.460, 0.604] |

**Spec 046 Confidence Signal Ablation (few-shot)**:

| Confidence Signal | AURC | AUGRC | vs llm baseline |
|-------------------|------|-------|-----------------|
| `llm` (evidence count) | 0.135 | 0.035 | — |
| `retrieval_similarity_mean` | **0.128** | 0.034 | **-5.4% AURC** |
| `retrieval_similarity_max` | **0.128** | 0.034 | **-5.4% AURC** |
| `hybrid_evidence_similarity` | 0.135 | 0.035 | +0.2% AURC |

**Key Findings**:
1. **Retrieval similarity improves AURC 5.4%**: `retrieval_similarity_mean` provides better ranking than evidence count alone
2. **AUGRC unchanged**: Improvement within noise (0.034 vs 0.035)
3. **Hybrid signal not helpful**: Multiplying evidence × similarity doesn't improve over either alone
4. **GitHub Issue #86 hypothesis partially validated**: Retrieval signals help AURC but don't substantially move AUGRC

**Interpretation**: The retrieval similarity signal provides modest but measurable improvement in selective prediction ranking. However, the AUGRC target of <0.020 (from Issue #86) was not achieved. Further improvements would require Phase 2 (verbalized confidence) or Phase 3 (multi-signal calibration) approaches.

---

### Run 10: Jan 3, 2026 - Confidence Suite (Specs 048–051) Attempt (INVALID)

**File**: `data/outputs/both_paper-test_20260103_182316.json`

**Log**: `data/outputs/run10_confidence_suite_20260103_111959.log`

**Run ID**: `3186a50d`

**Git Commit**: `064ed30` (dirty)

**Timestamp**: 2026-01-03T11:20:01

**Goal**: Emit confidence-suite signals (verbalized confidence, token-level CSFs, consistency) and re-evaluate AURC/AUGRC.

**What went wrong** (why this run is invalid for comparisons):

1. **Zero-shot had 2/41 hard failures** (PIDs 383, 427): `Exceeded maximum retries (3) for output validation`.
   - This was caused by deterministic malformed “JSON-like” outputs in the scoring step (pre-ANALYSIS-026 JSON hardening).
2. **Few-shot evaluated 0/41 participants**: every participant failed with:
   - `HuggingFace backend requires optional dependencies. Install with: pip install 'ai-psychiatrist[hf]'`
   - Root cause: the run used `EMBEDDING_BACKEND=huggingface` but `torch` was not installed, so query embeddings could not be computed.

**Results** (retain for debugging only; not a publication-quality run):

| Mode | N_eval | MAE_w | MAE_item | Coverage | Notes |
|------|--------|-------|----------|----------|-------|
| Zero-shot | 39/41 | 0.632 | 0.597 | 48.7% | Partial; biased by failures |
| Few-shot | 0/41 | n/a | n/a | n/a | Invalid (missing HF deps) |

**Selective prediction (zero-shot only; 39 participants)**:

Computed via:
`uv run python scripts/evaluate_selective_prediction.py --input data/outputs/both_paper-test_20260103_182316.json --mode zero_shot`

| Confidence | AURC | AUGRC | Cmax | Notes |
|------------|------|-------|------|-------|
| `llm` | 0.101 | 0.026 | 48.7% | Baseline for this partial run |
| `verbalized` | 0.092 | 0.026 | 48.7% | Lower AURC than `llm` |
| `token_pe` | 0.100 | 0.024 | 48.7% | Lower AUGRC than `llm` |

**Action items before Run 11**:
- Use a clean git state for the run (commit or stash).
- If using HuggingFace embeddings (`EMBEDDING_BACKEND=huggingface`), install deps first: `make dev-hf` (or `uv sync --extra hf`) and verify `uv run python -c "import torch"`.
- Re-run the confidence suite on a valid run artifact (both modes evaluated) before interpreting deltas.

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
# Optional (Spec 34): add `--write-item-tags` to generate a `.tags.json` sidecar for item-tag filtering, then set `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true` for runs.
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

- Statistical methodology: `docs/statistics/statistical-methodology-aurc-augrc.md`
- Feature index + defaults: `docs/pipeline-internals/features.md`
- RAG runtime features: `docs/rag/runtime-features.md`
- RAG debugging: `docs/rag/debugging.md`
- RAG artifact generation: `docs/rag/artifact-generation.md`
- Paper analysis: `docs/_archive/misc/paper-reproduction-analysis.md`
