# Spec 25: AURC/AUGRC Implementation for Selective Prediction Evaluation (Risk-Coverage, Bootstrap CIs)

> **STATUS**: IMPLEMENTED
>
> **Priority**: High - required for statistically valid reporting
>
> **GitHub Issue**: #66
>
> **Created**: 2025-12-27

---

## 1) Context and Problem

Our quantitative PHQ-8 system is a **selective prediction** system: it can abstain at the
PHQ-8 item level by outputting `"N/A"` (represented as `None` in code/JSON). This means
different runs/modes can operate at different **coverage** levels.

Comparing MAE between methods at *different* coverages is statistically invalid because:

- lower coverage selectively keeps easier cases (“cherry-picking”),
- higher coverage necessarily includes harder cases,
- MAE monotonically tends to worsen as coverage increases (in expectation).

We need a metrics + reporting suite that:

1. Evaluates the **full risk-coverage tradeoff** (not one arbitrary operating point),
2. Supports **matched-coverage comparisons**,
3. Provides **uncertainty estimates** (CIs) respecting participant-level clustering,
4. Is deterministic and reproducible.

---

## 2) Goals / Non-goals

### Goals

- Persist per-item confidence/evidence signals already present in domain objects.
- Implement, test, and report:
  - risk-coverage curve (RC curve),
  - **AURC** (Area Under the Risk-Coverage curve),
  - **AUGRC** (Area Under the Generalized Risk-Coverage curve; "joint risk"),
  - **MAE@coverage** (matched coverage),
  - max achievable coverage (**Cmax**),
  - participant-cluster bootstrap CIs for all reported scalars.
- Provide an evaluation script that turns saved run outputs into a metrics artifact + console summary.

### Non-goals (explicitly out of scope for this spec)

- Human clinician “baseline AURC”: humans typically produce a single operating point (coverage≈1) and do
  not emit a ranking signal, so “human AURC” is not defined without extra protocol design.
- Human+LLM assistance study design (valuable, but a different research question).
- Forcing “100% coverage” by changing prompts/logic (that defines a different system; we may analyze
  forced-coverage variants separately).

---

## 3) Repo-accurate Data Model (What We Evaluate)

### 3.1 Unit of evaluation

We evaluate **item instances**: one PHQ-8 item for one participant.

For each (participant_id, item):

- `pred` ∈ {0,1,2,3} or `None` (abstain)
- `gt` ∈ {0,1,2,3} (ground truth; required)
- `confidence` (scalar ranking signal; higher = more confident)

### 3.2 Inclusion rules (critical)

- Include only `success=True` participants from `scripts/reproduce_results.py` output.
- Include participants even if they have 0 predicted items (coverage 0 for that participant).
  - This is required; excluding them inflates coverage.
- Failed participants (`success=False`) are reported separately as a reliability statistic and are not
  included in selective prediction metrics (unless a future spec explicitly defines “end-to-end coverage”
  semantics).

### 3.3 Total item count (denominator)

Let:

- `P = number of included participants`
- `N = P * 8` (total items; includes abstentions)

All coverage computations must use `N` (never “predicted count”).

---

## 4) Confidence Signals (What We Rank By)

### 4.1 Signals available today

`ItemAssessment` already stores:

- `llm_evidence_count: int` - number of evidence quotes extracted by the LLM evidence step
- `keyword_evidence_count: int` - number of keyword-hit quotes injected into scorer evidence
- `evidence_source: "llm" | "keyword" | "both" | None`

Source: `src/ai_psychiatrist/domain/value_objects.py` (`ItemAssessment`)

### 4.2 Signals to persist (required)

We must persist per item:

- `llm_evidence_count`
- `keyword_evidence_count`
- `evidence_source`

Rationale:

- Confidence is low-cardinality and tied to evidence extraction; we need the raw components to interpret
  curve behavior and to support future refinements without rerunning expensive LLM calls.

### 4.3 Default ranking functions (required to support)

We will compute metrics for at least two ranking functions:

1. `conf_llm = llm_evidence_count` (mode-agnostic, stable across backfill)
2. `conf_total_evidence = llm_evidence_count + keyword_evidence_count` (reflects evidence presented to scorer)

This allows reporting robustness to the confidence definition.

### 4.4 Tie-breaking (must be deterministic)

Evidence counts are discrete; ties are common. To avoid creating *unachievable* operating points
(partial acceptance within a confidence plateau), we treat ties as **plateaus**:

- RC curves and all area/MAE@coverage metrics are computed at **working points defined by unique
  confidence values** (thresholding on `confidence`), so within-plateau ordering is irrelevant.
- When a deterministic total order is needed (debug-only), sort by:
  1. `confidence` descending
  2. `participant_id` ascending
  3. `item_index` ascending, where `item_index` follows `PHQ8Item.all_items()` order

### 4.5 Recommended future confidence signals (optional, but high-value)

Counts are low-cardinality; adding at least one **continuous** confidence signal improves curve
resolution and reduces tie artifacts.

Candidates (not required for this spec’s first implementation):

- Few-shot retrieval quality:
  - top-k cosine similarity (per item) from the embedding retrieval step
  - mean/median similarity across retrieved references
- Evidence grounding:
  - fraction of evidence quotes that are exact transcript substrings
  - evidence quote token/character length (longer grounded spans often correlate with confidence)

---

## 5) Output Schema Extension (Backward Compatible)

### 5.1 Current output

`scripts/reproduce_results.py` writes a run JSON shaped like:

```json
{
  "run_metadata": {"run_id": "...", "git_commit": "...", "...": "..."},
  "experiments": [
    {
      "provenance": {"mode": "few_shot", "split": "dev", "...": "..."},
      "results": {
        "mode": "few_shot",
        "model": "...",
        "...": "...",
        "results": [
          {
            "participant_id": 300,
            "success": true,
            "error": null,
            "ground_truth_total": 10,
            "predicted_total": 8,
            "available_items": 6,
            "na_items": 2,
            "mae_available": 0.5,
            "ground_truth_items": {"NoInterest": 2, "Depressed": 1, "...": 0},
            "predicted_items": {"NoInterest": 2, "Depressed": null, "...": null},
            "item_signals": {
              "NoInterest": {
                "llm_evidence_count": 3,
                "keyword_evidence_count": 0,
                "evidence_source": "llm"
              },
              "Depressed": {
                "llm_evidence_count": 0,
                "keyword_evidence_count": 1,
                "evidence_source": "keyword"
              }
            }
          }
        ]
      }
    }
  ]
}
```

Note: `item_signals` was added in Phase 1 (completed). This signal persistence allows selective
prediction evaluation without re-running the heavy inference steps.

### 5.2 Signal Persistence (Implemented)

The `item_signals` key is structured as follows:

```json
  "item_signals": {
    "NoInterest": {
      "llm_evidence_count": 3,
      "keyword_evidence_count": 0,
      "evidence_source": "llm"
    },
    ...
  }
```

Rules:

- For `success=True`, `item_signals` MUST contain all 8 PHQ-8 items.
- Keys under `item_signals` MUST match `PHQ8Item.value` (same as `predicted_items`/`ground_truth_items`).
- For `success=False`, preserve the minimal failure payload (no requirement to include signals).

Implementation detail (repo-accurate):

- These values come directly from `PHQ8Assessment.items[item]` returned by
  `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent.assess`).

---

## 6) Metric Definitions (Exact; no ambiguity)

### 6.1 Loss function

Primary (human-readable) loss:

- `abs_err = |pred - gt|` (range 0-3)

Normalized loss (recommended for bounded “generalized risk” metrics):

- `abs_err_norm = abs_err / 3` (range 0-1)

Unless explicitly stated, “risk” refers to MAE on the chosen loss.

### 6.2 Risk-Coverage curve (RC curve)

We define the RC curve using **confidence-threshold working points** (tie-plateau aware), matching
the intent of `_reference/fd-shifts/fd_shifts/analysis/rc_stats_utils.py` while adapting coverage to
include abstentions.

Given a list of `N` item instances (including abstentions):

1. Filter to predicted items `S = {i | pred_i is not None}`.
2. Let `K = |S|` and `Cmax = K / N` (max achievable coverage).
3. Group `S` by `confidence` value and sort unique confidence values descending:
   - `c1 > c2 > ... > cM`
4. Iterate `j = 1..M`, where the accepted set is all predicted items with confidence ≥ `cj`:
   - `A_j = {i ∈ S | confidence_i ≥ cj}`
   - `k_j = |A_j|`
   - `coverage_j = k_j / N`
   - `selective_risk_j = (1/k_j) * Σ(loss_i for i ∈ A_j)`
   - `generalized_risk_j = (1/N) * Σ(loss_i for i ∈ A_j)`  (a.k.a. “joint risk”)

The RC curve is the sequence of `M` working points:

- `[(coverage_j, selective_risk_j, generalized_risk_j, threshold=cj) for j in 1..M]`

Notes:

- `coverage_j` is strictly increasing and ends at `Cmax`.
- If `K == 0`, the curve is empty and `Cmax == 0`.

### 6.3 AURC (Area Under Risk-Coverage Curve)

We compute AURC as the area under the selective-risk RC curve over the **achievable** coverage
range `[0, Cmax]`.

Let the RC curve have working points `(coverage_j, selective_risk_j)` for `j=1..M`. Define the
right-continuous convention at 0 coverage:

- `selective_risk(0) = selective_risk_1` when `M > 0`

Then:

- `AURC = ∫_0^{Cmax} selective_risk(c) dc`

Estimator (linear/trapezoidal integration over working points):

- If `M == 0`: `AURC = 0.0`
- Else compute `trapz` over the augmented points:
  - `coverages = [0.0] + [coverage_1, ..., coverage_M]`
  - `risks = [selective_risk_1] + [selective_risk_1, ..., selective_risk_M]`
  - `AURC = numpy.trapz(risks, coverages)`

Optional normalized variant (often easier to interpret across runs):

- If `Cmax > 0`: `nAURC = AURC / Cmax` (mean selective risk over `[0, Cmax]`)

Implementation notes:

- Some implementations (e.g., fd-shifts) scale AURC by 1000 for display purposes. Our implementation
  returns raw values; multiply by 1000 for display if desired.
- Alternative estimators exist with lower finite-sample bias (e.g., harmonic-weighted `em_AURC` in
  `_reference/AsymptoticAURC/utils/estimators.py`). The trapezoidal estimator is the standard choice
  and matches fd-shifts default behavior.

### 6.4 AUGRC (Area Under Generalized Risk-Coverage Curve)

We compute AUGRC as the area under the **generalized-risk** (a.k.a. joint-risk) RC curve, which is
less prone to unintuitive weighting than AURC in some regimes.

Let the RC curve have working points `(coverage_j, generalized_risk_j)` for `j=1..M`, and define:

- `generalized_risk(0) = 0.0`

Then:

- `AUGRC = ∫_0^{Cmax} generalized_risk(c) dc`

Estimator (linear/trapezoidal integration over working points):

- If `M == 0`: `AUGRC = 0.0`
- Else compute `trapz` over:
  - `coverages = [0.0] + [coverage_1, ..., coverage_M]`
  - `risks = [0.0] + [generalized_risk_1, ..., generalized_risk_M]`
  - `AUGRC = numpy.trapz(risks, coverages)`

Implementation rules:

- Compute AUGRC on normalized loss (`abs_err_norm`) by default (keeps residuals in [0, 1], matching
  many reference implementations including fd-shifts).
- Also expose raw-loss AUGRC for internal debugging if needed.

Optional normalized variant:

- If `Cmax > 0`: `nAUGRC = AUGRC / Cmax`

### 6.5 Matched-coverage risk (MAE@coverage)

Define `risk_at_coverage(target_c)` (achievable by thresholding on `confidence`):

- Validate `0 < target_c <= 1`.
- Find the *smallest* working point `j` such that `coverage_j >= target_c`.
- If no such working point exists (`target_c > Cmax`), return `None`.
- If `target_c <= coverage_1` (smallest achievable coverage), return `selective_risk_1`
  (the highest-confidence operating point).
- Else return `selective_risk_j`.

We report MAE@coverage for a fixed grid of coverages (configurable), but only compare methods at
coverages where both methods have `Cmax >= target_c`.

### 6.6 Truncated areas for fair cross-method comparison (AURC@C, AUGRC@C)

When two methods have different `Cmax`, comparing “full” AURC (integrated over `[0, Cmax]`) is not a
clean apples-to-apples comparison because the integration domain differs.

We therefore also define **truncated areas** at a chosen coverage `C`:

- `C` must satisfy `0 < C <= 1`
- for paired method comparison, default to `C_common = min(Cmax_A, Cmax_B)`

Define:

- `C' = min(C, Cmax)`
- `AURC@C = ∫_0^{C'} selective_risk(c) dc`
- `AUGRC@C = ∫_0^{C'} generalized_risk(c) dc`

Estimator:

- Compute the full augmented curve points used for AURC/AUGRC (Section 6.3/6.4).
- If `C'` falls between two coverage points, linearly interpolate the corresponding risk value at `C'`.
- Compute `numpy.trapz` on the truncated arrays ending at `C'`.

We will report both:

- `AURC_full` / `AUGRC_full` (integrated over `[0, Cmax]`) plus `Cmax`
- `AURC@C_common` / `AUGRC@C_common` for method comparisons

### 6.7 Excess AURC-family metrics (optional, confidence-quality diagnostics)

To evaluate the *ranking quality* of the confidence signal (separately from the underlying model’s
prediction quality), implement the fd-shifts-style “excess” variants:

- `eAURC = AURC - AURC_optimal`
- `eAUGRC = AUGRC - AUGRC_optimal`

Where `*_optimal` is computed on the same set of residuals but with an “ideal” confidence ordering
(lowest loss treated as highest confidence). For non-binary residuals, mirror
`_reference/fd-shifts/fd_shifts/analysis/rc_stats.py`:

1. Sort predicted items by `loss` ascending.
2. Assign strictly decreasing synthetic confidences (e.g., `np.linspace(1, 0, K)`).
3. Recompute the metric (same integration method) to obtain `*_optimal`.

Notes:

- With confidence plateaus and finite samples, `eAURC`/`eAUGRC` can be slightly negative; do not clamp.

---

## 7) Statistical Inference (Publication-grade CIs)

### 7.1 Why participant-cluster bootstrap

PHQ-8 items within a participant are correlated. Treating items as i.i.d. leads to overly narrow CIs.
We therefore compute CIs via **cluster bootstrap by participant**:

- sample participant IDs with replacement,
- include all 8 items per sampled participant,
- recompute metrics on pooled items.

### 7.2 Paired bootstrap for method comparisons

For comparisons on the same participant set (e.g., zero-shot vs few-shot):

- sample participants once per replicate,
- compute metric for each method on that sample,
- store Δ = (method_B - method_A),
- report percentile CI for Δ.

### 7.3 Defaults (final reporting)

- `n_resamples = 10_000`
- `ci = 95%` (percentiles 2.5 and 97.5)
- `seed` is required, recorded into the metrics artifact for reproducibility.

### 7.4 Handling insufficient coverage in bootstrap replicates

For MAE@coverage:

- if a bootstrap replicate does not reach `target_c`, record `None`;
- exclude `None` replicates from percentile calculation and report the exclusion rate.

---

## 8) Implementation Plan (TDD)

### Phase 1 - Persist `item_signals` in `scripts/reproduce_results.py` [COMPLETED]

**Status**: Implemented.

- `EvaluationResult` dataclass extended with `item_signals`.
- `evaluate_participant()` populates signals from `assessment.items`.
- `ExperimentResults.to_dict()` persists signals for successful results.

### Phase 2 - Implement metrics module

Create:

- `src/ai_psychiatrist/metrics/__init__.py`
- `src/ai_psychiatrist/metrics/selective_prediction.py`

Core API (stable + typed):

- `@dataclass(frozen=True, slots=True) class ItemPrediction:`
  - `participant_id: int`
  - `item_index: int`
  - `pred: int | None`
  - `gt: int`
  - `confidence: float`
- `@dataclass(frozen=True, slots=True) class RiskCoverageCurve:`
  - `coverage: list[float]` (working points; increasing)
  - `selective_risk: list[float]`
  - `generalized_risk: list[float]`
  - `threshold: list[float]` (unique confidence values; decreasing)
  - `cmax: float`
- `compute_risk_coverage_curve(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"]) -> RiskCoverageCurve`
- `compute_aurc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"])`
- `compute_augrc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"])`
- `compute_aurc_at_coverage(items: Sequence[ItemPrediction], *, max_coverage: float, loss: ...)`
- `compute_augrc_at_coverage(items: Sequence[ItemPrediction], *, max_coverage: float, loss: ...)`
- `compute_risk_at_coverage(items: Sequence[ItemPrediction], *, target_coverage: float, loss: ...)`
- `compute_cmax(items: Sequence[ItemPrediction]) -> float`

Note: the spec requires that:

- RC curves are computed at **unique confidence thresholds** (tie-plateau aware).
- AURC uses `numpy.trapz` over the augmented selective-risk curve on `[0, Cmax]`.
- AUGRC uses `numpy.trapz` over the augmented generalized-risk curve on `[0, Cmax]`.

### Phase 3 - Bootstrap utilities

Create:

- `src/ai_psychiatrist/metrics/bootstrap.py`

Implement:

- `bootstrap_by_participant(...) -> BootstrapResult`
- `paired_bootstrap_delta_by_participant(...) -> BootstrapDeltaResult`

### Phase 4 - Add an evaluation script (separation of concerns)

Create:

- `scripts/evaluate_selective_prediction.py`

Responsibilities:

- Load one or more output JSON files produced by `scripts/reproduce_results.py`
- Select an experiment (mode) and build `ItemPrediction` rows using:
  - `predicted_items`
  - `ground_truth_items`
- `item_signals` + confidence function(s)
- Compute, for each configured confidence function (at minimum `conf_llm` and `conf_total_evidence`):
  - RC curve arrays (coverage, selective_risk, generalized_risk, threshold)
  - AURC_full, AUGRC_full, Cmax
  - AURC@C and AUGRC@C for a configurable `C` (default: `C_common` when comparing two runs)
  - MAE@coverage grid
  - bootstrap CIs for scalars
  - paired Δ CIs when two runs are provided
- Write a metrics JSON artifact plus a concise console report.

#### Phase 4a - CLI contract (required)

The script MUST support:

- `--input` (repeatable): one or two paths to `data/outputs/*.json` produced by `scripts/reproduce_results.py`.
- `--mode` (repeatable): experiment selection for each `--input` (`zero_shot` or `few_shot`).
  - If `--mode` is provided once, it applies to all inputs.
  - If `--mode` is provided multiple times, it MUST match the number of `--input` flags and is paired by position.
  - If `--mode` is omitted for an input:
    - If the run file contains exactly one experiment, select it.
    - Else error with a clear message listing available experiments and how to disambiguate.
- `--loss`: which loss to compute risk on:
  - `abs` (raw absolute error; range 0-3)
  - `abs_norm` (normalized absolute error; range 0-1; default)
- `--confidence`: which confidence function(s) to compute metrics for:
  - `llm` (llm_evidence_count)
  - `total_evidence` (llm_evidence_count + keyword_evidence_count)
  - `all` (default; computes both)
- `--coverage-grid`: comma-separated floats in `(0, 1]` (default: `0.1,0.2,...,0.9`).
- `--area-coverage`: float in `(0, 1]` for reporting AURC@C/AUGRC@C in single-run mode
  (default: `0.5`).
- `--bootstrap-resamples`: int (default: 10_000).
- `--seed`: int (required for any bootstrap computation; optional when `--bootstrap-resamples=0`).
- `--output`: optional path for metrics JSON; default under `data/outputs/`.
- `--intersection-only`: if two inputs have different participant sets, restrict evaluation to the overlap
  (default: strict error if sets differ).

Paired comparison mode:

- If exactly two inputs are provided, the script MUST compute paired deltas over overlapping participant IDs.
- Define `Δ = right - left`, where `left` is the first `--input` and `right` is the second.
- If participant sets differ and `--intersection-only` is not set, exit non-zero with a clear message.
- In paired mode, the script MUST compute per-method point estimates on the same participant set used for
  deltas:
  - Let `ids_left_total` / `ids_right_total` be all participant IDs present in each selected experiment.
  - If `--intersection-only` is set: `ids_overlap_total = ids_left_total ∩ ids_right_total`.
  - Else require `ids_left_total == ids_right_total` and set `ids_overlap_total = ids_left_total`.
  - Let `ids_left_success` / `ids_right_success` be success participants within `ids_overlap_total`.
  - Define the analysis set `ids_included = ids_left_success ∩ ids_right_success`.
  - Compute `cmax`, RC curves, AURC/AUGRC, and MAE@coverage for each method on `ids_included` only.

#### Phase 4b - Metrics artifact schema (required)

The output metrics JSON MUST include:

```json
{
  "schema_version": "1",
  "created_at": "2025-12-27T01:23:45Z",
  "inputs": [
    {"path": "data/outputs/...", "run_id": "...", "git_commit": "...", "mode": "few_shot"}
  ],
  "population": {
    "participants_total": 123,
    "participants_included": 123,
    "participants_failed": 0,
    "items_total": 984
  },
  "loss": {
    "name": "abs_norm",
    "definition": "abs(pred - gt) / 3",
    "raw_multiplier": 3
  },
  "confidence_variants": {
    "llm": {
      "cmax": 0.716,
      "aurc_full": 0.1234,
      "augrc_full": 0.0456,
      "aurc_at_c": {"requested": 0.50, "used": 0.50, "value": 0.0812},
      "augrc_at_c": {"requested": 0.50, "used": 0.50, "value": 0.0301},
      "mae_at_coverage": {
        "0.10": {"requested": 0.10, "achieved": 0.10, "value": 0.12},
        "0.20": {"requested": 0.20, "achieved": 0.33, "value": 0.20},
        "...": null
      },
      "bootstrap": {
        "seed": 7,
        "n_resamples": 10000,
        "ci95": {
          "aurc_full": [0.11, 0.14],
          "augrc_full": [0.04, 0.05],
          "aurc_at_c": [0.07, 0.10],
          "augrc_at_c": [0.02, 0.04],
          "cmax": [0.70, 0.73],
          "mae_at_coverage": {
            "0.10": [0.10, 0.14],
            "0.20": [0.18, 0.24],
            "...": null
          }
        },
        "drop_rate": {
          "mae_at_coverage": {
            "0.10": 0.0,
            "0.20": 0.0,
            "...": 1.0
          }
        }
      },
      "curve": {
        "coverage": [0.10, 0.33, "..."],
        "selective_risk": [0.0, 0.5, "..."],
        "generalized_risk": [0.0, 0.05, "..."],
        "threshold": [3, 2, "..."]
      }
    }
  },
  "comparison": {
    "enabled": false,
    "intersection_only": null,
    "participants_left_only": null,
    "participants_right_only": null,
    "participants_overlap_total": null,
    "participants_overlap_included": null,
    "participants_failed_left": null,
    "participants_failed_right": null,
    "deltas": null
  }
}
```

Rules:

- If `comparison.enabled == false`, `inputs` MUST have length 1.
- If `comparison.enabled == true`, `inputs` MUST have length 2.
- `curve.coverage` MUST be strictly increasing and end at `cmax` (within float tolerance).
- `curve.threshold` MUST have the same length as `curve.coverage` and be strictly decreasing.
- `population.participants_total == population.participants_included + population.participants_failed`.
- `population.items_total == population.participants_included * 8`.
- `mae_at_coverage` values MUST be `null` when `target_c > cmax`; otherwise they must include
  `requested`, `achieved`, and `value`.
- When bootstrap is enabled (`n_resamples > 0`), the artifact MUST include:
  - `bootstrap.ci95` for: `aurc_full`, `augrc_full`, `aurc_at_c`, `augrc_at_c`, `cmax`, and `mae_at_coverage`.
  - `bootstrap.drop_rate.mae_at_coverage[target_c]` for each requested `target_c`.
- Bootstrap CI arrays MUST be `[low, high]` percentiles.

If `comparison.enabled == true` (paired comparison mode), the artifact MUST include:

- `intersection_only: bool`
- `participants_left_only: int`
- `participants_right_only: int`
- `participants_overlap_total: int` (participant IDs present in both inputs, regardless of success)
- `participants_overlap_included: int` (participant IDs used for paired metrics; MUST equal `population.participants_included`)
- `participants_failed_left: int` (in `participants_overlap_total`, failed in left input)
- `participants_failed_right: int` (in `participants_overlap_total`, failed in right input)
- `deltas`: per confidence variant, a structure containing:
  - point estimates for Δ metrics (right - left),
  - bootstrap CI95 for Δ metrics.

---

## 9) Test Plan (must catch real bugs)

### 9.1 Unit tests (exact values; no “> 0”)

File:

- `tests/unit/metrics/test_selective_prediction.py`

Required tests:

- Perfect predictions → AURC=0 and AUGRC=0.
- Single prediction + abstentions:
  - verify `N` includes abstentions,
  - verify AURC reduces to `coverage_1 * selective_risk_1` (right-continuous convention at 0),
  - verify MAE@coverage uses the smallest achievable working point `coverage >= target`.
- Tie plateaus:
  - two predictions with identical `confidence` produce a single RC working point,
  - MAE@coverage at a target between plateaus snaps to the next plateau.
- Truncated areas:
  - verify `AURC@C` and `AUGRC@C` linearly interpolate within the segment containing `C`.
- AUGRC consistency at working points:
  - verify `generalized_risk_j == coverage_j * selective_risk_j` for the curve.
- All abstain (K=0):
  - verify `Cmax == 0`, `AURC == 0.0`, `AUGRC == 0.0`,
  - verify curve is empty (`len(coverage) == 0`).
- All same confidence (single plateau):
  - all predicted items have identical confidence,
  - verify single working point in curve (`len(coverage) == 1`),
  - verify `coverage[0] == Cmax`.
- C exactly at working point:
  - when truncation coverage `C` equals an exact working point coverage,
  - verify no interpolation occurs (value matches working point exactly).
- Single participant cluster bootstrap:
  - with `P=1`, cluster bootstrap should not crash,
  - all resamples are identical (only one cluster to draw),
  - CI degenerates to point estimate (low == high).

Canonical numeric test vector (MUST be included verbatim in unit tests):

Given `N=4` item instances with `(pred, gt, confidence)`:

1. `(2, 2, 2)`  -> `abs_err=0`
2. `(3, 1, 2)`  -> `abs_err=2`
3. `(1, 1, 1)`  -> `abs_err=0`
4. `(None, 0, 0)` -> abstain (excluded from `S`, but included in `N`)

Expected RC curve working points (ties treated as plateaus):

- `threshold = [2, 1]`
- `coverage = [2/4, 3/4] = [0.5, 0.75]`

Expected values for `loss="abs"`:

- `selective_risk = [1, 2/3]`
- `generalized_risk = [1/2, 1/2]`
- `AURC_full = 17/24`
- `AUGRC_full = 1/4`
- `risk_at_coverage(target_c=0.6)` returns `2/3` with achieved coverage `0.75`
- `AURC@0.6 = 89/150`
- `AUGRC@0.6 = 7/40`

Expected values for `loss="abs_norm"` (exactly scaled by 1/3):

- `selective_risk = [1/3, 2/9]`
- `generalized_risk = [1/6, 1/6]`
- `AURC_full = 17/72`
- `AUGRC_full = 1/12`
- `risk_at_coverage(target_c=0.6)` returns `2/9` with achieved coverage `0.75`
- `AURC@0.6 = 89/450`
- `AUGRC@0.6 = 7/120`

### 9.2 Integration test (parsing output schema)

File:

- `tests/integration/test_selective_prediction_from_output.py`

Fixture must include:

- at least 1 participant with all 8 items present,
- predicted + abstained mix,
- `item_signals` for all 8 items.

Assertions:

- `N == 8 * P` (not “predicted count”),
- `Cmax == predicted_count / N`,
- metrics computed without exceptions and match exact expected values for the fixture.

---

## 10) Acceptance Criteria

- `scripts/reproduce_results.py` persists `item_signals` for each successful participant.
- `EvaluationResult` includes `item_signals` field with all 8 PHQ-8 items for successful participants.
- Metrics module implements RC curve working points (unique confidence thresholds), AURC/AUGRC
  (`numpy.trapz` on `[0, Cmax]`), MAE@coverage (achievable by thresholding), Cmax, and truncated
  AURC@C/AUGRC@C.
- `compute_risk_coverage_curve()` returns empty curve (`cmax=0`) when all items abstain.
- `compute_aurc()` and `compute_augrc()` return `0.0` when `cmax=0`.
- Bootstrap module produces participant-level CIs and paired Δ CIs.
- Bootstrap with single participant does not crash (degenerates to point estimate).
- Evaluation script produces a machine-readable metrics artifact and console summary.
- Evaluation script supports per-input mode selection (`--mode` repeatable, paired-by-position with `--input`), `--loss`, and strict-vs-intersection participant handling.
- Metrics artifact includes top-level `loss`, plus bootstrap `ci95` and `drop_rate` for `mae_at_coverage` when bootstrapping is enabled.
- All tests in Section 9.1 pass with exact numeric assertions.
- `make ci` passes.

---

## Appendix A) Human baselines (informational; not implemented here)

If we ever want to compare to a human rater:

  - A human typically provides one point: (coverage≈100%, risk≈human MAE). Without a ranking signal,
  “human AURC/AUGRC” is not defined.
- Valid options (separate protocol design work):
  1. **Forced-coverage model**: define a model variant that always outputs 0-3 (no abstention), then
     compare MAE at coverage=1 against a human MAE at coverage=1.
  2. **Human selective protocol**: ask humans to abstain (or provide a calibrated confidence) so they
     also define a risk-coverage curve.
  3. **Compare at matched operating points** only (e.g., MAE@50%) if both systems can produce those
     operating points via an agreed abstention/confidence protocol.

None of these are required to produce robust LLM-only evaluation, which is the focus of this spec.

## 11) References

Definitions and critique of AURC + AUGRC:

1. [Overcoming Common Flaws in the Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032v1)

Population AURC background (optional; not required for the empirical estimator we implement here):

2. [A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)](https://arxiv.org/html/2410.15361v1)

Abstention background and reporting cautions:

3. [Know Your Limits: A Survey of Abstention in Large Language Models](https://arxiv.org/abs/2407.18418)

Local reference implementations (used for semantic cross-checks; not imported at runtime):

4. fd-shifts RC metrics (AURC/AUGRC/eAURC/eAUGRC + bootstrap CI utilities)
   - `_reference/fd-shifts/fd_shifts/analysis/rc_stats.py`
   - `_reference/fd-shifts/fd_shifts/analysis/rc_stats_utils.py`

5. AsymptoticAURC (finite-sample estimator bias context; alternative estimators)
   - `_reference/AsymptoticAURC/utils/estimators.py`
