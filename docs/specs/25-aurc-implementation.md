# Spec 25: Selective Prediction Evaluation Suite (Risk–Coverage, AURC, AUGRC, Bootstrap CIs)

> **STATUS**: Proposed
>
> **Priority**: High — required for statistically valid reporting
>
> **GitHub Issue**: #66
>
> **Created**: 2025-12-27
>
> **Supersedes**: `docs/specs/24-aurc-metric.md`

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

1. Evaluates the **full risk–coverage tradeoff** (not one arbitrary operating point),
2. Supports **matched-coverage comparisons**,
3. Provides **uncertainty estimates** (CIs) respecting participant-level clustering,
4. Is deterministic and reproducible.

---

## 2) Goals / Non-goals

### Goals

- Persist per-item confidence/evidence signals already present in domain objects.
- Implement, test, and report:
  - risk–coverage curve (RC curve),
  - **AURC** (Area Under the Risk–Coverage curve),
  - **AUGRC** (Area Under the Generalized Risk–Coverage curve; “joint risk”),
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

- `llm_evidence_count: int` — number of evidence quotes extracted by the LLM evidence step
- `keyword_evidence_count: int` — number of keyword-hit quotes injected into scorer evidence
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

Evidence counts are discrete; ties are common. To ensure exact reproducibility, sorting MUST be:

1. `confidence` descending
2. `participant_id` ascending
3. `item_index` ascending, where `item_index` follows `PHQ8Item.all_items()` order

---

## 5) Output Schema Extension (Backward Compatible)

### 5.1 Current output

`scripts/reproduce_results.py` currently persists (per successful participant) only:

```json
{
  "participant_id": 300,
  "predicted_items": {"NoInterest": 2, "Depressed": null, "...": null},
  "ground_truth_items": {"NoInterest": 2, "Depressed": 1, "...": 0}
}
```

This is insufficient for selective prediction evaluation because no ranking signal exists.

### 5.2 Required output (backward compatible)

Add a new key: `item_signals` (do not rename/remove existing keys):

```json
{
  "participant_id": 300,
  "predicted_items": {"NoInterest": 2, "Depressed": null, "...": null},
  "ground_truth_items": {"NoInterest": 2, "Depressed": 1, "...": 0},
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
```

Rules:

- For `success=True`, `item_signals` MUST contain all 8 PHQ-8 items.
- For `success=False`, preserve the minimal failure payload (no requirement to include signals).

Implementation detail (repo-accurate):

- These values come directly from `PHQ8Assessment.items[item]` returned by
  `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent.assess`).

---

## 6) Metric Definitions (Exact; no ambiguity)

### 6.1 Loss function

Primary (human-readable) loss:

- `abs_err = |pred - gt|` (range 0–3)

Normalized loss (recommended for bounded “generalized risk” metrics):

- `abs_err_norm = abs_err / 3` (range 0–1)

Unless explicitly stated, “risk” refers to MAE on the chosen loss.

### 6.2 Risk–Coverage curve (RC curve)

Given a list of `N` item instances (including abstentions):

1. Filter to predicted items (`pred is not None`).
2. Sort predicted items by confidence (desc) with deterministic tie-breaking.
3. Let `K = number of predicted items` and `Δ = 1 / N`.
4. For each `k = 1..K`:
   - `coverage_k = k / N`
   - `risk_k = (1/k) * sum(loss_i for i in top k)`
   - `joint_risk_k = (k/N) * risk_k = (1/N) * sum(loss_i for i in top k)`

Return a curve with `K` points:

- `[(coverage_k, risk_k, joint_risk_k) for k in 1..K]`

### 6.3 AURC (Area Under Risk–Coverage Curve)

We implement **empirical step AURC** (not trapezoidal). This matches the discrete integration
interpretation of selective risk aggregated over acceptance thresholds.

Let `Δ = 1 / N` and `risk_k` defined above. Then:

- `AURC = sum_{k=1..K} risk_k * Δ`

Properties:

- If `K == 0` (all abstained), `AURC = 0.0`.
- AURC integrates over `[0, Cmax]` where `Cmax = K/N`.

Optional normalized variant:

- If `Cmax > 0`: `nAURC = AURC / Cmax` (mean risk across acceptance levels).

### 6.4 AUGRC (Area Under Generalized Risk–Coverage Curve)

We implement **AUGRC** using *joint risk* (generalized risk) to reduce pathological weighting of
high-confidence failures.

Let `joint_risk_k` and `Δ = 1 / N`. Then:

- `AUGRC = sum_{k=1..K} joint_risk_k * Δ`

Implementation rules:

- Compute AUGRC on normalized loss (`abs_err_norm`) by default.
- Also expose a raw-loss AUGRC for internal debugging if needed.

Optional normalized variant:

- If `Cmax > 0`: `nAUGRC = AUGRC / Cmax`.

### 6.5 Matched-coverage risk (MAE@coverage)

Define `risk_at_coverage(target_c)`:

- Validate `0 < target_c <= 1`.
- `k* = ceil(target_c * N)`.
- If `k* > K`, return `None`.
- Else return `risk_{k*}`.

We report MAE@coverage for a fixed grid of coverages (configurable), but only compare methods at
coverages where both methods have `Cmax >= target_c`.

### 6.6 Truncated areas for fair cross-method comparison (AURC@C, AUGRC@C)

When two methods have different `Cmax`, comparing “full” AURC (integrated over `[0, Cmax]`) is not a
clean apples-to-apples comparison because the integration domain differs.

We therefore also define **truncated areas** at a chosen coverage `C`:

- `C` must satisfy `0 < C <= 1`
- for paired method comparison, default to `C_common = min(Cmax_A, Cmax_B)`

Define the step-function risk `r(c)` implied by the RC curve:

- `r(c) = risk_k` for `c ∈ ((k-1)/N, k/N]`

Then:

- `AURC@C = ∫_0^{min(C, Cmax)} r(c) dc`
- `AUGRC@C = ∫_0^{min(C, Cmax)} joint_risk(c) dc` (same step convention)

Discrete implementation (exact, for step integration):

1. Let `C' = min(C, Cmax)` and `x = C' * N`.
2. Let `k_full = floor(x)` and `frac = x - k_full` (so `0 <= frac < 1`).
3. `AURC@C = (sum_{k=1..k_full} risk_k + frac * risk_{k_full+1}) * (1/N)` (if `k_full == K`, the last term is omitted).
4. `AUGRC@C = (sum_{k=1..k_full} joint_risk_k + frac * joint_risk_{k_full+1}) * (1/N)`.

We will report both:

- `AURC_full` / `AUGRC_full` (integrated over `[0, Cmax]`) plus `Cmax`
- `AURC@C_common` / `AUGRC@C_common` for method comparisons

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
- store Δ = (method_B − method_A),
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

### Phase 1 — Persist `item_signals` in `scripts/reproduce_results.py`

Modify:

- `scripts/reproduce_results.py`

Changes:

- Extend `EvaluationResult` dataclass with:
  - `item_signals: dict[PHQ8Item, dict[str, object]] = field(default_factory=dict)`
- In `evaluate_participant()`, after `assessment = await agent.assess(transcript)`:
  - build `item_signals` from `assessment.items[item]` for all items in `PHQ8Item.all_items()`.
- In `ExperimentResults.to_dict()`, include `"item_signals"` under each successful result.

### Phase 2 — Implement metrics module

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
- `compute_risk_coverage_curve(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"])`
- `compute_aurc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"])`
- `compute_augrc(items: Sequence[ItemPrediction], *, loss: Literal["abs", "abs_norm"])`
- `compute_risk_at_coverage(items: Sequence[ItemPrediction], *, target_coverage: float, loss: ...)`
- `compute_cmax(items: Sequence[ItemPrediction]) -> float`

Note: the spec requires that:

- AURC uses empirical step integration.
- AUGRC uses joint risk (coverage × risk) and empirical step integration.

### Phase 3 — Bootstrap utilities

Create:

- `src/ai_psychiatrist/metrics/bootstrap.py`

Implement:

- `bootstrap_by_participant(...) -> BootstrapResult`
- `paired_bootstrap_delta_by_participant(...) -> BootstrapDeltaResult`

### Phase 4 — Add an evaluation script (separation of concerns)

Create:

- `scripts/evaluate_selective_prediction.py`

Responsibilities:

- Load one or more output JSON files produced by `scripts/reproduce_results.py`
- Select an experiment (mode) and build `ItemPrediction` rows using:
  - `predicted_items`
  - `ground_truth_items`
  - `item_signals` + chosen confidence function
- Compute:
  - RC curve arrays (coverage, risk, joint_risk)
  - AURC, AUGRC, Cmax
  - MAE@coverage grid
  - bootstrap CIs for scalars
  - paired Δ CIs when two runs are provided
- Write a metrics JSON artifact plus a concise console report.

---

## 9) Test Plan (must catch real bugs)

### 9.1 Unit tests (exact values; no “> 0”)

File:

- `tests/unit/metrics/test_selective_prediction.py`

Required tests:

- Perfect predictions → AURC=0 and AUGRC=0.
- Single prediction + abstentions:
  - verify `N` includes abstentions,
  - verify AURC = risk * (1/N) (step integration),
  - verify MAE@coverage uses ceil rule.
- Deterministic tie-breaking:
  - same confidence values reorder by participant_id and item_index only.
- AUGRC consistency:
  - verify `joint_risk_k == coverage_k * risk_k` for the curve.

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
- Metrics module implements RC curve, AURC (step), AUGRC (joint risk), MAE@coverage, Cmax.
- Bootstrap module produces participant-level CIs and paired Δ CIs.
- Evaluation script produces a machine-readable metrics artifact and console summary.
- `make ci` passes.

---

## 11) References

Definitions and critique of AURC + AUGRC:

1. Overcoming Common Flaws in the Evaluation of Selective Classification Systems
   - https://arxiv.org/html/2407.01032v1

Population AURC background (optional; not required for the empirical estimator we implement here):

2. A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)
   - https://arxiv.org/html/2410.15361v1

Abstention background and reporting cautions:

3. Know Your Limits: A Survey of Abstention in Large Language Models
   - https://arxiv.org/abs/2407.18418
