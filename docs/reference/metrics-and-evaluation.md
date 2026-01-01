# Metrics and Evaluation (Exact Definitions + Output Schema)

**Audience**: Researchers who need implementation-accurate metrics
**Last Updated**: 2026-01-01

This page is the canonical (non-archive) reference for how this repo computes and reports:
- **coverage**
- **risk-coverage curves**
- **AURC / AUGRC**
- **MAE@coverage**
- **bootstrap confidence intervals**

SSOT implementations:
- `src/ai_psychiatrist/metrics/selective_prediction.py`
- `src/ai_psychiatrist/metrics/bootstrap.py`
- `scripts/evaluate_selective_prediction.py`

---

## Unit of Evaluation (Critical)

We evaluate **item instances**: one PHQ-8 item for one participant.

For each `(participant_id, item)`:
- `gt` is always present (0–3)
- `pred` is either (0–3) or `None` (abstain)
- `confidence` is a scalar ranking signal (higher = more confident)

### Participant Failures

`scripts/reproduce_results.py` records per-participant success:
- `success=True` participants are included in selective prediction metrics.
- `success=False` participants are counted as reliability failures and excluded from AURC/AUGRC (by design).

This is implemented in `scripts/evaluate_selective_prediction.py:parse_items()`.

---

## Coverage and Cmax

Let:
- `P` = number of **included** participants (`success=True`)
- `N = P * 8` total item instances
- `K` = number of predicted items (`pred is not None`) across the `N` items

Then:
- `coverage = K / N`
- `Cmax = K / N` (same value; named “max achievable coverage” because abstentions bound the curve)

SSOT: `compute_cmax()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

---

## Confidence Variants

`scripts/evaluate_selective_prediction.py` currently supports two confidence variants:

1. `llm`:
   - `confidence = llm_evidence_count`
2. `total_evidence`:
   - `confidence = llm_evidence_count + keyword_evidence_count`

These are derived from `item_signals` in the run output JSON.

---

## Loss Functions

Two loss functions are supported:

- `abs`: `|pred - gt|`
- `abs_norm`: `|pred - gt| / 3` (range 0–1)

SSOT: `_compute_loss()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

---

## Risk-Coverage Curve (RC Curve)

### Inputs

Given all `N` item instances:
1. Filter to predicted items `S = {i | pred_i is not None}`.
2. Compute loss for each `i ∈ S`.
3. Sort `S` by `confidence` descending.

### Plateau (Tie) Handling

Confidence is often discrete (evidence counts). We compute working points by **grouping equal confidence values**:
- Each unique confidence value defines a working point.
- We add all items from that confidence plateau at once.

SSOT: `compute_risk_coverage_curve()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

### Working Point Metrics

At working point `j` after accepting `k_j` items:
- `coverage_j = k_j / N`
- `selective_risk_j = (sum loss of accepted) / k_j`
- `generalized_risk_j = (sum loss of accepted) / N`

---

## AURC and AUGRC (Integration Semantics)

We integrate using trapezoidal rule over `[0, Cmax]` with an explicit augmentation at `coverage=0`.

SSOT: `_integrate_curve()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

### AURC

- x-axis: coverage
- y-axis: selective risk
- augmentation: **right-continuous** at 0
  - `risk(0) = risk(coverage_1)`

### AUGRC

- x-axis: coverage
- y-axis: generalized risk
- augmentation:
  - `generalized_risk(0) = 0`

---

## Truncated Areas and MAE@Coverage

### Truncated AURC/AUGRC

We compute truncated areas up to a requested maximum coverage `C'`:
- `AURC@C'`
- `AUGRC@C'`

If `C' > Cmax`, the effective `C'` becomes `Cmax`.

SSOT: `_integrate_truncated()` in `src/ai_psychiatrist/metrics/selective_prediction.py` (includes linear interpolation to land exactly on `C'`).

### MAE@Coverage

`MAE@coverage=c` is defined as:
- take the first working point where `coverage >= c`
- return its selective risk

If no working point reaches the requested coverage (i.e., `c > Cmax`), the value is `None`.

SSOT: `compute_risk_at_coverage()` in `src/ai_psychiatrist/metrics/selective_prediction.py`.

---

## Bootstrap Confidence Intervals

We use **participant-cluster bootstrap**:
- resample participants with replacement
- include all 8 items per sampled participant
- recompute metrics on the resampled set

SSOT: `bootstrap_by_participant()` in `src/ai_psychiatrist/metrics/bootstrap.py`.

### Paired Deltas (Mode Comparisons)

When evaluating two modes on the same run artifact, we can compute paired deltas:
- `delta = metric_right - metric_left`
- bootstrap resamples are applied at the participant level across both inputs

SSOT: `paired_bootstrap_delta_by_participant()` in `src/ai_psychiatrist/metrics/bootstrap.py`.

---

## Metrics Artifact Output Schema

`scripts/evaluate_selective_prediction.py` produces a JSON artifact:

```json
{
  "schema_version": "1",
  "created_at": "2025-12-31T00:00:00Z",
  "inputs": [
    {"path": "...", "run_id": "...", "git_commit": "...", "mode": "few_shot"}
  ],
  "population": {
    "participants_included": 41,
    "participants_failed": 0,
    "participants_total": 41,
    "items_total": 328,
    "items_predicted": 215,
    "cmax": 0.655
  },
  "loss": {"name": "abs", "definition": "abs(pred - gt)", "raw_multiplier": 1},
  "confidence_variants": {
    "llm": {
      "cmax": 0.655,
      "aurc_full": 0.192,
      "augrc_full": 0.058,
      "aurc_at_coverage": 0.0,
      "augrc_at_coverage": 0.0,
      "mae_grid": {"0.10": {"requested": 0.1, "achieved": 0.123, "value": 0.5}},
      "bootstrap": {"ci95": {"cmax": [0.6, 0.7], "aurc_full": [0.1, 0.2]}}
    }
  },
  "comparison": {
    "enabled": false,
    "intersection_only": false,
    "deltas": null
  }
}
```

Exact keys and nesting are defined in `scripts/evaluate_selective_prediction.py` (constructs `artifact` near the end of `main()`).

---

## How To Run

```bash
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/your_run.json \
  --mode few_shot \
  --loss abs \
  --bootstrap-resamples 10000 \
  --seed 42
```

For paired comparisons:

```bash
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/your_run.json --mode zero_shot \
  --input data/outputs/your_run.json --mode few_shot \
  --loss abs \
  --seed 42
```

---

## Related Docs

- Why AURC/AUGRC matter: `docs/reference/statistical-methodology-aurc-augrc.md`
- Run output format / provenance: `docs/results/run-history.md`
