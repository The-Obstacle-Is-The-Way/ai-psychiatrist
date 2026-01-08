# Reproduction Run Output Schema (JSON + Registry)

**Audience**: Researchers parsing outputs and maintaining run provenance
**Last Updated**: 2026-01-08

This repo writes four primary provenance artifacts for quantitative reproduction runs:

1. `data/outputs/{mode}_{split}_{YYYYMMDD_HHMMSS}.json`
2. `data/experiments/registry.yaml` (append/update registry of runs)
3. `data/outputs/failures_{run_id}.json` (failure summary; Spec 056)
4. `data/outputs/telemetry_{run_id}.json` (structured telemetry summary)

SSOT implementation:
- `scripts/reproduce_results.py` (writer)
- `src/ai_psychiatrist/services/experiment_tracking.py` (filename + provenance helpers)
- `src/ai_psychiatrist/infrastructure/observability.py` (failure registry; Spec 056)
- `src/ai_psychiatrist/infrastructure/telemetry.py` (telemetry registry)

---

## Output Filename Format

`generate_output_filename()` produces:

```text
{mode}_{split}_{YYYYMMDD_HHMMSS}.json
```

Where:
- `mode` is `zero_shot`, `few_shot`, or `both`
- `split` is one of: `train`, `dev`, `train+dev`, `paper`, `paper-train`, `paper-val`, `paper-test`
  - `paper` is an alias for `paper-test` in `scripts/reproduce_results.py`

Note: `{mode}` in the filename refers to **scoring mode** (zero-shot vs few-shot vs both). Prediction evaluation mode (Specs 061/062: `item` vs `total` vs `binary`) is recorded inside the JSON as `experiments[].results.prediction_mode`.

---

## JSON Top-Level Shape

```json
{
  "run_metadata": { "...": "..." },
  "experiments": [
    {
      "provenance": { "...": "..." },
      "results": { "...": "..." }
    }
  ]
}
```

### `run_metadata`

Captured once per run:
- `run_id`
- `timestamp`
- `git_commit`, `git_dirty`
- `python_version`, `platform`
- `ollama_base_url`

### `experiments[].provenance`

Captured per mode (`zero_shot` / `few_shot`):
- split name
- model/backends used
- embedding artifact identity (path + checksums when applicable)
- participants requested vs evaluated

### `experiments[].results`

Per-mode aggregated metrics + per-participant results:
- counts: total/success/failed/evaluated
- MAE variants (`item_mae_weighted`, `item_mae_by_item`, `item_mae_by_subject`) for item-level scoring
- coverage (`prediction_coverage`)
- evaluation settings (Specs 061/062): `prediction_mode`, `total_score_min_coverage`, `binary_threshold`, `binary_strategy`
- per-item breakdowns
- per-participant predictions
- `item_signals` (Spec 25/046 confidence signals)
  - includes Spec 063 severity inference annotations when enabled: `inference_used`, `inference_type`, `inference_marker`

When enabled via `--prediction-mode` (Specs 061/062), additional aggregated metrics are included:
- `total_metrics` (total score evaluation)
- `binary_metrics` (binary depression classification)

For downstream AURC/AUGRC evaluation, `scripts/evaluate_selective_prediction.py` consumes:
- per-participant `success`
- per-item predictions
- `item_signals` evidence counts (`llm_evidence_count`)
- (Spec 046) retrieval stats (`retrieval_reference_count`, `retrieval_similarity_mean`, `retrieval_similarity_max`)
- (Specs 048–051) confidence suite signals:
  - `verbalized_confidence`
  - `token_msp`, `token_pe`, `token_energy`
  - `consistency_modal_confidence`, `consistency_score_std`, `consistency_na_rate`, `consistency_samples`

Note: Outputs are written as strict JSON (no `NaN`/`Infinity`). Undefined aggregate metrics are emitted as `null` (BUG-048).

### `experiments[].results.results[]` (Per-participant)

Each participant result includes:
- `participant_id`, `success`, `error`
- per-item maps: `ground_truth_items`, `predicted_items`
- totals/bounds: `ground_truth_total`, `predicted_total`, `predicted_total_min`, `predicted_total_max`
- severity bounds: `severity_lower_bound`, `severity_upper_bound` (and `severity` when available)
- binary fields (Spec 062): `ground_truth_binary`, `predicted_binary`, `binary_correct` (populated when `prediction_mode="binary"`)
- optional structured sections when enabled:
  - `total_score` (when `prediction_mode="total"`)
  - `binary_classification` (when `prediction_mode="binary"`)
  - `severity_tier` (when `prediction_mode="total"`)

---

## Experiment Registry (`data/experiments/registry.yaml`)

`scripts/reproduce_results.py` updates a YAML registry after each run to make it easy to:
- list historical runs
- associate output files with run ids and git commits
- compare summary metrics without opening large JSON artifacts

Treat the registry as a convenience index; the JSON output is the authoritative raw artifact.

---

## Failure Registry (`data/outputs/failures_{run_id}.json`) (Spec 056)

Runs can emit a compact failure summary to support postmortems and trend tracking without leaking transcript text.

Top-level shape:

```json
{
  "summary": { "...": "..." },
  "failures": [
    {
      "category": "scoring_pydantic_retry_exhausted",
      "severity": "fatal",
      "message": "Exceeded maximum retries (3) for output validation",
      "participant_id": 383,
      "phq8_item": null,
      "stage": "scoring",
      "timestamp": "2026-01-03T23:23:16.828218+00:00",
      "context": { "...": "..." }
    }
  ]
}
```

Design constraints:
- No transcript text or evidence quotes should be written to this file.
- Prefer categorical `category`/`severity`, numeric counts, stable hashes, and short messages.

---

## Telemetry Registry (`data/outputs/telemetry_{run_id}.json`)

Runs may also emit a compact telemetry summary for debugging/monitoring without leaking transcript text.

Design constraints:
- No transcript text, evidence quotes, or raw LLM outputs.
- Telemetry should be categorical + aggregate (counts, stable hashes), suitable for trend tracking.
