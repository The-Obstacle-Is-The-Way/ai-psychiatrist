# Reproduction Run Output Schema (JSON + Registry)

**Audience**: Researchers parsing outputs and maintaining run provenance
**Last Updated**: 2026-01-04

This repo writes three primary provenance artifacts for quantitative reproduction runs:

1. `data/outputs/{mode}_{split}_{YYYYMMDD_HHMMSS}.json`
2. `data/experiments/registry.yaml` (append/update registry of runs)
3. `data/outputs/failures_{run_id}.json` (failure summary; Spec 056)

SSOT implementation:
- `scripts/reproduce_results.py` (writer)
- `src/ai_psychiatrist/services/experiment_tracking.py` (filename + provenance helpers)
- `src/ai_psychiatrist/infrastructure/observability.py` (failure registry; Spec 056)

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
- MAE variants (`item_mae_weighted`, `item_mae_by_item`, `item_mae_by_subject`)
- coverage (`prediction_coverage`)
- per-item breakdowns
- per-participant predictions
- `item_signals` (Spec 25/046 confidence signals)

For downstream AURC/AUGRC evaluation, `scripts/evaluate_selective_prediction.py` consumes:
- per-participant `success`
- per-item predictions
- `item_signals` evidence counts (`llm_evidence_count`)
- (Spec 046) retrieval stats (`retrieval_reference_count`, `retrieval_similarity_mean`, `retrieval_similarity_max`)
- (Specs 048–051) confidence suite signals:
  - `verbalized_confidence`
  - `token_msp`, `token_pe`, `token_energy`
  - `consistency_modal_confidence`, `consistency_score_std`, `consistency_na_rate`, `consistency_samples`

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
