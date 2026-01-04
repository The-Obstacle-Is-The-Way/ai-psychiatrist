# Spec 060: Retry Telemetry Metrics (PydanticAI + JSON Parsing)

**Status**: ✅ Implemented (2026-01-04)
**Canonical Docs**: `docs/developer/error-handling.md`, `docs/rag/debugging.md`
**Priority**: High
**Risk**: Low (observability only; must not affect outputs)
**Effort**: Medium

---

## Problem

We repeatedly discover run invalidations late (hours in) due to:

- PydanticAI retry exhaustion (`UnexpectedModelBehavior: Exceeded maximum retries`)
- JSON parsing “repair” being applied (or not) without a durable record beyond logs

Today we have:
- Per-run **failure registry** (`data/outputs/failures_{run_id}.json`) for *terminal* failures (Spec 056)
- Structured logs that *may* show repair activity, but are not aggregated or persisted in a stable, machine-readable form

We need **privacy-safe, per-run telemetry** that answers:

1. How often did PydanticAI have to retry due to validation failures (by extractor + error type)?
2. How often were JSON repair paths used (fixups applied; python-literal fallback; json-repair fallback)?

This is necessary to:
- quantify brittleness improvements over time
- catch regressions quickly
- debug without transcript leakage

---

## Goals

- Provide deterministic, privacy-safe telemetry persisted alongside run outputs.
- Make “retry behavior” visible *even when the run succeeds*.
- Preserve SSOT: telemetry is orthogonal to evaluation outputs (no behavior changes).

---

## Non-Goals

- Changing scoring, retrieval, or evaluation behavior.
- Logging any transcript text or raw LLM outputs.
- Building dashboards; a JSON artifact + summary printout is sufficient.

---

## Requirements

### R1. New per-run telemetry artifact

Write `data/outputs/telemetry_{run_id}.json` with:

- run_id + timestamps
- counts by telemetry category
- top N (<=10) breakdowns where useful (e.g., extractor name)
- a **capped** event list (default cap: 5,000 events) plus `dropped_events` for any events beyond the cap

**Rationale**: aggregate summaries are the primary signal, but a capped event list enables post-hoc debugging without requiring log scraping. The cap prevents unbounded growth in long runs.

### R2. PydanticAI retry telemetry (attempt-level)

When an extractor raises `ModelRetry`, record a telemetry event:

- `category`: `pydantic_retry`
- `extractor`: one of `extract_quantitative`, `extract_judge_metric`, `extract_meta_review`, `extract_qualitative`
- `reason`: one of `json_parse`, `schema_validation`, `missing_structure`, `other`
- `error_type`: exception class name (`JSONDecodeError`, `ValidationError`, etc.)

**Privacy**: do not record the exception string if it may contain evidence text.

### R3. JSON repair telemetry (repair-path visibility)

When JSON parsing applies repairs, record events:

- tolerant fixups applied: `category=json_fixups_applied`, `fixes=[...]` (or one event per fix)
- python-literal fallback used: `category=json_python_literal_fallback`
- json-repair fallback used: `category=json_repair_fallback`

**Privacy**: allow only stable hashes + lengths, never raw text.

### R4. Optional / safe-by-default initialization

Telemetry collection must be:

- initialized by `scripts/reproduce_results.py` (and any other “run” entrypoints as needed)
- safe to call when uninitialized (no-op + debug log), matching `record_failure()` behavior

### R5. Must not affect experiment outputs

- Telemetry is purely additive (no changes to computed scores, coverage, AURC/AUGRC, etc.)
- Must not introduce new retry loops or alter existing ones

---

## Implementation Plan

### 1) Add a new telemetry registry (SSOT)

Create `src/ai_psychiatrist/infrastructure/telemetry.py`:

- `TelemetryCategory` enum
- `TelemetryEvent` dataclass
- `TelemetryRegistry` with:
  - `record(...)`
  - `summary()`
  - `save(output_dir)`
  - `print_summary()` (short)
  - `max_events` cap + `dropped_events` counter (memory safety; no unbounded event growth)
- `init_telemetry_registry(run_id)` + `get_telemetry_registry()` + `record_telemetry(...)`
  - Same contextvar pattern as `infrastructure/observability.py`

### 2) Wire into `scripts/reproduce_results.py`

- Initialize telemetry registry with `run_id` at start (next to `init_failure_registry`)
- At end of run:
  - print summary
  - save JSON artifact to `data/outputs/telemetry_{run_id}.json`

### 3) Instrument PydanticAI extractors

Update `src/ai_psychiatrist/agents/extractors.py`:

- In each `except ...: raise ModelRetry(...)` branch, call `record_telemetry(...)` first.
- Classify `reason`:
  - `JSONDecodeError` → `json_parse`
  - `ValidationError` → `schema_validation`
  - missing tags/structure → `missing_structure`
  - otherwise → `other`

### 4) Instrument JSON parsing

Update `src/ai_psychiatrist/infrastructure/llm/responses.py`:

- When `tolerant_json_fixups()` applies any fixups, record `json_fixups_applied`
- When `parse_llm_json()` succeeds via:
  - python-literal fallback → record `json_python_literal_fallback`
  - json-repair fallback → record `json_repair_fallback`

---

## Tests (TDD)

Create `tests/unit/infrastructure/test_telemetry.py`:

1. `TelemetryRegistry` records and summarizes events correctly.
2. `record_telemetry()` is a no-op when registry is uninitialized.
3. Registry enforces an event cap and increments `dropped_events` when exceeded.

Extend existing unit tests:

- `tests/unit/infrastructure/llm/test_tolerant_json_fixups.py`:
  - Assert telemetry increments when fallbacks are used (python literal + json-repair).

Avoid brittle assertions on exact log messages; test the telemetry artifact state.

---

## Acceptance Criteria

- [x] `data/outputs/telemetry_{run_id}.json` is written on reproduction runs
- [x] Telemetry contains hashes + counts only (no transcript text / raw LLM outputs)
- [x] Telemetry counts include pydantic retry triggers + JSON repair path usage
- [x] Telemetry event list is capped with `dropped_events` recorded (no unbounded growth)
- [x] All tests pass: `make ci`
