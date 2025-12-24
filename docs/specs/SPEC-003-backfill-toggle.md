# SPEC-003: Keyword Backfill Toggle + N/A Reason Tracking

**GitHub Issue**: [#49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
**Status**: Implemented (PR #51)
**Created**: 2025-12-23
**Last Updated**: 2025-12-24

---

## Problem Statement

The paper reports substantial abstention during quantitative scoring:

> “In ~50% of cases, the model was unable to provide a prediction due to insufficient evidence.”

This repository includes a keyword-based backfill step that can increase coverage by injecting
keyword-matched sentences as evidence when the LLM misses them. That is useful for clinical
utility, but it can materially change the **coverage / MAE tradeoff**, making direct paper
comparisons harder.

SPEC-003 makes this behavior **explicit and configurable**, and adds deterministic “why N/A?”
metadata for debugging and analysis.

---

## Goals

1. **Paper parity by default**: backfill OFF unless explicitly enabled.
2. **Configurable backfill**: allow higher-coverage runs when desired.
3. **Observability**: tag each N/A with a deterministic reason (optional).
4. **Reproducibility**: ensure scripts/server use the same settings source (`Settings.quantitative`).

---

## Configuration

Implemented in `src/ai_psychiatrist/config.py` as `QuantitativeSettings`, nested under
`Settings.quantitative`.

Environment variables (Pydantic Settings):

- `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` (bool, default `false`)
- `QUANTITATIVE_TRACK_NA_REASONS` (bool, default `true`)
- `QUANTITATIVE_KEYWORD_BACKFILL_CAP` (int, default `3`, range 1–10)

Paper parity mode requires **no configuration** (defaults apply).

---

## N/A Reason Taxonomy

Implemented as `NAReason` in `src/ai_psychiatrist/domain/enums.py`.

These reasons are computed deterministically from pipeline state:

| Reason | Description | When |
|--------|-------------|------|
| `NO_MENTION` | Neither the LLM extractor nor keyword matcher found any evidence. | `llm_count=0` and `keyword_hits=0` |
| `LLM_ONLY_MISSED` | Keywords would have matched, but backfill is disabled (paper parity), so the scorer saw no evidence. | `llm_count=0`, `keyword_hits>0`, `backfill=false` |
| `KEYWORDS_INSUFFICIENT` | Keywords matched and were provided as evidence (backfill enabled), but the scorer still produced N/A. | `llm_count=0`, `keyword_hits>0`, `backfill=true` |
| `SCORE_NA_WITH_EVIDENCE` | LLM extraction found evidence, but the scorer still produced N/A. | `llm_count>0` and `score is N/A` |

Notes:
- `na_reason` is only populated when `score is None` **and** `track_na_reasons=true`.
- These categories intentionally avoid “subjective” heuristics. They are derived from counts and flags.

---

## Evidence Source + Counts

Implemented as `ItemAssessment` extensions in `src/ai_psychiatrist/domain/value_objects.py`.

| Field | Type | Meaning |
|------|------|---------|
| `evidence_source` | `Literal["llm","keyword","both"] \| None` | Evidence provided to scorer (`None` means no evidence). |
| `llm_evidence_count` | `int` | Evidence count from LLM extractor (per item). |
| `keyword_evidence_count` | `int` | Keyword sentences added to the scorer evidence (per item). |

Important: `keyword_evidence_count` reflects **evidence added** (not “keyword hits that would have matched”).
Keyword hits are still computed for N/A reason tracking when enabled.

---

## Implementation Overview

### Agent Behavior

Implemented in `src/ai_psychiatrist/agents/quantitative.py`:

1. Run LLM evidence extraction → `llm_evidence` (dict keyed by `PHQ8_*` keys).
2. Optionally compute keyword hits:
   - computed when `enable_keyword_backfill=true` (needed to backfill), or
   - computed when `track_na_reasons=true` (needed to classify N/A).
3. If `enable_keyword_backfill=true`, merge keyword hits into evidence up to cap:
   - keyword matching: `_find_keyword_hits()`
   - evidence merge: `_merge_evidence()`
4. Score with LLM using evidence (+ optional few-shot references).
5. Construct `ItemAssessment` objects and populate:
   - `evidence_source`, `llm_evidence_count`, `keyword_evidence_count`
   - `na_reason` when enabled and score is N/A

### Server Wiring

Implemented in `server.py`:
- `QuantitativeAssessmentAgent(..., quantitative_settings=app_settings.quantitative)`

### Script Wiring

Implemented in `scripts/reproduce_results.py`:
- reads `get_settings().quantitative` and passes into `QuantitativeAssessmentAgent`.

---

## Acceptance Criteria

- [x] Backfill can be enabled/disabled with `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL`.
- [x] Default behavior is paper parity (`enable_keyword_backfill=false`).
- [x] When backfill is disabled, keyword matches are **not** injected into scorer evidence.
- [x] When backfill is enabled, evidence is merged up to `QUANTITATIVE_KEYWORD_BACKFILL_CAP`.
- [x] When `QUANTITATIVE_TRACK_NA_REASONS=true`, N/A items receive deterministic `na_reason`.
- [x] When `QUANTITATIVE_TRACK_NA_REASONS=false`, `na_reason` is never populated.
- [x] Docs and `.env.example` describe the toggle and defaults accurately.

---

## Tests

Minimum coverage expectations (unit-level):

- Backfill OFF produces `LLM_ONLY_MISSED` for keyword-hit-but-no-LLM-evidence case.
- Backfill ON increases `keyword_evidence_count` and sets `evidence_source="keyword"` for that case.
- `track_na_reasons=false` results in `na_reason is None` even when score is N/A.
- `NAReason` enum values match this spec exactly.

Existing test locations:
- `tests/unit/agents/test_quantitative_backfill.py`
- `tests/unit/domain/test_enums.py`
- `tests/unit/domain/test_value_objects.py`
