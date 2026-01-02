# Spec 045: Quantitative Severity Bounds for Partial PHQ-8 (BUG-045)

**Status**: Implemented
**Primary implementation**: `src/ai_psychiatrist/domain/entities.py` (`PHQ8Assessment`)
**Integration points**: `src/ai_psychiatrist/agents/quantitative.py`, `server.py`, `scripts/reproduce_results.py`
**Verification**: `uv run pytest tests/ --tb=short` (2026-01-02)

## 0. Problem Statement

The quantitative path supports **abstention** at the PHQ-8 item level by emitting `N/A` when there is insufficient
evidence. However, the current domain model derives a single PHQ-8 `total_score` and `severity` label by treating
`N/A` as `0`. This produces **systematic severity underestimation** whenever any items are unknown.

This is clinically misleading because PHQ-8 severity bands (Minimal/Mild/Moderate/Moderately Severe/Severe) are
defined for a **complete 8-item total**, not a partial lower bound.

## 1. Goals / Non-Goals

### 1.1 Goals

- Prevent misleading single-label severity classifications when the assessment is incomplete.
- Provide deterministic, auditable **bounds** for totals and severity:
  - `min_total_score` (lower bound): treat `N/A` as `0` (current behavior).
  - `max_total_score` (upper bound): treat `N/A` as `3` (max per item).
  - `severity_lower_bound` and `severity_upper_bound` derived from those bounds.
- Keep paper-parity item-level metrics unchanged (MAE is computed per-item excluding `N/A`).
- Make call sites (API + logs) explicitly surface “partial vs complete” classification.

### 1.2 Non-Goals

- Imputing missing items (e.g., scaling totals, probabilistic inference).
- Changing quantitative prompting strategy or the meaning of `N/A`.
- Changing the meta-review agent’s severity prediction workflow.

## 2. Definitions (First Principles)

Given per-item scores `s_i ∈ {0,1,2,3} ∪ {N/A}` for i=1..8:

- `min_total_score = Σ score_i` where `N/A → 0`
- `max_total_score = Σ score_i` where `N/A → 3`

Severity bounds:

- `severity_lower_bound = SeverityLevel.from_total_score(min_total_score)`
- `severity_upper_bound = SeverityLevel.from_total_score(max_total_score)`

Determinate severity:

- `severity` is only defined when `severity_lower_bound == severity_upper_bound`
  (i.e., missing items cannot change the band).

## 3. Domain API Changes (PHQ8Assessment)

Update `src/ai_psychiatrist/domain/entities.py`:

### 3.1 New properties

- `min_total_score: int` (lower bound; equals legacy `total_score`)
- `max_total_score: int` (upper bound)
- `total_score_bounds: tuple[int, int]` returning `(min_total_score, max_total_score)`
- `severity_lower_bound: SeverityLevel`
- `severity_upper_bound: SeverityLevel`
- `severity_bounds: tuple[SeverityLevel, SeverityLevel]`
- `is_complete: bool` (`na_count == 0`)

### 3.2 Updated `severity` semantics

- Change `PHQ8Assessment.severity` to return `SeverityLevel | None`.
- Return a `SeverityLevel` only when the severity bounds are equal; otherwise return `None`.

### 3.3 Backward compatibility of `total_score`

- Keep `PHQ8Assessment.total_score` behavior unchanged (lower bound) to avoid cascading breakage.
- Update docstrings to explicitly label it as a lower bound.

## 4. Integration Updates

### 4.1 Quantitative logging

Update `src/ai_psychiatrist/agents/quantitative.py` to avoid calling `.name` on an indeterminate severity.

Log fields should include:

- `total_score_min`, `total_score_max`
- `severity` (string or `None`)
- `severity_lower_bound`, `severity_upper_bound`
- `na_count`

### 4.2 API output (`/assess/quantitative`, `/full_pipeline`)

Update `server.py` response model `QuantitativeResult`:

- Make `severity` nullable (`str | None`).
- Add:
  - `total_score_min: int`
  - `total_score_max: int`
  - `severity_lower_bound: str`
  - `severity_upper_bound: str`

`total_score` remains present and equals `total_score_min` for compatibility.

### 4.3 Reproduction run outputs (optional but recommended)

Update `scripts/reproduce_results.py` JSON output to include:

- `predicted_total_min`
- `predicted_total_max`
- `severity_lower_bound`
- `severity_upper_bound`
- `severity` (nullable / determinate-only)

This is additive and should not break consumers.

## 5. Test Plan (TDD)

### 5.1 Domain unit tests

Add tests to `tests/unit/domain/test_entities.py`:

- Partial scoring yields correct `(min_total_score, max_total_score)` bounds.
- `severity` is `None` when bounds differ.
- `severity` is determinate (non-None) when bounds are equal even if items are missing.

### 5.2 Agent unit tests

Update `tests/unit/agents/test_quantitative.py`:

- Replace the current “severity is Mild” assertion for a partial assessment with:
  - `severity is None`
  - bounds match expected bands (e.g., `MILD..MODERATE` for 6 observed + 2 unknown).

### 5.3 Type safety

Because `.severity` becomes optional, update any `.severity.is_mdd` usages to guard with
`assert result.severity is not None` before attribute access.

## 6. Acceptance Criteria

- All tests pass: `uv run pytest tests/ -v --tb=short`
- Lint passes: `uv run ruff check`
- Types pass: `uv run mypy src tests scripts --strict`
- API responses never return a misleading single severity label for incomplete quantitative assessments:
  - `severity is None` unless bounds are equal
  - bounds are always present
