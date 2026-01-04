# Exception Reference (Domain + Runtime)

**Audience**: Maintainers and debugging-focused researchers
**Last Updated**: 2026-01-01

This page documents the exception taxonomy used across the repo and how callers should handle errors without corrupting research runs.

SSOT:
- `src/ai_psychiatrist/domain/exceptions.py`
- [Error-handling philosophy](error-handling.md)

---

## Domain Exceptions

All domain exceptions inherit from `DomainError`.

Defined in `src/ai_psychiatrist/domain/exceptions.py`:

- `DomainError`
  - `ValidationError`
  - `TranscriptError`
    - `EmptyTranscriptError`
  - `AssessmentError`
    - `PHQ8ItemError`
    - `InsufficientEvidenceError`
  - `EvaluationError`
    - `LowScoreError`
    - `MaxIterationsError`
  - `LLMError`
    - `LLMResponseParseError`
    - `LLMTimeoutError`
  - `EmbeddingError`
    - `EmbeddingDimensionMismatchError`
    - `EmbeddingArtifactMismatchError`
    - `EmbeddingValidationError` (NaN/Inf/zero detection - Spec 055)

---

## Evidence Validation Exceptions (Specs 053-054)

These are raised during evidence extraction validation:

- `EvidenceSchemaError`
  - Raised when extracted evidence JSON has wrong types (e.g., string instead of list)
  - Contains `violations` dict with per-field error details
  - SSOT: `src/ai_psychiatrist/services/evidence_validation.py`

- `EvidenceGroundingError`
  - Raised when all extracted evidence quotes are ungrounded (not found in transcript)
  - Prevents few-shot silently degrading to zero-shot
  - SSOT: `src/ai_psychiatrist/services/evidence_validation.py`

---

## Handling Boundaries (What Should Catch What)

### API Layer (FastAPI)

API endpoints should:
- catch `DomainError` and map to a safe error response (HTTP 4xx/5xx with clear message)
- allow unexpected exceptions to propagate (500 + stack trace in logs)

### Scripts (Research Runs)

Long-running scripts should be explicit about failure semantics:
- **Evaluation scripts**: participant failures are tracked and excluded from AURC/AUGRC metrics (do not silently drop participants).
- **Artifact generation**: strict-by-default (fail-fast) unless an explicit debug flag allows partial output.

### Agents / Services

Do not wrap unknown exceptions in generic `ValueError`. Preserve exception types (Spec 39) so timeouts vs parse errors vs validation errors remain distinguishable.

---

## Debugging Aids

Most log sites include:
- `error` (string)
- `error_type` (class name)

When triaging failures, start by grouping by `error_type`.

---

## Related Docs

- [Error-handling philosophy](error-handling.md)
- [Configuration (timeouts, backends)](../configs/configuration.md)
