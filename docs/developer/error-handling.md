# Error Handling and Fail-Fast Philosophy

**Audience**: Maintainers and researchers
**Last Updated**: 2026-01-04

This repo prioritizes **research-honest behavior**:
- broken features must not silently degrade
- failures must be diagnosable
- optional features must be truly optional (no hidden I/O)

---

## Core Principles

### 1) Skip If Disabled, Crash If Broken (Spec 38)

If a feature is disabled:
- do not read its files
- do not validate its artifacts
- do not warn about missing artifacts (because the feature is off)

If a feature is enabled:
- missing artifacts → crash with a clear error
- invalid artifacts → crash with a clear error

This prevents “runs that look successful” but silently used a different method.

### 2) Preserve Exception Types (Spec 39)

Do not catch `Exception` and rethrow `ValueError(...)`.
That masks whether a failure was:
- a timeout
- invalid JSON
- missing file
- schema mismatch

Instead:
- log the error and `error_type`
- re-raise the original exception

### 3) Fail-Fast Artifact Generation (Spec 40)

Embedding artifacts must be complete or the run is scientifically corrupted.
Therefore:
- embedding generation is strict by default
- “partial output” is an explicit debug mode only

---

## Pipeline Robustness (Specs 053-057)

These specs enforce fail-fast behavior at critical pipeline stages:

| Spec | What It Validates | Where | Failure Mode |
|------|------------------|-------|--------------|
| 053 | Evidence grounding | `_extract_evidence()` | `EvidenceGroundingError` if all quotes ungrounded |
| 054 | Evidence schema | `_extract_evidence()` | `EvidenceSchemaError` on wrong types |
| 055 | Embedding validity | Query/reference generation, similarity | `EmbeddingValidationError` on NaN/Inf/zero |
| 056 | Failure observability | Per-run | `failures_{run_id}.json` artifact |
| 057 | Dimension invariants | Reference store load | `EmbeddingDimensionMismatchError` by default |

SSOT:
- Evidence validation: `src/ai_psychiatrist/services/evidence_validation.py`
- Embedding validation: `src/ai_psychiatrist/infrastructure/validation.py`
- Failure registry: `src/ai_psychiatrist/infrastructure/observability.py`

---

## Failure Pattern Observability (Spec 056)

The `FailureRegistry` captures all failures with:
- consistent taxonomy (by category, severity, stage)
- per-run JSON artifacts (`data/outputs/failures_{run_id}.json`)
- privacy-safe context (hashes + counts, never transcript text)

Initialization:
```python
from ai_psychiatrist.infrastructure.observability import init_failure_registry
registry = init_failure_registry(run_id)
```

At end of run:
```python
registry.print_summary()
registry.save(Path("data/outputs"))
```

---

## Retry Telemetry (Spec 060)

The failure registry captures **terminal** failures (e.g., retry exhaustion), but runs can still be brittle even when they succeed.

Spec 060 adds a privacy-safe per-run telemetry artifact:

- `data/outputs/telemetry_{run_id}.json`

It records:
- PydanticAI retry triggers (`ModelRetry`) by extractor (`extract_quantitative`, etc.)
- JSON repair path usage (`tolerant_json_fixups`, python-literal fallback, `json-repair`)

The telemetry file includes a **capped** event list (default cap: 5,000) and reports `dropped_events` if the cap is exceeded.

Telemetry must not include transcript text or raw LLM outputs (hashes + counts only).

---

## Where Silent Fallbacks Are Allowed

Silent fallbacks are generally treated as research corruption.

The only allowed exceptions should be:
- explicit debug modes (e.g., `scripts/generate_embeddings.py --allow-partial`)
- explicitly documented, narrow "best-effort" helpers that cannot affect evaluation outputs

If a fallback changes an experiment's method, it must not be silent.

---

## Practical Debugging Guidance

When a run fails:
1. Identify the highest-level failure boundary (script vs service vs agent).
2. Group by `error_type` in logs.
3. Check whether the failure is “enabled feature broken” (should crash) vs “disabled feature” (should not touch files).

See: [RAG Debugging](../rag/debugging.md).

---

## Related Docs

- [Exception taxonomy](exceptions.md)
- [Feature reference](../pipeline-internals/features.md)
