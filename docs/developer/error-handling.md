# Error Handling and Fail-Fast Philosophy

**Audience**: Maintainers and researchers
**Last Updated**: 2026-01-01

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

## Where Silent Fallbacks Are Allowed

Silent fallbacks are generally treated as research corruption.

The only allowed exceptions should be:
- explicit debug modes (e.g., `scripts/generate_embeddings.py --allow-partial`)
- explicitly documented, narrow “best-effort” helpers that cannot affect evaluation outputs

If a fallback changes an experiment’s method, it must not be silent.

---

## Practical Debugging Guidance

When a run fails:
1. Identify the highest-level failure boundary (script vs service vs agent).
2. Group by `error_type` in logs.
3. Check whether the failure is “enabled feature broken” (should crash) vs “disabled feature” (should not touch files).

See: [Retrieval debugging](../embeddings/debugging-retrieval-quality.md).

---

## Related Docs

- [Exception taxonomy](exceptions.md)
- [Feature reference](../pipeline-internals/features.md)
