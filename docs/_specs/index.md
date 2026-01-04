# Specs

Implementation-ready (or implementation-planned) specifications for changes that require code modifications.

## Deferred

- **Spec 20**: [Keyword Fallback Improvements (Deferred)](../_archive/specs/20-keyword-fallback-improvements.md)

## Archived (Implemented)

Implemented specs are distilled into canonical (non-archive) documentation under `docs/`:

### Pipeline Robustness (Specs 053-057) - PR #92, 2026-01-03

| Spec | Title | Canonical Doc Location |
|------|-------|------------------------|
| **053** | [Evidence Hallucination Detection](../_archive/specs/spec-053-evidence-hallucination-detection.md) | [Evidence Extraction](../pipeline-internals/evidence-extraction.md#evidence-hallucination-detection-spec-053), [Features](../pipeline-internals/features.md#evidence-extraction-validation-specs-053-054) |
| **054** | [Strict Evidence Schema Validation](../_archive/specs/spec-054-strict-evidence-schema-validation.md) | [Evidence Extraction](../pipeline-internals/evidence-extraction.md#evidence-schema-validation-spec-054), [Exceptions](../developer/exceptions.md#evidence-validation-exceptions-specs-053-054) |
| **055** | [Embedding NaN Detection](../_archive/specs/spec-055-embedding-nan-detection.md) | [Artifact Generation](../rag/artifact-generation.md#embedding-validation-spec-055), [Debugging](../rag/debugging.md#step-6-diagnose-embedding-failures-spec-055) |
| **056** | [Failure Pattern Observability](../_archive/specs/spec-056-failure-pattern-observability.md) | [Error Handling](../developer/error-handling.md#failure-pattern-observability-spec-056), [Debugging](../rag/debugging.md#step-5-check-failure-registry-spec-056) |
| **057** | [Embedding Dimension Strict Mode](../_archive/specs/spec-057-embedding-dimension-strict-mode.md) | [Artifact Generation](../rag/artifact-generation.md#dimension-invariants-spec-057), [Configuration](../configs/configuration-philosophy.md) |

### Other Implemented Specs

- Quantitative severity bounds (BUG-045): [spec](../_archive/specs/spec-045-quantitative-severity-bounds.md) → [PHQ-8 docs](../clinical/phq8.md#severity-bounds-partial-assessments)
- Feature index + defaults: [features.md](../pipeline-internals/features.md)
- RAG runtime features (prompt format, CRAG, batch embedding): [runtime-features.md](../rag/runtime-features.md)
- RAG debugging workflow: [debugging.md](../rag/debugging.md)
- RAG artifact generation (embeddings + tags): [artifact-generation.md](../rag/artifact-generation.md)
- Chunk scoring setup + schema: [chunk-scoring.md](../rag/chunk-scoring.md)
- Error handling philosophy: [error-handling.md](../developer/error-handling.md)
- Exception taxonomy: [exceptions.md](../developer/exceptions.md)
- Metrics definitions + output schema: [metrics-and-evaluation.md](../statistics/metrics-and-evaluation.md)
- Selective prediction confidence signals (Spec 046): [spec](../_archive/specs/spec-046-selective-prediction-confidence-signals.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Verbalized confidence (Spec 048): [spec](../_archive/specs/spec-048-verbalized-confidence.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Supervised confidence calibrator (Spec 049): [spec](../_archive/specs/spec-049-supervised-confidence-calibrator.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Consistency-based confidence (Spec 050): [spec](../_archive/specs/spec-050-consistency-based-confidence.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Advanced CSFs from fd-shifts (Spec 051): [spec](../_archive/specs/spec-051-advanced-csf-from-fd-shifts.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Excess AURC/AUGRC metrics (Spec 052): [spec](../_archive/specs/spec-052-excess-aurc-augrc-metrics.md) → [metrics docs](../statistics/metrics-and-evaluation.md#optimal-and-excess-metrics-spec-052)
- Remove keyword backfill (Spec 047): [spec](../_archive/specs/spec-047-remove-keyword-backfill.md) → [configuration philosophy](../configs/configuration-philosophy.md)
- Configuration philosophy: [configuration-philosophy.md](../configs/configuration-philosophy.md)
- DAIC-WOZ transcript preprocessing + variants: [spec](../_archive/specs/daic-woz-transcript-preprocessing.md) → [user guide](../data/daic-woz-preprocessing.md)

Historical spec texts remain in `docs/_archive/specs/` for provenance, but the active documentation
should not require them.
