# Specs

Implementation-ready (or implementation-planned) specifications for changes that require code modifications.

## Ready to Implement

_No active specs._

## Deferred

- **Spec 20**: [Keyword Fallback Improvements (Deferred)](../_archive/specs/20-keyword-fallback-improvements.md)

## Archived (Implemented)

Implemented specs are distilled into canonical (non-archive) documentation under `docs/`:

- Quantitative severity bounds (BUG-045): [spec](../_archive/specs/spec-045-quantitative-severity-bounds.md) → [PHQ-8 docs](../clinical/phq8.md#severity-bounds-partial-assessments)
- Feature index + defaults: [features.md](../pipeline-internals/features.md)
- Few-shot reference prompt format: [few-shot-prompt-format.md](../embeddings/few-shot-prompt-format.md)
- Retrieval debugging workflow: [debugging-retrieval-quality.md](../embeddings/debugging-retrieval-quality.md)
- Item-tag filtering setup + schema: [item-tagging-setup.md](../embeddings/item-tagging-setup.md)
- Chunk scoring setup + schema: [chunk-scoring.md](../embeddings/chunk-scoring.md)
- CRAG validation setup: [crag-validation-guide.md](../statistics/crag-validation-guide.md)
- Embedding generation (fail-fast + partial): [embedding-generation.md](../embeddings/embedding-generation.md)
- Error handling philosophy: [error-handling.md](../developer/error-handling.md)
- Exception taxonomy: [exceptions.md](../developer/exceptions.md)
- Metrics definitions + output schema: [metrics-and-evaluation.md](../statistics/metrics-and-evaluation.md)
- Selective prediction confidence signals: [spec](spec-046-selective-prediction-confidence-signals.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Configuration philosophy: [configuration-philosophy.md](../configs/configuration-philosophy.md)
- DAIC-WOZ transcript preprocessing + variants: [spec](../_archive/specs/daic-woz-transcript-preprocessing.md) → [user guide](../data/daic-woz-preprocessing.md)

Historical spec texts remain in `docs/_archive/specs/` for provenance, but the active documentation
should not require them.
