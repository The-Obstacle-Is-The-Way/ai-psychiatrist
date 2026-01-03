# Specs

Implementation-ready (or implementation-planned) specifications for changes that require code modifications.

## Ready to Implement (AUGRC Improvement Suite)

These specs form the **gold-standard AUGRC improvement roadmap**, derived from [fd-shifts](https://github.com/IML-DKFZ/fd-shifts) (NeurIPS 2024), [AsymptoticAURC](https://arxiv.org/abs/2410.15361) (ICML 2025), and 2025-2026 LLM uncertainty research.

### Phase 2: Verbalized Confidence

- **Spec 048**: [Verbalized Confidence](spec-048-verbalized-confidence.md) — Prompt LLM to output confidence rating (1-5), apply temperature scaling calibration. Expected: 20-40% AUGRC reduction.

### Phase 3: Multi-Signal Calibration

- **Spec 049**: [Supervised Confidence Calibrator](spec-049-supervised-confidence-calibrator.md) — Train logistic/isotonic regression on multiple signals (evidence, retrieval similarity, verbalized confidence). Expected: 30-50% AUGRC reduction.

### Alternative: Consistency-Based Confidence

- **Spec 050**: [Consistency-Based Confidence](spec-050-consistency-based-confidence.md) — Multiple inference passes (N=5), measure agreement. Trade-off: 5x inference time. Expected: 20-35% AUGRC reduction.

### Enrichment: Advanced CSFs

- **Spec 051**: [Advanced CSFs from fd-shifts](spec-051-advanced-csf-from-fd-shifts.md) — Port token-level CSFs (MSP, entropy, energy) and secondary combinations from fd-shifts benchmark.

### Enrichment: Excess Metrics

- **Spec 052**: [Excess AURC/AUGRC Metrics](spec-052-excess-aurc-augrc-metrics.md) — Compute optimal baselines and excess metrics (e-AURC, e-AUGRC) for interpretable progress tracking.

### Implementation Priority

| Spec | Priority | Effort | Expected AUGRC |
|------|----------|--------|----------------|
| 048 | High | Medium | ~0.024 |
| 049 | High | Medium-High | ~0.018 |
| 050 | Medium | Medium-High | ~0.020 |
| 051 | Medium | Low-Medium | — |
| 052 | Low | Low | — |

**Recommended path**: Spec 048 → Spec 049 → Spec 052 (for tracking) → Spec 051 (enrichment)

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
- Selective prediction confidence signals (Spec 046): [spec](../_archive/specs/spec-046-selective-prediction-confidence-signals.md) → [metrics docs](../statistics/metrics-and-evaluation.md#confidence-variants)
- Remove keyword backfill (Spec 047): [spec](../_archive/specs/spec-047-remove-keyword-backfill.md) → [configuration philosophy](../configs/configuration-philosophy.md)
- Configuration philosophy: [configuration-philosophy.md](../configs/configuration-philosophy.md)
- DAIC-WOZ transcript preprocessing + variants: [spec](../_archive/specs/daic-woz-transcript-preprocessing.md) → [user guide](../data/daic-woz-preprocessing.md)

Historical spec texts remain in `docs/_archive/specs/` for provenance, but the active documentation
should not require them.
