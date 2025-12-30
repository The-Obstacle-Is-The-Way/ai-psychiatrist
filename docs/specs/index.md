# Specs

Implementation-ready (or implementation-planned) specifications for changes that require code modifications.

## Ready to Implement

- **Spec 37**: [Batch Query Embedding](./37-batch-query-embedding.md) — 8x reduction in API calls, fixes BUG-033/034/036
- **Spec 38**: [Graceful Degradation](./38-embedding-graceful-degradation.md) — tag validation fallback, fixes BUG-035
- **Spec 36**: [CRAG-style Reference Validation](./36-crag-reference-validation.md) — new method; cost + variance controls required

## Archived (Implemented)

See `docs/archive/specs/` for completed specs:

- **Spec 34**: Item-tagged reference embeddings ✅ (2025-12-29) — index-time tagging + retrieval-time filtering
- **Spec 33**: Retrieval quality guardrails ✅ (2025-12-29) — similarity threshold + context budget + XML fix
- **Spec 35**: Offline chunk-level PHQ-8 scoring ✅ (2025-12-30) — new method + protocol lock + strict validation
- **Spec 31**: Paper-parity few-shot reference formatting ✅ (2025-12-28) — 10% AURC improvement (closing delimiter later changed by Spec 33)
- **Spec 32**: Few-shot retrieval diagnostics ✅ (2025-12-28) — opt-in audit logging
