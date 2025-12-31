# Specs

Implementation-ready (or implementation-planned) specifications for changes that require code modifications.

## Ready to Implement

- (none) — see **Archived (Implemented)** below

## Deferred

- **Spec 20**: [Keyword Fallback Improvements (Deferred)](./20-keyword-fallback-improvements.md)

## Archived (Implemented)

See `docs/archive/specs/` for completed specs:

- **Spec 40**: [Fail-fast embedding generation](../archive/specs/40-fail-fast-embedding-generation.md) ✅ (2025-12-30) — strict by default, `--allow-partial` opt-in
- **Spec 37**: [Batch Query Embedding](../archive/specs/37-batch-query-embedding.md) ✅ (2025-12-30) — 8x reduction in embedding calls, configurable query timeout
- **Spec 38**: [Conditional Feature Loading](../archive/specs/38-conditional-feature-loading.md) ✅ (2025-12-30) — skip-if-disabled, crash-if-broken
- **Spec 39**: [Preserve Exception Types](../archive/specs/39-preserve-exception-types.md) ✅ (2025-12-30) — stop masking errors
- **Spec 34**: [Item-tagged reference embeddings](../archive/specs/34-item-tagged-reference-embeddings.md) ✅ (2025-12-29) — index-time tagging + retrieval-time filtering
- **Spec 33**: [Retrieval quality guardrails](../archive/specs/33-retrieval-quality-guardrails.md) ✅ (2025-12-29) — similarity threshold + context budget + XML fix
- **Spec 35**: [Offline chunk-level PHQ-8 scoring](../archive/specs/35-offline-chunk-level-phq8-scoring.md) ✅ (2025-12-30) — new method + protocol lock + strict validation
- **Spec 36**: [CRAG-style reference validation](../archive/specs/36-crag-reference-validation.md) ✅ (2025-12-30) — optional runtime validator (default OFF)
- **Spec 31**: [Paper-parity few-shot reference formatting](../archive/specs/31-paper-parity-reference-examples-format.md) ✅ (2025-12-28) — 10% AURC improvement (closing delimiter later changed by Spec 33)
- **Spec 32**: [Few-shot retrieval diagnostics](../archive/specs/32-few-shot-retrieval-diagnostics.md) ✅ (2025-12-28) — opt-in audit logging
