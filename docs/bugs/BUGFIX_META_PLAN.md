# Bugfix Meta Plan (BUG-004 through BUG-017)

This document defines the order and grouping for fixing bugs identified in
`docs/bugs/BUG-004` through `docs/bugs/BUG-017`. The goal is to prioritize
runtime correctness, paper fidelity, and integration stability while avoiding
unnecessary work on legacy-only code until the modern pipeline is stable.

## Guiding Principles

- Fix active runtime paths first (P0/P1).
- Fix configuration and model selection before data artifact generation.
- Fix embedding correctness before attempting paper-metric verification.
- Legacy cleanup happens last, after functional parity is confirmed.

## Scope and Current Status

- **OPEN**: BUG-006, BUG-008, BUG-009, BUG-010, BUG-011, BUG-013, BUG-015, BUG-016, BUG-017
- **RESOLVED**: BUG-004, BUG-005, BUG-007, BUG-012, BUG-014

## Chunked Fix Order

### Chunk 1: Runtime Entry Point Correctness (P0/P1) ✅ COMPLETED

**Bugs**: BUG-012, BUG-014
**Status**: RESOLVED (2025-12-19)

**Rationale**: The API currently routes through legacy agents and a missing
transcript path. This is a production correctness issue and blocks all
meaningful end-to-end validation.

**Primary Tasks**:
- ✅ Rewire `server.py` to the modern `src/ai_psychiatrist` agents/services.
- ✅ Replace the legacy transcript file dependency with `TranscriptService`.

**Exit Criteria**:
- ✅ API uses modern agents and config settings.
- ✅ All 583 tests pass.

---

### Chunk 2: Model and Parsing Correctness (P1/P3)

**Bugs**: BUG-008, BUG-011  
**Rationale**: Configuration is currently ignored for model selection, and
evidence parsing silently degrades. Fixing these ensures the pipeline uses the
intended models and maintains robustness under noisy LLM output.

**Primary Tasks**:
- Thread `ModelSettings` into all LLM/embedding calls (model, temperature, top_k, top_p).
- Apply tolerant parsing for evidence extraction (or add repair path).

**Exit Criteria**:
- Configuration overrides demonstrably control models and sampling.
- Evidence extraction survives common JSON formatting noise.

---

### Chunk 3: Embedding Retrieval Fidelity (P1/P2)

**Bugs**: BUG-009, BUG-010, BUG-006  
**Rationale**: Embedding retrieval is core to few-shot performance. Dimension
mismatch handling and similarity range semantics must be correct before
generating reference embeddings.

**Primary Tasks**:
- Decide and enforce similarity range semantics (either allow -1..1 or keep 0..1).
- Make dimension mismatches explicit (error or structured warning).
- Generate the reference embeddings artifact after the above is stable.

**Exit Criteria**:
- Dimension mismatches are explicit (no silent skip).
- Similarity semantics match domain constraints.
- `data/embeddings/participant_embedded_transcripts.pkl` exists and loads.

---

### Chunk 4: Pipeline Completion (P1/P2)

**Bugs**: BUG-016, BUG-017
**Rationale**: The full paper pipeline requires MetaReviewAgent (Spec 10) and
FeedbackLoopService integration. These were spec'd but never implemented or wired.
Without these, paper replication (Section 2.3.3, 78% accuracy) is not achievable.

**Primary Tasks**:
- Implement `MetaReviewAgent` from Spec 10 into `src/ai_psychiatrist/agents/meta_review.py`.
- Wire `FeedbackLoopService` into `server.py` for qualitative assessment iteration.
- Update `server.py` `/full_pipeline` to include meta-review step.
- Export `MetaReviewAgent` from agents `__init__.py`.

**Exit Criteria**:
- `MetaReviewAgent` exists and is exported.
- `/full_pipeline` runs: qualitative (with feedback loop) → quantitative → meta-review.
- Response includes severity prediction and explanation from meta-review.

---

### Chunk 5: Legacy Cleanup and Hygiene (P2)

**Bugs**: BUG-013, BUG-015  
**Rationale**: These are legacy-only concerns. Cleaning after the modern
pipeline is stable avoids losing reference behavior before parity is achieved.

**Primary Tasks**:
- Archive or remove legacy directories once parity is confirmed.
- Replace hardcoded absolute paths in retained scripts with `DataSettings`.

**Exit Criteria**:
- Legacy directories are archived or removed with no runtime references.
- Any retained scripts are portable and path-configurable.

## Notes

- BUG-004, BUG-005, BUG-007 are resolved and should not block any chunk.
- If Chunk 1 requires a different API shape, record that change in the specs
  (Spec 11 or a checkpoint addendum) for traceability.
