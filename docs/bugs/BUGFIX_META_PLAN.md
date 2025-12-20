# Bugfix Meta Plan (BUG-004 through BUG-018)

This document defines the order and grouping for fixing bugs identified in
`docs/bugs/BUG-004` through `docs/bugs/BUG-018`. The goal is to prioritize
runtime correctness, paper fidelity, and integration stability while avoiding
unnecessary work on legacy-only code until the modern pipeline is stable.

## Guiding Principles

- Fix active runtime paths first (P0/P1).
- Fix configuration and model selection before data artifact generation.
- Fix embedding correctness before attempting paper-metric verification.
- Legacy cleanup happens last, after functional parity is confirmed.

## Scope and Current Status

- **OPEN**: BUG-018
- **RESOLVED**: BUG-004, BUG-005, BUG-006, BUG-007, BUG-008, BUG-009, BUG-010, BUG-011, BUG-012, BUG-013, BUG-014, BUG-015, BUG-016, BUG-017

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

### Chunk 2: Model and Parsing Correctness (P1/P3) ✅ COMPLETED

**Bugs**: BUG-008, BUG-011
**Status**: RESOLVED (2025-12-19)

**Rationale**: Configuration is currently ignored for model selection, and
evidence parsing silently degrades. Fixing these ensures the pipeline uses the
intended models and maintains robustness under noisy LLM output.

**Primary Tasks**:

- ✅ Thread `ModelSettings` into all LLM/embedding calls (model, temperature, top_k, top_p).
- ✅ Apply tolerant parsing for evidence extraction (or add repair path).

**Exit Criteria**:

- ✅ Configuration overrides demonstrably control models and sampling.
- ✅ Evidence extraction survives common JSON formatting noise.
- ✅ All 583 tests pass.

---

### Chunk 3: Embedding Retrieval Fidelity (P1/P2) ✅ COMPLETED

**Bugs**: BUG-009, BUG-010, BUG-006
**Status**: RESOLVED (2025-12-20)

**Rationale**: Embedding retrieval is core to few-shot performance. Dimension
mismatch handling and similarity range semantics must be correct before
generating reference embeddings.

**Primary Tasks**:

- ✅ Decide and enforce similarity range semantics: Chose `(1 + cos) / 2` transformation.
- ✅ Make dimension mismatches explicit: Raises `EmbeddingDimensionMismatchError` on full mismatch,
  logs warnings for partial mismatches.
- ✅ Generate the reference embeddings artifact: Created `scripts/generate_embeddings.py`.

**Exit Criteria**:

- ✅ Dimension mismatches are explicit (no silent skip).
- ✅ Similarity semantics match domain constraints (transformed to [0, 1]).
- ✅ `scripts/generate_embeddings.py` exists and is ready to generate artifact.

---

### Chunk 4: Pipeline Completion (P1/P2) ✅ COMPLETED

**Bugs**: BUG-016, BUG-017
**Status**: RESOLVED (2025-12-20)

**Rationale**: The full paper pipeline requires MetaReviewAgent (Spec 10) and
FeedbackLoopService integration. These were spec'd but never implemented or wired.
Without these, paper replication (Section 2.3.3, 78% accuracy) is not achievable.

**Primary Tasks**:

- ✅ Implement `MetaReviewAgent` from Spec 10 into `src/ai_psychiatrist/agents/meta_review.py`.
- ✅ Wire `FeedbackLoopService` into `server.py` for qualitative assessment iteration.
- ✅ Update `server.py` `/full_pipeline` to include meta-review step.
- ✅ Export `MetaReviewAgent` from agents `__init__.py`.

**Exit Criteria**:

- ✅ `MetaReviewAgent` exists and is exported.
- ✅ `/full_pipeline` runs: qualitative (with feedback loop) → quantitative → meta-review.
- ✅ Response includes severity prediction and explanation from meta-review.

---

### Chunk 5: Legacy Cleanup and Hygiene (P2) ✅ COMPLETED

**Bugs**: BUG-013, BUG-015
**Status**: RESOLVED (2025-12-20)

**Rationale**: These are legacy-only concerns. Cleaning after the modern
pipeline is stable avoids losing reference behavior before parity is achieved.

**Primary Tasks**:

- ✅ Archive legacy directories to `_legacy/`:
  - `agents/`, `meta_review/`, `qualitative_assessment/`, `quantitative_assessment/`
  - `slurm/`, `assets/`, `visualization/`, `analysis_output/`
- ✅ Update `pyproject.toml` ruff exclude to use single `_legacy/` entry.
- ✅ Verify no active imports from legacy directories.

**Exit Criteria**:

- ✅ Legacy directories archived with no runtime references.
- ✅ 603 tests pass at 96.52% coverage.
- ✅ Lint passes with simplified exclude list.

---

### Chunk 6: Real Ollama Integration Testing (P1)

**Bugs**: BUG-018
**Rationale**: All 603 unit tests use mocks. We have **zero verification** that
the pipeline works with real Ollama. This is a critical gap that must be closed
before claiming operational readiness. The architecture is complete (Chunks 1-5),
but we cannot claim the pipeline works without real integration tests.

**Primary Tasks**:

- Create `tests/integration/` directory with real Ollama tests.
- Add pytest markers (`@pytest.mark.slow`, `@pytest.mark.integration`).
- Write vertical slice tests for each agent hitting real Ollama.
- Write E2E test for `/full_pipeline` with real DAIC-WOZ transcript.
- Verify embeddings generate with correct dimensions.
- Verify JSON parsing handles real LLM output (not just mock responses).

**Exit Criteria**:

- At least one real integration test per agent passes with Ollama running.
- `/full_pipeline` E2E test returns valid response structure.
- Any parsing/prompt issues discovered are documented as new bugs.
- Confidence that the pipeline actually works, not just appears to work.

---

## Notes

- **Chunks 1-5 COMPLETE**: All architectural work is done. MetaReviewAgent implemented,
  FeedbackLoopService wired, legacy code archived.
- **Chunk 6 is the final gate**: Real Ollama integration testing is required before
  claiming operational readiness.
- All 603 unit tests pass at 96.52% coverage - but these are mock-only tests.
- The codebase is clean and lint-free with legacy code properly archived.
