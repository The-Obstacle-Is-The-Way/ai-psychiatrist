# BUG-018: No Real Ollama Integration Tests

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

All 583 tests use `MockLLMClient` - **zero tests actually hit Ollama**. This means:
- We have no verification that prompts work with real models
- We have no verification that JSON parsing handles real LLM output
- We have no verification that embeddings are generated correctly
- We have no verification that the full pipeline produces valid assessments

This is a critical gap that violates vertical slice and integration testing principles. The codebase may appear functional but could fail catastrophically when connected to real infrastructure.

---

## Evidence

- All agent tests use `MockLLMClient` with canned responses (`tests/unit/agents/`)
- All service tests use mocked LLM clients (`tests/unit/services/`)
- No integration tests exist in `tests/integration/` that hit real Ollama
- No E2E tests exist that run the full pipeline with real models
- Spec 09.5 defines an integration checkpoint but no real integration tests were written

---

## Impact

- **False confidence**: 583 passing tests give illusion of correctness
- **Hidden failures**: Real LLM output may break JSON parsing, scoring, or evidence extraction
- **Paper fidelity risk**: Cannot verify 78% accuracy claim without real model runs
- **Deployment risk**: API may fail immediately when connected to real Ollama

---

## Scope & Disposition

- **Code Path**: All agent/service tests
- **Fix Category**: Testing infrastructure / integration testing
- **Recommended Action**: Fix after all current bug chunks are resolved, before Spec 10/11

---

## Recommended Fix

### Phase 1: Local Ollama Integration Tests (Slow, Marked)

Create `tests/integration/` with tests that:
1. Require Ollama running locally (skip if unavailable)
2. Are marked `@pytest.mark.slow` and `@pytest.mark.integration`
3. Are excluded from CI by default but runnable locally

### Phase 2: Vertical Slice Verification

For each agent, create one real integration test:
1. **QualitativeAssessmentAgent**: Real transcript → real LLM → valid `QualitativeAssessment`
2. **QuantitativeAssessmentAgent**: Real transcript → real LLM → valid PHQ-8 scores
3. **JudgeAgent**: Real assessment → real LLM → valid `JudgeFeedback`
4. **EmbeddingService**: Real text → real embedding → correct dimensions
5. **Full Pipeline**: Real transcript → all agents → complete response

### Phase 3: E2E API Test

Create a script that:
1. Starts Ollama (if not running)
2. Starts the FastAPI server
3. Hits `/full_pipeline` with a real DAIC-WOZ transcript
4. Validates the response structure and sanity

---

## Files Involved

- `tests/integration/` (to be created)
- `tests/conftest.py` (add integration fixtures)
- `pyproject.toml` (add pytest markers)
- Possibly a `scripts/integration_test.sh` runner

---

## Blocking Status

This bug should be resolved **after** all current bug chunks (Chunk 3, 4, 5) but **before** implementing Spec 10 (MetaReviewAgent) and Spec 11.

---

## Notes

- Ollama runs locally on M1 Max, so no external dependencies needed
- Integration tests will be slow (seconds per test) - mark appropriately
- Consider using a small model (e.g., `gemma3:4b`) for faster integration tests
- Real PHQ-8 accuracy validation requires ground truth labels from DAIC-WOZ
