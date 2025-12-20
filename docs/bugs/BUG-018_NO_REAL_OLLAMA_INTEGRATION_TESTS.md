# BUG-018: No Real Ollama Integration Tests

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

All LLM-facing tests are mock-based (either `MockLLMClient` or `respx`-mocked HTTP), and **no automated test hits a live Ollama server or runs the pipeline end-to-end with real models**. This means:
- Prompts are never validated against real model behavior
- Parsing and repair logic is never exercised against real LLM output
- Embeddings are never generated via `/api/embeddings` in a test run
- The full pipeline has never been executed with real infrastructure

This violates our own testing philosophy (Spec 01) and the intent of the 09.5 integration checkpoint. The architecture may be correct, but we cannot claim operational readiness without real Ollama validation.

---

## Evidence

- `tests/integration/test_qualitative_pipeline.py` and `tests/integration/test_dual_path_pipeline.py` use `MockLLMClient` and never touch real Ollama.
- `tests/unit/infrastructure/llm/test_ollama.py` explicitly mocks HTTP calls using `respx` (no live server).
- There is no test in `tests/` that invokes `OllamaClient` against a real host or calls `server.py` endpoints with a live model.
- Spec 01 explicitly requires E2E tests with **real** Ollama, and Spec 09.5 calls for integration validation before Meta-Review; current tests do not satisfy that requirement.

---

## Impact

- **False confidence**: Mock-only tests can pass while real model outputs fail parsing.
- **Hidden failures**: JSON repair paths, retry logic, and timeouts are untested under real latency/error conditions.
- **Paper fidelity risk**: We cannot validate MAE or N/A rates without running the actual models.
- **Deployment risk**: The API may fail at first real request (prompt drift, response format drift, model availability).

---

## Scope & Disposition

- **Code Path**: All LLM-facing agent/service paths (qualitative, judge, quantitative, embeddings, pipeline).
- **Fix Category**: Integration/E2E testing gap.
- **Recommended Action**: Treat as a hard gate at Checkpoint 09.5 (before Spec 10/11), after embedding artifacts are available.

---

## Recommended Fix (Ironclad Plan)

### Phase 1: Real Ollama Smoke Tests (Local, Gated)

Create real integration tests that **only run when explicitly enabled**:
- Use pytest markers: `@pytest.mark.ollama`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Skip by default unless `AI_PSYCHIATRIST_OLLAMA_TESTS=1` (or similar env var) is set
- Skip if `OllamaClient.ping()` fails or required models are missing

Minimum smoke tests:
1. `OllamaClient.simple_chat` returns non-empty content with the configured model
2. `OllamaClient.simple_embed` returns correct dimension and normalized vector

### Phase 2: Vertical Slice Tests (Real Models, Minimal Assertions)

Add one real test per agent with **structure-based assertions**, not exact outputs:
1. **QualitativeAssessmentAgent**  
   - Output contains required XML tags  
   - Parsed `QualitativeAssessment` has 8 PHQ-8 items and non-empty evidence list
2. **JudgeAgent**  
   - Response parse yields 4 metric scores  
   - Each score is in [1, 5]
3. **QuantitativeAssessmentAgent**  
   - All 8 items present  
   - Scores are in {0,1,2,3} or None  
   - Evidence strings are non-empty or "No relevant evidence found"
4. **EmbeddingService**  
   - Embedding dimension matches config (4096)  
   - L2 norm ≈ 1.0  
   - Retrieval returns <= top_k matches

### Phase 3: Full Pipeline E2E (Server)

Add a real E2E test or script:
1. Start FastAPI `server.py` (using `OLLAMA_HOST`/`OLLAMA_PORT`)
2. Call `/health` and `/full_pipeline`
3. Validate response schema (qual + quant + metadata)
4. Assert no unhandled exceptions and response fields conform to domain constraints

### Phase 4: Paper Fidelity Checkpoint

Run the paper’s quantitative metrics **once** on a small sample:
- Generate embeddings artifact (Spec 08 requirement)
- Measure MAE/N/A rate on a small subset of DAIC-WOZ (licensed data)
- Confirm outputs fall within the acceptable paper range

---

## Files Involved

- `tests/integration/` (new real Ollama tests)
- `tests/e2e/` or `scripts/` (E2E runner)
- `tests/conftest.py` (ollama availability fixture, model availability checks)
- `pyproject.toml` (pytest markers: `ollama`, `integration`, `slow`)
- `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md` (add explicit real-LLM gate)

---

## Blocking Status

This bug is a **hard gate at Checkpoint 09.5**. Do not proceed to Spec 10/11 until:
- At least one real Ollama test per agent passes locally
- The full pipeline E2E call succeeds on a real model

---

## Notes

- Ollama can run locally (M1 Max is sufficient) but tests will be slow.
- Use smaller models for smoke tests if needed; document model substitutions in `docs/models/MODEL_REGISTRY.md`.
- Real PHQ-8 accuracy validation requires DAIC-WOZ ground truth and is not CI-safe.
  
## Definition of Done

- Real Ollama tests exist and are gated by explicit opt-in.
- Each agent has at least one real integration test with structural assertions.
- Full pipeline E2E test passes against a live Ollama server.
- Documentation updated with exact commands to run locally.
