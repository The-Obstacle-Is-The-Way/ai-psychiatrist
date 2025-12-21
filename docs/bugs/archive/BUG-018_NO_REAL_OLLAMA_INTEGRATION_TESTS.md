# BUG-018: No Real Ollama Integration Tests

**Severity**: HIGH (P1)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-21
**Spec Reference**: `docs/specs/09.5_INTEGRATION_CHECKPOINT_QUANTITATIVE.md`

---

## Executive Summary

All LLM-facing tests are mock-based (either `MockLLMClient` or `respx`-mocked HTTP), and **no automated test hits a live Ollama server or runs the pipeline end-to-end with real models**. This means:
- Prompts are never validated against real model behavior
- Parsing and repair logic is never exercised against real LLM output
- Embeddings are never generated via Ollama’s `/api/embeddings` in a test run
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

### Phase 0: Test Isolation vs Real Integration (Stop the “TESTING” Trap)

Our current `tests/conftest.py` sets `TESTING=1` and clears env vars to prevent
local `.env` bleed-through. This is correct for unit/integration tests, but it
also prevents any real-LLM validation from using `.env` and developer-provided
settings.

**Policy**:
- Default test runs stay isolated (`TESTING=1`, `.env` ignored).
- Real Ollama tests are **opt-in** and run in a “real mode”:
  - `AI_PSYCHIATRIST_OLLAMA_TESTS=1`
  - `.env` is allowed (no forced `TESTING=1`)
  - env variables are not cleared

This ensures our unit tests remain deterministic while making real integration
possible without rewriting config logic.

### Phase 1: Real Ollama Smoke Tests (Local, Opt‑In)

Create real integration tests that **only run when explicitly enabled**:
- Use pytest markers: `@pytest.mark.ollama`, `@pytest.mark.e2e`, `@pytest.mark.slow`
- Skip by default unless `AI_PSYCHIATRIST_OLLAMA_TESTS=1` is set
- Fail fast (with actionable error) if Ollama is unreachable or required models are missing

Minimum smoke tests:
1. `OllamaClient.simple_chat` returns non-empty content with the configured model
2. `OllamaClient.simple_embed` returns correct dimension and L2-normalized vector

### Phase 2: Vertical Slice Tests (Real Models, Minimal Assertions)

Add one real test per agent with **structure-based assertions**, not exact outputs:
1. **QualitativeAssessmentAgent**  
   - Parsed `QualitativeAssessment` has non-empty core sections:
     `overall`, `phq8_symptoms`, `social_factors`, `biological_factors`, `risk_factors`
   - `supporting_quotes` may be empty (do not require quotes; they depend on transcript)
2. **JudgeAgent**  
   - Evaluation returns all 4 metrics with scores in [1, 5]
   - Score extraction succeeds on raw judge text (i.e., the response contains a parseable score)
3. **QuantitativeAssessmentAgent**  
   - All 8 items present  
   - Scores are in {0,1,2,3} or None  
   - Evidence strings are non-empty or "No relevant evidence found"
   - At least one score is numeric (guards against “total fallback skeleton” passing silently)
4. **EmbeddingService**  
   - Embedding dimension matches config (4096)  
   - L2 norm ≈ 1.0  
   - Retrieval returns <= top_k matches

### Phase 3: Full Pipeline E2E (Server)

Add a real E2E test (preferred) and a runner script (nice-to-have):
1. Instantiate the FastAPI `app` from `server.py` via ASGI transport (no subprocess server needed)
2. Call `/health`
3. Call `/full_pipeline` using `transcript_text` (avoids requiring licensed DAIC-WOZ data)
4. Validate response shape:
   - `qualitative`, `quantitative`, `evaluation`, `meta_review` are present
   - PHQ-8 item keys exist and scores are valid or null
   - Meta-review severity is clamped to [0, 4]
5. Assert no unhandled exceptions and response fields conform to domain constraints

### Phase 4: Paper Fidelity Checkpoint

Run the paper’s quantitative metrics **once** on a small sample:
- Generate embeddings artifact (Spec 08 requirement)
- Measure MAE/N/A rate on a small subset of DAIC-WOZ (licensed data)
- Confirm outputs fall within the acceptable paper range

---

## Files Involved

- `tests/e2e/` (new real Ollama tests)
- `scripts/` (optional E2E runner for manual use)
- `tests/conftest.py` (enable real-mode opt-in without breaking isolated tests)
- `pyproject.toml` (pytest markers: `ollama`, `e2e`, `slow`)
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
  
### How to Run (Local)

1. Start Ollama:
   - `brew install ollama`
   - `brew services start ollama` (or `ollama serve`)
2. Pull models (choose what your machine can run; paper-optimal defaults shown):
   - `ollama pull gemma3:27b`
   - `ollama pull alibayram/medgemma:27b`
   - `ollama pull qwen3-embedding:8b`
3. Run the opt-in tests:
   - `AI_PSYCHIATRIST_OLLAMA_TESTS=1 uv run pytest -m e2e --no-cov`
   - Optional: `AI_PSYCHIATRIST_OLLAMA_TESTS=1 uv run pytest -m \"e2e and ollama\" --no-cov`

### Proposed Test Inventory (What to Implement)

These are the concrete, minimal tests that close this bug without flakiness:

1. `tests/e2e/test_ollama_smoke.py`
   - `test_ollama_ping_and_models_present`
   - `test_simple_chat_returns_expected_xml_answer_tag`
   - `test_simple_embed_dimension_and_l2_norm`

2. `tests/e2e/test_agents_real_ollama.py`
   - `test_qualitative_agent_assess_real_ollama`
   - `test_judge_agent_evaluate_real_ollama_scores_parseable`
   - `test_quantitative_agent_assess_real_ollama_has_some_numeric_scores`
   - Optional: `test_embedding_service_build_reference_bundle_real_ollama` (uses a tiny temp pickle artifact)

3. `tests/e2e/test_server_real_ollama.py`
   - `test_server_health_real_ollama`
   - `test_server_full_pipeline_transcript_text_zero_shot_real_ollama`
   - Optional: `test_server_full_pipeline_few_shot_requires_embeddings_artifact` (skip if missing)
  
## Definition of Done

- Real Ollama tests exist and are gated by explicit opt-in.
- Each agent has at least one real integration test with structural assertions.
- Full pipeline E2E test passes against a live Ollama server.
- Documentation updated with exact commands to run locally.

---

## Resolution

Implemented an opt-in real-Ollama E2E test suite that runs locally without CI flakiness:

- Added `tests/e2e/` with live Ollama tests (skip unless `AI_PSYCHIATRIST_OLLAMA_TESTS=1`).
- Adjusted `tests/conftest.py` so “real mode” does not force `TESTING=1` and does not clear
  environment variables (allowing `.env` and model settings).
- Added pytest marker `ollama` (strict markers are enabled).
- Verified locally against a running Ollama daemon with paper-optimal models:
  - `gemma3:27b`
  - `alibayram/medgemma:27b`
  - `qwen3-embedding:8b`

### Additional Fix (Discovered During Real Runs)

Real Ollama tests surfaced a configuration gap: `OLLAMA_TIMEOUT_SECONDS` was not
being applied to the per-request timeout on `OllamaClient.simple_chat()` and
`OllamaClient.simple_embed()` because those helpers were constructing
`ChatRequest` / `EmbeddingRequest` without passing the configured timeout,
falling back to protocol defaults (180s chat / 120s embeddings).

Fix: `OllamaClient.simple_chat()` and `OllamaClient.simple_embed()` now pass
`timeout_seconds=self._default_timeout`, ensuring `OLLAMA_TIMEOUT_SECONDS`
controls real request timeouts end-to-end (agents and server included).

### Verification (Local)

```bash
# Smoke tests (chat + embeddings)
AI_PSYCHIATRIST_OLLAMA_TESTS=1 uv run pytest tests/e2e/test_ollama_smoke.py -v --no-cov

# Per-agent tests
AI_PSYCHIATRIST_OLLAMA_TESTS=1 uv run pytest tests/e2e/test_agents_real_ollama.py -v --no-cov

# Full pipeline E2E (FastAPI app via ASGI transport)
AI_PSYCHIATRIST_OLLAMA_TESTS=1 uv run pytest tests/e2e/test_server_real_ollama.py -v --no-cov
```

Note: Paper-optimal 27B models can be slow on local hardware. If you see timeouts
(especially in the quantitative agent with MedGemma), run with a larger timeout:

```bash
AI_PSYCHIATRIST_OLLAMA_TESTS=1 OLLAMA_TIMEOUT_SECONDS=600 \
  uv run pytest -m "e2e and ollama" --no-cov
```
