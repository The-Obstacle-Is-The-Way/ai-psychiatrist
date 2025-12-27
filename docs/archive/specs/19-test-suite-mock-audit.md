# Spec 19: Test Suite Mock Audit

> **STATUS: AUDIT COMPLETE — IMPROVEMENTS IMPLEMENTED (ARCHIVED)**
>
> **GitHub Issue**: #59
>
> **Priority**: Medium (test coverage + mock hygiene)
>
> **Last Updated**: 2025-12-26

---

## Executive Summary

A comprehensive audit of the test suite was conducted to evaluate:
1. Whether mocks are appropriate given extensive refactoring
2. Whether assertions match current codebase behavior
3. Whether the mocking strategy follows 2025 ML testing best practices

**Overall Verdict: The test suite is healthy and the prior high-impact gaps are now closed.**

The codebase went through significant refactoring (Pydantic AI integration, embedding backend decoupling, model wiring fixes), and the tests have been properly updated.

This spec originally identified two higher-impact gaps (P1/P2 below). Both have since been addressed:
1. Offline unit tests now exercise the Pydantic AI `.run(...)` path for all core agents (no network required).
2. Contract-bearing mocks were hardened (use of `spec_set` and real settings objects).

---

## Audit Scope

### Files Reviewed

| Category | Files Reviewed | Coverage |
|----------|---------------|----------|
| Mock Infrastructure | `tests/fixtures/mock_llm.py`, `tests/conftest.py` | 100% |
| Agent Unit Tests | `test_quantitative.py`, `test_judge.py`, `test_qualitative.py`, `test_meta_review.py` | 100% |
| Service Tests | `test_feedback_loop.py`, `test_embedding.py`, `test_reference_store.py` | 100% |
| Extractor Tests | `test_pydantic_ai_extractors.py` | 100% |
| Integration Tests | `test_dual_path_pipeline.py`, `test_qualitative_pipeline.py` | 100% |
| E2E Tests | `test_agents_real_ollama.py`, `test_ollama_smoke.py`, `test_server_real_ollama.py` | 100% |

### Standards Applied

1. **Clean Architecture** (Robert C. Martin): test doubles live in `tests/`, not `src/`
2. **Google Testing Best Practices**: test behavior, not implementation
3. **pytest Best Practices**: deterministic unit tests, strict markers, fixture isolation
4. **ML Testing Best Practices**: mock LLM I/O boundaries, stress parsing/validation paths

---

## Positive Findings (Things Done Right)

### 1. MockLLMClient Properly Located ✅

**Location**: `tests/fixtures/mock_llm.py`

**Standard**: Clean Architecture (test/prod separation)

Per BUG-001 resolution, `MockLLMClient` was moved from `src/` to `tests/fixtures/`. This ensures:
- Test doubles are not shipped in production artifacts
- Reduced risk of mock contamination in production paths
- Clear separation of test and production environments

### 2. Protocol-Based Testing ✅

**Location**: `test_quantitative.py:560-593`, `test_qualitative.py:386-417`

Tests verify that `MockLLMClient` implements `SimpleChatClient` protocol:

```python
def test_mock_client_implements_protocol(self) -> None:
    """MockLLMClient should implement SimpleChatClient protocol."""
    client = MockLLMClient()
    assert isinstance(client, SimpleChatClient)
```

This follows the Dependency Inversion Principle - agents depend on protocols, not implementations.

### 2.1 MockLLMClient Protocol Compliance ✅

**Location**: `tests/unit/infrastructure/llm/test_mock.py`

In addition to `SimpleChatClient` compatibility, the mock is validated against the protocol surface used by production services:
- `ChatClient`
- `EmbeddingClient`
- `LLMClient`

This meaningfully reduces the chance of “fossilized mocks” silently diverging from the real request/response contracts.

### 3. E2E Tests with Real Ollama ✅

**Location**: `tests/e2e/test_agents_real_ollama.py`, `tests/e2e/test_server_real_ollama.py`

The codebase has opt-in e2e tests that exercise real Ollama integration (agent-level checks and the FastAPI `/full_pipeline` endpoint):

```python
@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.slow
class TestAgentsRealOllama:
    async def test_qualitative_agent_assess_real_ollama(self, ...):
        ...
```

Run with: `AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e`

This addresses BUG-007's concern about "testing the mock" by having real LLM integration tests.

### 4. Pydantic AI Extractors Properly Tested ✅

**Location**: `tests/unit/agents/test_pydantic_ai_extractors.py`

Extractors that raise `ModelRetry` on validation failure are thoroughly tested:

```python
def test_extract_quantitative_missing_answer_tags_retries() -> None:
    with pytest.raises(ModelRetry):
        extract_quantitative("no answer tags here")

def test_extract_quantitative_invalid_json_retries() -> None:
    with pytest.raises(ModelRetry):
        extract_quantitative("<answer>{not valid json}</answer>")
```

This tests the Pydantic AI validation + retry contract independently.

### 5. Async/Sync Mock Separation ✅

**Location**: `tests/unit/services/test_feedback_loop.py:57-58`

Per BUG-005 resolution, `get_feedback_for_low_scores()` (sync method) is mocked as `Mock`, not `AsyncMock`:

```python
agent.get_feedback_for_low_scores = Mock(return_value={})
```

This avoids the "coroutine was never awaited" RuntimeWarning.

### 6. Test Isolation via conftest.py ✅

**Location**: `tests/conftest.py:1-84`

Proper test isolation is implemented:
- `TESTING=1` env var prevents `.env` loading in unit tests
- 37 environment variables explicitly cleared
- `get_settings.cache_clear()` ensures fresh config reads
- Auto-marker hook applies `@pytest.mark.unit/integration/e2e` based on path

### 7. Comprehensive Edge Case Coverage ✅

**Location**: `test_quantitative.py:262-411`, `test_embedding.py:574-763`

Tests cover:
- JSON parsing with smart quotes, trailing commas, markdown blocks
- Missing PHQ-8 items filled with defaults
- Embedding dimension mismatches
- Similarity transformation edge cases (cos=-1 → 0.0, cos=0 → 0.5, cos=1 → 1.0)

---

## Issues Found

### P1: Pydantic AI Execution Path Not Covered by Default Test Runs (CI Risk) — RESOLVED

**Severity**: P1 (High)

**What was wrong**: The core agents have a **dual-path** execution (legacy `simple_chat` vs Pydantic AI via `pydantic_ai.Agent`). Pydantic AI only activates when `ollama_base_url` is provided at construction time. Most unit/integration tests instantiated agents without `ollama_base_url`, which forced the legacy path even when Pydantic AI was enabled.

**Fix implemented**: Offline unit tests now exercise the Pydantic AI `.run(...)` path for all core agents (no network required) by patching `create_*_agent()` factories and providing `ollama_base_url`:
- `tests/unit/agents/test_quantitative.py` (`test_pydantic_ai_path_success`)
- `tests/unit/agents/test_judge.py` (`test_pydantic_ai_path_success`)
- `tests/unit/agents/test_meta_review.py` (`test_pydantic_ai_path_success`)
- `tests/unit/agents/test_qualitative.py` (`test_pydantic_ai_path_success`)

**Evidence** (representative; repeated across agents): when enabled but missing base URL, agents log and fall back:

```python
if self._pydantic_ai.enabled:
    if not self._ollama_base_url:
        logger.warning("Pydantic AI enabled but no ollama_base_url provided; falling back to legacy")
    else:
        self._scoring_agent = create_quantitative_agent(...)
```

---


### P2: Contract Drift Risk from Unspecced MagicMock / AsyncMock — RESOLVED

**Severity**: P2 (Medium)

**What was wrong**: Some tests used “free-form” mocks for collaborators that represent real interfaces (settings objects and agent/service boundaries). Unspecced mocks can silently accept new/incorrect attribute and method usage, masking breaking changes.

**Fix implemented**:
- `tests/unit/services/test_embedding.py` now uses real `EmbeddingSettings(...)` objects instead of free-form mocks.
- `tests/unit/services/test_feedback_loop.py` now uses `AsyncMock(spec_set=QualitativeAssessmentAgent)` and `AsyncMock(spec_set=JudgeAgent)`.
- `tests/unit/services/test_reference_store.py` now uses `MagicMock(spec_set=numpy.lib.npyio.NpzFile)`.
- Agent tests now use `AsyncMock(spec_set=pydantic_ai.Agent)`.

**Outcome**: Contract drift is less likely to slip through because mocks/settings now enforce real attribute/method surfaces.

---


### P3: Tests Rely on Enum Iteration Order — RESOLVED

**Severity**: P3 (Low)
**Location**: `tests/unit/agents/test_judge.py:88-107, 117-138`

**What was wrong**: Tests relied on `EvaluationMetric` enum iteration order, which is fragile if the enum changes.

**Fix implemented**: `tests/unit/agents/test_judge.py` now uses `MockLLMClient(chat_function=...)` to return responses based on prompt content, making the tests stable regardless of enum order:

```python
def response_by_metric(request: ChatRequest) -> str:
    if "coherence" in request.messages[-1].content.lower():
        return "Explanation: Good\nScore: 5"
    elif "completeness" in request.messages[-1].content.lower():
        return "Explanation: Bad\nScore: 2"
    ...

mock_client = MockLLMClient(chat_function=response_by_metric)
```

---


### P5: Large Hardcoded Response Strings

**Severity**: P5 (Cosmetic)
**Location**: `test_dual_path_pipeline.py:39-152`, `test_quantitative.py:63-91`

**Issue**: Large response strings like `SAMPLE_QUALITATIVE_RESPONSE`, `SAMPLE_QUANT_SCORING_RESPONSE` are defined in test files.

**Impact**: None functional. The strings are readable and explicit.

**Recommendation**: No action needed. Current approach is acceptable per "explicit is better than implicit" principle.

---

## Issues NOT Found (False Concerns Addressed)

### ❌ "Over-mocking / Testing the Mock"

**Status**: Mostly mitigated (note: e2e is opt-in)

Per BUG-020's validation, this was already addressed:
- E2E tests with real Ollama exist
- Unit tests verify protocol compatibility
- Integration tests exercise full pipelines

### ❌ "Low Coverage"

**Status**: Coverage is enforced; absolute % will drift

Coverage is enforced via `pyproject.toml` pytest `addopts` (`--cov-fail-under=80`). Treat any single percentage as a measurement at a point in time, not a permanent guarantee.

### ❌ "Mocks Don't Match Implementations"

**Status**: Mitigated

The `MockLLMClient` is validated against `ChatClient` / `EmbeddingClient` / `LLMClient` (and `SimpleChatClient` in agent tests), which strongly limits drift. Additional contract-bearing mocks were hardened as part of P2.

### ❌ "AsyncMock Warnings"

**Status**: Resolved in BUG-005

The feedback loop tests now use sync `Mock` for sync methods.

---

## Best Practices Assessment (2025)

| Practice | Status | Notes |
|----------|--------|-------|
| Test doubles in tests/, not src/ | ✅ | BUG-001 resolved |
| Protocol-based mocking | ✅ | `ChatClient` / `EmbeddingClient` / `LLMClient` + `SimpleChatClient` |
| Separate unit/integration/E2E | ✅ | Directory structure + markers |
| Real LLM integration tests | ✅ | opt-in E2E with Ollama |
| Test edge cases (parsing, validation) | ✅ | Extensive coverage |
| Async/sync mock separation | ✅ | BUG-005 resolved |
| Test isolation (env vars, caching) | ✅ | conftest.py |
| Auto-applied test markers | ✅ | `pytest_collection_modifyitems` in `tests/conftest.py` |
| Pydantic AI retry/validation testing | ✅ | Extractors tested |
| Avoid unspecced mocks for interfaces | ✅ | All core mocks now have `spec_set`. |

---

## Recommendations Summary

| Priority | Issue | Recommendation | Effort |
|----------|-------|----------------|--------|
| P1 | Pydantic AI path not covered in default runs | ✅ Implemented: offline unit tests cover `.run(...)` path | - |
| P2 | Unspecced mocks for contract interfaces | ✅ Implemented: real settings + `spec_set` mocks | - |
| P3 | Enum order reliance in judge tests | ✅ Implemented: `chat_function`-based responses | - |
| P5 | Large hardcoded response strings | No action needed | - |

---

## Remediation Summary

The remediations described in P1/P2/P3 are implemented and run in the default unit suite:

- P1: Offline `.run(...)`-path tests for all agents (patch `create_*_agent()` factories in each agent test module).
- P2: Hardened contract mocks (use `spec_set` for agent/service boundaries; use real settings objects where practical).
- P3: Removed enum-order reliance in judge tests via `MockLLMClient(chat_function=...)`.

---

## Acceptance Criteria

- [x] Audit completed with documented findings
- [x] 2025 best practices verified
- [x] P1 issue addressed (offline unit tests cover Pydantic AI `.run(...)` path)
- [x] P2 issue addressed (real settings + `spec_set` mocks for contract-bearing collaborators)
- [x] P3 issue addressed (no enum-order reliance in judge tests)

---

## References

- BUG-001: MockLLMClient in Production Path (resolved)
- BUG-005: AsyncMock Warning in Feedback Loop Tests (resolved)
- BUG-019: Code Quality Audit (resolved)
- BUG-020: Jules Audit Findings (resolved)
- Spec 13: Pydantic AI Structured Outputs (implemented)
- Spec 17: Test Suite Marker Consistency (implemented)
- Spec 18: Qualitative Agent Robustness (implemented)
- [pytest Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Clean Architecture - Test Doubles](https://blog.cleancoder.com/uncle-bob/2014/05/14/TheLittleMocker.html)
