# Spec 19: Test Suite Mock Audit

> **STATUS: AUDIT COMPLETE — IMPROVEMENTS RECOMMENDED**
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

**Overall Verdict: The test suite is broadly healthy, but it has a real coverage gap.**

The codebase went through significant refactoring (Pydantic AI integration, embedding backend decoupling, model wiring fixes), and the tests have been properly updated.

However, two findings are higher impact:
1. **Pydantic AI execution paths are not exercised by the default unit/integration runs**, and most e2e tests instantiate agents in a way that forces the legacy path.
2. **Some tests use unspecced `MagicMock` / `AsyncMock` for contract-bearing collaborators** (settings objects and agent/service boundaries), which can hide interface drift.

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

### P1: Pydantic AI Execution Path Not Covered by Default Test Runs (CI Risk)

**Severity**: P1 (High)

**What’s wrong**: The core agents have a **dual-path** execution (legacy `simple_chat` vs Pydantic AI via `pydantic_ai.Agent`). Pydantic AI is enabled by default (`PydanticAISettings.enabled = True`), but it only activates when `ollama_base_url` is provided at construction time.

Most unit/integration tests instantiate agents without `ollama_base_url`, which forces the legacy path even when Pydantic AI is enabled. Additionally, `tests/e2e/test_agents_real_ollama.py` instantiates agents without `ollama_base_url`, so those e2e tests also exercise the legacy path.

**Why it matters**:
- A regression in Pydantic AI integration (API changes, factory wiring, output adapter behavior) can slip through CI because the default suite won’t exercise the `.run(...)` path.
- The production entrypoints that matter most for paper reproduction (`server.py`, `scripts/reproduce_results.py`) *do* pass `ollama_base_url`, so real runs are more likely to hit the untested path.

**Evidence** (representative; repeated across agents): when enabled but missing base URL, agents log and fall back:

```python
if self._pydantic_ai.enabled:
    if not self._ollama_base_url:
        logger.warning("Pydantic AI enabled but no ollama_base_url provided; falling back to legacy")
    else:
        self._scoring_agent = create_quantitative_agent(...)
```

**Recommendation**:
1. Add **offline unit tests** (no network) that patch `create_*_agent()` factories and assert the `.run()` path is taken for each agent when `ollama_base_url` is provided, OR
2. Add a dedicated integration test around FastAPI lifespan wiring with patched `create_*_agent()` factories (no network), OR
3. Explicitly accept the risk and enforce `AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e` in CI (note: makes CI depend on a running Ollama).

**Effort**: ~1–2 hours (offline unit test approach)

---

### P2: Contract Drift Risk from Unspecced MagicMock / AsyncMock

**Severity**: P2 (Medium)

**What’s wrong**: Some tests use “free-form” mocks for collaborators that represent real interfaces (settings objects and agent/service boundaries). Unspecced mocks can silently accept new/incorrect attribute and method usage, masking breaking changes.

**Where it shows up**:
- `tests/unit/services/test_embedding.py` mocks embedding settings with `MagicMock()` (no `spec`/`spec_set`)
- `tests/unit/services/test_feedback_loop.py` mocks agents with `AsyncMock()` (no `spec`/`spec_set`)

**Why it matters**: This is a common failure mode for “fossilized” tests: the interface changes, tests keep passing, and the regression leaks into production.

**Suggested fix**:
- Prefer real settings objects where practical (e.g., use `EmbeddingSettings(...)` in tests), or use `MagicMock(spec_set=EmbeddingSettings)` to enforce attribute names.
- For agent/service mocks, use `AsyncMock(spec_set=QualitativeAssessmentAgent)` / `AsyncMock(spec_set=JudgeAgent)` (or a small Protocol representing the boundary) so the service cannot call non-existent methods without failing the test.

**Effort**: ~30–60 minutes (test-only cleanup)

---

### P3: Tests Rely on Enum Iteration Order

**Severity**: P3 (Low)
**Location**: `tests/unit/agents/test_judge.py:88-107, 117-138`

**Issue**: Comments acknowledge fragility:

```python
responses = [
    "Explanation: Good\nScore: 5",  # Coherence
    "Explanation: Bad\nScore: 2",   # Completeness
    "Explanation: Good\nScore: 5",  # Specificity
    "Explanation: Bad\nScore: 2",   # Accuracy
]
# NOTE: This relies on Enum order. If Enum order changes, this test might be flaky.
```

**Impact**: If `EvaluationMetric` enum order changes, tests could fail or produce false positives.

**Recommendation**: Use `MockLLMClient`'s `chat_function` parameter to return responses based on prompt content:

```python
def response_by_metric(request: ChatRequest) -> str:
    if "coherence" in request.messages[-1].content.lower():
        return "Explanation: Good\nScore: 5"
    elif "completeness" in request.messages[-1].content.lower():
        return "Explanation: Bad\nScore: 2"
    ...

mock_client = MockLLMClient(chat_function=response_by_metric)
```

**Effort**: ~30 minutes

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

**Status**: MockLLMClient aligns; some other mocks are still “loose”

The `MockLLMClient` is validated against `ChatClient` / `EmbeddingClient` / `LLMClient` (and `SimpleChatClient` in agent tests), which strongly limits drift. Some tests still use unspecced mocks for settings/services (see P2).

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
| Avoid unspecced mocks for interfaces | ⚠️ | Improve settings/agent mocks (see P2) |

---

## Recommendations Summary

| Priority | Issue | Recommendation | Effort |
|----------|-------|----------------|--------|
| P1 | Pydantic AI path not covered in default runs | Add offline unit tests around the `.run()` path | ~1–2 hrs |
| P2 | Unspecced mocks for contract interfaces | Use real settings or `spec_set` mocks | ~30–60 min |
| P3 | Enum order reliance in judge tests | Use `chat_function` for dynamic responses | ~30 min |
| P5 | Large hardcoded response strings | No action needed | - |

---

## Implementation Plan (Optional)

If the P1 issue is addressed:

### Option A: Mock Pydantic AI Agent in Unit Tests

Add to `tests/unit/agents/test_quantitative.py` (and replicate the pattern for `JudgeAgent`, `MetaReviewAgent`, and `QualitativeAssessmentAgent`):

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_pydantic_ai_path_success(sample_transcript: Transcript) -> None:
    """Should use Pydantic AI agent when enabled and configured."""
    mock_agent = AsyncMock()
    mock_agent.run.return_value = AsyncMock(output=QuantitativeOutput(...))

    with patch('ai_psychiatrist.agents.pydantic_agents.create_quantitative_agent', return_value=mock_agent):
        agent = QuantitativeAssessmentAgent(
            llm_client=MockLLMClient(),
            pydantic_ai_settings=PydanticAISettings(enabled=True),
            ollama_base_url="http://localhost:11434",
        )
        result = await agent.assess(sample_transcript)

    mock_agent.run.assert_called_once()
    assert result.total_score >= 0
```

### Option B: Accept Integration Coverage

Document that the Pydantic AI path is tested via:
1. `tests/unit/agents/test_pydantic_ai_extractors.py` (extractors)
2. `tests/e2e/test_server_real_ollama.py` (full pipeline via FastAPI lifespan wiring)

Note: `tests/e2e/test_agents_real_ollama.py` currently instantiates agents without `ollama_base_url`, so it exercises the legacy path even when Pydantic AI is enabled by default.

This is acceptable because:
- The Pydantic AI agent is a thin wrapper around extractors
- The extractors are thoroughly unit-tested
- E2E tests validate the integrated behavior

---

## Acceptance Criteria

- [x] Audit completed with documented findings
- [x] 2025 best practices verified
- [x] P1 issue addressed or explicitly accepted (Pydantic AI path coverage)
- [x] Optional: P2 issue addressed (specced mocks for contract-bearing collaborators)
- [x] Optional: P3 issue addressed (enum order reliance)

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
