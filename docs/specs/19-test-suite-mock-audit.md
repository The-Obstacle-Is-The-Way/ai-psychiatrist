# Spec 19: Test Suite Mock Audit

> **STATUS: AUDIT COMPLETE — MINOR IMPROVEMENTS RECOMMENDED**
>
> **GitHub Issue**: #59
>
> **Priority**: Low (test hygiene; production code unaffected)
>
> **Last Updated**: 2025-12-26

---

## Executive Summary

A comprehensive audit of the test suite was conducted to evaluate:
1. Whether mocks are appropriate given extensive refactoring
2. Whether assertions match current codebase behavior
3. Whether the mocking strategy follows 2025 ML testing best practices

**Overall Verdict: The test suite is in excellent health.**

The codebase went through significant refactoring (Pydantic AI integration, embedding backend decoupling, model wiring fixes), and the tests have been properly updated. Most issues found are low priority (P3-P5).

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
| Archived Bugs/Specs | 50+ documents | 100% |

### Standards Applied

1. **Clean Architecture** (Robert C. Martin): Test doubles as outer-circle concerns
2. **Google Testing Best Practices**: Test behavior, not implementation
3. **pytest Community Standards 2025**: Async/sync mock separation, fixtures
4. **ML Testing Best Practices**: Mock LLM boundaries, test parsing/validation edge cases
5. **ISO 27001 Control 8.31**: Separation of test and production environments

---

## Positive Findings (Things Done Right)

### 1. MockLLMClient Properly Located ✅

**Location**: `tests/fixtures/mock_llm.py`

**Standard**: Clean Architecture, ISO 27001

Per BUG-001 resolution, `MockLLMClient` was moved from `src/` to `tests/fixtures/`. This ensures:
- Test doubles are not shipped in production artifacts
- No risk of mock contamination in medical AI system
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

### 3. E2E Tests with Real Ollama ✅

**Location**: `tests/e2e/test_agents_real_ollama.py`

The codebase has opt-in E2E tests that exercise the full pipeline with real Ollama:

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
- 50+ environment variables explicitly cleared
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

### P2: Pydantic AI Dual-Path Not Unit-Tested

**Severity**: P2 (Medium)
**Location**: `tests/unit/agents/test_quantitative.py`, `test_judge.py`, `test_meta_review.py`

**Issue**: The agents now have a **dual-path** execution (legacy `simple_chat` + Pydantic AI). Unit tests only exercise the legacy path because:

1. `pydantic_ai_settings` is not passed to agents in unit tests
2. `ollama_base_url` is not provided
3. Agents fall back to legacy path

**Evidence**: From `quantitative.py:107-124`:

```python
if self._pydantic_ai.enabled:
    if not self._ollama_base_url:
        logger.warning("Pydantic AI enabled but no ollama_base_url provided; falling back to legacy")
    else:
        self._scoring_agent = create_quantitative_agent(...)
```

**Impact**: The integrated Pydantic AI path (agent creation, `agent.run()`, fallback on failure) is not covered in unit tests. Only extractors are tested independently.

**Recommendation**: Either:
1. Add unit tests that mock `pydantic_ai.Agent` behavior, OR
2. Accept that integration/E2E tests provide sufficient coverage for the Pydantic AI path

**Effort**: ~2 hours if implementing unit test approach

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

### P4: Missing Qualitative Output Extractor Tests

**Severity**: P4 (Low)
**Location**: `tests/unit/agents/test_pydantic_ai_extractors.py`

**Issue**: `extract_qualitative` is imported but has no dedicated test cases:

```python
from ai_psychiatrist.agents.extractors import (
    extract_judge_metric,
    extract_meta_review,
    extract_quantitative,
    extract_qualitative,  # Imported but not tested
)
```

**Context**: Per Spec 18, `extract_qualitative` is part of the planned Qualitative Agent robustness work. The function exists but is not yet used in production.

**Recommendation**: Add tests when Spec 18 is implemented. Track as part of Spec 18 acceptance criteria.

**Effort**: Part of Spec 18 (~30 min)

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

**Status**: Not an issue

Per BUG-020's validation, this was already addressed:
- E2E tests with real Ollama exist
- Unit tests verify protocol compatibility
- Integration tests exercise full pipelines

### ❌ "Low Coverage"

**Status**: Not an issue

Coverage is **96.52%** with 603+ tests passing. CI enforces 80%+ minimum.

### ❌ "Mocks Don't Match Implementations"

**Status**: Not an issue

The `MockLLMClient` properly implements `SimpleChatClient` protocol. Protocol-based testing ensures mocks and real implementations are interchangeable.

### ❌ "AsyncMock Warnings"

**Status**: Resolved in BUG-005

The feedback loop tests now use sync `Mock` for sync methods.

---

## Best Practices Assessment (2025)

| Practice | Status | Notes |
|----------|--------|-------|
| Test doubles in tests/, not src/ | ✅ | BUG-001 resolved |
| Protocol-based mocking | ✅ | `SimpleChatClient` protocol |
| Separate unit/integration/E2E | ✅ | Directory structure + markers |
| Real LLM integration tests | ✅ | opt-in E2E with Ollama |
| Test edge cases (parsing, validation) | ✅ | Extensive coverage |
| Async/sync mock separation | ✅ | BUG-005 resolved |
| Test isolation (env vars, caching) | ✅ | conftest.py |
| Auto-applied test markers | ✅ | Spec 17 implemented |
| Pydantic AI retry/validation testing | ✅ | Extractors tested |
| Document test data sources | ⚠️ | Could add more comments on DAIC-WOZ samples |

---

## Recommendations Summary

| Priority | Issue | Recommendation | Effort |
|----------|-------|----------------|--------|
| P2 | Pydantic AI dual-path not unit-tested | Add unit tests or accept integration coverage | ~2 hrs |
| P3 | Enum order reliance in judge tests | Use `chat_function` for dynamic responses | ~30 min |
| P4 | Missing `extract_qualitative` tests | Implement with Spec 18 | Part of Spec 18 |
| P5 | Large hardcoded response strings | No action needed | - |

---

## Implementation Plan (Optional)

If the P2 issue is addressed:

### Option A: Mock Pydantic AI Agent in Unit Tests

Add to `tests/unit/agents/test_quantitative.py`:

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
2. `tests/e2e/test_agents_real_ollama.py` (full path with real Ollama)

This is acceptable because:
- The Pydantic AI agent is a thin wrapper around extractors
- The extractors are thoroughly unit-tested
- E2E tests validate the integrated behavior

---

## Acceptance Criteria

- [x] Audit completed with documented findings
- [x] No P0/P1 issues found
- [x] 2025 best practices verified
- [ ] Optional: P2 issue addressed (Pydantic AI dual-path testing)
- [ ] Optional: P3 issue addressed (enum order reliance)

---

## References

- BUG-001: MockLLMClient in Production Path (resolved)
- BUG-005: AsyncMock Warning in Feedback Loop Tests (resolved)
- BUG-019: Code Quality Audit (resolved)
- BUG-020: Jules Audit Findings (resolved)
- Spec 13: Pydantic AI Structured Outputs (implemented)
- Spec 17: Test Suite Marker Consistency (implemented)
- Spec 18: Qualitative Agent Robustness (planned)
- [pytest Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Clean Architecture - Test Doubles](https://blog.cleancoder.com/uncle-bob/2014/05/14/TheLittleMocker.html)
