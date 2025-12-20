# BUG-019: Comprehensive Code Quality Audit

**Severity**: MIXED (P0-P4)
**Status**: OPEN
**Date Identified**: 2025-12-20
**Audit Scope**: Full codebase review

---

## Executive Summary

This document captures all issues identified during a comprehensive code quality audit of the ai-psychiatrist codebase. Issues are categorized by priority (P0-P4) and include: type safety violations, magic numbers, anti-patterns, SOLID/DRY violations, inconsistent defaults, and test suite concerns.

The codebase demonstrates generally good architecture (Clean Architecture, DDD patterns, proper separation of concerns) but has accumulated technical debt that should be addressed systematically.

---

## Priority Definitions

| Priority | Definition | Example |
|----------|------------|---------|
| P0 | Critical - System broken or data integrity at risk | Production crashes, silent data corruption |
| P1 | High - Significant impact on reliability/correctness | Type errors that could cause runtime failures |
| P2 | Medium - Technical debt affecting maintainability | Missing type annotations, inconsistent patterns |
| P3 | Low - Code quality issues, not urgent | Magic numbers, minor DRY violations |
| P4 | Trivial - Nice to have improvements | Documentation gaps, style inconsistencies |

---

## P1: Type Safety Violations (72 Mypy Errors)

### BUG-019.1: Test Suite Type Errors

**Status**: OPEN
**Location**: `tests/`
**Impact**: Type checker cannot verify test correctness, potential runtime failures

**Evidence**:
```
tests/conftest.py:99: error: Missing type parameters for generic type "dict"  [type-arg]
tests/unit/test_config.py:200: error: Argument "level" to "LoggingSettings" has incompatible type
tests/unit/services/test_transcript.py: 13+ errors for MockDataSettings type mismatch
tests/unit/services/test_ground_truth.py: 13+ errors for MockDataSettings type mismatch
tests/unit/agents/test_judge.py: list invariance errors
```

**Root Cause**: Tests use ad-hoc mock objects that don't satisfy the type contracts expected by production code.

**Fix**:
1. Create proper typed mock classes that inherit from production protocols
2. Use `Protocol` from `typing` to define mock interfaces
3. Add type annotations to test fixtures

**Pattern Violated**: Type Safety, Liskov Substitution Principle

---

### BUG-019.2: Script Type Errors

**Status**: OPEN
**Location**: `scripts/generate_embeddings.py`, `server.py`
**Impact**: Type checker cannot verify script correctness

**Evidence**:
```
scripts/generate_embeddings.py:190: error: Unexpected keyword argument "level" for "setup_logging"
scripts/generate_embeddings.py:190: error: Unexpected keyword argument "format" for "setup_logging"
server.py:40: error: Function is missing a return type annotation
server.py:178: error: Missing type parameters for generic type "dict"
```

**Root Cause**: Scripts were written before full type checking was enabled; `setup_logging` signature changed.

**Fix**:
1. Update `generate_embeddings.py` to use correct `setup_logging` signature
2. Add return type annotations to all `server.py` endpoint functions
3. Parameterize generic types (e.g., `dict[str, Any]`)

---

## P2: Inconsistent Default Values and Fallback Strategies

### BUG-019.3: Default Score of 3 on LLM Failure (Potential Bias)

**Status**: OPEN
**Severity**: P2
**Location**: `src/ai_psychiatrist/agents/judge.py:141, 157`

**Evidence**:
```python
# Line 141: On LLM failure
return EvaluationScore(
    metric=metric,
    score=3,  # MAGIC NUMBER: Middle of 1-5 scale
    explanation="LLM evaluation failed; default score used.",
)

# Line 157: On score extraction failure
score = 3  # MAGIC NUMBER
```

**Impact**:
- Using score=3 (middle of 1-5 Likert scale) biases results toward "acceptable"
- Score 3 is just under the threshold (<=3 triggers refinement per paper)
- This means LLM failures will trigger refinement, but the default itself may not be intentional

**Recommendation**:
- Make the fallback score configurable via `FeedbackLoopSettings`
- Consider score=2 (below threshold) to ensure refinement on failure
- Add explicit documentation for the fallback strategy
- Add metrics/logging to track fallback frequency

**Pattern Violated**: Fail-Fast Principle, Explicit Configuration

---

### BUG-019.4: Scattered Default Parameter Values

**Status**: OPEN
**Severity**: P2
**Location**: Multiple files

**Evidence**:
Hardcoded defaults appear in multiple locations instead of being centralized:

| Value | Locations | Purpose |
|-------|-----------|---------|
| `0.2` | `qualitative.py:89`, `quantitative.py:140,189`, `meta_review.py:100`, `ollama.py:292`, `protocols.py:63` | Default temperature |
| `0.8` | Same files | Default top_p |
| `20` | Same files | Default top_k |
| `0.1` | `quantitative.py:341` | Repair temperature |
| `0.0` | `judge.py:122` | Judge temperature |

**Impact**:
- Changing a default requires editing 10+ files
- Inconsistency risk if some locations are missed
- Violates DRY principle

**Recommendation**:
- Centralize all LLM parameter defaults in `config.py`
- Agents should always defer to `ModelSettings` or `get_settings()`
- Remove inline fallbacks or use a single `DEFAULT_LLM_PARAMS` constant

**Pattern Violated**: DRY (Don't Repeat Yourself), Single Source of Truth

---

## P3: Magic Numbers and Hardcoded Values

### BUG-019.5: Hardcoded Clinical Constants

**Status**: OPEN
**Severity**: P3
**Location**: Various

**Evidence**:

| Magic Number | Location | Purpose |
|--------------|----------|---------|
| `cap: int = 3` | `quantitative.py:232` | Max evidence items per domain |
| `200` | `quantitative.py:210`, `judge.py:155` | Response preview length for logging |
| `999_999` | `server.py:437` | Ad-hoc transcript participant ID |
| `4096` | Config default | Embedding dimension |
| `8` | Config default | Chunk size |

**DOMAIN_KEYWORDS Dictionary** (`prompts/quantitative.py:16-79`):
```python
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "PHQ8_NoInterest": [
        "can't be bothered",
        "no interest",
        # ... 80+ hardcoded keywords
    ],
    # ...
}
```

**Impact**:
- Domain keywords cannot be updated without code changes
- Paper references specific values but they're scattered in code
- Difficult to tune hyperparameters for optimization

**Recommendation**:
1. Move `DOMAIN_KEYWORDS` to a YAML/JSON config file
2. Create a `ClinicalConstants` class in config for magic numbers
3. Document paper references inline with constants:
   ```python
   # Paper Appendix D: chunk_size=8 optimal for DAIC-WOZ
   CHUNK_SIZE = 8
   ```

**Pattern Violated**: Configuration over Code, Self-Documenting Constants

---

### BUG-019.6: Participant ID Range Assumptions

**Status**: OPEN
**Severity**: P3
**Location**: `server.py:437`

**Evidence**:
```python
# Use a reserved, positive participant ID for ad-hoc transcripts to satisfy
# domain constraints (Transcript requires participant_id > 0) while avoiding
# collision with real DAIC-WOZ participants (300-492).
return transcript_service.load_transcript_from_text(
    participant_id=999_999,  # MAGIC NUMBER
    text=request.transcript_text,
)
```

**Impact**:
- If future datasets have IDs >= 999_999, collision could occur
- The valid ID range (300-492) is not enforced in domain layer
- Comments serve as documentation but aren't executable constraints

**Recommendation**:
1. Add `ADHOC_PARTICIPANT_ID` constant to config
2. Consider using negative IDs for ad-hoc (requires domain change)
3. Add validation in `Transcript` entity for known ID ranges

---

## P3: Global State Anti-Pattern

### BUG-019.7: Module-Level Globals in server.py

**Status**: OPEN
**Severity**: P3
**Location**: `server.py:31-36, 42-43`

**Evidence**:
```python
# --- Shared State (initialized at startup) ---
_ollama_client: OllamaClient | None = None
_transcript_service: TranscriptService | None = None
_embedding_service: EmbeddingService | None = None
_feedback_loop_service: FeedbackLoopService | None = None
_model_settings: ModelSettings | None = None
_settings: Settings | None = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ollama_client, _transcript_service, _embedding_service  # noqa: PLW0603
    global _feedback_loop_service, _model_settings, _settings  # noqa: PLW0603
```

**Impact**:
- Testing requires monkey-patching globals
- Thread safety concerns (though ASGI is single-threaded per worker)
- `noqa` comments suppress legitimate linter warnings
- Makes the code harder to reason about

**Recommendation**:
1. Use FastAPI's `app.state` for shared state:
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       app.state.ollama_client = OllamaClient(settings.ollama)
       # ...
   ```
2. Update dependency injection to use `request.app.state`
3. Consider a proper Dependency Injection container (e.g., `dependency-injector`)

**Pattern Violated**: Dependency Inversion, Testability, Single Responsibility

---

## P3: Pickle Security Concern

### BUG-019.8: Pickle Loading Without Validation

**Status**: OPEN
**Severity**: P3
**Location**: `services/reference_store.py:91-92`

**Evidence**:
```python
# SECURITY: This pickle file is a trusted internal artifact generated by our
# embedding pipeline (quantitative_assessment/embedding_batch_script.py).
# It is NOT user-supplied. The path is controlled via DataSettings which
# defaults to data/embeddings/participant_embedded_transcripts.pkl.
# If external/untrusted embeddings are ever needed, migrate to SafeTensors.
with self._embeddings_path.open("rb") as f:
    raw_data = pickle.load(f)
```

**Impact**:
- Pickle is known to be unsafe for untrusted data (arbitrary code execution)
- While currently internal-only, this creates a latent vulnerability
- Comment acknowledges the issue but doesn't enforce safety

**Recommendation**:
1. Add file hash validation before loading
2. Migrate to SafeTensors or NumPy `.npz` format (safer, also more portable)
3. Add a `verify_embedding_artifact()` function that checks:
   - File hash matches expected value
   - File is within expected data directory
   - File was generated by known pipeline version

**Pattern Violated**: Defense in Depth, Secure by Default

---

## P3: Test Suite Concerns

### BUG-019.9: Over-Mocking Risk (Vertical Integration Gap)

**Status**: OPEN
**Severity**: P3
**Location**: `tests/unit/`, `tests/integration/`

**Evidence**:
The test suite uses `MockLLMClient` extensively (correctly moved to `tests/fixtures/`), but:

1. **All integration tests mock the LLM layer**:
   ```python
   # tests/integration/test_dual_path_pipeline.py
   @pytest.fixture
   def qualitative_mock_client(self) -> MockLLMClient:
       return MockLLMClient(chat_responses=[...])
   ```

2. **Real Ollama tests are opt-in and separate** (BUG-018 fix was good)

3. **Mock responses are hand-crafted JSON/XML** that may drift from real model outputs

**Impact**:
- Tests pass but real LLM output format could differ
- Parsing/repair logic tested against synthetic data only
- Mock responses may be "too perfect" (no edge cases)

**Recommendation**:
1. Add "golden file" tests with real LLM outputs captured from dev runs
2. Create a `RealLLMOutputs` fixture with actual model responses
3. Run periodic validation against real models (weekly CI job)
4. Add fuzzing tests for parser robustness

**Pattern Violated**: Test Fidelity, Shift-Left Testing

---

### BUG-019.10: MockDataSettings Type Mismatch

**Status**: OPEN
**Severity**: P2
**Location**: `tests/unit/services/test_transcript.py`, `tests/unit/services/test_ground_truth.py`

**Evidence**:
```python
# tests/unit/services/test_transcript.py
class MockDataSettings:
    def __init__(self, path: Path | str):
        self.transcript_dir = Path(path) if isinstance(path, str) else path

# Type error:
service = TranscriptService(data_settings=mock_settings)
# error: Argument "data_settings" has incompatible type "MockDataSettings"; expected "DataSettings"
```

**Impact**:
- Mypy reports 24+ errors for this pattern
- Tests rely on duck typing that mypy can't verify
- Risk of tests passing but production failing

**Recommendation**:
1. Create a proper test double that implements the full interface:
   ```python
   class TestDataSettings(DataSettings):
       """Test-friendly DataSettings with in-memory paths."""
       model_config = ConfigDict(arbitrary_types_allowed=True)

       @classmethod
       def for_testing(cls, tmp_path: Path) -> "TestDataSettings":
           return cls(transcript_dir=tmp_path, ...)
   ```
2. Use `pytest` fixtures that return proper types
3. Alternatively, use `cast()` to suppress errors with documented rationale

---

## P4: Minor Code Quality Issues

### BUG-019.11: Missing Return Type Annotations

**Status**: OPEN
**Severity**: P4
**Location**: Various (see mypy output)

Files with missing annotations:
- `server.py`: 5 functions
- `tests/e2e/*.py`: Multiple test fixtures
- `scripts/generate_embeddings.py`: `main()` function

**Recommendation**: Add return type annotations during next touch.

---

### BUG-019.12: Inconsistent Error Handling Patterns

**Status**: OPEN
**Severity**: P4
**Location**: Various

**Pattern 1**: Silent fallback (quantitative agent)
```python
except (json.JSONDecodeError, ValueError):
    logger.warning("Failed to parse evidence JSON, using empty evidence")
    obj = {}
```

**Pattern 2**: Default value (judge agent)
```python
if score is None:
    logger.warning("Could not extract score, defaulting to 3")
    score = 3
```

**Pattern 3**: Re-raise with context (server.py)
```python
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Assessment failed: {e}") from e
```

**Recommendation**:
- Document the error handling strategy in a CONTRIBUTING.md or ADR
- Standardize on one pattern per error category
- Consider using Result types (`result` library) for explicit error handling

---

### BUG-019.13: Legacy Code Pollution (Resolved but Notable)

**Status**: RESOLVED (BUG-013, BUG-015)
**Location**: `_legacy/`

The `_legacy/` directory contains old implementations that are properly archived but still have TODOs:

```python
_legacy/assets/ollama_example.py:4:OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable
_legacy/meta_review/meta_review.py:40:    OLLAMA_NODE = "arctrdgn001" # TODO: Change this variable
_legacy/qualitative_assessment/qual_assessment.py:11:OLLAMA_NODE = "arctrddgxa003" # TODO: Change this variable
```

**Recommendation**: Add `noqa` comments or remove if truly archived.

---

## Summary Statistics

| Priority | Count | Status |
|----------|-------|--------|
| P0 | 0 | - |
| P1 | 2 | Type safety issues |
| P2 | 3 | Defaults, mock types |
| P3 | 7 | Magic numbers, globals, tests |
| P4 | 3 | Minor quality issues |
| **Total** | **15** | **Issues identified** |

---

## Recommended Action Plan

### Phase 1: Type Safety (Week 1)
1. Fix MockDataSettings type mismatches
2. Add return type annotations to server.py
3. Fix scripts/generate_embeddings.py logging call
4. Target: Zero mypy errors

### Phase 2: Centralize Configuration (Week 2)
1. Move all LLM parameter defaults to config.py
2. Create `ClinicalConstants` for domain values
3. Extract DOMAIN_KEYWORDS to YAML
4. Add fallback score configuration

### Phase 3: Refactor Globals (Week 3)
1. Migrate server.py to FastAPI `app.state`
2. Update dependency injection
3. Improve testability

### Phase 4: Test Hardening (Week 4)
1. Add golden file tests with real LLM outputs
2. Create typed test fixtures
3. Add parser fuzzing tests

---

## References

- [Clean Code (Robert C. Martin)](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Python Type Checking Guide](https://mypy.readthedocs.io/en/stable/)
- [FastAPI Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Pickle Security Warning](https://docs.python.org/3/library/pickle.html#module-pickle)
