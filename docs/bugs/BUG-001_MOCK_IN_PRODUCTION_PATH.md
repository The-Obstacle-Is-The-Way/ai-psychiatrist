# BUG-001: MockLLMClient in Production Code Path

**Severity**: HIGH
**Status**: RESOLVED
**Date Identified**: 2025-12-18
**Date Resolved**: 2025-12-18
**Spec Reference**: `docs/specs/04_LLM_INFRASTRUCTURE.md`

---

## Resolution Summary

**Option A was implemented**: MockLLMClient moved to `tests/fixtures/mock_llm.py`.

Changes made:
1. Created `tests/fixtures/mock_llm.py` with full MockLLMClient implementation
2. Created `tests/fixtures/__init__.py` with proper exports and documentation
3. Deleted `src/ai_psychiatrist/infrastructure/llm/mock.py`
4. Removed MockLLMClient from `src/ai_psychiatrist/infrastructure/llm/__init__.py`
5. Updated all test imports to use `from tests.fixtures.mock_llm import MockLLMClient`
6. Updated Spec 04 with Test Double Location Policy
7. Updated related specs (06, 07, 08) to use `tests/fixtures` import path

Verification:
```bash
rg -n "MockLLMClient" src/  # Returns only docstring reference explaining the policy
rg -n "infrastructure\\.llm\\.mock" docs/specs  # Only appears in commented WRONG example in Spec 04
```

---

## Current State (Post-Fix)

### File Location (Current)
```
src/ai_psychiatrist/infrastructure/llm/
├── __init__.py          # No MockLLMClient export
├── ollama.py            # Production client
├── protocols.py         # Abstractions
└── responses.py         # Parsing utilities

tests/fixtures/
├── __init__.py
└── mock_llm.py          # MockLLMClient (test-only)
```

### Import Pattern (Tests Only)
```python
from tests.fixtures.mock_llm import MockLLMClient
```

---

## Original Analysis (Historical)

### Executive Summary (Original Issue)

`MockLLMClient` was located in `src/ai_psychiatrist/infrastructure/llm/mock.py` and exported via `__init__.py`, making it part of the public package and importable from production code paths. This matched Spec 04 as written, but it also meant test code shipped in the production artifact.

For a **medical AI system evaluating psychiatric assessments**, any accidental use of a mock client in production would be a high-impact failure mode.

---

## Original State Analysis (Pre-Fix)

### File Location (Pre-Fix)
```
src/ai_psychiatrist/infrastructure/llm/
├── __init__.py          # EXPORTS MockLLMClient publicly
├── mock.py              # MockLLMClient lives here
├── ollama.py            # Production client
├── protocols.py         # Abstractions
└── responses.py         # Parsing utilities
```

### Export Analysis (Pre-Fix)
```python
# src/ai_psychiatrist/infrastructure/llm/__init__.py
from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient  # EXPORTED

__all__ = [
    ...
    "MockLLMClient",  # PUBLICLY AVAILABLE
    ...
]
```

### Risk Vector (Pre-Fix)
Any developer (or future agent) can do:
```python
from ai_psychiatrist.infrastructure.llm import MockLLMClient

# And accidentally use it in production code
client = MockLLMClient(chat_responses=["Patient is fine."])
```

If used in production, this would return **canned responses instead of real LLM analysis** for psychiatric assessments. There is no evidence this is currently happening; the concern is about risk surface.

---

## First Principles Analysis

### What Are We Building?

An AI system that:
1. Analyzes psychiatric interview transcripts (DAIC-WOZ dataset)
2. Generates PHQ-8 depression severity assessments
3. Produces clinical rationales for those assessments

**The stakes**: Wrong outputs could inform clinical decisions. A mock returning `"Patient is fine"` when the real system would flag severe depression is a **patient safety issue**.

### Clean Architecture Principles (Robert C. Martin)

Per [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html):

> "The overriding rule that makes this architecture work is **The Dependency Rule**. This rule says that source code dependencies can only point inwards. Nothing in an inner circle can know anything at all about something in an outer circle."

**Test doubles are outer circle concerns.** They exist to verify inner circle behavior, not to participate in it. Placing `MockLLMClient` in `src/` makes it easy for production code to depend on test infrastructure, which weakens the separation even if current production code does not depend on it today.

### ISO 27001 Control 8.31

Per [ISO 27001:2022 Control 8.31](https://www.isms.online/iso-27002/control-8-31-separation-of-development-test-and-production-environments/):

> "Development, testing and production environments should be separated to reduce the risks of unauthorized access or changes to the production environment."

`MockLLMClient` in `src/` blurs this boundary and could be flagged as a separation-of-environments concern in a stricter audit.

### 8th Light's Guidance

Per [8th Light - Don't Mix Test Code with Production Code](https://8thlight.com/insights/dont-mix-test-code-with-production-code):

> "Production code should not contain test code. For example, having `if (isTest)` blocks clutters the production code with test logic."

Shipping `MockLLMClient` in the production package is effectively shipping test code, even if it is not used at runtime.

---

## Why Did This Happen?

### Spec Analysis

Spec 04 explicitly defines `MockLLMClient` under `src/ai_psychiatrist/infrastructure/llm/mock.py` and uses imports like:

```
from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient
```

The implementation is therefore **spec-compliant**. If we want to remove the mock from the production package, this requires a **spec change**, not just an implementation change.

---

## Options Analysis

### Option A: Move to `tests/` (Recommended)

**Location**: `tests/fixtures/mock_llm.py` or `tests/conftest.py`

**Pros**:
- **Complete separation**: Not included in production artifacts if tests are excluded from packaging (default in this repo)
- **Follows Clean Architecture**: Test doubles stay in test layer
- **Follows ISO 27001**: Clearer environment separation
- **Rob Martin approved**: Dependency rule preserved

**Cons**:
- Requires imports from `src/` for protocol types (acceptable - inner circles don't know about outer)
- Slight refactor needed

**Implementation**:
```python
# tests/fixtures/mock_llm.py
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse
)

class MockLLMClient:
    """Test double for LLM clients. NEVER import in production."""
    ...
```

### Option B: Keep in `src/` but Remove from `__init__.py`

**Pros**:
- Minimal change
- Still accessible for tests via explicit import

**Cons**:
- **Still in production path**: `from ai_psychiatrist.infrastructure.llm.mock import MockLLMClient` works
- **Clean Architecture risk**: Production package contains test code
- **Audit risk**: Test code in production artifact could be questioned in stricter audits
- **Half-measure**: Reduces but doesn't eliminate risk

### Option C: Runtime Guard

**Implementation**:
```python
class MockLLMClient:
    def __init__(self, ...):
        if "pytest" not in sys.modules:
            raise RuntimeError("MockLLMClient cannot be used outside tests")
        ...
```

**Pros**:
- Catches accidental production usage at runtime

**Cons**:
- **Still ships in production**: The code exists, just with a guard
- **Can be bypassed**: `import pytest` before instantiation
- **May not trigger in all runners**: `unittest` or custom harnesses
- **Not Clean Architecture**: Guard is a code smell indicating wrong location

---

## Recommendation

**Option A: Move MockLLMClient to `tests/fixtures/mock_llm.py`**

This is the only option that:
1. Provides **complete separation** (not shipped in production artifacts when tests are excluded)
2. Follows **Clean Architecture** (dependency rule)
3. Strengthens **ISO 27001 Control 8.31** alignment (environment separation)
4. Eliminates the risk **entirely** in standard packaging setups

---

## Proposed Fix

### Step 1: Create test fixtures module
```
tests/
├── fixtures/
│   ├── __init__.py
│   └── mock_llm.py      # MockLLMClient moves here
├── conftest.py          # Expose fixtures globally
└── unit/
    └── infrastructure/
        └── llm/
            └── test_mock.py  # Update imports
```

### Step 2: Update Spec
Add to `04_LLM_INFRASTRUCTURE.md`:

```markdown
## Test Double Location Policy

MockLLMClient is a **test-only artifact** and MUST NOT exist in `src/`.

Location: `tests/fixtures/mock_llm.py`

Rationale:
- Clean Architecture: Test doubles are outer circle concerns
- ISO 27001 8.31: Separation of test and production environments
- Safety: Medical AI system cannot risk mock contamination
```

### Step 3: Remove from `src/`
- Delete `src/ai_psychiatrist/infrastructure/llm/mock.py`
- Remove export from `__init__.py`
- Update all test imports

---

## Affected Files

| File | Action |
|------|--------|
| `src/ai_psychiatrist/infrastructure/llm/mock.py` | DELETE |
| `src/ai_psychiatrist/infrastructure/llm/__init__.py` | Remove MockLLMClient export |
| `tests/fixtures/mock_llm.py` | CREATE (move content here) |
| `tests/fixtures/__init__.py` | CREATE |
| `tests/conftest.py` | Add fixture exposure |
| `tests/unit/infrastructure/llm/test_mock.py` | Update imports |
| `tests/unit/infrastructure/llm/test_responses.py` | Update imports |
| `docs/specs/04_LLM_INFRASTRUCTURE.md` | Add location policy |

---

## Risk if Not Fixed

| Scenario | Probability | Impact | Risk |
|----------|-------------|--------|------|
| Developer imports MockLLMClient in production code | Low | CRITICAL | HIGH |
| Future agent auto-generates code using MockLLMClient | Medium | CRITICAL | HIGH |
| Code review misses mock usage in complex PR | Medium | CRITICAL | HIGH |
| Mock responses used for actual patient assessments due to misconfiguration | Low | CATASTROPHIC | CRITICAL |

---

## Decision Required

**Awaiting senior consensus on**:

1. Is Option A (move to `tests/`) the correct approach?
2. Should we add a pre-commit hook to prevent `mock` imports in `src/`?
3. Should the spec be updated to include explicit location policies for all test artifacts?

---

## References

- [Robert C. Martin - The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [8th Light - Don't Mix Test Code with Production Code](https://8thlight.com/insights/dont-mix-test-code-with-production-code)
- [ISO 27001:2022 Control 8.31 - Separation of Environments](https://www.isms.online/iso-27002/control-8-31-separation-of-development-test-and-production-environments/)
- [Clean Code Episode 23 - Mocking (Robert Martin)](https://cleancoders.com/episode/clean-code-episode-23-p1)
- [Pytest Common Mocking Problems](https://pytest-with-eric.com/mocking/pytest-common-mocking-problems/)
- [Mark Seemann - Treat Test Code Like Production Code](https://blog.ploeh.dk/2025/12/01/treat-test-code-like-production-code/)
