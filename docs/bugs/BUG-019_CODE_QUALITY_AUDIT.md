# BUG-019: Comprehensive Code Quality Audit

**Severity**: MIXED (P3-P4)
**Status**: RESOLVED (audit closed; remaining items deferred)
**Date Identified**: 2025-12-20
**Audit Scope**: Full codebase review
**Last Updated**: 2025-12-21 (Corrected after implementation)

---

## Executive Summary

This document captures issues identified during a comprehensive code quality audit of the ai-psychiatrist codebase. After deep investigation of the specs and CI configuration, several initially-identified "issues" were found to be deliberate design decisions.

**Key Finding**: The codebase is in excellent health. CI passes cleanly, type checking on production code passes with zero errors, and most "magic numbers" are paper-documented hyperparameters properly centralized in `config.py`.

**Genuine Issues Found**: 2 P3 issues, 3 P4 issues (total: 5)

---

## Investigation Context

### CI Configuration Analysis

The CI pipeline (`ci.yml`) is configured to:
- **Type check `src tests scripts server.py`**: `uv run mypy src tests scripts server.py` (strict, 0 errors)
- **Lint `src tests scripts server.py`**: Passes with 0 errors
- **Test with coverage**: 96%+ coverage threshold met

This was intentionally broadened from a `src/`-only scope to prevent hidden
type errors in tests and scripts.

---

## Corrected Findings

### NOT BUGS (Initially Misidentified)

#### ~~BUG-019.3: Default Score of 3~~ → DESIGN DECISION

**Status**: ✅ NOT A BUG - Intentional per Spec 07

The `score=3` fallback in `judge.py:141, 157` is **intentional and correct**:

From **Spec 07** (lines 347-363):
```python
except LLMError as e:
    logger.error(...)
    return EvaluationScore(
        metric=metric,
        score=3,  # INTENTIONAL: At threshold, triggers refinement
        explanation="LLM evaluation failed; default score used.",
    )
```

**Why score=3 is correct**:
- Per paper Section 2.3.1: "scores < 4 trigger refinement" (i.e., score ≤ 3)
- Using 3 ensures that **failures trigger refinement** (fail-safe behavior)
- This is documented in the spec and is defensive coding

---

#### ~~BUG-019.4: Scattered Default Parameter Values~~ → DESIGN DECISION

**Status**: ✅ NOT A BUG - Values ARE centralized in `config.py`

**Evidence of centralization** (from `src/ai_psychiatrist/config.py`):
```python
class ModelSettings(BaseSettings):
    temperature: float = Field(default=0.2, ...)  # Paper-optimal
    temperature_judge: float = Field(default=0.0, ...)  # Deterministic
    top_k: int = Field(default=20, ...)
    top_p: float = Field(default=0.8, ...)
```

The "fallbacks" in agents (e.g., `qualitative.py:89`) are **defensive coding** for when `ModelSettings` is `None`:
```python
# Uses model settings if provided, otherwise falls back to paper defaults
temperature = self._model_settings.temperature if self._model_settings else 0.2
```

This is correct - agents accept `ModelSettings | None` to be testable without full config.

---

#### ~~BUG-019.5: Hardcoded Clinical Constants~~ → PAPER-DOCUMENTED VALUES

**Status**: ✅ NOT A BUG - All values from paper, documented in Spec 03

From **Spec 03** (Configuration & Logging):

| Value | Paper Reference | Location in Config |
|-------|-----------------|-------------------|
| `4096` | Paper Appendix D | `EmbeddingSettings.dimension` |
| `8` | Paper Appendix D | `EmbeddingSettings.chunk_size` |
| `2` | Paper Appendix D | `EmbeddingSettings.top_k_references` |
| `0.2` | Paper Section 2.2 | `ModelSettings.temperature` |
| `3` | Paper Section 2.3.1 | `FeedbackLoopSettings.score_threshold` |
| `10` | Paper Section 2.3.1 | `FeedbackLoopSettings.max_iterations` |

**Remaining valid concern**: `DOMAIN_KEYWORDS` dictionary could be externalized to YAML for easier updates. Retained as P4 below.

---

## Genuine Issues (Post-Fix)

### P3: Module-Level Globals in server.py

**Status**: ✅ RESOLVED
**Severity**: P3
**Location**: `server.py` (lifespan + dependency injection)

**Evidence**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama_client = OllamaClient(settings.ollama)
    # ...

def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client
```

**Per 2025 FastAPI Best Practices**:
From [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/):
> "Because of that, it's now recommended to instead use the lifespan... Doing startup/shutdown logic in separated functions that don't share logic or variables together is more difficult as you would need to store values in global variables or similar tricks."

From [Using FastAPI Like a Pro](https://medium.com/@hieutrantrung.it/using-fastapi-like-a-pro-with-singleton-and-dependency-injection-patterns-28de0a833a52):
> "Never store mutable state in globals — use `app.state` for shared singletons."

**Impact**:
- `noqa` comments suppress legitimate linter warnings
- Testing requires monkey-patching module globals
- Doesn't follow current best practices

**Recommendation**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama_client = OllamaClient(settings.ollama)
    app.state.settings = settings
    yield
    await app.state.ollama_client.close()

# In endpoints:
def get_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client
```

---

### P3: Missing `py.typed` Marker (PEP 561)

**Status**: ✅ RESOLVED
**Severity**: P3
**Location**: `src/ai_psychiatrist/py.typed`

**Evidence**:
When running `mypy tests/`, errors appear:
```
tests/unit/test_cli.py:3: error: Skipping analyzing "ai_psychiatrist": module is
installed, but missing library stubs or py.typed marker  [import-untyped]
```

**Per [PEP 561](https://peps.python.org/pep-0561/)**:
> Package maintainers who wish to support type checking of their code MUST add a marker file named `py.typed` to their package.

From [The Importance of py.typed](https://safjan.com/the-importance-of-adding-py-typed-file-to-your-typed-package/):
> Without a `py.typed` file, mypy will raise a "module is installed, but missing library stubs or py.typed marker" error.

**Impact**:
- Cannot run `mypy` on tests that import from the package
- Downstream consumers cannot get type info from this package
- Breaks type checking for scripts (`server.py`, `scripts/*.py`)

**Fix Applied**:
- Added `src/ai_psychiatrist/py.typed`
- Ensured it is included in wheel builds via hatchling `force-include`

---

### P4: DOMAIN_KEYWORDS Not Externalized

**Status**: ✅ RESOLVED
**Severity**: P4
**Location**: `src/ai_psychiatrist/resources/phq8_keywords.yaml`

**Fix Applied**:
- Moved keywords from Python dict to packaged YAML at `src/ai_psychiatrist/resources/phq8_keywords.yaml`
- Added PyYAML dependency
- Keywords now loaded via `_load_domain_keywords()` with LRU cache
- Domain experts can now review/update keywords without code changes

```yaml
# src/ai_psychiatrist/resources/phq8_keywords.yaml
PHQ8_NoInterest:
  - "can't be bothered"
  - "no interest"
  # ...
```

---

### P4: Ad-hoc Participant ID Magic Number

**Status**: ✅ RESOLVED
**Severity**: P4
**Location**: `server.py:32`

**Evidence**:
```python
AD_HOC_PARTICIPANT_ID = 999_999
```

**Impact**:
- Magic number not in config
- Comment documents intent but isn't executable

**Note**: This was resolved with a named constant in `server.py`. If this value
is needed outside the API layer in the future, consider promoting it to `config.py`.

---

### P4: Pickle Usage (Acknowledged in Code)

**Status**: ✅ RESOLVED
**Severity**: P4
**Location**: `src/ai_psychiatrist/services/reference_store.py`

**Fix Applied**:
- Replaced pickle with NPZ + JSON sidecar format
- NPZ file stores embeddings (numpy arrays, no code execution)
- JSON file stores text chunks (safe, standard format)
- Updated `scripts/generate_embeddings.py` to produce new format
- Updated `ReferenceStore` to load new format

```text
# New format:
data/embeddings/reference_embeddings.npz   # Embeddings as numpy arrays
data/embeddings/reference_embeddings.json  # Text chunks
```

---

## Summary Statistics (Corrected)

| Priority | Count | Status |
|----------|-------|--------|
| P0 | 0 | - |
| P1 | 0 | - |
| P2 | 0 | - |
| P3 | 2 | ✅ All resolved |
| P4 | 3 | ✅ All resolved |
| **Total** | **5** | **All resolved (0 open)** |

**Initially Reported**: 15 issues
**After Investigation**: 5 genuine issues (10 were design decisions or expected behavior)
**All Resolved**: 2025-12-21

---

## Recommended Action Plan

### Completed
1. ✅ Externalized `DOMAIN_KEYWORDS` to YAML
2. ✅ Replaced pickle embeddings with NPZ + JSON format

---

## References

- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [PEP 561 - Distributing Type Information](https://peps.python.org/pep-0561/)
- [mypy Documentation](https://mypy.readthedocs.io/en/stable/)
- [Using FastAPI Like a Pro - Singleton Pattern](https://medium.com/@hieutrantrung.it/using-fastapi-like-a-pro-with-singleton-and-dependency-injection-patterns-28de0a833a52)
- Project Specs: `docs/specs/01_PROJECT_BOOTSTRAP.md`, `docs/specs/03_CONFIG_LOGGING.md`, `docs/specs/07_JUDGE_AGENT.md`
