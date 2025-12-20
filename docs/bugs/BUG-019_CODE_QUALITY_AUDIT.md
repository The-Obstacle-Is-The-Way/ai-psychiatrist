# BUG-019: Comprehensive Code Quality Audit

**Severity**: MIXED (P3-P4)
**Status**: OPEN
**Date Identified**: 2025-12-20
**Audit Scope**: Full codebase review
**Last Updated**: 2025-12-20 (Corrected after deep investigation)

---

## Executive Summary

This document captures issues identified during a comprehensive code quality audit of the ai-psychiatrist codebase. After deep investigation of the specs and CI configuration, several initially-identified "issues" were found to be deliberate design decisions.

**Key Finding**: The codebase is in excellent health. CI passes cleanly, type checking on production code passes with zero errors, and most "magic numbers" are paper-documented hyperparameters properly centralized in `config.py`.

**Genuine Issues Found**: 2 P3 issues, 3 P4 issues (total: 5)

---

## Investigation Context

### CI Configuration Analysis

The CI pipeline (`ci.yml`) is intentionally configured to:
- **Type check only `src/`**: `uv run mypy src` (passes with 0 errors)
- **Lint `src tests scripts server.py`**: Passes with 0 errors
- **Test with coverage**: 96%+ coverage threshold met

This is a deliberate architectural choice per **Spec 01** which states:
> `typecheck: ## Run type checker (mypy)`
> `uv run mypy src`

### Why Tests Aren't Type-Checked

When running `mypy tests/`, errors appear because:
1. **No `py.typed` marker file** - See BUG-019.1 below
2. **Duck-typed test doubles** - Tests use lightweight mocks that don't implement full protocol contracts

This is a common Python pattern. Many projects only type-check production code due to:
- Test fixtures often use duck typing for flexibility
- External libraries may lack type stubs
- Speed of type checking matters more in CI

Per [mypy best practices](https://mypy.readthedocs.io/en/stable/existing_code.html), gradual adoption is recommended.

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

## Genuine Issues (5 Total)

### P3: Module-Level Globals in server.py

**Status**: OPEN
**Severity**: P3
**Location**: `server.py:31-43`

**Evidence**:
```python
# --- Shared State (initialized at startup) ---
_ollama_client: OllamaClient | None = None
_transcript_service: TranscriptService | None = None
# ... more globals ...

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ollama_client, _transcript_service  # noqa: PLW0603
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

**Status**: OPEN
**Severity**: P3
**Location**: `src/ai_psychiatrist/` (missing file)

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

**Fix**:
```bash
touch src/ai_psychiatrist/py.typed
# Add to pyproject.toml package_data if using setuptools
```

---

### P4: DOMAIN_KEYWORDS Not Externalized

**Status**: OPEN
**Severity**: P4
**Location**: `src/ai_psychiatrist/agents/prompts/quantitative.py:16-79`

**Evidence**:
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
- Clinical keywords cannot be updated without code changes
- Domain experts can't easily review/update keywords
- No version control for keyword changes separate from code

**Recommendation**:
Move to `data/keywords/phq8_keywords.yaml`:
```yaml
PHQ8_NoInterest:
  - "can't be bothered"
  - "no interest"
  # ...
```

---

### P4: Ad-hoc Participant ID Magic Number

**Status**: OPEN
**Severity**: P4
**Location**: `server.py:437`

**Evidence**:
```python
return transcript_service.load_transcript_from_text(
    participant_id=999_999,  # Magic number for ad-hoc transcripts
    text=request.transcript_text,
)
```

**Impact**:
- Magic number not in config
- Comment documents intent but isn't executable

**Recommendation**:
Add to `config.py`:
```python
class APISettings(BaseSettings):
    adhoc_participant_id: int = Field(
        default=999_999,
        description="Reserved ID for ad-hoc transcripts (avoid DAIC-WOZ range 300-492)",
    )
```

---

### P4: Pickle Usage (Acknowledged in Code)

**Status**: OPEN (LOW PRIORITY - Already documented)
**Severity**: P4
**Location**: `services/reference_store.py:86-92`

**Evidence**:
```python
# SECURITY: This pickle file is a trusted internal artifact generated by our
# embedding pipeline... If external/untrusted embeddings are ever needed,
# migrate to SafeTensors.
with self._embeddings_path.open("rb") as f:
    raw_data = pickle.load(f)
```

**Assessment**:
- The code already has a security comment acknowledging the risk
- File is internal-only, path controlled via `DataSettings`
- Low priority since this is documented and understood

**Future Consideration**:
When migrating to production, consider SafeTensors or `.npz` format.

---

## Summary Statistics (Corrected)

| Priority | Count | Status |
|----------|-------|--------|
| P0 | 0 | - |
| P1 | 0 | - |
| P2 | 0 | - |
| P3 | 2 | FastAPI globals, missing py.typed |
| P4 | 3 | DOMAIN_KEYWORDS, magic ID, pickle |
| **Total** | **5** | **Genuine issues** |

**Initially Reported**: 15 issues
**After Investigation**: 5 genuine issues (10 were design decisions or expected behavior)

---

## Recommended Action Plan

### Immediate (Optional)
1. Add `py.typed` marker file (5 minutes)
2. Add `ADHOC_PARTICIPANT_ID` to config (5 minutes)

### Short-term (If refactoring server.py)
1. Migrate module globals to `app.state`
2. Update dependency injection pattern

### Long-term (Nice to have)
1. Externalize `DOMAIN_KEYWORDS` to YAML
2. Consider SafeTensors for embeddings

---

## References

- [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)
- [PEP 561 - Distributing Type Information](https://peps.python.org/pep-0561/)
- [mypy Documentation](https://mypy.readthedocs.io/en/stable/)
- [Using FastAPI Like a Pro - Singleton Pattern](https://medium.com/@hieutrantrung.it/using-fastapi-like-a-pro-with-singleton-and-dependency-injection-patterns-28de0a833a52)
- Project Specs: `docs/specs/01_PROJECT_BOOTSTRAP.md`, `docs/specs/03_CONFIG_LOGGING.md`, `docs/specs/07_JUDGE_AGENT.md`
