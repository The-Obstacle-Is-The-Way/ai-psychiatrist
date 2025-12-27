# Spec 21: Broad Exception Handling Cleanup

> **STATUS: OPEN**
>
> **Priority**: Low — Code works correctly. This is maintainability/readability debt.
>
> **GitHub Issue**: #60 (parent tech-debt audit)
>
> **Created**: 2025-12-26
>
> **Last Verified Against Code**: 2025-12-27

---

## Problem Statement

The codebase contains **24** instances of `except Exception` (non-doc code). Some are **intentional** (best-effort helpers, graceful degradation paths), but others can mask unexpected errors and make debugging harder.

This spec enumerates **every current** `except Exception` site and defines what we should do for each:
- **Replace** with narrower exception types when the error surface is understood.
- **Keep** broad catches only when explicitly justified (best-effort / “never crash” zones), with an explicit comment.

Note: GitHub Issue #60 originally listed 9 sites; the repo has since grown (FastAPI endpoints + Pydantic AI fallback paths + provenance hashing), so this spec tracks the **current** inventory.

---

## Current State: All Instances

### Inventory Summary (Current)

- **24×** `except Exception` in non-doc code:
  - `server.py`: 6
  - `src/ai_psychiatrist/services/reference_store.py`: 3
  - `scripts/generate_embeddings.py`: 3
  - `scripts/reproduce_results.py`: 2
  - `scripts/prepare_dataset.py`: 2
  - `src/ai_psychiatrist/infrastructure/llm/huggingface.py`: 2
  - `src/ai_psychiatrist/agents/qualitative.py`: 2
  - `src/ai_psychiatrist/agents/{judge,quantitative,meta_review}.py`: 3
  - `src/ai_psychiatrist/infrastructure/logging.py`: 1

---

### Category A: “Never Crash / Best-Effort” Helpers (Intentionally Broad)

These should be maximally defensive. If we keep them broad, we should add a short comment explaining why.

| File | Symbol | Current | Why It Exists | Recommended |
|------|--------|---------|---------------|-------------|
| `src/ai_psychiatrist/infrastructure/logging.py` | `_stdout_isatty()` | `except Exception` | Logging must not crash due to odd stdout objects | **Keep broad** |
| `scripts/prepare_dataset.py` | `_sample_transcript_lines()` | `except Exception` | Validation helper: best-effort sample | **Keep broad** |

---

### Category B: Split/Artifact Provenance Hashing (Best-Effort, Should Narrow)

These are “best-effort” but should avoid masking programmer errors. Narrow to expected IO/parse failures.

| File | Symbol | Current | Recommended |
|------|--------|---------|-------------|
| `src/ai_psychiatrist/services/reference_store.py` | `_calculate_split_ids_hash()` | `except Exception: return None` | Catch `(OSError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError, ValueError, TypeError)` |
| `src/ai_psychiatrist/services/reference_store.py` | `_derive_artifact_ids_hash()` | `except Exception: return None` | Catch `(OSError, json.JSONDecodeError, KeyError, ValueError, TypeError)` |
| `scripts/generate_embeddings.py` | `calculate_split_ids_hash()` | `except Exception: return "error"` | Catch `(OSError, pd.errors.ParserError, pd.errors.EmptyDataError, KeyError, ValueError, TypeError)` |

---

### Category C: External Backend Wrappers (HuggingFace Client)

These wrap external service calls where many exception types are possible.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/ai_psychiatrist/infrastructure/llm/huggingface.py` | `HuggingFaceClient.chat()` | `except Exception` | Catch `(RuntimeError, ValueError, OSError)` **or keep broad with explicit justification** |
| `src/ai_psychiatrist/infrastructure/llm/huggingface.py` | `HuggingFaceClient.embed()` | `except Exception` | Catch `(RuntimeError, ValueError, OSError)` **or keep broad with explicit justification** |

**Important constraint**: this module uses lazy imports. Do **not** reference `torch.cuda.OutOfMemoryError` in an `except (...)` tuple unless you restructure to avoid importing torch eagerly (otherwise you defeat optional deps).

---

### Category D: Pydantic AI Fallback (Agent Layer, Intentional Degradation)

These are **intentional fallback patterns** where Pydantic AI is tried first, then legacy logic on failure.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/ai_psychiatrist/agents/judge.py` | `_evaluate_metric()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior, LLMError)` |
| `src/ai_psychiatrist/agents/quantitative.py` | `_score_items()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior, LLMError)` |
| `src/ai_psychiatrist/agents/qualitative.py` | `assess()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior, LLMError)` |
| `src/ai_psychiatrist/agents/qualitative.py` | `refine()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior, LLMError)` |
| `src/ai_psychiatrist/agents/meta_review.py` | `review()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior, LLMError)` |

**Note**: `asyncio.CancelledError` is a `BaseException` in Python 3.11+ and will not be caught by `except Exception`. The explicit `except asyncio.CancelledError: raise` blocks are therefore redundant but harmless.

**Decision Needed**: Is graceful degradation to legacy parsing acceptable for any exception, or should we only catch Pydantic AI / LLM specific errors?

---

### Category E: Metadata Loading / Validation (Service Layer)

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/ai_psychiatrist/services/reference_store.py` | `_load_embeddings()` | `except Exception` around meta load/validate | Catch `(OSError, json.JSONDecodeError, ValueError, TypeError)` and **let unexpected exceptions propagate**; keep explicit re-raise for `EmbeddingArtifactMismatchError` |

---

### Category F: Script Error Handling / Long-Run Robustness

Scripts often choose to continue on per-participant failures. That’s acceptable, but we should still avoid swallowing programmer errors.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `scripts/reproduce_results.py` | `evaluate_participant()` | `except Exception` | Keep broad (record failure), but use `logger.exception(...)` to preserve stack, OR narrow to known domain errors + re-raise unexpected |
| `scripts/reproduce_results.py` | `check_ollama_connectivity()` | `except Exception` | Catch `LLMError` (since `OllamaClient.ping()` wraps httpx into `LLMError`) |
| `scripts/generate_embeddings.py` | `process_participant()` transcript load | `except Exception` | Catch `(TranscriptError, EmptyTranscriptError)` |
| `scripts/generate_embeddings.py` | `process_participant()` embed chunk | `except Exception` | Catch `(LLMError, LLMTimeoutError, ValueError)` (client wrappers already convert httpx/TimeoutError) |
| `scripts/prepare_dataset.py` | transcript extraction loop | `except Exception` | Catch `(OSError, KeyError, ValueError)` (keep `zipfile.BadZipFile` explicit) |

---

### Category G: FastAPI Layer (server.py)

The API currently catches `Exception` in multiple endpoints/helpers. Prefer catching known domain errors and letting unexpected exceptions crash the request (FastAPI will return 500 + logs), rather than masking.

| File | Symbol | Current | Recommended |
|------|--------|---------|-------------|
| `server.py` | `health_check()` ollama ping | `except Exception` | Catch `LLMError` (Ollama ping wraps httpx) |
| `server.py` | `assess_quantitative()` | `except Exception` | Catch `LLMError`, `EmbeddingArtifactMismatchError`, `TranscriptError` / `EmptyTranscriptError` and map to 4xx/5xx; let unknown exceptions propagate |
| `server.py` | `assess_qualitative()` | `except Exception` | Same principle as above |
| `server.py` | `run_full_pipeline()` | `except Exception` | Same principle as above |
| `server.py` | `_resolve_transcript()` participant_id path | `except Exception` | Catch `(TranscriptError, EmptyTranscriptError)` |
| `server.py` | `_resolve_transcript()` ad-hoc text path | `except Exception` | Catch `(TranscriptError, EmptyTranscriptError, ValueError)` |

---

## Implementation Plan

### Phase 1: Update Spec Inventory (This Document)

- Ensure this spec stays in sync with `rg -n "except Exception" --glob '!docs/**'` output.

### Phase 2: High-Signal Narrowing (Low Risk)

- `server.py` transcript resolution: catch transcript-domain errors only.
- `scripts/reproduce_results.py` connectivity check: catch `LLMError`.
- `ReferenceStore` meta load: catch JSON/IO errors and let other exceptions propagate.

### Phase 3: Best-Effort Helpers

- Narrow provenance hash helpers to IO/parse error types (avoid masking bugs).

### Phase 4: Pydantic AI Fallback (Decision Required)

- Either keep broad with explicit comment, or narrow to Pydantic AI + validation + LLM errors.

### Phase 5: HuggingFace Client

- Decide whether to keep broad `except Exception` (document why), or narrow to `(RuntimeError, ValueError, OSError)` without importing torch eagerly.

Replace broad exceptions in `src/ai_psychiatrist/infrastructure/llm/huggingface.py`:

```python
# Before
except Exception as e:
    logger.error("HuggingFace chat failed", ...)
    raise LLMError(f"HuggingFace chat failed: {e}") from e

# After
except (RuntimeError, ValueError, OSError) as e:
    logger.error("HuggingFace chat failed", ...)
    raise LLMError(f"HuggingFace chat failed: {e}") from e
```

### Phase 6: Service Layer

Replace in `reference_store.py`:

```python
# Before
except Exception as e:
    logger.warning("Failed to load embedding metadata", ...)

# After
except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
    logger.warning("Failed to load embedding metadata", ...)
```

### Phase 7: Agent Layer (Decision Required)

**Option A**: Keep broad exception (defensive degradation)
- Pros: Maximum resilience, any Pydantic AI issue falls back gracefully
- Cons: Could hide unexpected bugs

**Option B**: Specific Pydantic AI exceptions
```python
from pydantic import ValidationError
from pydantic_ai import ModelRetry, UnexpectedModelBehavior

except (ValidationError, ModelRetry, UnexpectedModelBehavior) as e:
    logger.error("Pydantic AI call failed; falling back to legacy", ...)
```
- Pros: More precise, won't hide unrelated bugs
- Cons: May not catch all Pydantic AI failure modes

**Recommendation**: Option A (keep broad) for now. Add a note explaining the intent:
```python
# Intentionally broad: gracefully degrade to legacy on ANY Pydantic AI failure
except Exception as e:
    ...
```

### Phase 8: Scripts Layer (Low Impact)

Replace broad exceptions with specific ones as documented above.

---

## Acceptance Criteria

- [ ] Spec inventory matches `rg -n "except Exception" --glob '!docs/**'` (24 current sites)
- [ ] All “best-effort” sites that keep `except Exception` have an explicit comment
- [ ] All non-best-effort sites either narrow exception types or explicitly justify why broad is required
- [ ] No regressions in test suite
- [ ] No new `except Exception` introduced without justification

---

## Files To Modify

This spec is inventory-driven; expected touched files include:

```
server.py
src/ai_psychiatrist/infrastructure/logging.py
src/ai_psychiatrist/infrastructure/llm/huggingface.py
src/ai_psychiatrist/services/reference_store.py
src/ai_psychiatrist/agents/judge.py
src/ai_psychiatrist/agents/quantitative.py
src/ai_psychiatrist/agents/qualitative.py
src/ai_psychiatrist/agents/meta_review.py
scripts/reproduce_results.py
scripts/generate_embeddings.py
scripts/prepare_dataset.py
```

---

## References

- GitHub Issue: #60 (tech-debt: Code readability audit)
- Ruff rule: `BLE001` (blind except) and the broader “catching too much” concern from CodeRabbit
- Python docs: [Exception hierarchy](https://docs.python.org/3/library/exceptions.html#exception-hierarchy)
