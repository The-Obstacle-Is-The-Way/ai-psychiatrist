# Spec 21: Broad Exception Handling Cleanup

> **STATUS: IMPLEMENTED**
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

When this spec was created, the codebase contained **24** instances of `except Exception` (non-doc code). Some were **intentional** (best-effort helpers, graceful degradation paths), but others masked unexpected errors and made debugging harder.

This spec has now been implemented. The remaining `except Exception` sites are intentional and limited to cases where graceful degradation is desired (e.g., Pydantic AI → legacy fallback) or where a helper must never crash the process.

This spec enumerates **every current** `except Exception` site and defines what we should do for each:
- **Replace** with narrower exception types when the error surface is understood.
- **Keep** broad catches only when explicitly justified (best-effort / “never crash” zones), with an explicit comment.

Note: GitHub Issue #60 originally listed 9 sites; the repo has since grown (FastAPI endpoints + Pydantic AI fallback paths + provenance hashing), so this spec tracks the **current** inventory.

---

## Current State: All Instances

### Inventory Summary (Current)

- **8×** `except Exception` in non-doc code (as of 2025-12-27):
  - `src/ai_psychiatrist/infrastructure/logging.py`: 1
  - `scripts/reproduce_results.py`: 1
  - `scripts/prepare_dataset.py`: 1
  - `src/ai_psychiatrist/agents/judge.py`: 1
  - `src/ai_psychiatrist/agents/meta_review.py`: 1
  - `src/ai_psychiatrist/agents/quantitative.py`: 1
  - `src/ai_psychiatrist/agents/qualitative.py`: 2

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
| `src/ai_psychiatrist/services/reference_store.py` | `_calculate_split_ids_hash()` | `except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError): return None` | ✅ Implemented (no blind except) |
| `src/ai_psychiatrist/services/reference_store.py` | `_derive_artifact_ids_hash()` | `except (TypeError, ValueError, OSError): return None` | ✅ Implemented (no blind except) |
| `scripts/generate_embeddings.py` | `calculate_split_ids_hash()` | `except (KeyError, TypeError, ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError): return "error"` | ✅ Implemented (no blind except) |

---

### Category C: External Backend Wrappers (HuggingFace Client)

These wrap external service calls where many exception types are possible.

| File | Context | Current | Recommended |
|------|---------|---------|-------------|
| `src/ai_psychiatrist/infrastructure/llm/huggingface.py` | `HuggingFaceClient.chat()` | `except (RuntimeError, ValueError, OSError, TypeError)` | ✅ Implemented |
| `src/ai_psychiatrist/infrastructure/llm/huggingface.py` | `HuggingFaceClient.embed()` | `except (RuntimeError, ValueError, OSError, TypeError)` | ✅ Implemented |

**Important constraint**: this module uses lazy imports. Do **not** reference `torch.cuda.OutOfMemoryError` in an `except (...)` tuple unless you restructure to avoid importing torch eagerly (otherwise you defeat optional deps).

---

### Category D: Pydantic AI Fallback (Agent Layer, Intentional Degradation)

These are **intentional fallback patterns** where Pydantic AI is tried first, then legacy logic on failure.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/ai_psychiatrist/agents/judge.py` | `_evaluate_metric()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior)` (optionally include `httpx.HTTPError`) |
| `src/ai_psychiatrist/agents/quantitative.py` | `_score_items()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior)` (optionally include `httpx.HTTPError`) |
| `src/ai_psychiatrist/agents/qualitative.py` | `assess()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior)` (optionally include `httpx.HTTPError`) |
| `src/ai_psychiatrist/agents/qualitative.py` | `refine()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior)` (optionally include `httpx.HTTPError`) |
| `src/ai_psychiatrist/agents/meta_review.py` | `review()` | `except Exception` | Keep broad **with comment**, OR catch `(pydantic.ValidationError, pydantic_ai.ModelRetry, pydantic_ai.UnexpectedModelBehavior)` (optionally include `httpx.HTTPError`) |

**Note**: `asyncio.CancelledError` is a `BaseException` in Python 3.11+ and will not be caught by `except Exception`. The explicit `except asyncio.CancelledError: raise` blocks are therefore redundant but harmless.

**Decision Needed**: Is graceful degradation to legacy parsing acceptable for any exception, or should we only catch Pydantic AI / LLM specific errors?

---

### Category E: Metadata Loading / Validation (Service Layer)

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/ai_psychiatrist/services/reference_store.py` | `_load_embeddings()` | `except (OSError, TypeError, ValueError)` around meta load/validate | ✅ Implemented (with explicit type-check for dict metadata; mismatch errors propagate) |

---

### Category F: Script Error Handling / Long-Run Robustness

Scripts often choose to continue on per-participant failures. That’s acceptable, but we should still avoid swallowing programmer errors.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `scripts/reproduce_results.py` | `evaluate_participant()` | `except Exception` | ✅ Implemented (`logger.exception(...)` preserves stack; continue-on-failure behavior unchanged) |
| `scripts/reproduce_results.py` | `check_ollama_connectivity()` | `except LLMError` | ✅ Implemented |
| `scripts/generate_embeddings.py` | `process_participant()` transcript load | `except (DomainError, ValueError, OSError)` | ✅ Implemented (no blind except) |
| `scripts/generate_embeddings.py` | `process_participant()` embed chunk | `except (DomainError, ValueError, OSError)` | ✅ Implemented (no blind except) |
| `scripts/prepare_dataset.py` | transcript extraction loop | `except (OSError, KeyError, ValueError)` | ✅ Implemented (keeps `zipfile.BadZipFile` explicit) |

---

### Category G: FastAPI Layer (server.py)

The API no longer uses `except Exception` in endpoints/helpers. It catches known domain errors and lets unexpected exceptions propagate (FastAPI returns 500 and logs a stack trace), rather than masking failures.

| File | Symbol | Current | Recommended |
|------|--------|---------|-------------|
| `server.py` | `health_check()` ollama ping | `except (DomainError, OSError)` | ✅ Implemented |
| `server.py` | `assess_quantitative()` | `except DomainError` | ✅ Implemented (unexpected exceptions propagate) |
| `server.py` | `assess_qualitative()` | `except DomainError` | ✅ Implemented (unexpected exceptions propagate) |
| `server.py` | `run_full_pipeline()` | `except DomainError` | ✅ Implemented (unexpected exceptions propagate) |
| `server.py` | `_resolve_transcript()` participant_id path | `except (DomainError, ValueError)` | ✅ Implemented |
| `server.py` | `_resolve_transcript()` ad-hoc text path | `except (DomainError, ValueError)` | ✅ Implemented |

---

## Implementation Plan

All phases below have been completed; this plan is kept for historical reference.

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

- [x] Spec inventory matches `rg -n "except Exception" --glob '!docs/**'` (8 current sites)
- [x] All “best-effort” sites that keep `except Exception` have an explicit comment where applicable
- [x] All non-best-effort sites either narrow exception types or explicitly justify why broad is required
- [x] No regressions in test suite
- [x] No new `except Exception` introduced without justification

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
