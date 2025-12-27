# Spec 21: Broad Exception Handling Cleanup

> **STATUS: OPEN**
>
> **Priority**: Low — Code works correctly. This is maintainability/readability debt.
>
> **GitHub Issue**: #60 (parent tech-debt audit)
>
> **Created**: 2025-12-26

---

## Problem Statement

The codebase contains 15 instances of `except Exception` which can mask unexpected errors and make debugging harder. Specific exception types provide better error messages and prevent silently catching programming errors.

---

## Current State: All Instances

### Category A: LLM/Embedding Errors (Infrastructure Layer)

These wrap external service calls where many exception types are possible.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/.../llm/huggingface.py` | 123 | Chat generation | `except Exception` | `except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError)` |
| `src/.../llm/huggingface.py` | 148 | Embedding generation | `except Exception` | `except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError)` |

**Rationale**: HuggingFace can raise `RuntimeError` (model loading), `ValueError` (tokenization), or CUDA OOM errors. These should be wrapped in `LLMError` but we should catch specific types.

### Category B: Pydantic AI Fallback (Agent Layer)

These are **intentional fallback patterns** where Pydantic AI is tried first, then legacy logic on failure.

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/.../agents/judge.py` | 164 | Metric evaluation fallback | `except Exception` | Keep as-is OR `except (ValidationError, ModelRetry, UnexpectedModelBehavior)` |
| `src/.../agents/quantitative.py` | 292 | Scoring fallback | `except Exception` | Keep as-is OR `except (ValidationError, ModelRetry, UnexpectedModelBehavior)` |
| `src/.../agents/qualitative.py` | 153 | Assessment fallback | `except Exception` | Keep as-is OR `except (ValidationError, ModelRetry, UnexpectedModelBehavior)` |
| `src/.../agents/qualitative.py` | 224 | Assessment fallback | `except Exception` | Keep as-is OR `except (ValidationError, ModelRetry, UnexpectedModelBehavior)` |
| `src/.../agents/meta_review.py` | 156 | Integration fallback | `except Exception` | Keep as-is OR `except (ValidationError, ModelRetry, UnexpectedModelBehavior)` |

**Note**: All these re-raise `asyncio.CancelledError` correctly. The broad catch is to gracefully degrade to legacy parsing when Pydantic AI fails for any reason. This may be **intentionally broad** as a defensive pattern.

**Decision Needed**: Is graceful degradation to legacy parsing acceptable for any exception, or should we only catch Pydantic AI / LLM specific errors?

### Category C: Metadata Loading (Service Layer)

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/.../services/reference_store.py` | 258 | Metadata JSON loading | `except Exception` | `except (json.JSONDecodeError, OSError, KeyError)` |

### Category D: Script Error Handling

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `scripts/reproduce_results.py` | 291 | Client creation | `except Exception` | `except (httpx.ConnectError, LLMError, ValueError)` |
| `scripts/reproduce_results.py` | 569 | Ollama ping | `except Exception` | `except httpx.HTTPError` |
| `scripts/generate_embeddings.py` | 210 | Transcript loading | `except Exception` | `except (OSError, json.JSONDecodeError, pd.errors.ParserError)` |
| `scripts/generate_embeddings.py` | 228 | Embedding generation | `except Exception` | `except LLMError` |
| `scripts/prepare_dataset.py` | 204 | File extraction | `except Exception` | `except (OSError, zipfile.BadZipFile, tarfile.TarError)` |
| `scripts/prepare_dataset.py` | 254 | Dataset preparation | `except Exception` | `except (OSError, ValueError)` |

### Category E: Logging (Special Case)

| File | Line | Context | Current | Recommended |
|------|------|---------|---------|-------------|
| `src/.../infrastructure/logging.py` | 29 | Log formatting | `except Exception` | **Keep as-is** — logging should never crash |

**Rationale**: The logging module should be maximally defensive. A crash in logging is worse than a missed log message.

---

## Implementation Plan

### Phase 1: Infrastructure Layer (High Impact)

Replace broad exceptions in `src/ai_psychiatrist/infrastructure/llm/huggingface.py`:

```python
# Before
except Exception as e:
    logger.error("HuggingFace chat failed", ...)
    raise LLMError(f"HuggingFace chat failed: {e}") from e

# After
except (RuntimeError, ValueError) as e:
    logger.error("HuggingFace chat failed", ...)
    raise LLMError(f"HuggingFace chat failed: {e}") from e
except torch.cuda.OutOfMemoryError as e:
    logger.error("HuggingFace OOM", ...)
    raise LLMError(f"CUDA out of memory: {e}") from e
```

### Phase 2: Service Layer

Replace in `reference_store.py`:

```python
# Before
except Exception as e:
    logger.warning("Failed to load embedding metadata", ...)

# After
except (json.JSONDecodeError, OSError) as e:
    logger.warning("Failed to load embedding metadata", ...)
except KeyError as e:
    logger.warning("Incomplete embedding metadata", ...)
```

### Phase 3: Agent Layer (Decision Required)

**Option A**: Keep broad exception (defensive degradation)
- Pros: Maximum resilience, any Pydantic AI issue falls back gracefully
- Cons: Could hide unexpected bugs

**Option B**: Specific Pydantic AI exceptions
```python
from pydantic_ai import ValidationError, ModelRetry, UnexpectedModelBehavior

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

### Phase 4: Scripts Layer (Low Impact)

Replace broad exceptions with specific ones as documented above.

---

## Acceptance Criteria

- [ ] All Category C (service layer) instances use specific exceptions
- [ ] All Category D (scripts) instances use specific exceptions
- [ ] Category A (infrastructure) uses specific exceptions with proper error messages
- [ ] Category B (agents) has explicit comment if keeping broad exception
- [ ] Category E (logging) unchanged
- [ ] No regressions in test suite
- [ ] No new `except Exception` introduced without justification

---

## Files To Modify

```
src/ai_psychiatrist/infrastructure/llm/huggingface.py   (2 instances)
src/ai_psychiatrist/services/reference_store.py          (1 instance)
src/ai_psychiatrist/agents/judge.py                      (1 instance - review)
src/ai_psychiatrist/agents/quantitative.py               (1 instance - review)
src/ai_psychiatrist/agents/qualitative.py                (2 instances - review)
src/ai_psychiatrist/agents/meta_review.py                (1 instance - review)
scripts/reproduce_results.py                             (2 instances)
scripts/generate_embeddings.py                           (2 instances)
scripts/prepare_dataset.py                               (2 instances)
```

---

## References

- GitHub Issue: #60 (tech-debt: Code readability audit)
- Ruff rule: `BLE001` (blind except)
- Python docs: [Exception hierarchy](https://docs.python.org/3/library/exceptions.html#exception-hierarchy)
