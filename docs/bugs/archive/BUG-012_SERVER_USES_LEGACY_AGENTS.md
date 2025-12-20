# BUG-012: Server Uses Legacy Agents (Split Brain)

**Severity**: CRITICAL (P0)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/09_QUANTITATIVE_AGENT.md`, `docs/specs/07_JUDGE_AGENT.md`

---

## Executive Summary

The main entry point `server.py` imports and instantiates **legacy/deprecated agents** from the root `agents/` directory (e.g., `agents.quantitative_assessor_f`) instead of the re-implemented, paper-aligned agents in `src/ai_psychiatrist/agents/`. As a result, none of the recent improvements (robust parsing, embedding service, feedback loop) are actually exposed via the API. The system is effectively running old, unmaintained code.

---

## Evidence (Pre-Fix)

- `server.py` imports:
  ```python
  from agents.quantitative_assessor_f import QuantitativeAssessor as QuantitativeAssessorF
  from agents.qualitative_assessor_f import QualitativeAssessor
  ```
- The new, correct implementation is in `src/ai_psychiatrist/agents/quantitative.py` and `src/ai_psychiatrist/agents/qualitative.py`.
- `server.py` hardcodes logic like `if request.mode == 0: quantitative_assessor = quantitative_assessor_Z` which bypasses the new `AssessmentMode` enum and configuration injection.

---

## Impact

- **Zero Paper Fidelity**: The API runs the old "POC" code, not the rigor-checked implementation.
- **Silent Regression**: Improvements to parsing, error handling, and prompts in `src/` are completely ignored.
- **Configuration Useless**: The `config.py` settings are not used by the legacy agents.

---

## Scope & Disposition

- **Code Path**: Legacy entrypoint (`server.py` + root `agents/`), but **currently active runtime path**.
- **Fix Category**: Production correctness (P0).
- **Recommended Action**: Resolved; server.py now uses modern agents.

---

## Resolution

1. Refactored `server.py` to import from `ai_psychiatrist.agents`.
2. Instantiate `QuantitativeAssessmentAgent` and `QualitativeAssessmentAgent` using the `Settings` / dependency injection pattern with FastAPI lifespan.
3. Removed all imports from the root `agents/` directory.
4. Added new endpoints: `/assess/quantitative`, `/assess/qualitative`, `/full_pipeline`, `/health`.
5. Supports both `participant_id` (loads from DAIC-WOZ) and `transcript_text` (raw text input).

---

## Verification

```bash
ruff check server.py
pytest tests/ -v --no-cov
```

---

## Files Changed

- `server.py`
