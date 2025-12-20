# BUG-016: MetaReviewAgent Not Implemented

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/10_META_REVIEW_AGENT.md`, `docs/specs/11_FULL_PIPELINE.md`

---

## Executive Summary

Spec 10 defines a `MetaReviewAgent` that integrates qualitative and quantitative assessments to predict overall depression severity. The spec contains a complete implementation, but the file was never created. The domain model (`MetaReview`, `FullAssessment`) expects this agent, and Spec 11's full pipeline requires it, but the agent does not exist.

This is the same pattern as BUG-012: spec exists, domain models expect it, but the agent was never implemented.

---

## Evidence

- **Spec 10 Deliverables** state:
  ```
  1. `src/ai_psychiatrist/agents/meta_review.py` - Meta-review agent
  2. `tests/unit/agents/test_meta_review.py` - Tests
  ```
- **File does not exist**:
  ```bash
  ls src/ai_psychiatrist/agents/meta_review.py
  # -> No such file or directory
  ```
- **Domain model expects it**: `FullAssessment` in `domain/entities.py:383` requires a `MetaReview`
- **Agents __init__.py references it** in docstring (Section 2.3.4) but doesn't export it
- **Spec 11 pipeline expects it**: `dependencies.py` snippet shows `get_meta_review_agent()`
- **server.py /full_pipeline** only runs quantitative + qualitative, skips meta-review entirely

---

## Impact

- **Incomplete Pipeline**: The full assessment pipeline per the paper is not achievable
- **Paper Fidelity**: Section 2.3.3 and Section 3.3 (78% accuracy meta-review) cannot be reproduced
- **Domain Model Orphaned**: `MetaReview` entity exists but has no producer
- **API Incomplete**: `/full_pipeline` endpoint returns incomplete results

---

## Scope & Disposition

- **Code Path**: Modern codebase (`src/ai_psychiatrist/`)
- **Fix Category**: Feature implementation (missing deliverable from Spec 10)
- **Priority**: P1 - Required for paper replication

---

## Recommended Fix

1. Create `src/ai_psychiatrist/agents/meta_review.py` using the implementation in Spec 10
2. Create `src/ai_psychiatrist/agents/prompts/meta_review.py` for prompt templates
3. Create `tests/unit/agents/test_meta_review.py`
4. Export `MetaReviewAgent` from `src/ai_psychiatrist/agents/__init__.py`
5. Wire `MetaReviewAgent` into `server.py` `/full_pipeline` endpoint

---

## Spec 10 Contains Full Implementation

The spec file (`docs/specs/10_META_REVIEW_AGENT.md`) contains:
- `META_REVIEW_SYSTEM_PROMPT`
- `make_meta_review_prompt()` function
- `MetaReviewAgent` class with `review()` method
- `_format_quantitative()` helper
- `_parse_response()` with fallback logic

This should be extracted and placed in the proper file location.

---

## Files Involved

- `src/ai_psychiatrist/agents/meta_review.py` (TO CREATE)
- `src/ai_psychiatrist/agents/prompts/meta_review.py` (TO CREATE)
- `tests/unit/agents/test_meta_review.py` (TO CREATE)
- `src/ai_psychiatrist/agents/__init__.py` (UPDATE - add export)
- `server.py` (UPDATE - wire in meta-review)

---

## Related Bugs

- **BUG-012**: Same pattern (spec existed, agent not wired) - RESOLVED
- **BUG-017**: FeedbackLoop not wired in server.py (to be documented)
