# Spec: Remove Legacy Fallback Architecture

> **Status**: DRAFT
> **Date**: 2025-12-28
> **Goal**: Surgically remove the "legacy" fallback path from all agents, enforcing Pydantic AI as the single source of truth for LLM interactions.

## Context

The current agent architecture employs a "fallback" mechanism: if the Pydantic AI agent fails (validation error, timeout, etc.), it catches the exception and attempts a "legacy" run using `SimpleChatClient` and manual string parsing/repair.

As documented in `PYDANTIC_AI_FALLBACK_ARCHITECTURE.md`, this fallback is:
1.  **Redundant**: It calls the same underlying model.
2.  **Ineffective for Timeouts**: If the model times out via Pydantic AI, it will likely timeout via legacy.
3.  **Maintenance Burden**: It requires maintaining duplicate parsing logic (repair ladders, regex extractors) that drifts from the Pydantic AI extractors.
4.  **Behaviorally Divergent**: Legacy parsing is looser and may yield different results than the strictly validated Pydantic AI output.

## Objective

Remove all legacy fallback logic from `QuantitativeAssessmentAgent`, `QualitativeAssessmentAgent`, `JudgeAgent`, and `MetaReviewAgent`. The system should fail fast (raise exceptions) when Pydantic AI fails, rather than degrading to a deprecated codepath.

## Implementation Plan

### 1. `QuantitativeAssessmentAgent` (`src/ai_psychiatrist/agents/quantitative.py`)

**Action:** Remove fallback block in `_score_items` and delete unused legacy methods.

*   **Modify `_score_items`**:
    *   Remove `if self._scoring_agent is not None: try ... except ...`.
    *   Remove the legacy `self._llm.simple_chat` call.
    *   Unconditionally use `self._scoring_agent`.

*   **Remove Methods**:
    *   `_parse_response` (Used only by legacy scoring)
    *   `_llm_repair` (Used only by `_parse_response`)
    *   `_validate_and_normalize` (Used only by `_parse_response`)

*   **Keep Methods**:
    *   `_strip_json_block` (**CRITICAL**: Used by `_extract_evidence`)
    *   `_tolerant_fixups` (**CRITICAL**: Used by `_extract_evidence`)
    *   `_extract_evidence` (Step 1 of pipeline; not migrated to Pydantic AI yet)
    *   `_determine_na_reason`
    *   `_determine_evidence_source`
    *   `_from_quantitative_output`
    *   `_find_keyword_hits`
    *   `_merge_evidence`

### 2. `QualitativeAssessmentAgent` (`src/ai_psychiatrist/agents/qualitative.py`)

**Action:** Remove fallback block in `assess` and `refine` and delete unused legacy methods.

*   **Modify `assess`**:
    *   Remove `try ... except` fallback block.
    *   Remove `self._llm_client.simple_chat` call.
    *   Remove call to `_parse_response`.

*   **Modify `refine`**:
    *   Same as `assess`.

*   **Remove Methods/Constants**:
    *   `ASSESSMENT_TAGS` (ClassVar)
    *   `_parse_response`
    *   `_extract_quotes`
    *   `_clean_quote_line` (static)
    *   `_extract_inline_quotes` (static)

*   **Keep Methods**:
    *   `_from_qualitative_output`
    *   `_get_llm_params`

### 3. `JudgeAgent` (`src/ai_psychiatrist/agents/judge.py`)

**Action:** Remove fallback block in `_evaluate_metric`.

*   **Modify `_evaluate_metric`**:
    *   Remove `try ... except` fallback.
    *   Remove `self._llm_client.simple_chat` call.
    *   Remove `extract_score_from_text` usage.
    *   **Behavior Change**: Previously returned default score `3` on `LLMError`. Now it will raise. This is intentional (fail fast).

*   **Keep Methods**:
    *   `evaluate` (Orchestrator)
    *   `get_feedback_for_low_scores`

### 4. `MetaReviewAgent` (`src/ai_psychiatrist/agents/meta_review.py`)

**Action:** Remove fallback block in `review`.

*   **Modify `review`**:
    *   Remove `try ... except` fallback.
    *   Remove `self._llm.simple_chat` call.
    *   Remove call to `_parse_response`.

*   **Remove Methods**:
    *   `_parse_response`

*   **Keep Methods**:
    *   `_format_quantitative` (Used for prompt construction)

### 5. `SimpleChatClient` Usage

*   `SimpleChatClient` remains required for `QuantitativeAssessmentAgent._extract_evidence`. Do NOT delete it or its imports.

### 6. Config & Initialization

*   For now, we will enforce `self._agent is not None` in execution methods implicitly by attempting to call it. If it's None, it will raise `AttributeError`, which is acceptable for a misconfigured system (or we can add a check).

## Verification

1.  **Static Analysis**: Ensure no undefined references.
2.  **Unit Tests**: Run `pytest tests/unit/agents` to verify agents function with mocked Pydantic AI backend.
3.  **Manual Check**: Verify that `_extract_evidence` in Quantitative agent still works (since it uses `SimpleChatClient` and the retained helper methods).

## Risks

*   **Strictness**: Pydantic AI validation is stricter. "Good enough" responses that legacy parsing accepted might now fail. This is acceptable for correctness.
*   **Availability**: If Pydantic AI factory fails (e.g. `ImportError` or config issue), the system will break immediately instead of silently falling back. This is desired.
