# Spec: Remove Legacy Fallback Architecture

> **Status**: IMPLEMENTED (BREAKING)
> **Date**: 2025-12-29
> **Goal**: Remove the legacy fallback path (and legacy primary path) for all **structured-output** agent calls, making Pydantic AI the only scoring/evaluation mechanism.

## Context

The current agent architecture employs a "fallback" mechanism: if the Pydantic AI agent fails (validation error, timeout, etc.), it catches the exception and attempts a "legacy" run using `SimpleChatClient` and manual string parsing/repair.

As documented in `PYDANTIC_AI_FALLBACK_ARCHITECTURE.md`, this fallback is:
1.  **Redundant**: It calls the same underlying model.
2.  **Ineffective for Timeouts**: If the model times out via Pydantic AI, it will likely timeout via legacy.
3.  **Maintenance Burden**: It requires maintaining duplicate parsing logic (repair ladders, regex extractors) that drifts from the Pydantic AI extractors.
4.  **Behaviorally Divergent**: Legacy parsing is looser and may yield different results than the strictly validated Pydantic AI output.

## Scope (Explicit)

### In Scope

- Remove **all legacy scoring/evaluation/meta-review paths** from these agents:
  - `QuantitativeAssessmentAgent` (scoring step only; evidence extraction stays legacy for now)
  - `QualitativeAssessmentAgent` (assess + refine)
  - `JudgeAgent` (metric evaluation)
  - `MetaReviewAgent` (review)
- Remove the associated legacy parsing/repair helpers that become unreachable.
- Update configuration text and call sites so Pydantic AI is always configured (no “silent fallback”).
- Rewrite affected tests (unit, integration, e2e) to mock Pydantic AI agents instead of legacy parsing.

### Out of Scope (Non-Goals)

- Rewriting `QuantitativeAssessmentAgent._extract_evidence()` to use Pydantic AI. It still uses `SimpleChatClient` + tolerant JSON parsing and is NOT removed by this spec.
- Changing the research/clinical prompts themselves (beyond removing legacy-only parsing helpers).
- Implementing “smart fallback” by exception type. This spec removes fallback entirely.

## Objective

Remove all legacy scoring/evaluation/meta-review logic from `QuantitativeAssessmentAgent`, `QualitativeAssessmentAgent`, `JudgeAgent`, and `MetaReviewAgent`. The system should fail fast (raise exceptions) when Pydantic AI fails, rather than degrading to a deprecated codepath.

## Preconditions / Required Post-Conditions

These are **hard requirements** to prevent accidental silent legacy execution:

1. **No “fall back to legacy” in `__init__`**
   - Today, each agent logs `"Pydantic AI enabled but no ollama_base_url provided; falling back to legacy"`.
   - After implementation, this must become a hard failure: raise `ValueError` (or a domain `ConfigurationError`) with an actionable message.
2. **No legacy `simple_chat()` calls for structured outputs**
   - After implementation, the following must NOT happen:
     - Quantitative scoring via `self._llm.simple_chat(...)`
     - Qualitative assess/refine via `self._llm_client.simple_chat(...)`
     - Judge evaluation via `self._llm_client.simple_chat(...)`
     - Meta-review via `self._llm.simple_chat(...)`
3. **If the Pydantic AI `Agent.run()` raises, the agent method raises**
   - No `except Exception: ... fall back ...` blocks remain.
   - `asyncio.CancelledError` must continue to propagate unmodified.

## Implementation Plan

### 1. `QuantitativeAssessmentAgent` (`src/ai_psychiatrist/agents/quantitative.py`)

**Action:** Remove fallback block in `_score_items` and delete unused legacy methods.

*   **Modify `__init__`** (lines 111-116):
    *   Replace the warning `"Pydantic AI enabled but no ollama_base_url provided; falling back to legacy"` with a raised configuration error (no legacy).

*   **Modify `_score_items`** (lines 275-313):
    *   Remove `if self._scoring_agent is not None: try ... except ...` (lines 283-303).
    *   Remove the legacy `self._llm.simple_chat` call (lines 304-309).
    *   Remove call to `_parse_response` (line 313).
    *   Replace with:
        1) explicit guard `if self._scoring_agent is None: raise ...`
        2) direct `await self._scoring_agent.run(...)` with no broad fallback.

*   **Remove Methods**:
    *   `_parse_response` (lines 445-483; used only by legacy scoring)
    *   `_llm_repair` (lines 485-523; used only by `_parse_response`)
    *   `_validate_and_normalize` (lines 586-634; used only by `_parse_response`)

*   **Keep Methods** (**CRITICAL** - verify these are NOT removed):
    *   `_strip_json_block` (lines 525-562; used by `_extract_evidence` at line 355)
    *   `_tolerant_fixups` (lines 564-584; used by `_extract_evidence` at line 356)
    *   `_extract_evidence` (lines 329-376; Step 1 of pipeline, uses `SimpleChatClient`)
    *   `_determine_na_reason` (lines 636-649)
    *   `_determine_evidence_source` (lines 651-661)
    *   `_from_quantitative_output` (lines 316-327)
    *   `_find_keyword_hits` (lines 378-408)
    *   `_merge_evidence` (lines 410-443)

### 2. `QualitativeAssessmentAgent` (`src/ai_psychiatrist/agents/qualitative.py`)

**Action:** Remove fallback block in `assess` and `refine` and delete unused legacy methods.

*   **Modify `__init__`** (lines 89-94):
    *   Replace the warning `"Pydantic AI enabled but no ollama_base_url provided; falling back to legacy"` with a raised configuration error (no legacy).

*   **Modify `assess`** (lines 116-182):
    *   Remove `try ... except` fallback block (lines 155-163).
    *   Remove `self._llm_client.simple_chat` call (lines 165-171).
    *   Remove call to `_parse_response` (line 174).

*   **Modify `refine`** (lines 184-256):
    *   Remove `try ... except` fallback block (lines 232-240).
    *   Remove `self._llm_client.simple_chat` call (lines 242-247).
    *   Remove call to `_parse_response` (line 249).

*   **Remove Methods/Constants**:
    *   `ASSESSMENT_TAGS` (ClassVar, lines 59-65)
    *   `_parse_response` (lines 274-303)
    *   `_extract_quotes` (lines 304-336)
    *   `_clean_quote_line` (static, lines 339-346)
    *   `_extract_inline_quotes` (static, lines 349-365)

*   **Remove Imports** (will become unused):
    *   `extract_xml_tags` from `ai_psychiatrist.infrastructure.llm.responses` (line 27)
    *   `re` (line 15) - only used by `_extract_inline_quotes`

*   **Keep Methods**:
    *   `_from_qualitative_output` (lines 258-272)
    *   `_get_llm_params` (lines 106-114)

### 3. `JudgeAgent` (`src/ai_psychiatrist/agents/judge.py`)

**Action:** Remove fallback block in `_evaluate_metric` and clean up unused imports.

*   **Modify `__init__`** (lines 57-61):
    *   Replace the warning `"Pydantic AI enabled but no ollama_base_url provided; falling back to legacy"` with a raised configuration error (no legacy).

*   **Modify `_evaluate_metric`**:
    *   Remove `try ... except` fallback block (lines 150-175).
    *   Remove `self._llm_client.simple_chat` call (lines 177-182).
    *   Remove `extract_score_from_text` usage and score extraction logic (lines 196-211).
    *   Remove `LLMError` catch block that returns default score 3 (lines 183-194).
    *   **Behavior Change**: Previously returned default score `3` on parsing failure or `LLMError`. Now it will raise. This is intentional (fail fast).

*   **Remove Imports** (will become unused):
    *   `extract_score_from_text` from `ai_psychiatrist.infrastructure.llm.responses` (line 18)
    *   `LLMError` from `ai_psychiatrist.domain.exceptions` (line 16)

*   **Keep Methods**:
    *   `evaluate` (Orchestrator)
    *   `get_feedback_for_low_scores`

### 4. `MetaReviewAgent` (`src/ai_psychiatrist/agents/meta_review.py`)

**Action:** Remove fallback block in `review` and clean up unused imports.

*   **Modify `__init__`** (lines 76-80):
    *   Replace the warning `"Pydantic AI enabled but no ollama_base_url provided; falling back to legacy"` with a raised configuration error (no legacy).

*   **Modify `review`** (lines 93-191):
    *   Remove `try ... except` fallback block (lines 158-167).
    *   Remove `self._llm.simple_chat` call (lines 169-174).
    *   Remove call to `_parse_response` (line 176).

*   **Remove Methods**:
    *   `_parse_response` (lines 217-253)

*   **Remove Imports** (will become unused):
    *   `extract_xml_tags` from `ai_psychiatrist.infrastructure.llm.responses` (line 30)

*   **Keep Methods**:
    *   `_format_quantitative` (lines 193-215; used for prompt construction at line 117)

### 5. `SimpleChatClient` Usage

*   `SimpleChatClient` remains required for `QuantitativeAssessmentAgent._extract_evidence`. Do NOT delete it or its imports.

### 6. Config & Initialization

**This spec MUST NOT rely on `AttributeError`.** That’s a footgun and makes failures non-actionable.

Required config/docs updates:

*   `src/ai_psychiatrist/config.py` (`PydanticAISettings.enabled` docstring at lines 347-354):
    *   Remove the claim: “Fallback to legacy parsing occurs automatically on failure.”
    *   Update scope list to include qualitative agents (it’s currently incomplete).
*   Any call site that constructs these agents must pass `ollama_base_url` when Pydantic AI is enabled (production already does; tests must be updated).

### 7. Test Impacts

This is the biggest blast radius. Today, most tests intentionally exercise the **legacy** paths by constructing agents without `ollama_base_url`, which triggers “falling back to legacy”.

After implementing this spec, those tests must be rewritten to **mock Pydantic AI agent outputs** instead of mocking raw LLM strings.

**Affected test files (must be updated):**

*   `tests/unit/agents/test_quantitative.py` (multiple legacy-parsing suites, plus `test_score_items_fallback_on_cancel`)
*   `tests/unit/agents/test_quantitative_backfill.py` (all tests assume legacy scoring response JSON)
*   `tests/unit/agents/test_quantitative_coverage.py` (all parse/repair coverage tests; keep/adapt evidence-extraction coverage)
*   `tests/unit/agents/test_qualitative.py` (all XML parsing + quote extraction tests; keep prompt-template tests)
*   `tests/unit/agents/test_judge.py` (all legacy “Score: X” parsing tests)
*   `tests/unit/agents/test_meta_review.py` (all legacy severity XML parsing + fallback tests)
*   `tests/integration/test_qualitative_pipeline.py` (feedback loop integration currently uses legacy)
*   `tests/integration/test_dual_path_pipeline.py` (dual-path integration currently uses legacy)
*   `tests/e2e/test_agents_real_ollama.py` (currently runs legacy because no `ollama_base_url` is passed)

**Agent constructor call-site inventory (line-accurate as of 2025-12-29):**

*   Quantitative: `tests/unit/agents/test_quantitative.py` (lines 121, 139, 158, 173, 188, 204, 220, 237, 267, 282, 318, 344, 366, 392, 410, 433, 473, 618, 641, 659)
*   Quantitative: `tests/unit/agents/test_quantitative_backfill.py` (lines 34, 65, 91, 117, 151, 178, 204)
*   Quantitative: `tests/unit/agents/test_quantitative_coverage.py` (lines 36, 60, 78, 95, 121, 153, 191)
*   Quantitative: `tests/integration/test_dual_path_pipeline.py` (lines 226, 262, 285, 305, 331, 392, 431, 465, 485, 505, 522)
*   Quantitative: `tests/e2e/test_agents_real_ollama.py` (line 76)

*   Qualitative: `tests/unit/agents/test_qualitative.py` (lines 108, 126, 156, 169, 185, 208, 225, 249, 269, 296, 312, 343, 452)
*   Qualitative: `tests/integration/test_qualitative_pipeline.py` (line 98)
*   Qualitative: `tests/integration/test_dual_path_pipeline.py` (lines 196, 256, 325, 385, 430)
*   Qualitative: `tests/e2e/test_agents_real_ollama.py` (lines 31, 52)

*   Judge: `tests/unit/agents/test_judge.py` (lines 72, 105, 140, 175, 194, 221, 243)
*   Judge: `tests/integration/test_qualitative_pipeline.py` (line 99)
*   Judge: `tests/integration/test_dual_path_pipeline.py` (lines 197, 257, 326)
*   Judge: `tests/e2e/test_agents_real_ollama.py` (line 56)

*   Meta-review: `tests/unit/agents/test_meta_review.py` (lines 86, 120, 142, 156, 178, 201, 221, 242, 270, 298, 323, 337, 366)

**Tests to remove (legacy-only behavior):**

*   `tests/unit/agents/test_judge.py`:
    *   `test_default_score_on_failure` (lines 186-199) — legacy “default to 3” parsing
    *   `test_default_score_on_llm_error` (lines 201-226) — legacy `LLMError` fallback
*   `tests/unit/agents/test_quantitative.py`:
    *   `test_score_items_fallback_on_cancel` (lines 279-288) — specifically tests fallback behavior
    *   Entire `TestQuantitativeAgentParsing` class (starts at line 290) — legacy JSON parsing/repair
*   `tests/unit/agents/test_quantitative_coverage.py`:
    *   `test_parse_response_answer_block_failure` (line 43)
    *   `test_parse_response_llm_repair_failure` (line 68)
    *   `test_parse_response_non_dict_json` (line 86)
    *   `test_validate_and_normalize_float_scores` (line 102)
    *   `test_validate_and_normalize_score_types` (line 134)
    *   `test_strip_json_block_variations` (line 166) — currently exercises legacy scoring parsing, not evidence extraction
*   `tests/unit/agents/test_qualitative.py`:
    *   All tests that assert XML-tag parsing/quote extraction results (most of `TestQualitativeAssessmentAgent`), since those methods are removed.
*   `tests/unit/agents/test_meta_review.py`:
    *   All tests that assert XML severity parsing, clamping, and fallback to quantitative severity (lines 100-229).

*   **`tests/unit/infrastructure/llm/test_responses.py`**:
    *   Tests for `extract_score_from_text` (lines 198-256): **Keep** - this function is still used elsewhere (E2E tests, potentially other code). Do NOT remove.

**E2E note:** `tests/e2e/test_agents_real_ollama.py` will need refactoring:
* If Judge now returns `score` structurally via Pydantic AI (and `explanation` no longer embeds “Score: X”), then the assertion `extract_score_from_text(score.explanation) is not None` is no longer a valid requirement.

## Verification

1.  **Static Analysis**: Ensure no undefined references.
2.  **Unit Tests**: Run `pytest tests/unit/agents` after rewriting mocks to use Pydantic AI outputs (no legacy parsing).
3.  **Manual Check**: Verify that `_extract_evidence` in Quantitative agent still works (since it uses `SimpleChatClient` and the retained helper methods).

## Estimated Diff Size

| Agent | Lines Removed | Lines Modified |
|-------|---------------|----------------|
| `quantitative.py` | ~200 (`_parse_response`, `_llm_repair`, `_validate_and_normalize`, fallback block) | ~20 |
| `qualitative.py` | ~120 (`_parse_response`, `_extract_quotes`, `_clean_quote_line`, `_extract_inline_quotes`, `ASSESSMENT_TAGS`, fallback blocks) | ~30 |
| `judge.py` | ~35 (fallback block, `LLMError` handling, score extraction) | ~15 |
| `meta_review.py` | ~50 (`_parse_response`, fallback block) | ~15 |
| **Tests** | **Large rewrite** (unit + integration + e2e) | **Large rewrite** |
| **Total** | **~400-500 lines removed** | **~600-1200 lines changed** |

Net reduction: **significant code removal**, but expect churn in tests due to migrating assertions from “parse raw LLM text” → “mock structured outputs”.

## Risks

*   **Strictness**: Pydantic AI validation is stricter. "Good enough" responses that legacy parsing accepted might now fail. This is acceptable for correctness.
*   **Availability**: If Pydantic AI factory fails (e.g. `ImportError` or config issue), the system will break immediately instead of silently falling back. This is desired.
