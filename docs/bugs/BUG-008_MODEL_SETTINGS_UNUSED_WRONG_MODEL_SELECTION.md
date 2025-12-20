# BUG-008: Model Settings Unused and Wrong Model Selection

**Severity**: HIGH (P1)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/03_CONFIG_LOGGING.md`, `docs/specs/09_QUANTITATIVE_AGENT.md`, `docs/specs/04_LLM_INFRASTRUCTURE.md`

---

## Executive Summary

`ModelSettings` and feature flags (`enable_medgemma`, `enable_few_shot`) are defined in config but **never applied** when agents make LLM calls. As a result, the QuantitativeAssessmentAgent **does not use MedGemma** (paper Appendix F) and instead defaults to `gemma3:27b` via `OllamaClient.simple_chat`. This breaks paper fidelity and makes configuration changes ineffective.

---

## Evidence

- `ModelSettings` defines `qualitative_model`, `judge_model`, `meta_review_model`, `quantitative_model`, `embedding_model`, plus sampling settings (`temperature`, `top_k`, `top_p`), but no code consumes these values. (`src/ai_psychiatrist/config.py:59-105`)
- Quantitative agent never passes a `model=` argument to `simple_chat` for evidence extraction or scoring. (`src/ai_psychiatrist/agents/quantitative.py:133-137`, `src/ai_psychiatrist/agents/quantitative.py:171-173`)
- `OllamaClient.simple_chat` defaults to `gemma3:27b` when `model` is None. (`src/ai_psychiatrist/infrastructure/llm/ollama.py:285-288`)
- `EmbeddingService` only passes `dimension=` to `simple_embed`; there is no path that passes `model=` from config. (`src/ai_psychiatrist/services/embedding.py:104-113`)
- `Settings.enable_medgemma` and `Settings.enable_few_shot` are defined but not used anywhere. (`src/ai_psychiatrist/config.py:281-288`)

---

## Impact

- Quantitative agent runs on the **wrong model** (Gemma 3 27B) instead of MedGemma 27B, invalidating Appendix F performance claims.
- Changing `MODEL__QUANTITATIVE_MODEL` or `enable_medgemma` has **no effect**, causing misleading configuration behavior.
- Embedding model overrides via config are ignored because `simple_embed` defaults to `qwen3-embedding:8b` and no caller passes `model=`.
- Sampling controls from `ModelSettings` (temperature/top_k/top_p) are ignored because agents hardcode their own values.

---

## Scope & Disposition

- **Code Path**: Current implementation (`src/ai_psychiatrist/...`).
- **Fix Category**: Configuration correctness and paper fidelity.
- **Recommended Action**: Fix now; treat `ModelSettings` as SSOT and wire through to all LLM/embedding calls.

---

## Recommended Fix

- Thread `ModelSettings` into agent construction and **pass explicit `model=`** to `simple_chat` and `simple_embed` calls.
- Use `Settings.enable_medgemma` to select the quantitative model, or remove the flag entirely.
- Use `Settings.enable_few_shot` to set `AssessmentMode` when constructing the quantitative agent.
- Add integration tests verifying model selection from config.

---

## Files Involved

- `src/ai_psychiatrist/config.py`
- `src/ai_psychiatrist/agents/quantitative.py`
- `src/ai_psychiatrist/agents/qualitative.py`
- `src/ai_psychiatrist/agents/judge.py`
- `src/ai_psychiatrist/services/embedding.py`
- `src/ai_psychiatrist/infrastructure/llm/ollama.py`
- `server.py`

---

## Resolution

Threaded `ModelSettings` through the entire agent/service layer:

1. **QuantitativeAssessmentAgent**: Added `model_settings` parameter and uses it for all LLM calls
   (evidence extraction, scoring, and repair). Uses `quantitative_model` (MedGemma per Appendix F).

2. **QualitativeAssessmentAgent**: Added `model_settings` parameter and uses it for `assess()` and
   `refine()` calls. Uses `qualitative_model`.

3. **JudgeAgent**: Added `model_settings` parameter and uses it for metric evaluation. Uses
   `judge_model` and `temperature_judge` (0.0 for deterministic evaluation per Spec 07).

4. **EmbeddingService**: Added `model_settings` parameter and uses `embedding_model` for all
   embedding generation.

5. **server.py**: Initializes `ModelSettings` at startup and passes it to all agents/services
   via FastAPI dependency injection.

All configuration values now flow from `ModelSettings` to LLM calls, making configuration
changes effective.

---

## Verification

```bash
ruff check src/ai_psychiatrist/agents/ src/ai_psychiatrist/services/ server.py
pytest tests/ -v --no-cov
# 583 passed, 1 skipped
```
