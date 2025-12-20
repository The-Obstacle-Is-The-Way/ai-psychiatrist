# BUG-008: Model Settings Unused and Wrong Model Selection

**Severity**: HIGH (P1)
**Status**: OPEN
**Date Identified**: 2025-12-19
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

## Recommended Fix

- Thread `ModelSettings` into agent construction and **pass explicit `model=`** to `simple_chat` and `simple_embed` calls.
- Use `Settings.enable_medgemma` to select the quantitative model, or remove the flag entirely.
- Use `Settings.enable_few_shot` to set `AssessmentMode` when constructing the quantitative agent.
- Add integration tests verifying model selection from config.

---

## Files Involved

- `src/ai_psychiatrist/config.py`
- `src/ai_psychiatrist/agents/quantitative.py`
- `src/ai_psychiatrist/infrastructure/llm/ollama.py`
