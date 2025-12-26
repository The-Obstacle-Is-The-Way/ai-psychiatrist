# Hardcoded Technical Debt

> **ARCHIVED**: retained for provenance; all items are resolved as of 2025-12-26.

This file tracks hardcoded values and implementation discrepancies identified during audits.

## Infrastructure

### LLM Clients

1.  **HuggingFaceClient Timeouts** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py`
    *   **Issue:** `simple_chat` has hardcoded `timeout_seconds=180`. `simple_embed` has hardcoded `timeout_seconds=120`.
    *   **Resolution:** Introduced `HuggingFaceSettings` with `default_chat_timeout` and `default_embed_timeout` in `config.py`. Updated client to use these settings.

2.  **OllamaClient Default Models** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/ollama.py`
    *   **Issue:** `simple_chat` defaults to string literal `"gemma3:27b"` and `simple_embed` to `"qwen3-embedding:8b"` when `model` arg is `None`.
    *   **Resolution:** `simple_chat` and `simple_embed` now use `get_model_name()` helper, which reads from `ModelSettings` or falls back to config defaults (no hardcoded strings in client code).

3.  **Client Default Model Discrepancy** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py` vs `ollama.py`
    *   **Issue:** `HuggingFaceClient.simple_chat` defaults to `self._model_settings.qualitative_model`. `OllamaClient` defaults to the string literal `"gemma3:27b"`.
    *   **Resolution:** Both clients now resolve default models from `ModelSettings` (HuggingFace via `self._model_settings.*`; Ollama via `get_model_name()`), ensuring consistent config-driven defaults across all LLM backends.

### API Server

4.  **Assessment Mode Validation** (RESOLVED)
    *   **Location:** `server.py:170-173`
    *   **Issue:** `AssessmentRequest.mode` has hardcoded `le=1` validation.
    *   **Resolution:** Updated `AssessmentRequest` to handle `AssessmentMode` enum directly, with backward compatibility for legacy integer inputs (0/1) via a validator.

5.  **Magic Number Participant ID** (RESOLVED)
    *   **Location:** `server.py:37`
    *   **Issue:** `AD_HOC_PARTICIPANT_ID = 999_999` is defined as a constant.
    *   **Resolution:** Moved to `ServerSettings.ad_hoc_participant_id` in `config.py`.

## Agents

### Pydantic AI Agent Fallbacks

6.  **Hardcoded Model Fallbacks in Agents** (RESOLVED)
    *   **Locations:**
        - `src/ai_psychiatrist/agents/qualitative.py:101` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/quantitative.py:119` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/judge.py:68` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/meta_review.py:88` - `"gemma3:27b"`
    *   **Issue:** When `model_settings` is `None`, these agents fall back to hardcoded string literal `"gemma3:27b"` instead of reading from config.
    *   **Resolution:** Implemented `get_model_name` helper in `config.py`. Agents now use this helper to resolve model names from settings or defaults.

## Services

### Embedding Service

7.  **EmbeddingService Hardcoded Default** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/services/embedding.py:118`
    *   **Issue:** Falls back to `"qwen3-embedding:8b"` when `model_settings` is `None`.
    *   **Resolution:** Updated to use `get_model_name(..., "embedding")`.

## Additional HuggingFace Hardcoded Values

8.  **HuggingFace max_new_tokens** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py:324`
    *   **Issue:** `"max_new_tokens": 1024` is hardcoded in `_generate_text`.
    *   **Resolution:** Added `max_new_tokens` to `HuggingFaceSettings` in `config.py`.

9.  **HuggingFace Quantization group_size** (RESOLVED)
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py:271`
    *   **Issue:** `group_size=128` is hardcoded for int4 quantization.
    *   **Resolution:** Added `quantization_group_size` to `HuggingFaceSettings` in `config.py`.
