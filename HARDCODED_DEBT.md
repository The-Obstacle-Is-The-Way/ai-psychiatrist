# Hardcoded Technical Debt

This file tracks hardcoded values and implementation discrepancies identified during audits.

## Infrastructure

### LLM Clients

1.  **HuggingFaceClient Timeouts**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py`
    *   **Issue:** `simple_chat` has hardcoded `timeout_seconds=180`. `simple_embed` has hardcoded `timeout_seconds=120`.
    *   **Impact:** These helper methods are not configurable via `Settings`; callers cannot tune timeouts without using the lower-level `chat(...)` / `embed(...)` APIs.

2.  **OllamaClient Default Models**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/ollama.py`
    *   **Issue:** `simple_chat` defaults to string literal `"gemma3:27b"` and `simple_embed` to `"qwen3-embedding:8b"` when `model` arg is `None`.
    *   **Impact:** Bypasses `ModelSettings`. If `config.py` defaults change, these hardcoded fallbacks will drift.

3.  **Client Default Model Discrepancy**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py` vs `ollama.py`
    *   **Issue:** `HuggingFaceClient.simple_chat` defaults to `self._model_settings.qualitative_model`. `OllamaClient` defaults to the string literal `"gemma3:27b"`.
    *   **Impact:** Inconsistent behavior depending on backend choice.

### API Server

4.  **Assessment Mode Validation**
    *   **Location:** `server.py:170-173`
    *   **Issue:** `AssessmentRequest.mode` has hardcoded `le=1` validation.
    *   **Impact:** If `AssessmentMode` enum expands (e.g., to add `FINE_TUNED = 2`), the API will reject valid requests until this hardcoded constraint is updated.

5.  **Magic Number Participant ID**
    *   **Location:** `server.py:37`
    *   **Issue:** `AD_HOC_PARTICIPANT_ID = 999_999` is defined as a constant.
    *   **Impact:** Potential collision if the dataset expands or numbering scheme changes. Should be configurable or handled via a separate Transcript subclass.

## Agents

### Pydantic AI Agent Fallbacks

6.  **Hardcoded Model Fallbacks in Agents**
    *   **Locations:**
        - `src/ai_psychiatrist/agents/qualitative.py:101` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/quantitative.py:119` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/judge.py:68` - `"gemma3:27b"`
        - `src/ai_psychiatrist/agents/meta_review.py:88` - `"gemma3:27b"`
    *   **Issue:** When `model_settings` is `None`, these agents fall back to hardcoded string literal `"gemma3:27b"` instead of reading from config.
    *   **Impact:** Bypasses `ModelSettings`. Same pattern as OllamaClient defaults.

## Services

### Embedding Service

7.  **EmbeddingService Hardcoded Default**
    *   **Location:** `src/ai_psychiatrist/services/embedding.py:118`
    *   **Issue:** Falls back to `"qwen3-embedding:8b"` when `model_settings` is `None`.
    *   **Impact:** Same pattern as OllamaClient/agent defaults - bypasses config.

## Additional HuggingFace Hardcoded Values

8.  **HuggingFace max_new_tokens**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py:324`
    *   **Issue:** `"max_new_tokens": 1024` is hardcoded in `_generate_text`.
    *   **Impact:** Cannot be configured. May truncate long responses or waste compute on short ones.

9.  **HuggingFace Quantization group_size**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py:271`
    *   **Issue:** `group_size=128` is hardcoded for int4 quantization.
    *   **Impact:** Cannot be tuned for different model architectures or memory constraints.
