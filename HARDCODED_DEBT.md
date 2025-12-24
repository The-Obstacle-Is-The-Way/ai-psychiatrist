# Hardcoded Technical Debt

This file tracks hardcoded values and implementation discrepancies identified during audits.

## Infrastructure

### LLM Clients

1.  **HuggingFaceClient Timeouts**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py`
    *   **Issue:** `simple_chat` has hardcoded `timeout_seconds=180`. `simple_embed` has hardcoded `timeout_seconds=120`.
    *   **Impact:** Ignores any timeout configuration. Long generations on slower GPUs will timeout regardless of settings.

2.  **OllamaClient Default Models**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/ollama.py`
    *   **Issue:** `simple_chat` defaults to string literal `"gemma3:27b"` and `simple_embed` to `"qwen3-embedding:8b"` when `model` arg is `None`.
    *   **Impact:** Bypasses `ModelSettings`. If `config.py` defaults change, these hardcoded fallbacks will drift.

3.  **Client Default Model Discrepancy**
    *   **Location:** `src/ai_psychiatrist/infrastructure/llm/huggingface.py` vs `ollama.py`
    *   **Issue:** `HuggingFaceClient.simple_chat` defaults to `self._model_settings.qualitative_model`. `OllamaClient` defaults to the string literal `"gemma3:27b"`.
    *   **Impact:** Inconsistent behavior depending on backend choice.

### Logging

4.  **ANSI Colors**
    *   **Location:** `src/ai_psychiatrist/infrastructure/logging.py`
    *   **Issue:** `colors=True` is hardcoded in `ConsoleRenderer`.
    *   **Impact:** Log files contain ANSI escape codes, making them hard to grep/read (See BUG-025).

### API Server

5.  **Assessment Mode Validation**
    *   **Location:** `server.py`
    *   **Issue:** `AssessmentRequest.mode` has hardcoded `le=1` validation.
    *   **Impact:** If `AssessmentMode` enum expands (e.g., to add `FINE_TUNED = 2`), the API will reject valid requests until this hardcoded constraint is updated.

6.  **Magic Number Participant ID**
    *   **Location:** `server.py`
    *   **Issue:** `AD_HOC_PARTICIPANT_ID = 999_999` is defined as a constant.
    *   **Impact:** Potential collision if the dataset expands or numbering scheme changes. Should be configurable or handled via a separate Transcript subclass.
