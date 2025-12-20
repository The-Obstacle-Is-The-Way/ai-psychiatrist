# Code Audit & Bug Documentation

This document logs issues, anti-patterns, security risks, and technical debt identified during the code audit of the `ai-psychiatrist` repository.

**Date:** 2025-02-24
**Scope:** `src/ai_psychiatrist` and `tests/`

## Executive Summary

The codebase has transitioned from a research script to a more structured Python application, but retains significant "hacky" implementations characteristic of research code.

**Key Issues:**
*   **Security (P0):** Use of `pickle` for loading reference embeddings.
*   **Reliability (P1):** Brittle, regex-based JSON parsing for LLM outputs.
*   **Maintainability (P2):** Hardcoded "magic" values (prompts, keywords, model parameters) scattered throughout the code.
*   **Testing (P2):** Low coverage (~30-50%) and reliance on "over-mocking" in integration tests.

---

## P0: Critical Security Risks

### BUG-001: Unsafe Deserialization via `pickle`
**Location:** `src/ai_psychiatrist/services/reference_store.py`
**Description:** The `ReferenceStore` loads embeddings using `pickle.load()`.
```python
# src/ai_psychiatrist/services/reference_store.py
with self._embeddings_path.open("rb") as f:
    raw_data = pickle.load(f)
```
**Risk:** Arbitrary code execution. If an attacker replaces the embeddings file (even if it's "internal"), they can execute malicious code when the application starts.
**Recommendation:** Replace `pickle` with a safe format like `SafeTensors`, `Parquet`, or standard `JSON` (if size permits).

---

## P1: Major Reliability & Logic Issues

### BUG-002: Brittle JSON Parsing & "Repair" Logic
**Location:** `src/ai_psychiatrist/agents/quantitative.py`
**Description:** The agent attempts to parse LLM output using a chain of brittle methods:
1.  `_strip_json_block`: Regex/string slicing to find ` ```json ` or `<answer>` tags.
2.  `_tolerant_fixups`: Regex replacements to fix common JSON errors (trailing commas, smart quotes).
3.  `_llm_repair`: A recursive call to the LLM to "fix" the JSON if parsing fails.
**Impact:** This is non-deterministic and computationally expensive (extra LLM calls). It is a workaround for not using structured outputs (JSON mode) or constrained decoding which modern LLM providers support.
**Recommendation:** Use Pydantic models with constrained generation (e.g., via `instructor` or Ollama's JSON mode) instead of regex parsing.

### BUG-003: Naive Sentence Splitting
**Location:** `src/ai_psychiatrist/agents/quantitative.py` -> `_keyword_backfill`
**Description:**
```python
parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
```
**Impact:** This naive regex splits on periods, question marks, and exclamation points. It fails on abbreviations (e.g., "Dr. Smith", "e.g.") leading to fragmented sentences and potentially missing context for evidence extraction.
**Recommendation:** Use a proper NLP library (like `spacy` or `nltk` sent_tokenize) for sentence boundary detection.

---

## P2: Design Anti-Patterns & Technical Debt

### BUG-004: Hardcoded Domain Knowledge (Magic Data)
**Location:** `src/ai_psychiatrist/agents/prompts/quantitative.py`
**Description:** `DOMAIN_KEYWORDS` is a massive dictionary hardcoded in a Python file.
```python
DOMAIN_KEYWORDS = {
    "PHQ8_NoInterest": ["can't be bothered", "no interest", ...],
    ...
}
```
**Impact:** Changing these keywords requires a code deploy. This is configuration/data, not code.
**Recommendation:** Move these to a YAML/JSON configuration file or the `DataSettings` class.

### BUG-005: Hardcoded Model Parameters (Magic Numbers)
**Location:**
*   `src/ai_psychiatrist/config.py`: Default values for `temperature`, `top_k`, `chunk_size` are hardcoded defaults in Pydantic fields.
*   `src/ai_psychiatrist/agents/judge.py`: Hardcoded `temperature=0.0`.
*   `src/ai_psychiatrist/services/embedding.py`: Custom cosine similarity normalization `(1.0 + raw_cos) / 2.0` (Magic Formula).
**Impact:** Hides "paper-optimal" hyperparameters deep in the code logic or default args, making experimentation difficult without modifying source.
**Recommendation:** Ensure all hyperparameters are strictly controlled via `config.py` and environment variables, removing local overrides in agent classes unless explicitly passed.

### BUG-006: God Class / Mixed Concerns
**Location:** `src/ai_psychiatrist/agents/quantitative.py`
**Description:** `QuantitativeAssessmentAgent` handles:
*   LLM interaction
*   Embedding generation (via helper)
*   Retrieval logic
*   JSON parsing/repairing
*   Score normalization
**Impact:** High coupling and low cohesion. Hard to unit test specific logic (like the parsing) without mocking the entire LLM.
**Recommendation:** Extract `ResponseParser` into a separate service/utility. Move retrieval logic entirely to `EmbeddingService`.

---

## P3: Testing & Code Quality

### BUG-007: Over-Mocking in Integration Tests
**Location:** `tests/integration/`
**Description:** Integration tests rely heavily on `tests/fixtures/mock_llm.py`. While useful for unit tests, "integration" tests that mock the core integration point (the LLM) provide false confidence.
**Impact:** We are testing our mock, not the system's behavior with a real (or realistically flaky) LLM.
**Recommendation:** Create a true e2e test suite (using `pytest.mark.ollama`) that runs against a local Ollama instance, or record/replay real LLM interactions using tools like VCR.py (though non-determinism makes this hard).

### BUG-008: Low Test Coverage
**Status:** ~30-50% coverage.
**Description:** Key logic in `services/chunking.py`, `services/reference_store.py`, and `domain/entities.py` (validation logic) is under-tested.
**Recommendation:** Target 80% coverage for the `domain` and `services` layers as they are deterministic and easy to test.

### BUG-009: Error Handling Swallow
**Location:** Multiple files (`logging.py`, `quantitative.py`)
**Description:** `try...except` blocks often catch broad `Exception` or specific errors and log them as warnings without re-raising or properly handling the failure state (returning empty objects/defaults).
**Impact:** Failures fail silently, leading to "N/A" scores or empty assessments without clear root cause visibility in the application flow.
