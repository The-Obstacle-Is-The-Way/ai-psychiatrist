# Code Audit & Bug Documentation

This document logs issues, anti-patterns, security risks, and technical debt identified during the code audit of the `ai-psychiatrist` repository.

**Date:** 2025-02-24
**Scope:** `src/ai_psychiatrist`, `server.py`, and `tests/`
**Auditors:** Jules (Self), Claude (PR #23)

## Executive Summary

The codebase has transitioned from a research script to a more structured Python application but retains significant "hacky" implementations characteristic of research code. While unit tests pass, the system exhibits critical security flaws, brittle logic, and violations of SOLID principles (specifically SRP and DIP).

**Key Issues:**
*   **Security (P0):** Arbitrary Code Execution (RCE) vulnerability via `pickle`.
*   **Reliability (P1):** Brittle regex-based logic for critical tasks (JSON parsing, sentence splitting).
*   **Maintainability (P2):** Hardcoded configuration (Magic Numbers/Strings) and mixed responsibilities (God Classes).
*   **Architecture (P3):** Global mutable state and improper dependency management.
*   **Testing (P2):** Low coverage (~30-50%) and "Testing the Mock" anti-pattern.

---

## P0: Critical Security Risks

### BUG-001: Unsafe Deserialization via `pickle`
**Location:** `src/ai_psychiatrist/services/reference_store.py`
**Description:**
The `ReferenceStore` loads embeddings using `pickle.load()` from a path defined in `DataSettings`.
```python
# src/ai_psychiatrist/services/reference_store.py
with self._embeddings_path.open("rb") as f:
    raw_data = pickle.load(f)
```
**Risk:** **Remote Code Execution (RCE).** While marked as "trusted internal artifact" in comments, the file path is configurable via environment variables (`DATA__EMBEDDINGS_PATH`). In any deployment where this path is modifiable or the storage is shared/untrusted, an attacker can replace the `.pkl` file to execute arbitrary code upon application startup.
**Recommendation:** **Immediate Refactor.** Replace `pickle` with a secure serialization format like `SafeTensors` (for tensors), `Parquet` (for dataframes), or standard `JSON`.

---

## P1: Major Reliability & Logic Issues

### BUG-002: Brittle JSON Parsing & "Repair" Logic
**Location:** `src/ai_psychiatrist/agents/quantitative.py`
**Description:**
The agent attempts to parse LLM output using a chain of heuristic string manipulations:
1.  `_strip_json_block`: Regex/slicing to find code blocks.
2.  `_tolerant_fixups`: Regex replacements to fix syntax errors (e.g., trailing commas).
3.  `_llm_repair`: A recursive LLM call to "fix" the JSON.
**Impact:** **Non-deterministic Failure.** This "defensive coding" creates an unpredictable control flow. It masks underlying prompt engineering failures and adds latency/cost. It violates the Robustness Principle by accepting malformed input that should be guaranteed by the provider (e.g., JSON mode).
**Recommendation:** Implement **Structured Outputs**. Use Pydantic models with constrained generation (e.g., via `instructor` or native provider JSON modes) to guarantee valid schema compliance at the generation step.

### BUG-003: Naive Sentence Splitting
**Location:** `src/ai_psychiatrist/agents/quantitative.py` (`_keyword_backfill`)
**Description:**
```python
parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
```
**Impact:** **Data Corruption.** This regex splits on *any* period followed by whitespace. It incorrectly splits abbreviations like "Dr. Smith" or "e.g. example" into separate sentences (`['Dr.', 'Smith']`), destroying the semantic context needed for evidence extraction.
**Recommendation:** Use a specialized NLP tokenizer (e.g., `spacy`, `nltk.sent_tokenize`) that handles abbreviations and sentence boundaries correctly.

---

## P2: Design Anti-Patterns & SOLID Violations

### BUG-004: Hardcoded Domain Knowledge (Magic Data)
**Location:** `src/ai_psychiatrist/agents/prompts/quantitative.py`
**Description:** `DOMAIN_KEYWORDS` is a large dictionary of clinical terms hardcoded into the source.
**Impact:** **Violation of Open/Closed Principle.** Modifying clinical criteria requires changing source code and redeploying. This is configuration data, not logic.
**Recommendation:** Externalize domain data to a configuration file (YAML/JSON) or database, loaded via `DataSettings`.

### BUG-005: Hardcoded Hyperparameters (Magic Numbers)
**Location:** `src/ai_psychiatrist/config.py`, `agents/judge.py`, `services/embedding.py`
**Description:**
*   Default Pydantic values (`temperature=0.2`) act as hidden defaults.
*   `judge.py` hardcodes `temperature=0.0`.
*   `embedding.py` uses a custom normalization formula: `(1.0 + raw_cos) / 2.0`.
**Impact:** **Rigidity.** Hyperparameters identified in the research paper are buried in code, making it difficult to tune the system without code changes.
**Recommendation:** Centralize all hyperparameters in `config.py`. Remove local overrides in classes unless injected via constructor.

### BUG-006: QuantitativeAgent SRP Violation (Mixed Concerns)
**Location:** `src/ai_psychiatrist/agents/quantitative.py`
**Description:** `QuantitativeAssessmentAgent` is responsible for:
1.  LLM interaction/Prompting.
2.  Response parsing/JSON repair (Infrastructure).
3.  Evidence backfilling (Business Logic).
4.  Score normalization (Domain Logic).
**Impact:** **High Coupling / Low Cohesion.** Testing the scoring logic requires mocking the LLM and the parser.
**Recommendation:** Refactor into specialized components: `ResponseParser` (infra), `EvidenceExtractor` (domain), and `QuantitativeScorer` (domain).

---

## P3: Architecture & Concurrency

### BUG-010: Module-level Global State
**Location:** `server.py`
**Description:**
```python
_ollama_client: OllamaClient | None = None
# ... (6 global variables)
```
**Impact:** **Race Conditions & Test Pollution.** Using `global` variables for application state prevents running multiple instances (e.g., parallel tests) and introduces thread-safety issues during startup/shutdown.
**Recommendation:** Attach state to `app.state` (FastAPI standard) or use a proper Dependency Injection framework.

---

## P4: Packaging & Minor Issues

### BUG-011: Missing Type Marker
**Location:** `src/ai_psychiatrist/`
**Description:** Missing `py.typed` marker file.
**Impact:** **Type Erasure.** Consumers of this package cannot verify types using `mypy`.
**Recommendation:** Add empty `src/ai_psychiatrist/py.typed`.

### BUG-012: Ad-Hoc Magic Numbers
**Location:** `server.py`
**Description:** `participant_id=999_999` used as a sentinel value.
**Impact:** **Confusion.** Magic numbers obscure intent.
**Recommendation:** Define a constant `AD_HOC_PARTICIPANT_ID`.

---

## P2: Testing Strategy

### BUG-007: "Testing the Mock" (Over-mocking)
**Location:** `tests/integration/`
**Description:** Integration tests rely almost exclusively on `MockLLMClient`.
**Impact:** **False Confidence.** The tests verify that the *code works with the mock*, not that the *system works with an LLM*. They fail to catch prompt-related issues or real-world API behaviors.
**Recommendation:** Implement true E2E tests (using `pytest.mark.ollama` or VCR.py) to validate the integration boundary.

### BUG-008: Low Coverage on Deterministic Logic
**Status:** ~30-50% coverage.
**Description:** Critical deterministic logic in `services/chunking.py`, `services/reference_store.py`, and `domain/entities.py` (validation) is under-tested.
**Recommendation:** Prioritize 80%+ coverage for `domain` and `services` layers.

### BUG-009: Error Handling Swallow
**Location:** `server.py`, `infrastructure/logging.py`
**Description:** Broad `try...except Exception` blocks that log and continue or return generic error messages.
**Impact:** **Observability Loss.** Critical failures may be masked, leaving the system in an inconsistent state or returning "N/A" results without clear root cause.
**Recommendation:** Catch specific exceptions. Ensure failures bubble up or result in explicit error states (e.g., `Result.failure()`).
