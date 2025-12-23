# GEMINI.md

This file provides context and guidance for Gemini when working with the `ai-psychiatrist` repository.

## Project Overview

**Name:** `ai-psychiatrist`
**Version:** 2.0.0
**Purpose:** An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews. It is a production-grade, clean-room implementation of the methodology described in Greene et al. "AI Psychiatrist Assistant".
**Core Function:** Analyzes clinical interview transcripts to predict PHQ-8 depression scores and severity levels using a four-agent pipeline with iterative self-refinement.

## Technology Stack

*   **Language:** Python 3.11+
*   **Package Manager:** `uv`
*   **LLM Backend:** Ollama (Local) or HuggingFace (Optional)
*   **Web Framework:** FastAPI
*   **Configuration/Validation:** Pydantic v2
*   **Logging:** structlog
*   **Testing:** pytest
*   **Linting/Formatting:** ruff
*   **Type Checking:** mypy (strict)

## Operational Commands

Use `make` for most operations. The project relies on `uv` for dependency management.

### Setup
*   `make install`: Install production dependencies.
*   `make dev`: Install all dependencies (dev + docs) and pre-commit hooks.
*   `make dev-hf`: Install dev dependencies + HuggingFace extras.

### Development & Server
*   `make serve`: Start the development server (FastAPI) on port 8000 with hot reload.
*   `make repl`: Start a Python REPL with the project loaded.

### Testing
*   `make test`: Run all tests with coverage (target: 80%+).
*   `make test-unit`: Run fast unit tests only.
*   `make test-integration`: Run integration tests.
*   `make test-e2e`: Run end-to-end tests.
*   `make test-parallel`: Run tests in parallel using `pytest-xdist`.
*   **Real Ollama Tests:** `AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e` (Opt-in).

### Quality Assurance
*   `make ci`: Run full CI pipeline (format-check -> lint -> typecheck -> test).
*   `make lint-fix`: Auto-fix linting issues with `ruff`.
*   `make format`: Format code with `ruff`.
*   `make typecheck`: Run strict type checking with `mypy`.

## Architecture & Code Structure

### Source Directory: `src/ai_psychiatrist/`
*   `agents/`: Implementation of the four assessment agents (Qualitative, Judge, Quantitative, Meta-Review) and their prompts.
*   `domain/`: Core business logic, entities (UUIDs), value objects (frozen dataclasses), enums, and exceptions.
*   `services/`: Application services like `FeedbackLoopService`, `EmbeddingService`, and `TranscriptService`.
*   `infrastructure/`: External interface implementations (e.g., `OllamaClient`, logging).
*   `config.py`: Pydantic settings management.

### The Four-Agent Pipeline
1.  **QualitativeAssessmentAgent:** narrative analysis of social/biological/risk factors.
2.  **JudgeAgent:** Evaluates qualitative output for coherence and accuracy.
3.  **FeedbackLoopService:** Orchestrates the loop between Qualitative and Judge agents (up to 10 iterations) until a score >= 4 is achieved.
4.  **QuantitativeAssessmentAgent:** Predicts PHQ-8 scores (supports zero-shot and few-shot with embeddings).
5.  **MetaReviewAgent:** Synthesizes all outputs into a final severity classification.

## Development Guidelines

*   **Legacy Code:** **DO NOT** modify or depend on code in `_legacy/`, `_literature/`, or `_reference/`. These are archived.
*   **Strict Typing:** Mypy is configured for strict mode. All code must be fully annotated.
*   **Testing Strategy:**
    *   Use `tests/fixtures/mock_llm.py` for unit/integration tests to mock LLM calls.
    *   Use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`.
    *   Respect the `TESTING=1` environment variable.
*   **Logging:** Use the project's structured logging wrapper. Instantiate with `from ai_psychiatrist.infrastructure.logging import get_logger` then `logger = get_logger(__name__)`.

## Configuration

Configuration is managed via environment variables and `.env`.

*   **Models:** Defaults are `gemma3:27b` (Qualitative/Quantitative) and `qwen3-embedding:8b` (Embeddings).
*   **Key Parameters:**
    *   `EMBEDDING_DIMENSION`: 4096 (Paper Appendix D).
    *   `FEEDBACK_SCORE_THRESHOLD`: 3 (Paper Section 2.3.1).
    *   `OLLAMA_TIMEOUT_SECONDS`: 300 (for large transcripts).

## Key Files
*   `server.py`: Entry point for the FastAPI application.
*   `CLAUDE.md`: Contains detailed instructions for AI coding assistants (highly relevant reference).
*   `docs/specs/`: Implementation specifications.
