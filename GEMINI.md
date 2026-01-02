# GEMINI.md

This file provides context and guidance for Gemini when working with the `ai-psychiatrist` repository.

## Critical Context

**IMPORTANT**: This is a **robust, independent implementation** that fixes severe methodological flaws in the original research paper (Greene et al.). Do NOT use "paper-parity" terminology.

The original paper has documented failures:
- **#81**: Participant-level PHQ-8 scores assigned to individual chunks (semantic mismatch)
- **#69**: Few-shot retrieval attaches participant scores to arbitrary text chunks
- **#66**: Paper uses invalid statistical comparison (MAE at different coverages)
- **#47, #46**: Paper does not specify quantization or sampling parameters
- **#45**: Paper uses undocumented custom split

Our implementation fixes these issues. Use "baseline defaults" or "validated configuration" instead of "paper-parity".

## Project Overview

**Name:** `ai-psychiatrist`
**Version:** 0.1.0
**Purpose:** An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews. This is a production-grade implementation that fixes the methodology described in Greene et al.
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

## Configuration

**CRITICAL**: Copy `.env.example` to `.env` before running evaluations. Code defaults are conservative baselines for testing only.

### Recommended Settings (from .env.example)

| Setting | Value | Purpose |
|---------|-------|---------|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `chunk` | Use chunk-level scores (fixes core flaw) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `true` | Filter refs by PHQ-8 item domain |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.3` | Drop low-quality refs |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `500` | Limit ref context per item |
| `DATA_TRANSCRIPTS_DIR` | `data/transcripts_participant_only` | Participant-only preprocessing |
| `EMBEDDING_BACKEND` | `huggingface` | FP16 embeddings (better quality) |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | Large transcripts need extended time |

## Development Guidelines

*   **Legacy Code:** **DO NOT** modify or depend on code in `_legacy/`, `_literature/`, or `_reference/`. These are archived.
*   **Strict Typing:** Mypy is configured for strict mode. All code must be fully annotated.
*   **Testing Strategy:**
    *   Use `tests/fixtures/mock_llm.py` for unit/integration tests to mock LLM calls.
    *   Use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`.
    *   Respect the `TESTING=1` environment variable.
*   **Logging:** Use the project's structured logging wrapper. Instantiate with `from ai_psychiatrist.infrastructure.logging import get_logger` then `logger = get_logger(__name__)`.

## Key Files
*   `server.py`: Entry point for the FastAPI application.
*   `CLAUDE.md`: Contains detailed instructions for AI coding assistants (highly relevant reference).
*   `docs/_specs/`: Implementation specifications.

## Few-Shot RAG Pipeline (Specs 33-46)

Our implementation fixes the original methodology's core flaw: participant-level PHQ-8 scores were assigned to arbitrary chunks.

### Spec Status

| Spec | Description | Status |
|------|-------------|--------|
| 33 | Retrieval guardrails | Enabled |
| 34 | Item-tag filtering | Enabled |
| 35 | Chunk-level scoring | Enabled |
| 36 | CRAG runtime validation | Disabled by default |
| 46 | Retrieval similarity confidence signals | Implemented |

### Running the Full Evaluation

```bash
# In tmux for long runs (~2-3 hours):
tmux new -s run9
uv run python scripts/reproduce_results.py --split paper-test \
  2>&1 | tee data/outputs/run9_$(date +%Y%m%d_%H%M%S).log
```

## Evaluation Metrics (CRITICAL)

**WARNING: The original paper's methodology is not reproducible. Our implementation fixes the flaws.**

### Primary Metrics: AURC and AUGRC (NOT MAE)

**You cannot compare MAE values at different coverage levels.** Use coverage-aware metrics:

| Metric | What It Measures | Lower = Better |
|--------|------------------|----------------|
| **AURC** | Area Under Risk-Coverage curve | ✓ |
| **AUGRC** | Area Under Generalized Risk-Coverage curve (preferred) | ✓ |
| **Cmax** | Maximum coverage | Higher = more predictions |

**Why AUGRC over AURC?** AURC puts excessive weight on high-confidence failures. AUGRC provides holistic silent failure risk assessment. See [Traub et al. 2024](https://arxiv.org/html/2407.01032v1).

### When MAE Is Acceptable

MAE comparisons are ONLY valid when coverage is similar between conditions.

```bash
# Compute AURC/AUGRC with new confidence signals
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<your_run>.json \
  --mode few_shot \
  --confidence hybrid_evidence_similarity \
  --bootstrap-resamples 1000
```
