# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GitHub Repository

**IMPORTANT**: This is a FORK. Always use the fork for issues and PRs.

- **Fork (use this)**: https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist

When creating GitHub issues or PRs, ALWAYS specify the fork:
```bash
gh issue create --repo The-Obstacle-Is-The-Way/ai-psychiatrist ...
gh pr create --repo The-Obstacle-Is-The-Way/ai-psychiatrist ...
```

## Project Overview

LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews. Implements a research paper's methodology using four specialized agents to analyze transcripts and predict PHQ-8 depression scores.

## Commands

```bash
# Setup
make dev                    # Install all deps + pre-commit hooks
make install                # Production deps only

# Development
make test                   # All tests with coverage (80% minimum)
make test-unit              # Fast unit tests only
make test-parallel          # Parallel test execution
make ci                     # Full CI: format-check → lint → typecheck → test
make serve                  # Dev server on :8000 with reload

# Single test file
uv run pytest tests/unit/agents/test_qualitative.py -v

# Single test
uv run pytest tests/unit/agents/test_qualitative.py::test_assess_parses_response -v

# E2E with real Ollama (opt-in)
AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e

# Code quality
make lint-fix               # Auto-fix linting
make format                 # Format code
make typecheck              # mypy strict mode
```

## Architecture

### Four-Agent Pipeline (Paper Section 2.3)
1. **QualitativeAssessmentAgent**: Narrative analysis → social/biological/risk factors
2. **JudgeAgent**: Evaluates qualitative output (coherence, completeness, specificity, accuracy)
3. **FeedbackLoopService**: Iterates qualitative + judge until score ≥4 (max 10 iterations)
4. **QuantitativeAssessmentAgent**: PHQ-8 scoring (zero-shot or few-shot with embeddings)
5. **MetaReviewAgent**: Integrates all assessments → final severity classification

### Layer Structure
```
src/ai_psychiatrist/
├── agents/           # Four agents + prompts/
├── domain/           # Entities, enums, value_objects, exceptions
├── services/         # EmbeddingService, TranscriptService, FeedbackLoopService
├── infrastructure/   # OllamaClient (LLM), logging
└── config.py         # Pydantic settings (11 config groups)
```

### Key Patterns
- **Protocol-based abstractions**: LLM clients implement `ChatClient`/`EmbeddingClient` protocols
- **Domain entities have UUIDs**: `PHQ8Assessment`, `QualitativeAssessment`, `MetaReview`
- **Value objects are frozen**: `@dataclass(frozen=True)` for `ItemAssessment`, `SimilarityMatch`
- **Structured logging**: Use `get_logger(__name__)` with `logger.info("task", key="value")`

## Configuration

Environment-driven via `.env` (copy from `.env.example`). Key settings:

| Setting | Default | Paper Reference |
|---------|---------|-----------------|
| `MODEL_QUANTITATIVE_MODEL` | `gemma3:27b` | Section 2.2 (MedGemma in Appendix F produces more N/A) |
| `EMBEDDING_DIMENSION` | 4096 | Appendix D (optimal) |
| `EMBEDDING_TOP_K_REFERENCES` | 2 | Appendix D |
| `FEEDBACK_SCORE_THRESHOLD` | 3 | Section 2.3.1 (score <4 triggers refinement) |
| `OLLAMA_TIMEOUT_SECONDS` | 300 | Large transcripts need extended time |

Environment variable prefix: `OLLAMA_HOST=custom` sets `OllamaSettings.host`

## Testing

- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.ollama`
- **Test isolation**: `TESTING=1` env var prevents `.env` loading (set automatically in conftest)
- **Mock clients**: `tests/fixtures/mock_llm.py` provides `MockLLMClient`, `MockEmbeddingClient`
- **Real Ollama**: Set `AI_PSYCHIATRIST_OLLAMA_TESTS=1` to enable opt-in E2E tests

## Important Notes

- **Legacy code is archived**: `_legacy/`, `_literature/`, `_reference/` are not production code
- **Entry point**: `server.py` (root). Run with `make serve` or `uv run uvicorn server:app`
- **All hyperparameters are paper-justified**: Check docstrings for section references
- **Strict typing enforced**: mypy strict mode, full annotations required
- **Pre-commit hooks**: ruff lint/format, mypy, trailing whitespace cleanup
