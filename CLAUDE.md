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

LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews. This is a **robust, independent implementation** that fixes severe methodological flaws in the original research paper. We use four specialized agents to analyze transcripts and predict PHQ-8 depression scores.

**IMPORTANT**: Do NOT use "paper-parity" terminology. The original paper has documented methodological failures (see closed GitHub issues #81, #69, #66, #47, #46, #45). Our implementation diverges intentionally to fix these issues.

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

### Four-Agent Pipeline
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

**CRITICAL**: Copy `.env.example` to `.env` before running evaluations. Code defaults are conservative baselines for testing only.

Environment-driven via `.env`. See `docs/configs/configuration-philosophy.md` for details.

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

## Testing

- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.ollama`
- **Test isolation**: `TESTING=1` env var prevents `.env` loading (set automatically in conftest)
- **Mock clients**: `tests/fixtures/mock_llm.py` provides `MockLLMClient`, `MockEmbeddingClient`
- **Real Ollama**: Set `AI_PSYCHIATRIST_OLLAMA_TESTS=1` to enable opt-in E2E tests

## Few-Shot RAG Pipeline (Specs 33-36)

Our implementation fixes the original methodology's core flaw: participant-level PHQ-8 scores were assigned to arbitrary chunks. See `docs/_archive/misc/HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md`.

### Current Spec Status

| Spec | Description | Status | Config |
|------|-------------|--------|--------|
| 33 | Retrieval guardrails (similarity threshold, char limits) | Enabled | `EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3` |
| 34 | Item-tag filtering | Enabled | `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true` |
| 35 | Chunk-level scoring | Enabled | `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk` |
| 36 | CRAG runtime validation | Disabled | `EMBEDDING_ENABLE_REFERENCE_VALIDATION` |
| 46 | Retrieval similarity confidence signals | Implemented | New confidence variants |

### Running the Full Evaluation

```bash
# In tmux for long runs (~2-3 hours):
tmux new -s run9
uv run python scripts/reproduce_results.py \
  --split paper-test \
  2>&1 | tee data/outputs/run9_$(date +%Y%m%d_%H%M%S).log
```

## Evaluation Metrics (CRITICAL)

**WARNING: The original paper's methodology is not reproducible. Our implementation fixes the flaws.**

### Primary Metrics: AURC and AUGRC (NOT MAE)

**You cannot compare MAE values at different coverage levels.** The paper had grossly different coverages between zero-shot and few-shot modes, making their MAE comparisons invalid.

Use these coverage-aware metrics instead:

| Metric | What It Measures | Lower = Better |
|--------|------------------|----------------|
| **AURC** | Area Under Risk-Coverage curve | ✓ |
| **AUGRC** | Area Under Generalized Risk-Coverage curve (preferred) | ✓ |
| **Cmax** | Maximum coverage (fraction of items with predictions) | Higher = more predictions |

**Why AUGRC over AURC?** AURC puts excessive weight on high-confidence failures. AUGRC provides a holistic assessment of silent failure risk across all predictions. See [Traub et al. 2024](https://arxiv.org/html/2407.01032v1).

### When MAE Is Acceptable

MAE comparisons are ONLY valid when coverage is similar between conditions. If zero-shot and few-shot have ~50% coverage each, MAE becomes a reasonable secondary metric.

### Running Selective Prediction Evaluation

```bash
# Compute AURC/AUGRC with bootstrap CIs
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<your_run>.json \
  --mode few_shot \
  --confidence llm \
  --bootstrap-resamples 1000

# Try new retrieval-based confidence signals (Spec 046)
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<your_run>.json \
  --mode few_shot \
  --confidence hybrid_evidence_similarity \
  --bootstrap-resamples 1000
```

### Key Results (Run 8: 2026-01-02)

| Mode | MAE_item | Coverage | AURC | AUGRC |
|------|----------|----------|------|-------|
| Zero-shot | 0.776 | 48.8% | 0.141 | 0.031 |
| Few-shot | 0.609 | 50.9% | 0.125 | 0.031 |

Coverage is now similar (~50%), so MAE comparisons are valid.

## Important Notes

- **Legacy code is archived**: `_legacy/`, `_literature/`, `_reference/` are NOT production code
- **Entry point**: `server.py` (root). Run with `make serve` or `uv run uvicorn server:app`
- **Strict typing enforced**: mypy strict mode, full annotations required
- **Pre-commit hooks**: ruff lint/format, mypy, trailing whitespace cleanup
- **Do NOT reference "paper-parity"**: Use "baseline defaults" or "validated configuration" instead
