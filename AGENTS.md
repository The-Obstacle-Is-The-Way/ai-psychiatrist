# Repository Guidelines

## Project Structure

- `src/ai_psychiatrist/`: Production package (agents, services, domain, infrastructure, config).
- `server.py`: FastAPI app + orchestration endpoints (`/health`, `/assess/*`, `/full_pipeline`).
- `scripts/`: Utilities (e.g., dataset prep, embeddings generation, reproduction helpers).
- `tests/`: `unit/`, `integration/`, and `e2e/` tests (e2e is opt-in).
- `docs/`: MkDocs documentation; `docs/archive/` contains historical artifacts.
- `data/`: Local-only DAIC-WOZ artifacts (gitignored due to licensing).
- `_legacy/`: Archived original research code/scripts (reference only; not part of the production package).

## Build, Test, and Development Commands

Use `uv` and the Makefile targets:

- `make dev`: Install dev + docs deps and pre-commit hooks.
- `make ci`: Run formatting check, lint, mypy, and full test suite.
- `make test`: Run all tests with coverage.
- `make serve`: Run the API locally via Uvicorn.
- `make docs` / `make docs-serve`: Build/serve MkDocs.

E2E tests require a running Ollama instance:
- `AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e`

## Coding Style & Naming

- Format + lint: `ruff format` and `ruff check` (don’t hand-format).
- Typing: `mypy` is strict across `src/`, `tests/`, and `scripts/`.
- Prefer explicit names and small, single-purpose functions/classes.
- Public interfaces live in `src/ai_psychiatrist/...`; avoid shipping test doubles in `src/`.

## Testing Guidelines

- Framework: `pytest` with markers (`unit`, `integration`, `e2e`).
- Add tests for behavior (not “smoke-only” calls). Prefer deterministic unit tests with mocks at I/O boundaries.
- Keep coverage ≥ 80% (CI enforces).

## Commit & Pull Requests

- Use conventional commits (e.g., `feat(spec-09): …`, `fix(embedding): …`, `docs: …`).
- PRs should include: clear summary, linked bug/spec docs when relevant, and `make ci` passing.

## Security & Configuration

- Do not commit DAIC-WOZ data or secrets; configure via `.env` (see `.env.example`).
- Prefer running commands via `uv run …` to ensure the repo's venv is used.

## Few-Shot RAG Pipeline (Specs 33-36)

The few-shot methodology has a known flaw: participant-level PHQ-8 scores are assigned to arbitrary chunks regardless of content. See `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md`.

### Spec Status

| Spec | Description | Status |
|------|-------------|--------|
| 33 | Retrieval guardrails | Enabled (`EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3`) |
| 34 | Item-tag filtering | Enabled (`EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`) |
| 35 | Chunk-level scoring | Ready (needs preprocessing via `score_reference_chunks.py`) |
| 36 | CRAG runtime validation | Disabled by default (runtime cost) |

### Production Run: Spec 35

```bash
# Step 1: Generate chunk scores (one-time)
python scripts/score_reference_chunks.py \
  --embeddings-file ollama_qwen3_8b_paper_train \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat \
  --allow-same-model

# Step 2: Enable in .env
# EMBEDDING_REFERENCE_SCORE_SOURCE=chunk

# Step 3: Run evaluation
python scripts/reproduce_results.py --mode both --split paper-test
```

**Note**: `--allow-same-model` is acceptable per 2025 research (SELF-ICL, ACL findings). Ablate with MedGemma if desired.
