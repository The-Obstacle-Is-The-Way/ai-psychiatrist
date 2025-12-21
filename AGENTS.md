# Repository Guidelines

## Project Structure & Module Organization

- `src/ai_psychiatrist/`: production package
  - `domain/`: entities, enums, value objects, exceptions (no infrastructure imports)
  - `infrastructure/`: Ollama client + logging adapters
  - `agents/`: qualitative/judge/quantitative/meta-review agents
  - `services/`: transcript loading, embeddings/reference store, feedback loop, etc.
- `server.py`: FastAPI entrypoint (`uvicorn server:app`)
- `tests/`: pytest suite (unit/integration/e2e)
- `scripts/`: local utilities (e.g., dataset prep, embedding generation)
- `docs/`: specs, bug docs, and reference material
- `data/`: local-only artifacts (DAIC-WOZ, generated embeddings); not committed
- `_legacy/`: archived prototype/research code (excluded from lint/typecheck)

## Build, Test, and Development Commands

Use the venv via `uv run` (the `Makefile` already does this):

- `make dev`: install dev deps + `pre-commit install`
- `make ci`: format-check, lint, mypy (strict), and tests w/ coverage gate
- `make test` / `make test-unit` / `make test-integration` / `make test-e2e`
- `make serve`: run the API locally (`uvicorn server:app --reload`)
- `make format`, `make lint`, `make typecheck`

## Coding Style & Naming Conventions

- Python `>=3.11`, 4-space indentation, max line length 100
- Format/lint with Ruff: `ruff format` and `ruff check`
- Prefer `pathlib.Path` over `os.path`
- Naming: `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Keep domain logic in `src/ai_psychiatrist/domain/` and depend on protocols, not concrete clients

## Testing Guidelines

- Framework: `pytest` (+ `pytest-asyncio`, `respx`)
- Markers: `unit`, `integration`, `e2e`, `ollama`, `slow` (see `pyproject.toml`)
- Coverage: enforced at `--cov-fail-under=80`
- Live Ollama tests are opt-in: `uv run pytest -m ollama`

## Commit & Pull Request Guidelines

- Commit prefixes used here: `feat(spec-XX): ...`, `fix(chunk-N): ...`, `fix(types): ...`
- PRs should link the relevant spec/bug doc, summarize behavior changes, and include verification (`make ci`)
- Avoid modifying `_legacy/` unless the PR is explicitly for cleanup/archival
