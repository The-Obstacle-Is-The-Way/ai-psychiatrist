# BUG-021: `uv sync --dev` vs `uv sync --all-extras` Confusion

**Status**: RESOLVED (documentation issue)
**Severity**: LOW (user education)
**Found**: 2025-12-21
**Found by**: Claude Code during PHQ-8 YAML clinical validation

## Summary

The command `uv sync --dev` does NOT install optional dev dependencies defined
in `pyproject.toml` under `[project.optional-dependencies].dev`. The correct
command is `uv sync --all-extras` (which the Makefile already uses correctly).

This caused test collection to fail with `ModuleNotFoundError: No module named 'respx'`
when running `uv sync --dev` instead of `make dev`.

## Root Cause

`uv sync --dev` refers to uv's internal "dev dependencies" concept (for workspace
development mode), NOT the project's `[project.optional-dependencies].dev` extras.

### Correct Commands

```bash
# Install all extras (dev + docs) - CORRECT:
uv sync --all-extras

# Install only dev extras - CORRECT:
uv sync --extra dev

# Using Makefile - CORRECT (uses --all-extras):
make dev
```

### Incorrect Command

```bash
# This does NOT install [project.optional-dependencies].dev:
uv sync --dev  # WRONG - this is for uv workspace dev mode, not extras
```

## Resolution

The Makefile already uses the correct command:

```makefile
dev: ## Install all dependencies (including dev)
	uv sync --all-extras
	uv run pre-commit install
```

**Always use `make dev`** instead of running `uv sync` directly.

## Verification

```bash
make dev
uv run python -c "import respx; print('OK')"
uv run pytest tests/unit/infrastructure/llm/test_ollama.py -v
```

All commands now pass.

## References

- [uv optional dependencies](https://docs.astral.sh/uv/concepts/dependencies/#optional-dependencies)
- Makefile `dev` target (line 11-13)
