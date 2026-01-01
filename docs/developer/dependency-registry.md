# Dependency Registry

> **Last Updated:** 2025-12-26
> **Lock File:** `uv.lock` (232 packages resolved)

This document tracks all direct dependencies, their version constraints, and current locked versions for reproducibility and audit purposes.

## Quick Status

| Category | Count | Status |
|----------|-------|--------|
| Core | 12 | All latest stable |
| Dev | 10 | All latest stable |
| Docs | 3 | All latest stable |
| HF (optional) | 3 | All latest stable |

---

## Core Dependencies

Production dependencies required for the main application.

| Package | Constraint | Locked | PyPI Latest | Purpose |
|---------|------------|--------|-------------|---------|
| [fastapi](https://pypi.org/project/fastapi/) | `>=0.127.0` | 0.127.1 | 0.127.1 | ASGI web framework |
| [uvicorn](https://pypi.org/project/uvicorn/) | `>=0.40.0` | 0.40.0 | 0.40.0 | ASGI server |
| [pydantic](https://pypi.org/project/pydantic/) | `>=2.12.0` | 2.12.5 | 2.12.5 | Data validation |
| [pydantic-settings](https://pypi.org/project/pydantic-settings/) | `>=2.12.0` | 2.12.0 | 2.12.0 | Settings management |
| [pydantic-ai](https://pypi.org/project/pydantic-ai/) | `>=1.39.0` | 1.39.0 | 1.39.0 | Structured LLM outputs |
| [httpx](https://pypi.org/project/httpx/) | `>=0.28.0` | 0.28.1 | 0.28.1 | Async HTTP client |
| [structlog](https://pypi.org/project/structlog/) | `>=25.5.0` | 25.5.0 | 25.5.0 | Structured logging |
| [orjson](https://pypi.org/project/orjson/) | `>=3.11.0` | 3.11.5 | 3.11.5 | Fast JSON serialization |
| [pandas](https://pypi.org/project/pandas/) | `>=2.3.0` | 2.3.3 | 2.3.3 | Data analysis |
| [numpy](https://pypi.org/project/numpy/) | `>=2.4.0` | 2.4.0 | 2.4.0 | Numerical computing |
| [scikit-learn](https://pypi.org/project/scikit-learn/) | `>=1.8.0` | 1.8.0 | 1.8.0 | ML utilities (cosine similarity) |
| [pyyaml](https://pypi.org/project/pyyaml/) | `>=6.0.0` | 6.0.3 | 6.0.3 | YAML parsing |

---

## Dev Dependencies

Development and testing dependencies (`[project.optional-dependencies.dev]`).

| Package | Constraint | Locked | PyPI Latest | Purpose |
|---------|------------|--------|-------------|---------|
| [pytest](https://pypi.org/project/pytest/) | `>=9.0.0` | 9.0.2 | 9.0.2 | Test framework |
| [pytest-cov](https://pypi.org/project/pytest-cov/) | `>=7.0.0` | 7.0.0 | 7.0.0 | Coverage reporting |
| [pytest-asyncio](https://pypi.org/project/pytest-asyncio/) | `>=1.3.0` | 1.3.0 | 1.3.0 | Async test support |
| [pytest-xdist](https://pypi.org/project/pytest-xdist/) | `>=3.8.0` | 3.8.0 | 3.8.0 | Parallel test execution |
| [hypothesis](https://pypi.org/project/hypothesis/) | `>=6.148.0` | 6.148.8 | 6.148.8 | Property-based testing |
| [mypy](https://pypi.org/project/mypy/) | `>=1.19.0` | 1.19.1 | 1.19.1 | Static type checking |
| [types-pyyaml](https://pypi.org/project/types-pyyaml/) | `>=6.0.0` | 6.0.12.20250915 | 6.0.12.20250915 | PyYAML type stubs |
| [ruff](https://pypi.org/project/ruff/) | `>=0.14.0` | 0.14.10 | 0.14.10 | Linting + formatting |
| [pre-commit](https://pypi.org/project/pre-commit/) | `>=4.5.0` | 4.5.1 | 4.5.1 | Git hooks |
| [respx](https://pypi.org/project/respx/) | `>=0.22.0` | 0.22.0 | 0.22.0 | HTTPX mocking |

---

## Docs Dependencies

Documentation generation (`[project.optional-dependencies.docs]`).

| Package | Constraint | Locked | PyPI Latest | Purpose |
|---------|------------|--------|-------------|---------|
| [mkdocs](https://pypi.org/project/mkdocs/) | `>=1.6.0` | 1.6.1 | 1.6.1 | Documentation generator |
| [mkdocs-material](https://pypi.org/project/mkdocs-material/) | `>=9.5.0` | 9.7.1 | 9.7.1 | Material theme |
| [mkdocstrings](https://pypi.org/project/mkdocstrings/) | `>=0.27.0` | 1.0.0 | 1.0.0 | API docs from docstrings |

---

## HuggingFace Dependencies (Optional)

Heavy ML dependencies for local embeddings (`[project.optional-dependencies.hf]`).

| Package | Constraint | Locked | PyPI Latest | Purpose |
|---------|------------|--------|-------------|---------|
| [torch](https://pypi.org/project/torch/) | `>=2.4.0` | 2.9.1 | 2.9.1 | Deep learning framework |
| [transformers](https://pypi.org/project/transformers/) | `>=4.51.0` | 4.57.3 | 4.57.3 | HuggingFace models |
| [sentence-transformers](https://pypi.org/project/sentence-transformers/) | `>=2.7.0` | 5.2.0 | 5.2.0 | Embedding models |

---

## Python Version

```toml
requires-python = ">=3.11"
```

Tested on: Python 3.11, 3.12

---

## Package Manager

```toml
[tool.uv]
required-version = ">=0.9.18"
```

We use [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

---

## Updating Dependencies

```bash
# Check if lock is current
uv lock --check

# Update all to latest compatible
uv lock --upgrade

# Update specific package
uv lock --upgrade-package fastapi

# Sync environment
uv sync --all-extras

# Verify tests pass
uv run pytest tests/unit -q
```

---

## Audit Checklist

When auditing dependencies:

1. **Security:** Check for CVEs via `pip-audit` or Snyk
2. **Freshness:** Compare locked vs PyPI latest (this doc)
3. **Compatibility:** Run full test suite after updates
4. **Lock consistency:** `uv lock --check` should pass
5. **Python version:** Ensure all deps support our Python range

---

## Version History

| Date | Change | By |
|------|--------|-----|
| 2025-12-26 | Updated fastapi 0.124→0.127, uvicorn 0.38→0.40, numpy 2.3→2.4 | dependency-audit |
| 2025-12-24 | Added pydantic-ai 1.39.0 | PR #62 |
| 2025-12-23 | Initial v0.1.0 release | initial |
