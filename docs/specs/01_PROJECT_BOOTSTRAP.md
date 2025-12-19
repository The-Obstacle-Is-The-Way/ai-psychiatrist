# Spec 01: Project Bootstrap

## Objective

Establish modern Python project structure with uv, Ruff, pytest, and Makefile automation. This is the foundation for all subsequent specs.

## Testing Philosophy

**CRITICAL: No Mock Abuse**

This codebase follows a strict testing philosophy to avoid the "mock everything" anti-pattern:

### What to Test (Real Behavior)
- **Unit tests**: Test pure functions, data transformations, validation logic with REAL inputs
- **Integration tests**: Test component interactions with REAL dependencies where possible
- **E2E tests**: Test full pipeline with REAL Ollama (can use smaller/faster models for CI)

### When Mocking is Acceptable
Only mock at **external I/O boundaries** that would make tests slow, flaky, or require external services:
- HTTP calls to Ollama API (mock the HTTP layer, not the client logic)
- File system operations for tests that shouldn't touch disk
- Time-dependent operations (sparingly; prefer injecting clocks)

### When Mocking is FORBIDDEN
- **Never mock business logic** - if you need to mock it, your design is wrong
- **Never mock to make tests pass** - fix the code, not the test
- **Never mock domain models** - use real instances with test data
- **Never mock internal functions** - test them directly or through public API

### Test Data vs Mocks
- **Test data fixtures** (`sample_transcript`, `sample_phq8_scores`): GOOD - real data structures
- **Response fixtures** (`sample_ollama_response`): GOOD - real API response structure for parsing tests
- **Mock objects that replace real behavior**: BAD unless at I/O boundary

### Example: Testing Ollama Client
```python
# GOOD: Mock HTTP, test real client logic
@pytest.fixture
def mock_httpx_client(sample_ollama_response):
    with respx.mock:
        respx.post("http://localhost:11434/api/chat").respond(json=sample_ollama_response)
        yield

async def test_ollama_client_parses_response(mock_httpx_client):
    client = OllamaClient(base_url="http://localhost:11434")
    result = await client.chat(messages=[...])  # Real client, mocked HTTP
    assert result.content == "..."  # Test real parsing logic

# BAD: Mocking the client itself
from unittest import mock

def test_bad_mock_everything() -> None:
    mock_client = mock.Mock()  # NEVER DO THIS
    mock_client.chat.return_value = "whatever"  # Tests nothing real
```

## As-Is Implementation (Repo)

This repository now contains the `uv`/`pyproject.toml` bootstrap defined in this spec **alongside** the original research code. The legacy research execution path is primarily driven by:

- Conda environment file: `assets/env_reqs.yml`
- HPC execution: `slurm/job_ollama.sh`, `slurm/job_assess.sh`
- Local demo server: `server.py` (requires an Ollama daemon + a transcript file via `TRANSCRIPT_PATH`)

The production rewrite should live under `src/ai_psychiatrist/` and incrementally replace the legacy scripts while preserving paper parity.

## Deliverables

1. `pyproject.toml` - Single source of truth for project config
2. `uv.lock` - Locked dependencies for reproducibility
3. `Makefile` - Developer workflow automation
4. `.github/workflows/ci.yml` - CI/CD pipeline
5. `.pre-commit-config.yaml` - Local developer guardrails
6. `.env.example` - Paper-optimal configuration template
7. `src/ai_psychiatrist/__init__.py` - Package skeleton
8. `src/ai_psychiatrist/cli.py` - CLI entry point
9. `tests/conftest.py` - pytest fixtures
10. `tests/unit/test_bootstrap.py` - Bootstrap tests
11. `tests/unit/test_cli.py` - CLI smoke test

## Implementation

### 1. Project Configuration (pyproject.toml)

```toml
[project]
name = "ai-psychiatrist"
version = "2.0.0"
description = "LLM-based Multi-Agent System for Depression Assessment"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "TReNDS Center", email = "vcalhoun@gsu.edu" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]
keywords = ["llm", "mental-health", "depression", "phq-8", "multi-agent"]

dependencies = [
    "fastapi>=0.124.0",
    "uvicorn[standard]>=0.38.0",
    "pydantic>=2.12.0",
    "pydantic-settings>=2.12.0",
    "httpx>=0.28.0",
    "structlog>=25.5.0",
    "orjson>=3.11.0",
    "pandas>=2.3.0",
    "numpy>=2.3.0",
    "scikit-learn>=1.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.0",
    "pytest-cov>=7.0.0",
    "pytest-asyncio>=1.3.0",
    "pytest-xdist>=3.8.0",
    "hypothesis>=6.148.0",
    "mypy>=1.19.0",
    "ruff>=0.14.0",
    "pre-commit>=4.5.0",
    "respx>=0.22.0",  # httpx mocking
]
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.27.0",
]

[project.scripts]
ai-psychiatrist = "ai_psychiatrist.cli:main"

[project.urls]
Homepage = "https://github.com/trendscenter/ai-psychiatrist"
Documentation = "https://trendscenter.github.io/ai-psychiatrist"
Repository = "https://github.com/trendscenter/ai-psychiatrist"
Issues = "https://github.com/trendscenter/ai-psychiatrist/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ai_psychiatrist"]

# ============== RUFF ==============
[tool.ruff]
target-version = "py311"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "ERA",    # eradicate (commented code)
    "PL",     # pylint
    "RUF",    # ruff-specific
]
ignore = [
    "PLR0913",  # Too many arguments (agents need many params)
    "PLR2004",  # Magic value comparison
]

[tool.ruff.lint.isort]
known-first-party = ["ai_psychiatrist"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "ARG001", "PLR2004"]

# ============== PYTEST ==============
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "--strict-config",
    "-ra",
    "--cov=ai_psychiatrist",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=80",
]
markers = [
    "unit: Unit tests (fast, no I/O)",
    "integration: Integration tests (may use mocks)",
    "e2e: End-to-end tests (full system)",
    "slow: Slow tests (skip with -m 'not slow')",
]

# ============== MYPY ==============
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
follow_imports = "silent"
ignore_missing_imports = false

[[tool.mypy.overrides]]
module = [
    "pandas.*",
    "sklearn.*",
    "numpy.*",
]
ignore_missing_imports = true

# ============== COVERAGE ==============
[tool.coverage.run]
branch = true
source = ["src/ai_psychiatrist"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
]
```

### 2. Makefile

```makefile
.PHONY: help install dev test lint format typecheck clean docs serve

# Self-documenting help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============== Setup ==============
install: ## Install production dependencies
	uv sync --no-dev

dev: ## Install all dependencies (including dev)
	uv sync --all-extras
	uv run pre-commit install

# ============== Testing ==============
test: ## Run all tests with coverage
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest -m unit

test-integration: ## Run integration tests only
	uv run pytest -m integration

test-e2e: ## Run end-to-end tests only
	uv run pytest -m e2e

test-fast: ## Run tests excluding slow ones
	uv run pytest -m "not slow"

test-parallel: ## Run tests in parallel
	uv run pytest -n auto

# ============== Code Quality ==============
lint: ## Run linter (ruff)
	uv run ruff check src tests

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix src tests

format: ## Format code (ruff)
	uv run ruff format src tests

format-check: ## Check formatting without changes
	uv run ruff format --check src tests

typecheck: ## Run type checker (mypy)
	uv run mypy src

# ============== All Quality Checks ==============
check: lint typecheck test ## Run all checks (lint, typecheck, test)

ci: format-check lint typecheck test ## CI pipeline checks

# ============== Development ==============
serve: ## Run development server
	uv run uvicorn ai_psychiatrist.api.main:app --reload --host 0.0.0.0 --port 8000

repl: ## Start Python REPL with project loaded
	uv run python -i -c "from ai_psychiatrist import *"

# ============== Documentation ==============
docs: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

# ============== Cleanup ==============
clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

## Environment Variables (As-Is + Target)

### As-Is (Used by Current Repo)

- `TRANSCRIPT_PATH`: Read by `agents/interview_simulator.py` to locate the transcript file for `server.py`.

### HPC / SLURM (Configured in `slurm/job_ollama.sh`)

These are set in the SLURM script (not in Python) and should be documented because they materially affect Ollama behavior:

- `CUDA_VISIBLE_DEVICES`
- `OLLAMA_HOST_MEMORY`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_MMAP`
- `GGML_CUDA_FORCE_CUBLAS`
- `GGML_CUDA_FORCE_MMQ`
- `OLLAMA_HOST`
- `OLLAMA_FLASH_ATTENTION`
- `OLLAMA_MODELS`
- `OLLAMA_BACKEND`


### 3. Directory Structure

```bash
# Create src layout
mkdir -p src/ai_psychiatrist/{api,agents,domain,services,infrastructure}
mkdir -p tests/{unit,integration,e2e}

# Create __init__.py files
touch src/ai_psychiatrist/__init__.py
touch src/ai_psychiatrist/api/__init__.py
touch src/ai_psychiatrist/agents/__init__.py
touch src/ai_psychiatrist/domain/__init__.py
touch src/ai_psychiatrist/services/__init__.py
touch src/ai_psychiatrist/infrastructure/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/e2e/__init__.py
```

### 4. Package Init (src/ai_psychiatrist/__init__.py)

```python
"""AI Psychiatrist: LLM-based Multi-Agent System for Depression Assessment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-psychiatrist")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
__all__ = ["__version__"]
```

### 5. Test Configuration (tests/conftest.py)

```python
"""Shared pytest fixtures and configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def sample_transcript() -> str:
    """Return a sample interview transcript for testing."""
    return """
    Ellie: How are you doing today?
    Participant: Not great, I've been feeling really down lately.
    Ellie: Can you tell me more about that?
    Participant: I just can't seem to enjoy anything anymore.
    I used to love going out with friends, but now I can't be bothered.
    Ellie: How long has this been going on?
    Participant: A few months now. I'm also not sleeping well.
    I wake up at 3 or 4 in the morning and can't get back to sleep.
    """.strip()


@pytest.fixture(scope="session")
def sample_phq8_scores() -> dict[str, int]:
    """Return sample PHQ-8 ground truth scores."""
    return {
        "PHQ8_NoInterest": 2,
        "PHQ8_Depressed": 2,
        "PHQ8_Sleep": 2,
        "PHQ8_Tired": 1,
        "PHQ8_Appetite": 0,
        "PHQ8_Failure": 1,
        "PHQ8_Concentrating": 1,
        "PHQ8_Moving": 0,
    }


@pytest.fixture
def sample_ollama_response() -> dict:
    """Return sample Ollama API response structure for testing.

    NOTE: This is TEST DATA, not a mock. We use real data structures to test
    parsing/validation logic. Only mock external I/O boundaries (HTTP calls),
    never business logic.
    """
    return {
        "model": "alibayram/medgemma:27b",  # Paper-optimal (Appendix F)
        "message": {
            "role": "assistant",
            "content": '{"PHQ8_NoInterest": {"evidence": "can\'t be bothered", "score": 2}}'
        },
        "done": True,
    }
```

### 6. CI/CD Pipeline (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

env:
  UV_CACHE_DIR: /tmp/.uv-cache

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Check formatting
        run: uv run ruff format --check src tests

      - name: Lint
        run: uv run ruff check src tests

      - name: Type check
        run: uv run mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
          enable-cache: true

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run tests
        run: uv run pytest --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: true
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}

  # Prevent merging if CI fails
  ci-success:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - name: CI Passed
        run: echo "All CI checks passed!"
```

### 7. Pre-commit Configuration (.pre-commit-config.yaml)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: mypy
        name: mypy (uv)
        entry: uv run mypy src
        language: system
        pass_filenames: false

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key
```

### 8. Environment Template (.env.example)

```bash
# AI Psychiatrist Configuration (Paper-Optimal)
# Copy to .env and fill in values

# ============== Required ==============
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434

# ============== LLM Models (Paper-Optimal) ==============
# Paper baseline (Section 2.2): Gemma 3 27B for the multi-agent system
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b

# Paper Appendix F: MedGemma improves quantitative MAE (0.505 vs 0.619) but makes fewer predictions
MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b

# Embedding model family (Section 2.2): Qwen 3 8B Embedding
# Note: the paper does not specify quantization; Q8_0 is a recommended implementation default.
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b

# Alternative (paper baseline quantitative model):
# MODEL_QUANTITATIVE_MODEL=gemma3:27b

# ============== Feedback Loop (Paper Section 2.3.1) ==============
FEEDBACK_ENABLED=true
FEEDBACK_MAX_ITERATIONS=10
# Paper: "score below four" triggers refinement (threshold=3 means <4)
FEEDBACK_SCORE_THRESHOLD=3

# ============== Embedding/Few-Shot (Paper Appendix D) ==============
# Paper optimal hyperparameters:
EMBEDDING_DIMENSION=4096
EMBEDDING_TOP_K_REFERENCES=2
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_CHUNK_STEP=2

# ============== Server ==============
OLLAMA_TIMEOUT_SECONDS=180
API_HOST=0.0.0.0
API_PORT=8000

# ============== Logging ==============
LOG_LEVEL=INFO
LOG_FORMAT=json  # json or console
```

## Acceptance Criteria

### Tests (tests/unit/test_bootstrap.py)

```python
"""Tests for project bootstrap."""

import subprocess
from pathlib import Path

import ai_psychiatrist
import pytest


class TestProjectStructure:
    """Test project structure is correct."""

    def test_pyproject_toml_exists(self) -> None:
        """pyproject.toml should exist at project root."""
        assert Path("pyproject.toml").exists()

    def test_src_layout(self) -> None:
        """src/ai_psychiatrist package should exist."""
        assert Path("src/ai_psychiatrist/__init__.py").exists()

    def test_tests_directory(self) -> None:
        """tests directory should exist with subdirs."""
        assert Path("tests/unit").is_dir()
        assert Path("tests/integration").is_dir()
        assert Path("tests/e2e").is_dir()

    def test_package_subdirectories(self) -> None:
        """Package should have expected subdirectories."""
        subdirs = ["api", "agents", "domain", "services", "infrastructure"]
        for subdir in subdirs:
            assert Path(f"src/ai_psychiatrist/{subdir}/__init__.py").exists()


class TestImports:
    """Test that package can be imported."""

    def test_import_package(self) -> None:
        """Package should be importable with version from metadata."""
        assert isinstance(ai_psychiatrist.__version__, str)
        assert ai_psychiatrist.__version__  # Not empty


class TestMakefile:
    """Test Makefile targets work."""

    @pytest.mark.slow
    def test_make_help(self) -> None:
        """make help should run without error."""
        result = subprocess.run(
            ["make", "help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "help" in result.stdout


class TestEnvExample:
    """Test .env.example contains paper-optimal values."""

    def test_env_example_exists(self) -> None:
        """.env.example should exist."""
        assert Path(".env.example").exists()

    def test_env_example_has_paper_optimal_values(self) -> None:
        """.env.example should contain paper-optimal configuration."""
        content = Path(".env.example").read_text()

        # Paper-optimal models
        assert "gemma3:27b" in content
        assert "alibayram/medgemma:27b" in content
        assert "qwen3-embedding:8b" in content

        # Paper-optimal hyperparameters
        assert "EMBEDDING_DIMENSION=4096" in content
        assert "EMBEDDING_TOP_K_REFERENCES=2" in content
        assert "EMBEDDING_CHUNK_SIZE=8" in content
        assert "FEEDBACK_SCORE_THRESHOLD=3" in content
        assert "FEEDBACK_MAX_ITERATIONS=10" in content
```

## Migration Notes

### From Old Structure

1. Move `agents/*.py` → `src/ai_psychiatrist/agents/`
2. Move `server.py` → `src/ai_psychiatrist/api/main.py`
3. Keep `assets/env_reqs.yml` temporarily for research reproducibility; remove only after parity is proven
4. Keep `slurm/` for HPC compatibility (documented in Specs 01/04)

### Breaking Changes

- Python >= 3.11 required for the refactor (`pyproject.toml`)
- Import path changes: `from agents.x import Y` → `from ai_psychiatrist.agents.x import Y`
- Refactor path uses `uv`; legacy scripts may still use conda until migrated

## Verification Steps

```bash
# 1. Install all dependencies (incl. dev/docs extras)
uv sync --all-extras

# 2. Run full CI check locally
make ci

# 3. (Optional) Run pre-commit on everything
uv run pre-commit run --all-files
```

## Dependencies on Other Specs

- **None** - This is the foundation spec

## Specs That Depend on This

- **All subsequent specs** (02-12)
