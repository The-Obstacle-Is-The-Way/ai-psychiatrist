.PHONY: help install dev dev-hf test test-unit test-integration test-e2e test-fast test-parallel lint lint-fix format format-check typecheck check ci serve repl docs docs-serve clean

# Self-documenting help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============== Setup ==============
install: ## Install production dependencies
	uv sync --no-dev

dev: ## Install all dependencies (including dev)
	uv sync --extra dev --extra docs
	uv run pre-commit install

dev-hf: ## Install dev deps + HuggingFace backend extras
	uv sync --extra dev --extra docs --extra hf
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
	uv run ruff check src tests scripts server.py

lint-fix: ## Fix linting issues automatically
	uv run ruff check --fix src tests scripts server.py

format: ## Format code (ruff)
	uv run ruff format src tests scripts server.py

format-check: ## Check formatting without changes
	uv run ruff format --check src tests scripts server.py

typecheck: ## Run type checker (mypy)
	uv run mypy src tests scripts server.py

# ============== All Quality Checks ==============
check: lint typecheck test ## Run all checks (lint, typecheck, test)

ci: format-check lint typecheck test ## CI pipeline checks

# ============== Development ==============
serve: ## Run development server
	uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000

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
