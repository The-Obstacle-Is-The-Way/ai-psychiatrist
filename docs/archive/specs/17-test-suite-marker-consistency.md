# Spec 17: Test Suite Marker Consistency

> **STATUS**: ✅ IMPLEMENTED (ARCHIVED)
>
> **GitHub Issue**: [#58](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/58)
>
> **Priority**: Low (maintainability/organization; tests can still be run by directory)
>
> **Implemented**: 2025-12-26
>
> **Last Updated**: 2025-12-26

---

## Problem Statement

Our test suite is organized by directory (`tests/unit/`, `tests/integration/`, `tests/e2e/`),
but our Makefile targets use markers:

```bash
make test-unit        # uv run pytest -m unit
make test-integration # uv run pytest -m integration
make test-e2e         # uv run pytest -m e2e
```

Before this spec was implemented, most tests were **not marked**, so marker-based selection did **not** match directory-based selection.

**Observed on 2025-12-26 (recompute via commands below):**

| Test Category | Test Files | Marker Coverage | `pytest -m ...` Coverage |
|---|---:|---:|---:|
| Unit (`tests/unit/`) | 30 | 3 files manually marked | `-m unit` selects 82 / 646 tests |
| Integration (`tests/integration/`) | 2 | 0 files marked | `-m integration` selects 0 / 13 tests |
| E2E (`tests/e2e/`) | 3 | 3 files marked | ✅ works |

**Consequence (pre-fix)**: `make test-unit` and `make test-integration` were misleading (and `make test-integration` ran nothing).

---

## Root Cause

- The project defines markers in `pyproject.toml` (`unit`, `integration`, `e2e`), and runs with `--strict-markers`.
- Tests are organized by directory, but we rely on **manual decorators** (`@pytest.mark.unit`, etc.).
- Manual marking is currently inconsistent, so `-m unit` / `-m integration` do not select the intended tests.

---

## Solution

Automatically apply markers based on test location during collection.

Implement in `tests/conftest.py` using `pytest_collection_modifyitems`.

Important ordering detail:
- pytest applies `-m ...` deselection in its own `pytest_collection_modifyitems` hook
  (`_pytest/mark/__init__.py`), so our marker auto-application must run **first**.
- Use `@pytest.hookimpl(tryfirst=True)` to guarantee our hook runs before deselection.

---

## Implementation

### Step 1 — Auto-mark tests in `tests/conftest.py`

Append this hook at the end of `tests/conftest.py`:

```python
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Apply markers based on directory structure so Makefile targets behave correctly."""
    for item in items:
        # pytest uses pathlib.Path for `item.path` on modern versions; normalize for all OSes.
        raw_path = getattr(item, "path", item.fspath)
        path = str(raw_path).replace("\\", "/")

        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/tests/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
```

### Step 2 — Optional cleanup

After Step 1 is verified, optionally remove redundant `@pytest.mark.unit` decorators from:
- `tests/unit/agents/test_quantitative.py`
- `tests/unit/agents/test_quantitative_backfill.py`
- `tests/unit/infrastructure/llm/test_huggingface.py`

Keep functional markers like `@pytest.mark.asyncio`.

---

## Verification Plan

Note: this repo’s pytest config includes coverage enforcement (`--cov-fail-under=80`) via `addopts`.
For `--collect-only` checks, override it with `-o addopts=''`.

### Baseline (before)

```bash
uv run pytest -o addopts='' tests/unit --collect-only -q | tail -2
uv run pytest -o addopts='' -m unit --collect-only -q | tail -2

uv run pytest -o addopts='' tests/integration --collect-only -q | tail -2
uv run pytest -o addopts='' -m integration --collect-only -q || true  # currently exits 5 (no tests)
```

### After implementation

```bash
uv run pytest -o addopts='' tests/unit --collect-only -q | tail -2
uv run pytest -o addopts='' -m unit --collect-only -q | tail -2

uv run pytest -o addopts='' tests/integration --collect-only -q | tail -2
uv run pytest -o addopts='' -m integration --collect-only -q | tail -2
```

Then run:

```bash
make test-unit
make test-integration
make test
```

---

## Acceptance Criteria

- [x] `uv run pytest -o addopts='' -m unit --collect-only` collects the same number of tests as `uv run pytest -o addopts='' tests/unit --collect-only`. ✅ (653 tests as of 2025-12-26)
- [x] `uv run pytest -o addopts='' -m integration --collect-only` collects the same number of tests as `uv run pytest -o addopts='' tests/integration --collect-only`. ✅ (13 tests)
- [x] `make test-unit` runs all unit tests under `tests/unit/`.
- [x] `make test-integration` runs all integration tests under `tests/integration/`.
- [x] `make test` still passes.

---

## References

- pytest marker hook: https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
- pytest `-m` deselection implementation: `_pytest/mark/__init__.py`
