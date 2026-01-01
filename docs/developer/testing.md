# Testing Conventions (Markers, Fixtures, and Test Doubles)

**Audience**: Maintainers and contributors
**Last Updated**: 2026-01-01

This project treats the test suite as part of the research SSOT. Tests should be readable, deterministic, and accurately labeled.

---

## Test Layout

- `tests/unit/`: deterministic unit tests (mock I/O boundaries)
- `tests/integration/`: integration tests (slower, more dependencies)
- `tests/e2e/`: end-to-end tests (opt-in; may require external services)

---

## Pytest Markers (Explicit by Design)

All tests should be explicitly marked to match their directory:

- `pytest.mark.unit`
- `pytest.mark.integration`
- `pytest.mark.e2e`

Preferred pattern (module-level):

```python
import pytest

pytestmark = pytest.mark.unit
```

Rationale:
- makes test intent obvious during review
- avoids “magic” implicit marking behavior
- supports `pytest -m unit|integration|e2e` reliably

---

## Test Doubles Policy

Test doubles (mock clients, fake services) must live under `tests/`, not in `src/`.

Example: `MockLLMClient` lives in `tests/fixtures/mock_llm.py` and must not be exported from production modules.

Why:
- prevents accidental production dependencies on test-only code
- keeps `src/` focused on deployable package behavior

---

## Commands

```bash
make test
make test-unit
make test-integration
AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e
```
