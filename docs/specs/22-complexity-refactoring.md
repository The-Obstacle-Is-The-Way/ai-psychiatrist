# Spec 22: Complexity Refactoring (noqa Suppressions)

> **STATUS: OPEN**
>
> **Priority**: Low - Code works correctly. This is maintainability/readability debt.
>
> **GitHub Issue**: #60 (parent tech-debt audit)
>
> **Created**: 2025-12-26
>
> **Last Verified Against Code**: 2025-12-27

---

## Problem Statement

Three functions have Ruff pylint-complexity suppressions (`PLR0912` / `PLR0915`) indicating excessive branching/statements:

- **PLR0912**: Too many branches (if/elif/else)
- **PLR0915**: Too many statements

These functions are harder to test in isolation and have higher cognitive load for maintainers.

---

## Current State: All Instances

### 1. `ReferenceStore._validate_metadata()` (`src/ai_psychiatrist/services/reference_store.py`)

**Suppression (current)**: `# noqa: PLR0912` (too many branches)

**What it does**: Validates embedding artifact metadata against current config (backend, model, dimension, chunk params, split integrity).

**Current structure**:
```python
def _validate_metadata(self, metadata: dict[str, Any]) -> None:
    errors: list[str] = []

    # Backend check
    # Model check (resolved backend-specific ID)
    # Dimension check
    # Chunk params check
    # Split integrity check (split_ids_hash preferred; CSV hash is audit-only)

    if errors:
        raise EmbeddingArtifactMismatchError(...)
```

**Complexity driver**: repeated “missing field vs mismatch” checks and split-integrity branching.

### 2. `ReferenceStore._load_embeddings()` (`src/ai_psychiatrist/services/reference_store.py`)

**Suppressions**: `PLR0912, PLR0915` (too many branches + statements)

**What it does**: Loads pre-computed embeddings from NPZ + JSON files with validation.

**Current structure**:
```python
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    # Early return if cached
    # Check NPZ exists
    # Check JSON exists
    # Load and validate metadata
    # Load texts from JSON
    # Load embeddings from NPZ
    # Normalize and combine (loop with conditions)
    # Log statistics
```

**Complexity driver**: multiple early-return branches + metadata probing + normalization loop.

### 3. `main_async()` (`scripts/generate_embeddings.py`)

**Suppressions**: `PLR0915` (too many statements)

**What it does**: Main script entry point for generating embeddings.

**Current structure**:
```python
async def main_async(args: argparse.Namespace) -> int:
    # Setup logging
    # Load settings
    # Override backend if specified
    # Resolve model name
    # Determine output path
    # Print banner
    # Dry run check
    # Get participant IDs
    # Create transcript service
    # Create embedding client
    # Process participants loop
    # Save embeddings (NPZ, JSON, meta)
    # Print summary
```

**Line count**: ~150 lines
**Branch count**: Low (mostly sequential)

---

## Refactoring Strategy

### Strategy 1: Extract Validation Helpers (reference_store.py)

For `_validate_metadata()`, extract field-level validators:

```python
# Before: one large function with many conditional branches

# After:
def _validate_metadata(self, metadata: dict[str, Any]) -> None:
    errors: list[str] = []
    errors.extend(self._validate_backend(metadata))
    errors.extend(self._validate_model(metadata))
    errors.extend(self._validate_dimension(metadata))
    errors.extend(self._validate_chunk_params(metadata))
    errors.extend(self._validate_split_integrity(metadata))

    if errors:
        raise EmbeddingArtifactMismatchError(...)

def _validate_backend(self, metadata: dict[str, Any]) -> list[str]:
    stored = metadata.get("backend")
    current = self._embedding_backend.backend.value
    if stored is None:
        logger.debug("Metadata missing 'backend' field")
        return []
    if stored != current:
        return [f"backend mismatch: artifact='{stored}', config='{current}'"]
    return []

# Similar for _validate_model, _validate_dimension, etc.
```

**Important**: preserve the current split semantics:
- Prefer `split_ids_hash` (semantic) when present.
- Treat `split_csv_hash` mismatch as warning-only when IDs match.

**Benefits**:
- Each validator is testable in isolation
- Main function becomes 15-20 lines
- Individual validators are 8-12 lines each
- Easy to add new validations

### Strategy 2: Extract Loading Phases (reference_store.py)

For `_load_embeddings()`, extract phases:

```python
# Before: 124 lines

# After:
def _load_embeddings(self) -> dict[int, list[tuple[str, list[float]]]]:
    if self._embeddings is not None:
        return self._embeddings

    paths = self._resolve_embedding_paths()
    if paths is None:
        return {}

    self._validate_artifact_metadata(paths.meta)
    texts = self._load_texts_json(paths.json)
    embeddings = self._load_embeddings_npz(paths.npz)

    self._embeddings = self._combine_and_normalize(texts, embeddings)
    return self._embeddings

@dataclass
class EmbeddingPaths:
    npz: Path
    json: Path
    meta: Path | None

def _resolve_embedding_paths(self) -> EmbeddingPaths | None:
    """Check files exist and return paths."""
    ...

def _load_texts_json(self, path: Path) -> dict[str, list[str]]:
    """Load text chunks from JSON sidecar."""
    ...

def _combine_and_normalize(
    self,
    texts: dict[str, list[str]],
    embeddings: np.lib.npyio.NpzFile
) -> dict[int, list[tuple[str, list[float]]]]:
    """Combine texts with embeddings and normalize."""
    ...
```

**Benefits**:
- Each phase is testable
- Clear separation of concerns
- Easier to follow data flow

### Strategy 3: Extract Script Phases (generate_embeddings.py)

For `main_async()`, extract setup/execute/save phases:

```python
# Before: 150 lines

# After:
async def main_async(args: argparse.Namespace) -> int:
    config = prepare_config(args)

    if args.dry_run:
        print_dry_run_banner(config)
        return 0

    result = await generate_embeddings(config)
    save_embeddings(config.output_path, result)
    print_summary(result)
    return 0

@dataclass
class GenerationConfig:
    backend: str
    model: str
    dimension: int
    chunk_size: int
    chunk_step: int
    min_chars: int
    split: str
    output_path: Path
    participant_ids: list[int]

def prepare_config(args: argparse.Namespace) -> GenerationConfig:
    """Load settings and resolve configuration."""
    ...

async def generate_embeddings(config: GenerationConfig) -> GenerationResult:
    """Process all participants and generate embeddings."""
    ...

def save_embeddings(output_path: Path, result: GenerationResult) -> None:
    """Save NPZ, JSON, and metadata files."""
    ...
```

**Benefits**:
- Main function becomes ~15 lines
- Each phase is testable
- Configuration is explicit (dataclass)

---

## Implementation Plan

### Phase 1: reference_store.py Validators (Medium Impact)

1. Create `_validate_*` helper methods for each field
2. Refactor `_validate_metadata()` to use helpers
3. Add unit tests for each validator
4. Remove `PLR0912` suppression

**Estimated changes**: ~100 lines added, ~50 lines refactored

### Phase 2: reference_store.py Loading (Medium Impact)

1. Create `EmbeddingPaths` dataclass
2. Extract `_resolve_embedding_paths()`
3. Extract `_load_texts_json()`
4. Extract `_combine_and_normalize()`
5. Add unit tests for each phase
6. Remove `PLR0912, PLR0915` suppressions

**Estimated changes**: ~80 lines added, ~60 lines refactored

### Phase 3: generate_embeddings.py (Lower Impact - Script)

1. Create `GenerationConfig` and `GenerationResult` dataclasses
2. Extract `prepare_config()`
3. Extract `generate_embeddings()`
4. Extract `save_embeddings()`
5. Remove `PLR0915` suppression

**Estimated changes**: ~60 lines added, ~40 lines refactored

---

## Acceptance Criteria

- [ ] The three current suppressions are removed:
  - `ReferenceStore._validate_metadata()` no longer needs `# noqa: PLR0912`
  - `ReferenceStore._load_embeddings()` no longer needs `# noqa: PLR0912, PLR0915`
  - `scripts/generate_embeddings.py:main_async()` no longer needs `# noqa: PLR0915`
- [ ] `uv run ruff check .` passes with no new ignores
- [ ] Refactor preserves behavior (including split integrity validation semantics)
- [ ] No regressions in existing tests
- [ ] mypy passes with no new errors

---

## Files To Modify

```
src/ai_psychiatrist/services/reference_store.py    (2 instances, ~180 lines)
scripts/generate_embeddings.py                      (1 instance, ~150 lines)
tests/unit/services/test_reference_store.py        (add validator tests)
tests/unit/scripts/test_generate_embeddings.py     (new: add unit tests for extracted pure helpers)
```

---

## Complexity Metrics (Before/After)

Exact numbers are expected to drift; the goal is to remove the `PLR0912/PLR0915`
violations by extracting single-purpose helpers and adding unit tests.

---

## References

- GitHub Issue: #60 (tech-debt: Code readability audit)
- Ruff rules: [PLR0912](https://docs.astral.sh/ruff/rules/too-many-branches/), [PLR0915](https://docs.astral.sh/ruff/rules/too-many-statements/)
- Clean Code: Functions should do one thing (Uncle Bob)
