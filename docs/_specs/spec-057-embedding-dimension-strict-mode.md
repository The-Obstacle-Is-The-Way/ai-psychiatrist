# Spec 057: Embedding Dimension Strict Mode Default

**Status**: Ready to Implement
**Priority**: Medium
**Complexity**: Low
**Related**: PIPELINE-BRITTLENESS.md, Spec 055

---

## Problem Statement

When the embedding backend produces vectors with fewer dimensions than expected, the system silently truncates or skips chunks without warning:

```python
# Current behavior in reference_store.py
if len(emb) < self._config_dimension:
    if strict_mode:
        raise EmbeddingArtifactMismatchError(...)
    else:
        logger.warning("Skipping chunk with insufficient dimension")
        continue  # SILENT SKIP
```

**Default**: `strict_mode=False` → silent data loss

**Problem**: If config expects 4096-dim embeddings but backend produces 1024-dim, most chunks get silently skipped, leaving minimal references for few-shot mode.

---

## Current State

### Configuration

```python
# config.py - EmbeddingSettings
embedding_dimension_strict: bool = Field(
    default=False,  # <-- PROBLEM: Silent failures allowed by default
    description="Raise error if embedding dimension doesn't match config"
)
```

### Behavior Matrix

| Backend Dim | Config Dim | Strict Mode | Result |
|-------------|------------|-------------|--------|
| 4096 | 4096 | Any | ✅ Works |
| 4096 | 1024 | Any | ✅ Truncated (acceptable) |
| 1024 | 4096 | False | ⚠️ Silently skipped |
| 1024 | 4096 | True | ✅ Raises error |

---

## Proposed Solution

Change the default to `strict_mode=True` so dimension mismatches fail loudly.

---

## Implementation

### Configuration Change

```python
# config.py - EmbeddingSettings

embedding_dimension_strict: bool = Field(
    default=True,  # CHANGED: Fail loudly by default
    description=(
        "Raise error if embedding dimension doesn't match config. "
        "Set to False only for debugging or when intentionally mixing backends."
    )
)
```

### Logging Improvement

```python
# reference_store.py - _load_embeddings()

if len(emb) < self._config_dimension:
    if self._strict_mode:
        raise EmbeddingArtifactMismatchError(
            f"Embedding dimension mismatch for participant {pid}, chunk {idx}: "
            f"got {len(emb)}, expected {self._config_dimension}. "
            f"This usually means embeddings were generated with a different model/backend. "
            f"Re-generate embeddings or set EMBEDDING_DIMENSION_STRICT=false to skip."
        )
    else:
        logger.warning(
            "embedding_dimension_mismatch_skipped",
            participant_id=pid,
            chunk_index=idx,
            actual_dim=len(emb),
            expected_dim=self._config_dimension,
        )
        skipped_count += 1
        continue

# At end of loading:
if skipped_count > 0:
    logger.warning(
        "embedding_chunks_skipped_summary",
        skipped_count=skipped_count,
        total_chunks=total_count,
        skip_rate=round(skipped_count / total_count, 3),
    )
```

---

## Migration Guide

### For Existing Deployments

If you're currently relying on silent dimension skipping (unlikely but possible):

```bash
# Before upgrade: Check if you have dimension mismatches
uv run python -c "
from ai_psychiatrist.services.reference_store import ReferenceStore
from ai_psychiatrist.config import get_settings
settings = get_settings()
store = ReferenceStore(settings.embedding)
# Will log warnings if mismatches exist
"

# If mismatches exist, either:
# 1. Regenerate embeddings with correct backend/model
# 2. Set EMBEDDING_DIMENSION_STRICT=false (not recommended)
```

### For New Deployments

No change needed - strict mode is the correct default.

---

## Environment Variable

```bash
# .env.example - add documentation
EMBEDDING_DIMENSION_STRICT=true  # Default: fail if dimensions mismatch
```

---

## Testing

```python
# tests/unit/services/test_reference_store.py

def test_dimension_mismatch_raises_by_default(mock_embeddings_4096, config_expects_8192):
    """Strict mode is default - dimension mismatch should raise."""
    store = ReferenceStore(config_expects_8192)
    with pytest.raises(EmbeddingArtifactMismatchError) as exc:
        store.load()
    assert "dimension mismatch" in str(exc.value).lower()
    assert "8192" in str(exc.value)
    assert "4096" in str(exc.value)


def test_dimension_mismatch_skips_when_not_strict(mock_embeddings_4096, config_expects_8192):
    """With strict=False, mismatches are skipped with warning."""
    config_expects_8192.embedding_dimension_strict = False
    store = ReferenceStore(config_expects_8192)

    with pytest.warns(UserWarning, match="skipped"):
        store.load()

    # Store should have fewer chunks than source
    assert store.chunk_count < mock_embeddings_4096.chunk_count


def test_matching_dimensions_works(mock_embeddings_4096, config_expects_4096):
    """Matching dimensions work regardless of strict mode."""
    store = ReferenceStore(config_expects_4096)
    store.load()  # Should not raise
    assert store.chunk_count > 0
```

---

## Risk Assessment

### Breaking Change Risk

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Existing configs with wrong dimension | Medium | Clear error message with fix instructions |
| CI/CD pipelines | Low | Tests should have matching dimensions |
| Production runs | Low | Should already have correct config |

### Rollback Plan

If issues arise:
```bash
# Quick fix: disable strict mode
export EMBEDDING_DIMENSION_STRICT=false
```

---

## Rollout Plan

1. **Phase 1**: Update default in config.py
2. **Phase 2**: Improve error message with actionable guidance
3. **Phase 3**: Update .env.example documentation
4. **Phase 4**: Add migration guide to docs

---

## Success Criteria

1. Dimension mismatches fail by default with clear error
2. Error message explains cause and fix
3. Skip mode still available via config for debugging
4. No silent data loss in standard operation

---

## Documentation Update

Add to `docs/configs/configuration-philosophy.md`:

```markdown
## Strict Mode Defaults

As of Spec 057, the following strict modes are enabled by default:

| Setting | Default | Purpose |
|---------|---------|---------|
| `EMBEDDING_DIMENSION_STRICT` | `true` | Fail if embedding dimensions don't match config |

These defaults follow the "fail loudly" principle from ANALYSIS-026. Silent failures
that corrupt research results are worse than explicit errors that block progress.

To debug dimension issues, temporarily set `EMBEDDING_DIMENSION_STRICT=false` and
check logs for skip warnings.
```
