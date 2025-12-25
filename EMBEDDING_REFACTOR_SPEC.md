# Embedding Backend Refactor Specification

> **Status**: REVISED - Incorporates senior review feedback (Gemini + Codex)
> **Author**: Claude Code
> **Date**: 2024-12-24
> **Related**: MODEL_WIRING.md (design target), GH-46 (sampling params)

---

## Executive Summary

The codebase currently uses a single `LLM_BACKEND` for both chat AND embeddings. This spec details how to implement a **separate** `EMBEDDING_BACKEND` to enable:
- Ollama for chat (fast, local, QAT model)
- HuggingFace for embeddings (FP16 precision, better similarity scores)

---

## Document Conventions

- **"Current State"** = What the code does TODAY (verified against source)
- **"Target State"** = What we're implementing (this spec)
- Symbol references (e.g., `ModelSettings.embedding_model`) preferred over line numbers

---

## Current State (Verified)

### Embedding Artifacts

```
data/embeddings/
├── paper_reference_embeddings.npz   (101 MB)
└── paper_reference_embeddings.json  (2.9 MB)
```

**Provenance**: Participants match `paper_split_train.csv` (seeded 58/43/41 split). Model is **inferred** from script defaults (`qwen3-embedding:8b` via Ollama). No metadata in artifact proves this - adding metadata is the mechanism that upgrades inference to proof.

**Note**: Paper-published participant IDs are unknown; our "paper-style" split is seeded but not confirmed to match the paper authors' exact membership.

### Code Defaults (Verified)

| Setting | Location | Default | Notes |
|---------|----------|---------|-------|
| `LLM_BACKEND` | `BackendSettings.backend` | `ollama` | Used for ALL clients |
| `embedding_model` | `ModelSettings.embedding_model` | `qwen3-embedding:8b` | Ollama model name |
| Chat models | `ModelSettings.*_model` | `gemma3:27b` | All 4 agents |
| `EMBEDDING_BACKEND` | N/A | N/A | **NOT IMPLEMENTED** |

### Hardcoded Ollama Usage (Impacted Call Sites)

| File | Symbol/Line | What It Does |
|------|-------------|--------------|
| `scripts/generate_embeddings.py` | `OllamaClient(ollama_settings)` (~L263) | Embedding generation |
| `scripts/reproduce_results.py` | `from ... import OllamaClient` (L68) | Reproduction pipeline |
| `scripts/reproduce_results.py` | `init_embedding_service(..., ollama_client)` (L593) | Few-shot embedding service |

### Existing Infrastructure (Leverage, Don't Duplicate)

| Component | Location | Purpose |
|-----------|----------|---------|
| `resolve_model_name()` | `model_aliases.py` | Canonical name → backend-specific ID |
| `HuggingFaceClient` | `huggingface.py` | Already implements `EmbeddingClient` protocol |
| `OllamaClient` | `ollama.py` | Already implements `EmbeddingClient` protocol |

---

## Target State

```bash
# Separate backends (what we're implementing)
LLM_BACKEND=ollama                    # Chat agents → Ollama
EMBEDDING_BACKEND=huggingface         # Embeddings → HuggingFace

# Model resolved via existing alias infrastructure
MODEL_EMBEDDING_MODEL=qwen3-embedding  # Canonical name, resolved per backend
```

### Why Separate Backends?

| Backend | Precision | Quality | Speed |
|---------|-----------|---------|-------|
| Ollama `qwen3-embedding:8b` | Q4_K_M (4-bit) | Good | Fast |
| HuggingFace `Qwen/Qwen3-Embedding-8B` | FP16 (16-bit) | Better | Slower |

Embedding precision matters more than chat precision because similarity scores are sensitive to numerical precision. Embeddings are generated once, stored, reused many times.

---

## Implementation Plan

### Phase 1: Config Changes

**File**: `src/ai_psychiatrist/config.py`

#### 1.1 Add `EmbeddingBackend` Enum

```python
class EmbeddingBackend(str, Enum):
    """Embedding backend selection (separate from LLM backend)."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
```

#### 1.2 Add `EmbeddingBackendSettings`

```python
class EmbeddingBackendSettings(BaseSettings):
    """Embedding backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.HUGGINGFACE,
        description="Embedding backend (huggingface for FP16, ollama for speed)"
    )
```

#### 1.3 Update `ModelSettings` (Use Alias Infrastructure)

```python
# Keep single canonical name, resolved via model_aliases.py
embedding_model: str = Field(
    default="qwen3-embedding",  # Canonical name
    description="Embedding model (resolved to backend-specific ID)"
)

# DEPRECATED: Don't add embedding_model_ollama/embedding_model_hf
# Use resolve_model_name(embedding_model, backend) instead
```

#### 1.4 Update `EmbeddingSettings`

```python
# Add embeddings artifact selection
embeddings_file: str = Field(
    default="paper_reference_embeddings",  # Keep existing default for backward compat
    description="Reference embeddings basename (no extension)"
)
```

#### 1.5 Keep `DataSettings.embeddings_path` Simple

**IMPORTANT**: DataSettings cannot depend on EmbeddingSettings (separate Pydantic models).

```python
# DataSettings.embeddings_path stays as a simple Path field
# ReferenceStore computes full path using both DataSettings and EmbeddingSettings
embeddings_path: Path = Field(
    default=Path("data/embeddings/paper_reference_embeddings.npz"),
    description="Pre-computed embeddings (NPZ + .json sidecar).",
)
```

The path computation logic moves to ReferenceStore or a helper function that has access to both settings.

#### 1.6 Add to `Settings` Aggregator

```python
class Settings(BaseSettings):
    # ... existing ...
    embedding_backend: EmbeddingBackendSettings = Field(default_factory=EmbeddingBackendSettings)
```

---

### Phase 2: Factory Changes

**File**: `src/ai_psychiatrist/infrastructure/llm/factory.py`

#### 2.1 Add `create_embedding_client()` Function

**CRITICAL**: Match actual constructor signatures. Use lazy imports for HF.

```python
from ai_psychiatrist.domain.enums import EmbeddingBackend
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create embedding client based on EMBEDDING_BACKEND.

    Separate from create_llm_client() to allow different backends
    for chat vs embeddings.
    """
    backend = settings.embedding_backend.backend

    if backend == EmbeddingBackend.OLLAMA:
        return OllamaClient(settings.ollama)

    if backend == EmbeddingBackend.HUGGINGFACE:
        # Lazy import to avoid requiring HF deps when using Ollama
        try:
            from ai_psychiatrist.infrastructure.llm.huggingface import HuggingFaceClient
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires: pip install 'ai-psychiatrist[hf]'"
            ) from e

        # HuggingFaceClient takes (backend_settings, model_settings)
        return HuggingFaceClient(settings.backend, settings.model)

    raise ValueError(f"Unknown embedding backend: {backend}")
```

---

### Phase 3: Server Wiring

**File**: `server.py`

#### 3.1 Create Separate Embedding Client + Proper Cleanup

**CRITICAL**: OllamaClient is async context manager; HuggingFaceClient is not.

```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Chat client (for agents)
    llm_client = create_llm_client(settings)

    # Embedding client (may be different backend)
    embedding_client = create_embedding_client(settings)

    # ... rest of initialization ...

    try:
        yield
    finally:
        # Close BOTH clients
        await llm_client.close()
        await embedding_client.close()
```

**Note**: Both OllamaClient and HuggingFaceClient have `async def close()`. Use that directly instead of relying on context managers, since HuggingFaceClient doesn't implement `__aenter__`/`__aexit__`.

---

### Phase 4: Script Updates

**Files**: `scripts/generate_embeddings.py`, `scripts/reproduce_results.py`

#### 4.1 Add Backend Selection

```python
parser.add_argument(
    "--backend",
    choices=["ollama", "huggingface"],
    default=None,  # Use EMBEDDING_BACKEND from env
    help="Override embedding backend"
)
```

#### 4.2 Use Factory + Handle Client Lifecycle

**CRITICAL**: Don't use `async with` for HuggingFaceClient (not a context manager).

```python
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client

# Create client
embedding_client = create_embedding_client(settings)

try:
    # ... generation logic using embedding_client.embed() ...
finally:
    await embedding_client.close()
```

#### 4.3 Store Metadata in SEPARATE File

**CRITICAL**: Adding `_metadata` key to the JSON sidecar will break ReferenceStore because it does `int(pid_str)` for all keys.

**Solution**: Write metadata to a separate `.meta.json` file.

```python
def save_embeddings_with_metadata(
    output_path: Path,
    npz_arrays: dict,
    json_texts: dict,
    metadata: dict,
) -> None:
    """Save embeddings with provenance metadata."""
    # NPZ file (unchanged)
    np.savez_compressed(str(output_path), **npz_arrays)

    # Text chunks JSON (unchanged format - no _metadata key)
    json_path = output_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(json_texts, f, indent=2)

    # Metadata in SEPARATE file
    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
```

**Metadata schema**:
```json
{
    "backend": "huggingface",
    "model": "Qwen/Qwen3-Embedding-8B",
    "model_revision": "main",
    "dimension": 4096,
    "chunk_size": 8,
    "chunk_step": 2,
    "split": "paper-train",
    "participant_count": 58,
    "generated_at": "2024-12-24T10:00:00Z",
    "generator_script": "scripts/generate_embeddings.py",
    "split_csv_hash": "abc123..."
}
```

#### 4.4 Filename Convention

```python
def slugify_model(model: str) -> str:
    """Deterministic model name slugification."""
    # Qwen/Qwen3-Embedding-8B -> qwen3_8b
    # qwen3-embedding:8b -> qwen3_8b
    slug = model.split("/")[-1].split(":")[0].lower()
    slug = slug.replace("-embedding", "").replace("_embedding", "")
    slug = slug.replace("-", "_")
    return slug


def get_output_filename(backend: str, model: str, split: str) -> str:
    """Generate standardized output filename.

    Format: {backend}_{model_slug}_{split}
    Example: hf_qwen3_8b_paper_train
    """
    model_slug = slugify_model(model)
    split_slug = split.replace("-", "_")
    return f"{backend}_{model_slug}_{split_slug}"
```

---

### Phase 5: Validation at Startup

**File**: `src/ai_psychiatrist/services/reference_store.py`

#### 5.1 Load and Validate Metadata

**CRITICAL**: Validate more than just backend - include model, dimension, chunk params.

```python
def _load_embeddings(self) -> None:
    # Load optional metadata from .meta.json
    meta_path = self._npz_path.with_suffix(".meta.json")
    metadata = {}
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)

    # Validate if metadata exists
    if metadata:
        self._validate_metadata(metadata)

    # Load JSON sidecar (no _metadata key - all keys are participant IDs)
    json_path = self._npz_path.with_suffix(".json")
    with json_path.open() as f:
        texts_data = json.load(f)

    # ... rest of loading logic (unchanged) ...


def _validate_metadata(self, metadata: dict) -> None:
    """Validate embedding artifact matches current config."""
    errors = []

    # Backend check
    stored_backend = metadata.get("backend")
    current_backend = self._embedding_backend.backend.value
    if stored_backend and stored_backend != current_backend:
        errors.append(
            f"backend mismatch: artifact='{stored_backend}', config='{current_backend}'"
        )

    # Dimension check
    stored_dim = metadata.get("dimension")
    current_dim = self._embedding_settings.dimension
    if stored_dim and stored_dim != current_dim:
        errors.append(
            f"dimension mismatch: artifact={stored_dim}, config={current_dim}"
        )

    # Chunk params check
    stored_chunk = metadata.get("chunk_size")
    current_chunk = self._embedding_settings.chunk_size
    if stored_chunk and stored_chunk != current_chunk:
        errors.append(
            f"chunk_size mismatch: artifact={stored_chunk}, config={current_chunk}"
        )

    if errors:
        raise EmbeddingArtifactMismatchError(
            f"Embedding artifact validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors) +
            f"\nRegenerate embeddings or update config to match."
        )
```

---

### Phase 6: Migration Path

#### 6.1 Keep Old Embeddings (Backward Compatible)

Existing `paper_reference_embeddings.*` files continue to work. No `.meta.json` = validation skipped.

#### 6.2 Generate New Embeddings with Metadata

```bash
# Generate HuggingFace embeddings (will create .meta.json)
EMBEDDING_BACKEND=huggingface python scripts/generate_embeddings.py --split paper-train

# Output:
#   data/embeddings/hf_qwen3_8b_paper_train.npz
#   data/embeddings/hf_qwen3_8b_paper_train.json
#   data/embeddings/hf_qwen3_8b_paper_train.meta.json
```

#### 6.3 Update Config to Use New Embeddings

```bash
# .env
EMBEDDING_BACKEND=huggingface
EMBEDDING_EMBEDDINGS_FILE=hf_qwen3_8b_paper_train
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/ai_psychiatrist/config.py` | Add `EmbeddingBackend` enum, `EmbeddingBackendSettings`, `EmbeddingSettings.embeddings_file` |
| `src/ai_psychiatrist/infrastructure/llm/factory.py` | Add `create_embedding_client()` with lazy HF import |
| `server.py` | Create separate embedding client, close both in lifespan |
| `scripts/generate_embeddings.py` | Add `--backend`, use factory, write `.meta.json` |
| `scripts/reproduce_results.py` | Use factory instead of hardcoded OllamaClient |
| `src/ai_psychiatrist/services/reference_store.py` | Load + validate `.meta.json` |
| `pyproject.toml` | Ensure `[project.optional-dependencies]` includes HF deps |

---

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| No `.meta.json` | Validation skipped, loads normally |
| `EMBEDDING_BACKEND` not set | Defaults to `huggingface` (paper-closest) |
| Legacy `MODEL_EMBEDDING_MODEL` env var | Still works via `ModelSettings.embedding_model` |
| Old `paper_reference_embeddings.*` files | Still loadable, just no validation |

---

## Testing Plan

### Unit Tests

1. `test_embedding_backend_enum` - enum values
2. `test_create_embedding_client_ollama` - factory creates OllamaClient
3. `test_create_embedding_client_hf` - factory creates HuggingFaceClient (mock import)
4. `test_create_embedding_client_hf_missing_deps` - raises ImportError with helpful message
5. `test_metadata_validation_pass` - matching config passes
6. `test_metadata_validation_fail_backend` - mismatched backend raises
7. `test_metadata_validation_fail_dimension` - mismatched dimension raises
8. `test_no_metadata_skips_validation` - legacy files load without error

### Integration Tests

1. Generate embeddings with Ollama, verify `.meta.json` contents
2. Generate embeddings with HuggingFace, verify `.meta.json` contents
3. Load with matching config (success)
4. Load with mismatched config (error with helpful message)

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Naming convention | `{backend}_{model_slug}_{split}` with deterministic slugify |
| Old files | Keep as-is (backward compatible), no forced rename |
| Validation strictness | Hard error on mismatch (fail fast) |
| Default backend | HuggingFace (paper-closest quality) |
| HuggingFace deps | Optional via `pip install 'ai-psychiatrist[hf]'` |
| Metadata storage | Separate `.meta.json` file (avoids ReferenceStore crash) |

---

## Appendix: Review Feedback Incorporated

### From Gemini Review

- ✅ Lazy imports for HuggingFace dependencies
- ✅ Backward compatibility for legacy `MODEL_EMBEDDING_MODEL`
- ✅ Check `pyproject.toml` optional deps

### From Codex Review (More Critical Issues)

- ✅ **ReferenceStore crash**: `int("_metadata")` would fail → use separate `.meta.json`
- ✅ **HuggingFaceClient not async context manager**: use `close()` directly, not `async with`
- ✅ **HuggingFaceClient constructor**: takes `(backend_settings, model_settings)`, not `(model_id=...)`
- ✅ **Phase 1.5 impossible dependency**: DataSettings can't access EmbeddingSettings
- ✅ **Model aliases**: leverage existing `resolve_model_name()` infrastructure
- ✅ **Impacted call sites**: added reproduce_results.py to list
- ✅ **Validation scope**: validate backend, dimension, chunk_size (not just backend)
- ✅ **Server cleanup**: close both clients in lifespan finally block
- ✅ **Provenance claim**: existing embeddings model is INFERRED, not proven

---

**END OF SPEC - READY FOR IMPLEMENTATION**
