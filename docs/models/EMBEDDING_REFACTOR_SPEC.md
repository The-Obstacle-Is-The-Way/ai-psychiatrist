# Embedding Backend Refactor Specification

> **Status**: IMPLEMENTED
> **Author**: Claude Code
> **Date**: 2024-12-24
> **Related**: MODEL_WIRING.md (design target), GH-46 (sampling params)

---

## Executive Summary

The codebase now uses a separate `EMBEDDING_BACKEND` to enable:
- Ollama for chat (fast, local, QAT model)
- HuggingFace for embeddings (FP16 precision, better similarity scores)

---

## Document Conventions

- **"Previous State"** = What the code did before refactor
- **"Implemented State"** = What is currently in the codebase
- Symbol references (e.g., `ModelSettings.embedding_model`) preferred over line numbers

---

## Previous State (Historical)

### Embedding Artifacts

```text
data/embeddings/
├── paper_reference_embeddings.npz   (101 MB)
└── paper_reference_embeddings.json  (2.9 MB)
```

**Provenance**: Participants match `paper_split_train.csv` (seeded 58/43/41 split). Model was **inferred** from script defaults (`qwen3-embedding:8b` via Ollama).

### Code Defaults

| Setting | Location | Default | Notes |
|---------|----------|---------|-------|
| `LLM_BACKEND` | `BackendSettings.backend` | `ollama` | Used for ALL clients |
| `embedding_model` | `ModelSettings.embedding_model` | `qwen3-embedding:8b` | Ollama model name |
| Chat models | `ModelSettings.*_model` | `gemma3:27b` | All 4 agents |
| `EMBEDDING_BACKEND` | N/A | N/A | **Was NOT IMPLEMENTED** |

---

## Implemented State

```bash
# Separate backends
LLM_BACKEND=ollama                    # Chat agents → Ollama
EMBEDDING_BACKEND=huggingface         # Embeddings → HuggingFace

# Model resolved via existing alias infrastructure
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b  # Canonical name, resolved per backend
```

### Why Separate Backends?

| Backend | Precision | Quality | Speed |
|---------|-----------|---------|-------|
| Ollama `qwen3-embedding:8b` | Q4_K_M (4-bit) | Good | Fast |
| HuggingFace `Qwen/Qwen3-Embedding-8B` | FP16 (16-bit) | Better | Slower |

Embedding precision matters more than chat precision because similarity scores are sensitive to numerical precision.

---

## Implementation Details

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
    default="qwen3-embedding:8b",  # Canonical name
    description="Embedding model (resolved to backend-specific ID)"
)
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

```python
# DataSettings.embeddings_path stays as a simple Path field
embeddings_path: Path = Field(
    default=Path("data/embeddings/paper_reference_embeddings.npz"),
    description="Pre-computed embeddings (NPZ + .json sidecar).",
)
```

#### 1.6 Add to `Settings` Aggregator

```python
class Settings(BaseSettings):
    # ... existing ...
    # Note: named `embedding_config` to avoid env var collision with `EMBEDDING_BACKEND`.
    embedding_config: EmbeddingBackendSettings = Field(default_factory=EmbeddingBackendSettings)
```

---

### Phase 2: Factory Changes

**File**: `src/ai_psychiatrist/infrastructure/llm/factory.py`

#### 2.1 Add `create_embedding_client()` Function

**CRITICAL**: Match actual constructor signatures. Use lazy imports for HF.

```python
from ai_psychiatrist.config import EmbeddingBackend
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingClient


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create embedding client based on EMBEDDING_BACKEND.

    Separate from create_llm_client() to allow different backends
    for chat vs embeddings.
    """
    backend = settings.embedding_config.backend

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

**Solution**: Write metadata to a separate `.meta.json` file.

**Metadata schema**:
```json
{
    "backend": "huggingface",
    "model": "Qwen/Qwen3-Embedding-8B",
    "model_canonical": "qwen3-embedding:8b",
    "dimension": 4096,
    "chunk_size": 8,
    "chunk_step": 2,
    "min_evidence_chars": 8,
    "split": "paper-train",
    "participant_count": 58,
    "generated_at": "2024-12-24T10:00:00Z",
    "generator_script": "scripts/generate_embeddings.py",
    "split_csv_hash": "abc123..."
}
```

#### 4.4 Filename Convention

```python
import re

def slugify_model(model: str) -> str:
    """Deterministic model name slugification."""
    # Qwen/Qwen3-Embedding-8B -> qwen3_8b
    # qwen3-embedding:8b -> qwen3_8b
    raw = model.split("/")[-1].lower()

    name_part, tag_part = raw, ""
    if ":" in raw:
        name_part, tag_part = raw.split(":", 1)

    base = name_part.replace("-embedding", "").replace("_embedding", "")
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    tag_part = re.sub(r"[^a-z0-9]+", "_", tag_part).strip("_")

    if tag_part and not base.endswith(f"_{tag_part}"):
        base = f"{base}_{tag_part}"

    return base


def get_output_filename(backend: str, model: str, split: str) -> str:
    """Generate standardized output filename.

    Format: {backend}_{model_slug}_{split}
    Example: huggingface_qwen3_8b_paper_train
    """
    model_slug = slugify_model(model)
    split_slug = split.replace("-", "_")
    return f"{backend}_{model_slug}_{split_slug}"
```

---

### Phase 5: Validation at Startup

**File**: `src/ai_psychiatrist/services/reference_store.py`

#### 5.1 Load and Validate Metadata

```python
def _load_embeddings(self) -> None:
    # Load optional metadata from .meta.json
    meta_path = self._npz_path.with_suffix(".meta.json")
    # ...
    # Validate if metadata exists
    if metadata:
        self._validate_metadata(metadata)
    # ...

def _validate_metadata(self, metadata: dict) -> None:
    """Validate embedding artifact matches current config."""
    # Checks backend, model, dimension, chunk_size, chunk_step, min_evidence_chars, split hash
    # Raises EmbeddingArtifactMismatchError on failure
```

---

## Migration Path

### 6.1 Keep Old Embeddings (Backward Compatible)

Existing `paper_reference_embeddings.*` files continue to work. No `.meta.json` = validation skipped.

### 6.2 Generate New Embeddings with Metadata

```bash
# Generate HuggingFace embeddings (will create .meta.json)
EMBEDDING_BACKEND=huggingface python scripts/generate_embeddings.py --split paper-train

# Output:
#   data/embeddings/huggingface_qwen3_8b_paper_train.npz
#   data/embeddings/huggingface_qwen3_8b_paper_train.json
#   data/embeddings/huggingface_qwen3_8b_paper_train.meta.json
```

### 6.3 Update Config to Use New Embeddings

```bash
# .env
EMBEDDING_BACKEND=huggingface
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train
```

---

**END OF SPEC - IMPLEMENTATION COMPLETE**
