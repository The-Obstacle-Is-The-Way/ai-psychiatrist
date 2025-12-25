# Embedding Backend Refactor Specification

> **Status**: DRAFT - Awaiting senior review
> **Author**: Claude Code
> **Date**: 2024-12-24
> **Related**: MODEL_WIRING.md, GH-46 (sampling params)

---

## Executive Summary

The codebase currently uses a single `LLM_BACKEND` for both chat AND embeddings. MODEL_WIRING.md specifies a **separate** `EMBEDDING_BACKEND` to enable:
- Ollama for chat (fast, local, QAT model)
- HuggingFace for embeddings (FP16 precision, better similarity scores)

This spec details exactly what must change to implement swappable embedding backends.

---

## Current State

### Files That Exist

```
data/embeddings/
├── paper_reference_embeddings.npz   (101 MB) - Legitimate, paper-split embeddings
└── paper_reference_embeddings.json  (2.9 MB) - Text chunks sidecar
```

**These embeddings are LEGITIMATE** - created from `paper-train` split using Ollama's `qwen3-embedding:8b` (Q4 quantized).

### Current Code Defaults

| Setting | File:Line | Default | Issue |
|---------|-----------|---------|-------|
| `embedding_model` | config.py:142-145 | `qwen3-embedding:8b` | Ollama model name |
| Chat models | config.py:129-141 | `gemma3:27b` | Not QAT version |
| `LLM_BACKEND` | config.py:54-57 | `ollama` | Used for EVERYTHING |
| `EMBEDDING_BACKEND` | N/A | N/A | **NOT IMPLEMENTED** |

### Current Script

`scripts/generate_embeddings.py`:
- **Hardcoded** to use `OllamaClient` (line 263)
- No backend selection flag
- No metadata stored about which backend/model created embeddings
- Output naming: `paper_reference_embeddings.npz` (no model identifier)

---

## Desired State (per MODEL_WIRING.md)

```bash
# Separate backends
LLM_BACKEND=ollama                    # Chat agents → Ollama
EMBEDDING_BACKEND=huggingface         # Embeddings → HuggingFace

# Model identifiers differ by backend
MODEL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B  # HuggingFace model ID
# vs
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b       # Ollama model name
```

### Why HuggingFace for Embeddings?

| Backend | Precision | Quality | Speed |
|---------|-----------|---------|-------|
| Ollama `qwen3-embedding:8b` | Q4_K_M (4-bit) | Good | Fast |
| HuggingFace `Qwen/Qwen3-Embedding-8B` | FP16 (16-bit) | Better | Slower |

Embedding precision matters more than chat precision because:
1. Similarity scores are sensitive to numerical precision
2. Embeddings are generated once, stored, reused many times
3. Paper likely used full-precision embeddings

---

## Naming Convention for Embeddings

### Problem

Current: `paper_reference_embeddings.npz` - no indication of which model/backend created it.

If we regenerate with HuggingFace, we need:
1. A way to distinguish old vs new embeddings
2. Config to select which embeddings to load
3. Validation that loaded embeddings match current backend

### Proposed Naming Scheme

```
data/embeddings/
├── ollama_qwen3_paper_train.npz      # Ollama Q4, paper split
├── ollama_qwen3_paper_train.json
├── hf_qwen3_paper_train.npz          # HuggingFace FP16, paper split
├── hf_qwen3_paper_train.json
└── [deprecated]/
    └── paper_reference_embeddings.*  # Old files, kept for reference
```

**Pattern**: `{backend}_{model_short}_{split}.{npz|json}`

### Config Setting

```python
# In EmbeddingSettings
embeddings_file: str = Field(
    default="hf_qwen3_paper_train",  # Base name, no extension
    description="Reference embeddings file (without .npz/.json extension)"
)
```

---

## Implementation Plan

### Phase 1: Config Changes

**File**: `src/ai_psychiatrist/config.py`

#### 1.1 Add `EmbeddingBackend` Enum

```python
# Near LLMBackend enum (around line 50)
class EmbeddingBackend(str, Enum):
    """Embedding backend selection."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
```

#### 1.2 Add `EmbeddingBackendSettings`

```python
# After BackendSettings class
class EmbeddingBackendSettings(BaseSettings):
    """Embedding backend configuration (separate from LLM backend)."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.HUGGINGFACE,  # Paper-closest: FP16
        description="Embedding backend (huggingface for FP16, ollama for speed)"
    )
```

#### 1.3 Update `ModelSettings`

```python
# In ModelSettings class
embedding_model_ollama: str = Field(
    default="qwen3-embedding:8b",
    description="Embedding model for Ollama backend"
)
embedding_model_hf: str = Field(
    default="Qwen/Qwen3-Embedding-8B",
    description="Embedding model for HuggingFace backend"
)

# Deprecate or alias old field
# embedding_model → computed property that returns correct one based on backend
```

#### 1.4 Update `EmbeddingSettings`

```python
# Add embeddings file selection
embeddings_file: str = Field(
    default="hf_qwen3_paper_train",
    description="Reference embeddings basename (no extension)"
)
```

#### 1.5 Update `DataSettings`

```python
# Make embeddings_path dynamic
@property
def embeddings_path(self) -> Path:
    # This should now use EmbeddingSettings.embeddings_file
    return self.base_dir / "embeddings" / f"{self.embeddings_file}.npz"
```

---

### Phase 2: Factory Changes

**File**: `src/ai_psychiatrist/infrastructure/llm/factory.py`

#### 2.1 Add `create_embedding_client()` Function

```python
def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Create embedding client based on EMBEDDING_BACKEND.

    Separate from create_llm_client() to allow different backends
    for chat vs embeddings.
    """
    backend = settings.embedding_backend.backend  # NEW setting

    if backend == EmbeddingBackend.OLLAMA:
        return OllamaClient(settings.ollama)

    if backend == EmbeddingBackend.HUGGINGFACE:
        # Import model name for HF
        model = settings.model.embedding_model_hf
        return HuggingFaceClient(
            model_id=model,
            device=settings.huggingface.device,
            # ... other HF settings
        )

    raise ValueError(f"Unknown embedding backend: {backend}")
```

---

### Phase 3: Server Wiring

**File**: `server.py`

#### 3.1 Create Separate Embedding Client

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Chat client (for agents)
    llm_client = create_llm_client(settings)

    # Embedding client (may be different backend)
    embedding_client = create_embedding_client(settings)  # NEW

    reference_store = ReferenceStore(settings.data, settings.embedding)
    app.state.embedding_service = EmbeddingService(
        llm_client=embedding_client,  # Use embedding-specific client
        reference_store=reference_store,
        settings=settings.embedding,
        model_settings=app.state.model_settings,
    )
```

---

### Phase 4: Script Updates

**File**: `scripts/generate_embeddings.py`

#### 4.1 Add Backend Selection

```python
parser.add_argument(
    "--backend",
    choices=["ollama", "huggingface"],
    default=None,  # Use EMBEDDING_BACKEND from env
    help="Override embedding backend"
)
```

#### 4.2 Use Factory Instead of Hardcoded Ollama

```python
# Replace line 263:
# async with OllamaClient(ollama_settings) as client:

# With:
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client

embedding_client = create_embedding_client(settings)
async with embedding_client as client:
    # ... rest of generation logic
```

#### 4.3 Auto-Generate Output Filename with Metadata

```python
def get_output_filename(backend: str, model: str, split: str) -> str:
    """Generate standardized output filename.

    Format: {backend}_{model_short}_{split}
    Example: hf_qwen3_paper_train
    """
    model_short = model.split("/")[-1].split(":")[0].lower()
    model_short = model_short.replace("-embedding", "").replace("_embedding", "")
    return f"{backend}_{model_short}_{split}"
```

#### 4.4 Store Metadata in JSON Sidecar

```python
# In the JSON output, add metadata header
json_output = {
    "_metadata": {
        "backend": "huggingface",  # or "ollama"
        "model": "Qwen/Qwen3-Embedding-8B",
        "dimension": 4096,
        "chunk_size": 8,
        "chunk_step": 2,
        "generated_at": "2024-12-24T10:00:00Z",
        "split": "paper-train",
        "participant_count": 58,
    },
    "302": ["chunk1", "chunk2", ...],
    "304": [...],
    # ...
}
```

---

### Phase 5: Validation at Startup

**File**: `src/ai_psychiatrist/services/reference_store.py`

#### 5.1 Add Backend Validation

```python
def _load_embeddings(self) -> None:
    # Load JSON sidecar
    json_path = self._npz_path.with_suffix(".json")
    with json_path.open() as f:
        data = json.load(f)

    # Check metadata if present
    metadata = data.get("_metadata", {})
    if metadata:
        stored_backend = metadata.get("backend")
        current_backend = self._settings.backend.value  # from config

        if stored_backend and stored_backend != current_backend:
            raise EmbeddingBackendMismatchError(
                f"Embeddings were created with '{stored_backend}' but "
                f"EMBEDDING_BACKEND is '{current_backend}'. "
                f"Regenerate embeddings or change EMBEDDING_BACKEND."
            )
```

---

### Phase 6: Migration Path for Existing Embeddings

#### 6.1 Keep Old Embeddings (Deprecated)

```bash
# Rename existing files to deprecated namespace
mv data/embeddings/paper_reference_embeddings.npz \
   data/embeddings/deprecated_ollama_qwen3_paper_train.npz
mv data/embeddings/paper_reference_embeddings.json \
   data/embeddings/deprecated_ollama_qwen3_paper_train.json
```

#### 6.2 Generate New HuggingFace Embeddings

```bash
# With new script
EMBEDDING_BACKEND=huggingface python scripts/generate_embeddings.py \
    --split paper-train \
    --output data/embeddings/hf_qwen3_paper_train.npz
```

#### 6.3 Update Default Config

```python
# In EmbeddingSettings
embeddings_file: str = Field(
    default="hf_qwen3_paper_train",  # New default
    ...
)
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/ai_psychiatrist/config.py` | Add `EmbeddingBackend` enum, `EmbeddingBackendSettings`, update `ModelSettings`, `EmbeddingSettings`, `DataSettings` |
| `src/ai_psychiatrist/infrastructure/llm/factory.py` | Add `create_embedding_client()` function |
| `server.py` | Use separate embedding client |
| `scripts/generate_embeddings.py` | Add `--backend` flag, use factory, auto-generate filenames, store metadata |
| `src/ai_psychiatrist/services/reference_store.py` | Add backend validation on load |
| `data/embeddings/` | Rename old files, generate new ones |
| `MODEL_WIRING.md` | Update status from "not implemented" to "implemented" |

---

## Backwards Compatibility

### For Users with Existing Setup

1. **Old embeddings still work** - if `EMBEDDING_BACKEND=ollama` and pointing to old files
2. **Warning on mismatch** - if loading embeddings created with different backend
3. **Clear migration path** - documented steps to regenerate

### Environment Variable Defaults

```bash
# Default (after implementation)
EMBEDDING_BACKEND=huggingface
EMBEDDING_EMBEDDINGS_FILE=hf_qwen3_paper_train

# To use old Ollama embeddings
EMBEDDING_BACKEND=ollama
EMBEDDING_EMBEDDINGS_FILE=deprecated_ollama_qwen3_paper_train
```

---

## Testing Plan

### Unit Tests

1. `test_embedding_backend_enum` - enum values work
2. `test_create_embedding_client_ollama` - factory creates OllamaClient
3. `test_create_embedding_client_hf` - factory creates HuggingFaceClient
4. `test_backend_mismatch_error` - ReferenceStore raises on mismatch

### Integration Tests

1. Generate embeddings with Ollama backend
2. Generate embeddings with HuggingFace backend
3. Load embeddings with matching backend (success)
4. Load embeddings with mismatched backend (error)

### E2E Tests

1. Full pipeline with HuggingFace embeddings
2. Compare MAE: Ollama embeddings vs HuggingFace embeddings

---

## Open Questions for Review

1. **Naming convention**: Is `{backend}_{model_short}_{split}` clear enough?
2. **Old files**: Delete immediately or keep in `deprecated/` folder?
3. **Validation strictness**: Hard error on mismatch, or warning + continue?
4. **Default backend**: HuggingFace (paper-closest) or Ollama (faster)?
5. **HuggingFace dependencies**: Should they be optional (extras)?

---

## Appendix A: What the Paper Actually Used

From `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`:

> "For the embedding-based few-shot prompting approach, we used **Qwen 3 8B Embedding** due to its superior performance on the Massive Text Embedding Benchmark (MTEB) leaderboard."

The paper doesn't specify Ollama vs HuggingFace, but:
- MTEB benchmarks use full-precision models
- Paper's A100 SLURM scripts suggest they ran at higher precision
- HuggingFace FP16 is closest to paper's likely setup

---

## Appendix B: Quick Reference Commands

```bash
# Check current embedding files
ls -la data/embeddings/

# Generate with Ollama (fast, existing default)
EMBEDDING_BACKEND=ollama python scripts/generate_embeddings.py --split paper-train

# Generate with HuggingFace (paper-closest quality)
EMBEDDING_BACKEND=huggingface python scripts/generate_embeddings.py --split paper-train

# Verify which model created existing embeddings
python -c "import json; print(json.load(open('data/embeddings/paper_reference_embeddings.json')).get('_metadata', 'No metadata'))"
```

---

**END OF SPEC - AWAITING SENIOR REVIEW**
