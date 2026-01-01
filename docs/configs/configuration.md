# Configuration Reference

Complete reference for all AI Psychiatrist configuration options.

---

## Overview

Configuration is managed via Pydantic Settings with three sources (in priority order):

1. **Environment variables** (highest priority)
2. **`.env` file** (recommended for development)
3. **Code defaults** (paper-optimal values)

```bash
# Copy template and customize
cp .env.example .env
```

---

## Configuration Groups

### LLM Backend Settings

Selects which runtime implementation is used for chat.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_BACKEND` | string | `ollama` | Backend: `ollama` (local HTTP) or `huggingface` (Transformers) |
| `LLM_HF_DEVICE` | string | `auto` | HuggingFace device: `auto`, `cpu`, `cuda`, `mps` |
| `LLM_HF_QUANTIZATION` | string | *(unset)* | Optional HuggingFace quantization: `int4` or `int8` |
| `LLM_HF_CACHE_DIR` | path | *(unset)* | Optional HuggingFace cache directory |
| `LLM_HF_TOKEN` | string | *(unset)* | Optional HuggingFace token (prefer `huggingface-cli login`) |

**Notes:**
- HuggingFace dependencies are optional; install with `make dev-hf` (or `pip install 'ai-psychiatrist[hf]'`).
- Canonical model names like `gemma3:27b` are resolved to backend-specific IDs when possible.
- Official MedGemma weights are HuggingFace-only; there is no official MedGemma in the Ollama library.
- The `LLM_HF_*` settings are used when HuggingFace is selected for either chat (`LLM_BACKEND=huggingface`) or embeddings (`EMBEDDING_BACKEND=huggingface`).

**Example:**
```bash
LLM_BACKEND=huggingface
LLM_HF_DEVICE=mps
MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

### Embedding Backend Settings

Selects which runtime implementation is used for embeddings (separate from `LLM_BACKEND`).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_BACKEND` | string | `huggingface` | Embedding backend: `ollama` (fast, local) or `huggingface` (FP16/BF16 precision) |

### Ollama Settings

Connection settings for the Ollama LLM server.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_HOST` | string | `127.0.0.1` | Ollama server hostname |
| `OLLAMA_PORT` | int | `11434` | Ollama server port |
| `OLLAMA_TIMEOUT_SECONDS` | int | `600` | Request timeout (min 10s). Recommend `3600` for slow GPU research runs. |

**Derived properties:**
- `base_url`: `http://{host}:{port}`
- `chat_url`: `{base_url}/api/chat`
- `embeddings_url`: `{base_url}/api/embeddings`

**Timeout Notes:**
- Default 600s may still timeout on very slow GPUs / long transcripts; use `3600` for research runs.
- `OLLAMA_TIMEOUT_SECONDS` applies to the legacy Ollama client and (by default) syncs to the Pydantic AI path if `PYDANTIC_AI_TIMEOUT_SECONDS` is unset.
- Timeout sync is implemented in `Settings.validate_consistency()` in `src/ai_psychiatrist/config.py`.

**Example:**
```bash
# Remote Ollama server with generous timeout
OLLAMA_HOST=192.168.1.100
OLLAMA_PORT=11434
OLLAMA_TIMEOUT_SECONDS=3600  # 1 hour for research runs
```

---

### Model Settings

LLM model selection and sampling parameters.

| Variable | Type | Default | Paper Reference |
|----------|------|---------|-----------------|
| `MODEL_QUALITATIVE_MODEL` | string | `gemma3:27b` | Section 2.2 |
| `MODEL_JUDGE_MODEL` | string | `gemma3:27b` | Section 2.2 |
| `MODEL_META_REVIEW_MODEL` | string | `gemma3:27b` | Section 2.2 |
| `MODEL_QUANTITATIVE_MODEL` | string | `gemma3:27b` | Section 2.2 (MedGemma in Appendix F) |
| `MODEL_EMBEDDING_MODEL` | string | `qwen3-embedding:8b` | Section 2.2 |
| `MODEL_TEMPERATURE` | float | `0.0` | Clinical AI best practice ([Issue #46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46)) |

**Sampling Parameters (Evidence-Based)**:

All agents use `temperature=0.0`. We do NOT set `top_k` or `top_p` because:
1. At temp=0, they're irrelevant (greedy decoding)
2. Best practice: "use temperature only, not both" ([Anthropic](https://www.prompthub.us/blog/using-anthropic-best-practices-parameters-and-large-context-windows))
3. Claude APIs error if you set both temp and top_p

See [Agent Sampling Registry](../reference/agent-sampling-registry.md) for full rationale with citations

**Model Options:**

| Model | Size | Use Case | Performance |
|-------|------|----------|-------------|
| `gemma3:27b-it-qat` | ~17GB | All agents (Ollama recommended) | QAT 4-bit variant (same size, better quality/speed vs standard Q4) |
| `gemma3:27b` | ~16GB | All agents (default) | Paper Section 2.2 |
| `medgemma:27b` | ~16GB | Quantitative (HuggingFace only) | Appendix F, 18% better MAE but more N/A |
| `qwen3-embedding:8b` | ~4GB | Embeddings | Paper standard |

**Note**: `gemma3:27b-it-qat` is an Ollama tag; use it only with `LLM_BACKEND=ollama`. For HuggingFace, use canonical `gemma3:27b` (resolved to `google/gemma-3-27b-it`).

> **Note**: MedGemma is not available in Ollama officially. Use HuggingFace backend for official weights.
> See [Model Registry](../models/model-registry.md) for HuggingFace setup.

**Precision Comparison (Ollama vs HuggingFace):**

| Model | Ollama Precision | HuggingFace Precision | Impact |
|-------|------------------|----------------------|--------|
| `gemma3:27b` | Q4_K_M (4-bit) | FP16/BF16 (16-bit) | Higher quality responses |
| `qwen3-embedding:8b` | Q4_K_M (4-bit) | FP16/BF16 (16-bit) | More accurate similarity matching |

For best chat quality, use `LLM_BACKEND=huggingface`.
For best embedding quality (similarity), use `EMBEDDING_BACKEND=huggingface` (default).

**Example:**
```bash
# Canonical names (recommended): resolved per backend
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=gemma3:27b

# HuggingFace backend + MedGemma (Appendix F evaluation)
LLM_BACKEND=huggingface
MODEL_QUANTITATIVE_MODEL=medgemma:27b

# Clinical AI: temp=0 for reproducibility
MODEL_TEMPERATURE=0.0
```

---

### Embedding Settings

Few-shot retrieval configuration.

| Variable | Type | Default | Paper Reference |
|----------|------|---------|-----------------|
| `EMBEDDING_DIMENSION` | int | `4096` | Appendix D (optimal) |
| `EMBEDDING_CHUNK_SIZE` | int | `8` | Appendix D (optimal) |
| `EMBEDDING_CHUNK_STEP` | int | `2` | Section 2.4.2 |
| `EMBEDDING_TOP_K_REFERENCES` | int | `2` | Appendix D (optimal) |
| `EMBEDDING_MIN_EVIDENCE_CHARS` | int | `8` | Minimum text for embedding |
| `EMBEDDING_EMBEDDINGS_FILE` | string | `huggingface_qwen3_8b_paper_train` | Reference embeddings basename (no extension), resolved under `{DATA_BASE_DIR}/embeddings/` |
| `EMBEDDING_ENABLE_RETRIEVAL_AUDIT` | bool | `false` | Spec 32 (retrieval audit logging) |
| `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING` | bool | `true` | Spec 37 (batch query embedding; performance-only) |
| `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS` | int | `300` | Spec 37 (query embedding timeout; stability-only) |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | float | `0.0` | Spec 33 (drop low-similarity references; 0 disables) |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | int | `0` | Spec 33 (per-item reference context budget; 0 disables) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | bool | `false` | Spec 34 (filter refs by item tags; requires `{name}.tags.json`) |
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | string | `participant` | Spec 35: `participant` (paper-parity) or `chunk` (experimental) |
| `EMBEDDING_ALLOW_CHUNK_SCORES_PROMPT_HASH_MISMATCH` | bool | `false` | Spec 35 circularity control bypass (unsafe) |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | bool | `false` | Spec 36 (CRAG-style runtime validation; adds LLM calls) |
| `EMBEDDING_VALIDATION_MODEL` | string | *(unset)* | Spec 36 validation model (if unset, runners fall back to `MODEL_JUDGE_MODEL`) |
| `EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM` | int | `2` | Spec 36 max accepted refs per item after validation |

**Note on artifact naming**: `scripts/generate_embeddings.py` defaults to writing a namespaced artifact like
`data/embeddings/{backend}_{model_slug}_{split}.npz`. After generating, set `EMBEDDING_EMBEDDINGS_FILE` to that basename
(or pass `--output` to write to `paper_reference_embeddings.npz`).

**Optional item tags (Spec 34)**: `scripts/generate_embeddings.py --write-item-tags` writes a sibling `{name}.tags.json`
sidecar. At runtime, enable tag-based filtering with `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`.

**Chunk-level scoring (Spec 35)**: By default, retrieved chunks carry the participant's overall PHQ-8 score. Set
`EMBEDDING_REFERENCE_SCORE_SOURCE=chunk` to use per-chunk scores (requires `scripts/score_reference_chunks.py` output).
**WARNING**: This is NOT paper-parity; label runs as "experimental".

**CRAG validation (Spec 36)**: Set `EMBEDDING_ENABLE_REFERENCE_VALIDATION=true` to have the LLM validate each retrieved
reference at runtime (CRAG-style). Adds latency but filters irrelevant references.

**Paper optimization results (Appendix D):**
- Embedding dimension 4096 performed best among the tested dimensions (64, 256, 1024, 4096)
- Chunk size 8 optimal for clinical interviews
- Top-k=2 references balances context and noise

**Example:**
```bash
# More references for difficult cases
EMBEDDING_TOP_K_REFERENCES=3

# Larger chunks for longer utterances
EMBEDDING_CHUNK_SIZE=10
EMBEDDING_CHUNK_STEP=3
```

---

### Feedback Loop Settings

Iterative refinement configuration.

| Variable | Type | Default | Paper Reference |
|----------|------|---------|-----------------|
| `FEEDBACK_ENABLED` | bool | `true` | Enable/disable refinement |
| `FEEDBACK_MAX_ITERATIONS` | int | `10` | Section 2.3.1 |
| `FEEDBACK_SCORE_THRESHOLD` | int | `3` | Scores ≤3 trigger refinement |
| `FEEDBACK_TARGET_SCORE` | int | `4` | Minimum acceptable score |

**Threshold logic:**
- Score ≤ `threshold` (default 3) → needs improvement
- Score ≥ `target` (default 4) → acceptable

**Example:**
```bash
# Disable feedback loop for faster inference
FEEDBACK_ENABLED=false

# More strict quality requirements
FEEDBACK_SCORE_THRESHOLD=3
FEEDBACK_MAX_ITERATIONS=15
```

---

### Data Settings

File path configuration.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_BASE_DIR` | path | `data` | Base data directory |
| `DATA_TRANSCRIPTS_DIR` | path | `data/transcripts` | Transcript files |
| `DATA_EMBEDDINGS_PATH` | path | `data/embeddings/huggingface_qwen3_8b_paper_train.npz` | Full-path override for reference embeddings (takes precedence over `EMBEDDING_EMBEDDINGS_FILE`) |
| `DATA_TRAIN_CSV` | path | `data/train_split_Depression_AVEC2017.csv` | Training ground truth |
| `DATA_DEV_CSV` | path | `data/dev_split_Depression_AVEC2017.csv` | Development ground truth |

**Directory structure expected:**
```text
	data/
	├── transcripts/
	│   ├── 300_P/
	│   │   └── 300_TRANSCRIPT.csv
	│   └── .../
	├── embeddings/
	│   ├── huggingface_qwen3_8b_paper_train.npz         # default reference knowledge base (paper-train)
	│   ├── huggingface_qwen3_8b_paper_train.json
	│   ├── huggingface_qwen3_8b_paper_train.meta.json   # provenance metadata (backend/model/dim/chunking)
	│   ├── huggingface_qwen3_8b_paper_train.tags.json   # optional per-chunk PHQ-8 item tags (Spec 34)
	│   ├── paper_reference_embeddings.npz               # legacy/compat filename (paper-train)
	│   ├── paper_reference_embeddings.json
	│   └── paper_reference_embeddings.meta.json         # provenance metadata (legacy/compat)
	├── train_split_Depression_AVEC2017.csv
	└── dev_split_Depression_AVEC2017.csv
```

**Example:**
```bash
# Custom data location
DATA_BASE_DIR=/mnt/datasets/daic-woz
DATA_TRANSCRIPTS_DIR=/mnt/datasets/daic-woz/transcripts
```

---

### Logging Settings

Structured logging configuration.

| Variable | Type | Default | Options |
|----------|------|---------|---------|
| `LOG_LEVEL` | string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT` | string | `json` | `json`, `console` |
| `LOG_INCLUDE_TIMESTAMP` | bool | `true` | Add timestamp to logs |
| `LOG_INCLUDE_CALLER` | bool | `true` | Add file:line info |

**Formats:**
- `json`: Structured JSON for production/parsing
- `console`: Human-readable for development

**Example:**
```bash
# Debug mode with readable output
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

**Sample output:**
```json
{"event": "Starting qualitative assessment", "participant_id": 300, "word_count": 1234, "level": "info", "timestamp": "2025-12-21T10:00:00Z"}
```

---

### API Settings

HTTP server configuration.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_HOST` | string | `0.0.0.0` | Bind address |
| `API_PORT` | int | `8000` | Server port |
| `API_RELOAD` | bool | `false` | Hot reload (dev only) |
| `API_WORKERS` | int | `1` | Worker processes (1-16) |
| `API_CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins |

`API_CORS_ORIGINS` exists in configuration, but `server.py` does not currently install
FastAPI/Starlette `CORSMiddleware`. If you need CORS today, configure it at a reverse proxy
(recommended) or add `CORSMiddleware` in `server.py`.

**Example:**
```bash
# Production settings
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4
API_CORS_ORIGINS=["https://myapp.com"]

# Development settings
API_RELOAD=true
API_WORKERS=1
```

---

### Quantitative Assessment Settings

These settings control the quantitative assessment behavior (evidence extraction + scoring):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` | bool | `false` | Enable keyword backfill for evidence extraction (paper parity: OFF) |
| `QUANTITATIVE_TRACK_NA_REASONS` | bool | `true` | Track why items return N/A |
| `QUANTITATIVE_KEYWORD_BACKFILL_CAP` | int | `3` | Max keyword-matched sentences per domain |

**Default Behavior (Paper Parity):**

Keyword backfill is **OFF by default** to match paper methodology (~50% coverage).

> ⚠️ **Deprecated**: Keyword backfill is a historical ablation feature. Enabling it is not recommended for new runs. See `POST-ABLATION-DEFAULTS.md` in the project root for rationale.

See [Paper Parity Guide](../guides/paper-parity-guide.md) for reproduction guidance.

---

### Feature Flags

System-wide toggles.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_FEW_SHOT` | bool | `true` | Use embedding-based few-shot |

**Note:** `ENABLE_FEW_SHOT=true` requires pre-computed embeddings (resolved from `DATA_EMBEDDINGS_PATH` or `EMBEDDING_EMBEDDINGS_FILE`).

---

### Pydantic AI Settings

Structured validation + automatic retries for agent outputs (Spec 13).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PYDANTIC_AI_ENABLED` | bool | `true` | Enable Pydantic AI `TextOutput` validation + retry loop |
| `PYDANTIC_AI_RETRIES` | int | `3` | Retry count when validation fails (`0` disables retries) |
| `PYDANTIC_AI_TIMEOUT_SECONDS` | float | unset | Timeout override for Pydantic AI calls (unset = library default) |

**Notes:**
- This preserves existing prompt formats (e.g., `<thinking>...</thinking>` + `<answer>...</answer>`) and adds validation after generation.
- Legacy parsing fallbacks are disabled (fail-fast research behavior). If `PYDANTIC_AI_ENABLED=false`, agents will raise because no legacy path exists.

**Timeout Notes (BUG-027):**
- Unset `PYDANTIC_AI_TIMEOUT_SECONDS` uses the pydantic_ai library default (600s).
- Set `PYDANTIC_AI_TIMEOUT_SECONDS=3600` for 1-hour research runs on throttled GPUs.
- If only one of `{PYDANTIC_AI_TIMEOUT_SECONDS, OLLAMA_TIMEOUT_SECONDS}` is set, Settings syncs the other to match; if both are set and differ, a warning is emitted.

---

## Nested Delimiter

Most configuration uses the explicit group prefixes shown above (e.g., `MODEL_TEMPERATURE`,
`OLLAMA_HOST`). For advanced settings management, Pydantic also supports nested environment
variables using double underscores:

```bash
# Set nested values
MODEL__TEMPERATURE=0.5
EMBEDDING__TOP_K_REFERENCES=3
```

---

## `.env.example`

See the repo-root `.env.example` for an up-to-date template, including:
- Separate `LLM_BACKEND` (chat) and `EMBEDDING_BACKEND` (embeddings)
- Reference embeddings selection via `EMBEDDING_EMBEDDINGS_FILE` / `DATA_EMBEDDINGS_PATH`

---

## Programmatic Access

```python
from ai_psychiatrist.config import get_settings, Settings

# Get singleton settings
settings = get_settings()

# Access nested groups
print(settings.ollama.base_url)
print(settings.model.quantitative_model)
print(settings.embedding.dimension)
print(settings.feedback.max_iterations)

# Direct instantiation (for testing)
custom = Settings(
    ollama=OllamaSettings(host="custom-host"),
    model=ModelSettings(temperature=0.0),
)
```

---

## Validation

Settings are validated on load:

```python
# Port range validation
OLLAMA_PORT=99999  # Error: ge=1, le=65535

# Temperature validation
MODEL_TEMPERATURE=3.0  # Error: ge=0.0, le=2.0

# Chunk size validation
EMBEDDING_CHUNK_SIZE=1  # Error: ge=2, le=20
```

**Warnings:**
- Missing data directories log warnings but don't fail
- Few-shot enabled without embeddings logs warning

---

## Environment-Specific Configs

### Development
```bash
LOG_LEVEL=DEBUG
LOG_FORMAT=console
API_RELOAD=true
FEEDBACK_MAX_ITERATIONS=3  # Faster iteration
```

### Testing
```bash
# Tests automatically set TESTING=1 which skips .env loading
# Use code defaults for reproducibility
```

### Production
```bash
LOG_LEVEL=INFO
LOG_FORMAT=json
API_WORKERS=4
API_CORS_ORIGINS=["https://production-domain.com"]
OLLAMA_TIMEOUT_SECONDS=600
```

---

## See Also

- [Quickstart](../getting-started/quickstart.md) - Initial setup
- [Architecture](../architecture/architecture.md) - How settings are used
- `.env.example` (repository root) - Environment template
