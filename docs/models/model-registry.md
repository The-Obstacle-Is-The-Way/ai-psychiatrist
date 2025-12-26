# AI Psychiatrist Model Registry

Last Updated: 2025-12-26
Purpose: Paper-aligned, reproducible model configuration for this repo.

---

## Quick Reference: Which Setup Should I Use?

Chat and embeddings can be configured separately via:
- `LLM_BACKEND` (chat models for agents)
- `EMBEDDING_BACKEND` (embeddings only)

| Setup | Chat Backend (`LLM_BACKEND`) | Embedding Backend (`EMBEDDING_BACKEND`) | Quality | Hardware Needed | Use Case |
|-------|------------------------------|----------------------------------------|---------|-----------------|----------|
| **Default (Recommended)** | Ollama | HuggingFace | Better similarity | 16GB+ RAM + HF deps | Paper-parity chat + FP16 embeddings |
| **Paper Parity (Pure Ollama)** | Ollama | Ollama | Good | Any Mac/Linux | Reproduce paper results without HF deps |
| **High Quality (Full HF)** | HuggingFace | HuggingFace | Best | 32GB+ RAM, CUDA/MPS | Best possible MAE |
| **Development** | Ollama | Ollama | Fast | Any | Quick iteration |

Note: The codebase intentionally fails fast when a configured backend can’t run; there is no automatic fallback (see `model-wiring.md`).

---

## Paper-Optimal Models (Reproduction)

These models match the paper's methodology and are required for paper-accurate runs.

| Role | Model family | Params | Ollama tag | Paper reference | Notes |
|------|--------------|--------|------------|-----------------|-------|
| Qualitative Agent | Gemma 3 | 27B | `gemma3:27b` or `gemma3:27b-it-qat` | Section 2.2 | Used for qualitative assessment |
| Judge Agent | Gemma 3 | 27B | `gemma3:27b` or `gemma3:27b-it-qat` | Section 2.2 | Used for feedback loop |
| Meta-Review Agent | Gemma 3 | 27B | `gemma3:27b` or `gemma3:27b-it-qat` | Section 2.2 | Used for final review |
| Quantitative Agent | Gemma 3 | 27B | `gemma3:27b` or `gemma3:27b-it-qat` | Section 2.2 | **Default** (see MedGemma note below) |
| Embedding | Qwen3 Embedding | 8B | `qwen3-embedding:8b` | Section 2.2 | 4096-dim embeddings (Appendix D) |

### Quantization Note

The paper authors likely used full-precision BF16 weights. Both Ollama variants are quantized:
- `gemma3:27b` - Standard Ollama GGUF quantization (Q4_K_M)
- `gemma3:27b-it-qat` - QAT (Quantization-Aware Training) optimized, faster inference

Both are acceptable for reproduction. Use `-it-qat` for faster runs, or `27b` for closer naming parity with the paper.

Approximate disk for paper-optimal pulls: ~32 GB.

### MedGemma Note (Appendix F)

The paper's Appendix F evaluates MedGemma 27B as an **alternative** for the quantitative agent:
- **Better item-level MAE**: 0.505 vs 0.619 (18% improvement)
- **BUT produces more N/A**: "fewer predictions overall" - conservative on uncertain evidence

**⚠️ Warning**: There is NO official MedGemma in Ollama. The `alibayram/medgemma:27b` is a community upload with Q4_K_M quantization that may behave differently from official weights.

For official MedGemma, use **HuggingFace** (see below).

## Ollama Compatibility Notes

- `qwen3-embedding:8b` supports `/api/embeddings` and returns 4096 dimensions.
- The legacy tag `dengcao/Qwen3-Embedding-8B:Q8_0` does not support `/api/embeddings` in current Ollama. Avoid it for production.
- If you switch embedding models, update `EMBEDDING_DIMENSION` to match the model output.

## Development / Local Alternatives (Optional)

Use these for fast local testing only. They do not reproduce paper metrics.

| Role | Model | Params | Ollama tag | Embedding dim |
|------|-------|--------|------------|---------------|
| All Agents (chat) | Gemma 2 | 9B | `gemma2:9b` | - |
| Embedding (fast) | mxbai-embed-large | 335M | `mxbai-embed-large` | 1024 |
| Embedding (small) | Nomic Embed Text | 137M | `nomic-embed-text` | 768 |

## Installation Commands

### Ollama (Paper-optimal)

```bash
# Recommended (QAT-optimized, faster):
ollama pull gemma3:27b-it-qat
ollama pull qwen3-embedding:8b

# Alternative (standard quantization):
ollama pull gemma3:27b
```

### Ollama (Development - smaller/faster)

```bash
ollama pull gemma2:9b
ollama pull mxbai-embed-large
ollama pull nomic-embed-text
```

---

## HuggingFace Backend (Official Models)

For accessing official Google models (including MedGemma), use HuggingFace Transformers.

### Official Model IDs

| Canonical Name | HuggingFace Model ID | Access | Notes |
|----------------|---------------------|--------|-------|
| `gemma3:27b` | `google/gemma-3-27b-it` | Open | Instruction-tuned; loaded via Transformers `AutoModelForCausalLM` in this repo |
| `medgemma:27b` | `google/medgemma-27b-text-it` | **Gated** | Text-only, use `AutoModelForCausalLM` |
| `qwen3-embedding:8b` | `Qwen/Qwen3-Embedding-8B` | Open | Use `SentenceTransformer` (see model card for evaluation details) |

### HuggingFace Installation

```bash
# Install the optional HuggingFace backend dependencies:
make dev-hf
# Or, if installing via pip:
pip install "ai-psychiatrist[hf]"
```

**Optional (quantization):**
- `int8` quantization requires `bitsandbytes` support on your platform.

### MedGemma Access (Gated Model)

MedGemma requires accepting Google's Health AI Developer Foundations terms:

1. Go to: https://huggingface.co/google/medgemma-27b-text-it
2. Log in to HuggingFace
3. Click "Accept" on the terms (instant approval)
4. Login via CLI: `huggingface-cli login`

### HuggingFace Usage Examples

**Chat Model (MedGemma/Gemma)**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b-text-it")
```

**Embedding Model (Qwen3)**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
embeddings = model.encode(["Your text here"])
```

---

## Configuration (.env)

### Few-shot Embeddings Artifact Selection

Few-shot retrieval loads a precomputed artifact from `{DATA_BASE_DIR}/embeddings/`:

- `EMBEDDING_EMBEDDINGS_FILE` selects `{name}.npz` + `{name}.json` (+ optional `{name}.meta.json`).
- `DATA_EMBEDDINGS_PATH` overrides with a full `.npz` path.

If `{name}.meta.json` exists (all newly generated artifacts have it), the server validates backend/model/dimension/chunking against current config and fails fast on mismatch.

### Default (Recommended)

```bash
# Backend selection (defaults to Ollama chat + HuggingFace embeddings)
LLM_BACKEND=ollama
EMBEDDING_BACKEND=huggingface

# Models (all default to gemma3:27b for chat, qwen3-embedding:8b for embeddings)
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIMENSION=4096

# Embeddings artifact (default: huggingface_qwen3_8b_paper_train)
# Only set if you want to override the default HF embeddings
# EMBEDDING_EMBEDDINGS_FILE=paper_reference_embeddings  # Use Ollama embeddings instead
```

### Paper Parity (Pure Ollama)

```bash
LLM_BACKEND=ollama
EMBEDDING_BACKEND=ollama
EMBEDDING_EMBEDDINGS_FILE=paper_reference_embeddings  # Use Ollama Q4_K_M embeddings
```

### With MedGemma (Appendix F - HuggingFace backend required)

```bash
# Use the HuggingFace backend to access official MedGemma weights.
LLM_BACKEND=huggingface
MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

---

## High-Quality Setup (Recommended for Production)

For users with capable hardware (32GB+ RAM, Apple Silicon or NVIDIA GPU), use HuggingFace for **best quality**:

### Why HuggingFace is Better

| Component | Ollama | HuggingFace | Improvement |
|-----------|--------|-------------|-------------|
| **Chat (Quantitative)** | `gemma3:27b` (Q4_K_M) | `google/medgemma-27b-text-it` (FP16) | 18% better MAE (Appendix F) |
| **Embeddings** | `qwen3-embedding:8b` (Q4_K_M) | `Qwen/Qwen3-Embedding-8B` (FP16) | Higher precision similarity |

**Key insight**: Both Ollama models use **Q4_K_M quantization** (4-bit). HuggingFace provides **FP16/BF16** (16-bit) - 4x more precision.

### High-Quality Configuration

```bash
# Option A: FP16 embeddings (keep chat on Ollama)
LLM_BACKEND=ollama
EMBEDDING_BACKEND=huggingface
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b  # → Qwen/Qwen3-Embedding-8B

# Option B: Full HuggingFace (chat + embeddings)
# LLM_BACKEND=huggingface
# EMBEDDING_BACKEND=huggingface
# MODEL_QUANTITATIVE_MODEL=medgemma:27b    # → google/medgemma-27b-text-it (18% better MAE)
```

### Requirements

1. **Hardware**: 32GB+ unified memory (Apple Silicon) or 24GB+ VRAM (NVIDIA)
2. **Dependencies**: `pip install 'ai-psychiatrist[hf]'`
3. **MedGemma access**: Accept terms at [HuggingFace](https://huggingface.co/google/medgemma-27b-text-it)

### Pending: Graceful Fallback

[Issue #42](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/42) will add automatic fallback to Ollama if HuggingFace fails (missing deps, OOM, etc.).

---

## Sources

### Paper
- `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`

### Ollama
- https://ollama.com/library/gemma3
- https://ollama.com/library/qwen3-embedding
- https://ollama.com/library/mxbai-embed-large
- https://ollama.com/library/nomic-embed-text

### HuggingFace (Official)
- https://huggingface.co/google/gemma-3-27b-it
- https://huggingface.co/google/medgemma-27b-text-it (Gated)
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
