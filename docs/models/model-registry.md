# AI Psychiatrist Model Registry

Last Updated: 2025-12-22
Purpose: Paper-aligned, reproducible model configuration for this repo.

## Paper-Optimal Models (Reproduction)

These models match the paper's methodology and are required for paper-accurate runs.

| Role | Model family | Params | Ollama tag | Paper reference | Notes |
|------|--------------|--------|------------|-----------------|-------|
| Qualitative Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for qualitative assessment |
| Judge Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for feedback loop |
| Meta-Review Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for final review |
| Quantitative Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | **Default** (see MedGemma note below) |
| Embedding | Qwen3 Embedding | 8B | `qwen3-embedding:8b` | Section 2.2 | 4096-dim embeddings (Appendix D) |

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
ollama pull gemma3:27b
ollama pull qwen3-embedding:8b
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
| `gemma3:27b` | `google/gemma-3-27b-it` | Open | Multimodal, use `Gemma3ForConditionalGeneration` |
| `medgemma:27b` | `google/medgemma-27b-text-it` | **Gated** | Text-only, use `AutoModelForCausalLM` |
| `qwen3-embedding:8b` | `Qwen/Qwen3-Embedding-8B` | Open | Use `SentenceTransformer`, #1 on MTEB |

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

### Paper-optimal (Default)

```bash
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIMENSION=4096
```

### With MedGemma (Appendix F - HuggingFace backend required)

```bash
# Use the HuggingFace backend to access official MedGemma weights.
LLM_BACKEND=huggingface
MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

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
- https://huggingface.co/Qwen/Qwen3-Embedding-8B (#1 MTEB)
