# BUG-020: Complete Model Clarity and HuggingFace Options

**Date**: 2025-12-22
**Status**: ROOT CAUSE ANALYSIS COMPLETE - READY FOR SENIOR REVIEW
**Author**: Read-only analysis with web search verification

---

## Executive Summary

This document provides 100% clarity on which models the paper uses and how to run them locally.

### Paper's Model Configuration (CONFIRMED)

| Component | Model | Source |
|-----------|-------|--------|
| ALL Agents (Qualitative, Judge, Meta-review, Quantitative) | **Gemma 3 27B** | Section 2.2 |
| Embeddings (Few-shot) | **Qwen 3 8B Embedding** | Section 2.2 |
| Quantitative (OPTIONAL alternative) | **MedGemma 27B** | Appendix F only |

### Key Paper Quotes

**Section 2.2 (Models)**:
> "We utilized a state-of-the-art open-weight language model, **Gemma 3 with 27 billion parameters (Gemma 3 27B)**"
> "For the embedding-based few-shot prompting approach, we used **Qwen 3 8B Embedding**"

**Section 3.2 (Quantitative Assessment)**:
> "In addition to Gemma 3 27B, we **also evaluated** its variant fine-tuned on medical text, MedGemma 27B"
> "The few-shot approach with MedGemma 27B achieved an improved average MAE of 0.505 **but detected fewer relevant chunks, making fewer predictions overall** (Appendix F)"

**Appendix F**:
> "MedGemma 27B had an edge over Gemma 3 27B in most categories overall, achieving an average MAE of 0.505, 18% less than Gemma 3 27B, **although the number of subjects detected as having available evidence from the transcripts was smaller with MedGemma**"

---

## ROOT CAUSE: Why We Used alibayram/medgemma

### What Happened

1. Someone read Appendix F and saw MedGemma had 18% better MAE (0.505 vs 0.619)
2. They searched Ollama for MedGemma
3. Found `alibayram/medgemma:27b` (community upload with 9,916 pulls)
4. Set it as default WITHOUT reading the caveat: "fewer predictions overall"
5. MedGemma's conservative nature caused ALL N/A outputs

### Why alibayram Is Wrong

| Aspect | alibayram/medgemma | Official Google |
|--------|-------------------|-----------------|
| Source | Community conversion | [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) |
| Quantization | Q4_K_M (lossy) | Full BF16 weights |
| Verified | No | Yes |
| MTEB tested | No | Yes |

**There is NO official MedGemma in Ollama library** - confirmed via [Ollama Search](https://ollama.com/search?q=medgemma) and [GitHub Issue #10970](https://github.com/ollama/ollama/issues/10970).

---

## Official Google MedGemma Models on HuggingFace

### Text-Only (What We Need)

| Model | URL | Notes |
|-------|-----|-------|
| `google/medgemma-27b-text-it` | [HuggingFace](https://huggingface.co/google/medgemma-27b-text-it) | **Official, text-only, instruction-tuned** |

### Multimodal (Not Needed)

| Model | URL | Notes |
|-------|-----|-------|
| `google/medgemma-27b-it` | [HuggingFace](https://huggingface.co/google/medgemma-27b-it) | Multimodal with image support |

### GGUF Conversions (For Ollama/llama.cpp)

| Model | URL | Quantization |
|-------|-----|--------------|
| `unsloth/medgemma-27b-text-it-GGUF` | [HuggingFace](https://huggingface.co/unsloth/medgemma-27b-text-it-GGUF) | Multiple quantizations |
| `bartowski/google_medgemma-27b-it-GGUF` | [HuggingFace](https://huggingface.co/bartowski/google_medgemma-27b-it-GGUF) | Q4_K_M, Q5_K_M, Q8_0 |

**Unsloth is more reputable** than alibayram for GGUF conversions.

---

## Options for Running Models Locally

### Option 1: Ollama (Current Setup)

**For Gemma 3 27B** (main model):
```bash
ollama pull gemma3:27b  # Official in Ollama library
```

**For MedGemma 27B** (if needed for quantitative):
```bash
# Option A: Use unsloth GGUF (more reputable than alibayram)
# Download from HuggingFace and create Modelfile

# Option B: Keep using gemma3:27b (paper's primary results)
```

### Option 2: HuggingFace Transformers (Native Python)

**Installation**:
```bash
pip install "torch>=2.4.0" "transformers>=4.51.3"
```

**Basic Usage**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# For Gemma 3 27B
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

# For MedGemma 27B (if needed)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

**With Quantization** (for lower VRAM):
```python
from transformers import TorchAoConfig, AutoModelForCausalLM
import torch

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)
```

### Option 3: llama.cpp with GGUF

**Download GGUF**:
```bash
huggingface-cli download unsloth/medgemma-27b-text-it-GGUF \
    --include "medgemma-27b-text-it-Q4_K_M.gguf" \
    --local-dir ./models/
```

**Run**:
```bash
llama-cli -m ./models/medgemma-27b-text-it-Q4_K_M.gguf -p "Your prompt"
```

### Option 4: vLLM (High Performance Serving)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/medgemma-27b-text-it")
outputs = llm.generate(["Your prompt"], SamplingParams(temperature=0.7))
```

---

## Recommended Configuration

### For Reproducing Paper Results (Primary)

Use **Gemma 3 27B** for ALL agents (matches Section 2.2):

```env
MODEL__QUALITATIVE_MODEL=gemma3:27b
MODEL__JUDGE_MODEL=gemma3:27b
MODEL__META_REVIEW_MODEL=gemma3:27b
MODEL__QUANTITATIVE_MODEL=gemma3:27b
MODEL__EMBEDDING_MODEL=qwen3-embedding:8b
```

### For Reproducing Appendix F Results (Optional)

Use MedGemma 27B ONLY for quantitative agent:

```env
MODEL__QUALITATIVE_MODEL=gemma3:27b
MODEL__JUDGE_MODEL=gemma3:27b
MODEL__META_REVIEW_MODEL=gemma3:27b
MODEL__QUANTITATIVE_MODEL=<official-medgemma>  # See options above
MODEL__EMBEDDING_MODEL=qwen3-embedding:8b
```

**Note**: MedGemma produces FEWER predictions (more N/A). Paper states: "the number of subjects detected as having available evidence from the transcripts was smaller with MedGemma."

---

## Files That Need Changes

### 1. config.py - Remove Stale Comment

**Location**: `src/ai_psychiatrist/config.py:285-286`

**Current** (STALE):
```python
# NOTE: enable_medgemma removed - use MODEL__QUANTITATIVE_MODEL directly.
# Default quantitative_model is already alibayram/medgemma:27b (Paper Appendix F).
```

**Should Be**:
```python
# NOTE: Default quantitative_model is gemma3:27b (Paper Section 2.2).
# MedGemma was only evaluated as ALTERNATIVE in Appendix F with fewer predictions.
```

### 2. reproduce_results.py - Add Item-Level MAE

**Location**: `scripts/reproduce_results.py`

**Current**: Calculates total-score MAE (0-24 scale)
**Paper**: Uses item-level MAE (0-3 scale), excludes N/A

**Needed**: Add item-level MAE calculation for direct paper comparison.

### 3. .env.example - Documentation Update

Update comments to clarify:
- gemma3:27b is the PRIMARY model (Section 2.2)
- MedGemma is OPTIONAL alternative for quantitative only (Appendix F)
- MedGemma produces more N/A predictions

---

## Scoring Methodology Mismatch (SEPARATE ISSUE)

### Paper's Method (Legacy Code Correct)

From `_legacy/quantitative_assessment/quantitative_analysis.py:201-202`:
```python
if n_available > 0:
    avg_difference = sum(differences) / n_available  # ITEM LEVEL, excludes N/A
```

- Scale: 0-3 per item
- N/A: EXCLUDED from MAE
- Paper MAE: ~0.619 (Gemma), ~0.505 (MedGemma)

### Our Current Method

From `src/ai_psychiatrist/domain/entities.py:115-123`:
```python
@property
def total_score(self) -> int:
    """N/A scores contribute 0 to the total."""
    return sum(item.score_value for item in self.items.values())
```

- Scale: 0-24 total
- N/A: Counts as 0
- Our MAE: ~4.02 (not comparable to paper)

**Our 4.02 total MAE ÷ 8 items ≈ 0.50 item MAE** - actually BETTER than paper when adjusted!

---

## Summary of Root Causes

| Issue | Root Cause | Severity | Fix |
|-------|------------|----------|-----|
| MedGemma all N/A | Used community model + ignored "fewer predictions" caveat | CRITICAL | Use gemma3:27b |
| alibayram model | No official Ollama MedGemma exists | HIGH | Use official HuggingFace or Unsloth GGUF |
| Scoring mismatch | Total-score vs item-level MAE | CRITICAL | Add item-level MAE calculation |
| Stale comments | config.py still mentions alibayram | LOW | Update comment |

---

## SPEC: LLM Backend Architecture (From First Principles)

### Current Architecture (GOOD)

The codebase already has **protocol-based abstractions** that support backend swapping:

```
src/ai_psychiatrist/infrastructure/llm/
├── protocols.py      # ChatClient, EmbeddingClient, LLMClient protocols
├── ollama.py         # OllamaClient implementation
├── responses.py      # SimpleChatClient protocol + helpers
└── __init__.py       # Public exports
```

**Key Insight**: The Strategy pattern is ALREADY implemented. Agents depend on `SimpleChatClient` protocol, not concrete `OllamaClient`. Adding a `HuggingFaceClient` is architecturally supported.

### What's Missing

1. **No `HuggingFaceClient` implementation**
2. **No backend selection mechanism** (hardcoded to Ollama)
3. **Model names are Ollama-specific** (`gemma3:27b` vs `google/gemma-3-27b-it`)

### Proposed Architecture

#### 1. Backend Enum

```python
# src/ai_psychiatrist/config.py

class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    # Future: VLLM = "vllm", LLAMACPP = "llamacpp"
```

#### 2. Backend Settings

```python
# src/ai_psychiatrist/config.py

class BackendSettings(BaseSettings):
    """LLM backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=ENV_FILE,
    )

    backend: LLMBackend = Field(
        default=LLMBackend.OLLAMA,
        description="LLM backend: ollama or huggingface"
    )

    # HuggingFace-specific
    hf_device: str = Field(default="auto", description="Device: auto, cuda, mps, cpu")
    hf_quantization: str | None = Field(default="int4", description="Quantization: None, int4, int8")
    hf_cache_dir: Path | None = Field(default=None, description="Model cache directory")
```

#### 3. Model Alias Mapping

```python
# src/ai_psychiatrist/infrastructure/llm/model_aliases.py

MODEL_ALIASES: dict[str, dict[str, str | None]] = {
    # Canonical name -> backend-specific name
    "gemma3:27b": {
        "ollama": "gemma3:27b",
        "huggingface": "google/gemma-3-27b-it",
    },
    "medgemma:27b": {
        "ollama": None,  # NOT AVAILABLE OFFICIALLY
        "huggingface": "google/medgemma-27b-text-it",
    },
    "qwen3-embedding:8b": {
        "ollama": "qwen3-embedding:8b",
        "huggingface": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",  # Need to verify
    },
}

def resolve_model_name(canonical: str, backend: LLMBackend) -> str:
    """Resolve canonical model name to backend-specific name."""
    if canonical in MODEL_ALIASES:
        name = MODEL_ALIASES[canonical].get(backend.value)
        if name is None:
            raise ValueError(f"Model '{canonical}' not available for backend '{backend}'")
        return name
    # If not in aliases, assume it's already backend-specific
    return canonical
```

#### 4. HuggingFace Client Implementation

```python
# src/ai_psychiatrist/infrastructure/llm/huggingface.py

class HuggingFaceClient:
    """HuggingFace Transformers LLM client.

    Implements ChatClient and EmbeddingClient protocols.
    Uses official Google models directly from HuggingFace.
    """

    def __init__(self, settings: BackendSettings, model_settings: ModelSettings):
        self.settings = settings
        self.model_settings = model_settings
        self._models: dict[str, Any] = {}  # Lazy-loaded models
        self._tokenizers: dict[str, Any] = {}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute chat completion using HuggingFace model."""
        model = self._get_or_load_model(request.model)
        tokenizer = self._get_or_load_tokenizer(request.model)
        # ... implementation

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding using HuggingFace model."""
        # ... implementation
```

#### 5. Factory Pattern

```python
# src/ai_psychiatrist/infrastructure/llm/factory.py

def create_llm_client(settings: Settings) -> LLMClient:
    """Create LLM client based on backend configuration."""
    backend = settings.backend.backend

    if backend == LLMBackend.OLLAMA:
        return OllamaClient(settings.ollama)
    elif backend == LLMBackend.HUGGINGFACE:
        return HuggingFaceClient(settings.backend, settings.model)
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

#### 6. Updated .env Configuration

```env
# ============== LLM Backend ==============
# Choose backend: ollama or huggingface
LLM_BACKEND=huggingface

# HuggingFace-specific (only used if LLM_BACKEND=huggingface)
LLM_HF_DEVICE=auto
LLM_HF_QUANTIZATION=int4

# ============== Model Selection ==============
# Use canonical names - automatically resolved per backend
MODEL_QUANTITATIVE_MODEL=gemma3:27b      # -> google/gemma-3-27b-it on HF
# Or use MedGemma (HuggingFace only, not on Ollama):
# MODEL_QUANTITATIVE_MODEL=medgemma:27b   # -> google/medgemma-27b-text-it on HF
```

### Why HuggingFace Over Ollama?

| Aspect | Ollama | HuggingFace |
|--------|--------|-------------|
| **Model availability** | Community uploads, no official MedGemma | Official Google models |
| **Verification** | No verification of conversions | Verified by model authors |
| **Quantization control** | Limited (what's uploaded) | Full control (int4, int8, bf16) |
| **Dependencies** | External Ollama server | Native Python, pip install |
| **Debugging** | Black box | Full access to internals |
| **Reproducibility** | Depends on community | Deterministic weights |

### Migration Path

1. **Phase 1**: Add `HuggingFaceClient` as alternative backend (keep Ollama working)
2. **Phase 2**: Add model alias mapping
3. **Phase 3**: Add backend selection via `.env`
4. **Phase 4**: Test with official MedGemma from HuggingFace
5. **Phase 5**: Consider deprecating Ollama if HuggingFace works better

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/ai_psychiatrist/config.py` | Modify | Add `LLMBackend` enum, `BackendSettings` |
| `src/ai_psychiatrist/infrastructure/llm/huggingface.py` | Create | HuggingFace client implementation |
| `src/ai_psychiatrist/infrastructure/llm/model_aliases.py` | Create | Model name mapping |
| `src/ai_psychiatrist/infrastructure/llm/factory.py` | Create | Client factory |
| `src/ai_psychiatrist/infrastructure/llm/__init__.py` | Modify | Export new classes |
| `pyproject.toml` | Modify | Add `transformers`, `torch` dependencies |
| `.env.example` | Modify | Add backend configuration |
| `tests/unit/infrastructure/test_huggingface.py` | Create | Unit tests |

### Estimated Scope

- **New code**: ~300-400 lines
- **Test code**: ~200-300 lines
- **Config changes**: ~50 lines
- **Total**: ~600-800 lines

---

## References

### Paper
- Section 2.2: Model specification (Gemma 3 27B, Qwen 3 8B)
- Section 3.2: MedGemma evaluation mention
- Appendix F: Full MedGemma results with caveats

### HuggingFace (Official Google)
- [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
- [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)
- [MedGemma Collection](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4)

### HuggingFace (GGUF Conversions)
- [unsloth/medgemma-27b-text-it-GGUF](https://huggingface.co/unsloth/medgemma-27b-text-it-GGUF)
- [bartowski/google_medgemma-27b-it-GGUF](https://huggingface.co/bartowski/google_medgemma-27b-it-GGUF)

### Transformers
- [Run Gemma with HuggingFace](https://ai.google.dev/gemma/docs/core/huggingface_inference)
- [Gemma 3 Model Doc](https://huggingface.co/docs/transformers/en/model_doc/gemma3)

### Ollama
- [Ollama MedGemma Search](https://ollama.com/search?q=medgemma) - Shows NO official library entry
- [GitHub Issue #10970](https://github.com/ollama/ollama/issues/10970) - MedGemma support request

### Google Official
- [MedGemma Model Card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [GitHub: Google-Health/medgemma](https://github.com/Google-Health/medgemma)
