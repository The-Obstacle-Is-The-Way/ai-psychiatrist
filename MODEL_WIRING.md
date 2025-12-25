# Model Wiring: Current State

**Purpose**: Document exactly how models and backends are wired in the codebase.
**Last Updated**: 2025-12-24
**Status**: Implemented. `LLM_BACKEND` for chat, `EMBEDDING_BACKEND` for embeddings.

---

## TL;DR - The Simple Truth

| Component | Backend | Default Model | Precision |
|-----------|---------|---------------|-----------|
| Chat (all agents) | `LLM_BACKEND=ollama` | ⭐ `gemma3:27b-it-qat` | QAT 4-bit (trained for quantization) |
| Chat (quant alt) | `LLM_BACKEND=huggingface` | `medgemma:27b` | FP16 (16-bit) |
| **Embedding** | `EMBEDDING_BACKEND=huggingface` | `Qwen3-Embedding-8B` | **FP16 (16-bit)** |

**Key decisions:**
- **Chat**: Ollama default (paper parity). MedGemma is hard toggle for quant agent.
- **Embedding**: HuggingFace default (better precision). Ollama is opt-out fallback.

### Default vs Hard Toggle (The Simple Version)

| Component | Default | Hard Toggle Option |
|-----------|---------|-------------------|
| Qualitative Agent | Ollama (`gemma3:27b-it-qat`) | — |
| Judge Agent | Ollama (`gemma3:27b-it-qat`) | — |
| Meta-Review Agent | Ollama (`gemma3:27b-it-qat`) | — |
| **Quant Agent** | Ollama (`gemma3:27b-it-qat`) | HF (`medgemma:27b`) |
| **Embeddings** | **HF** (`Qwen3-Embedding-8B` FP16) | Ollama (`qwen3-embedding:8b` Q4) |

**Why this mix?**
- Ollama = local, no external deps, paper parity
- HF embeddings = FP16 quality matters for similarity scores
- MedGemma = only available officially on HF (Ollama version is community upload)

---

## Gemma 3 27B: All Official Options (Dec 2025)

### Hardware Requirements

| Hardware | VRAM/Memory | Max Model Size |
|----------|-------------|----------------|
| M1 Max 64GB | 64GB unified | ~54GB (BF16) ✅ |
| M1 Pro 32GB | 32GB unified | ~29GB (Q8_0) ✅ |
| RTX 4090 | 24GB VRAM | ~17GB (Q4) ✅ |

### Ollama Options (Official Google Models)

| Tag | Quantization | Size | M1 Max 64GB | M1 Pro 32GB | RTX 4090 24GB | Quality |
|-----|--------------|------|-------------|-------------|---------------|---------|
| `gemma3:27b` | Q4_K_M | 17GB | ✅ | ✅ | ✅ | Good |
| `gemma3:27b-it-qat` | Q4_0 (QAT) | 17GB | ✅ | ✅ | ✅ | **Better** (QAT-trained) |
| `gemma3:27b-it-q8_0` | Q8_0 | 29GB | ✅ | ❌ | ❌ | Better |

### What Do These Abbreviations Mean?

| Abbreviation | Full Name | Bits | What It Means |
|--------------|-----------|------|---------------|
| **BF16** | Brain Float 16 | 16-bit | Full precision. Each weight is a 16-bit float. No quality loss. Huge memory. |
| **Q8_0** | Quantized 8-bit | 8-bit | Weights compressed to 8-bit integers. ~2x smaller than BF16. Small quality loss. |
| **Q4_K_M** | Quantized 4-bit (K-quant Medium) | 4-bit | Weights compressed to 4-bit. ~4x smaller than BF16. Noticeable quality loss. |
| **QAT** | Quantization-Aware Training | 4-bit | Model was **trained knowing it would be quantized**. Same 4-bit size but better quality than post-hoc Q4. |

### How Quantization Works (Simple Version)

**Original model (BF16)**: Each of 27 billion weights stored as 16-bit float → 54GB

**Post-hoc quantization (Q4_K_M, Q8_0)**: Take trained model, compress weights after training.
- Like compressing a JPEG after taking the photo
- Some information lost in compression

**Quantization-Aware Training (QAT)**: Train the model knowing it will be compressed.
- Like shooting a photo knowing it will be JPEG - you optimize for the output format
- Google claims this preserves BF16 quality at Q4 size

### Quality Ranking (Best → Worst)

```
BF16 (54GB) > Q8_0 (29GB) > QAT Q4 (17GB) ≈ Q4_K_M (17GB)
     ↑              ↑              ↑              ↑
  Perfect      Very Good    Good (smart)   Good (dumb)
```

**Bottom line**: QAT is the sweet spot - same size as Q4_K_M but trained smarter.

### HuggingFace Options (Full Precision)

| Model | HuggingFace ID | Precision | Size | M1 Max 64GB | RTX 4090 | Access |
|-------|----------------|-----------|------|-------------|----------|--------|
| Gemma 3 27B | `google/gemma-3-27b-it` | BF16 | ~54GB | ✅ | ❌ | Open |
| MedGemma 27B | `google/medgemma-27b-text-it` | BF16 | ~54GB | ✅ | ❌ | Gated |

### Other Models (Embedding + Community)

| Model | Backend | Tag/ID | Quantization | Size |
|-------|---------|--------|--------------|------|
| Qwen3 Embedding 8B | Ollama | `qwen3-embedding:8b` | Q4_K_M | 4.7GB |
| Qwen3 Embedding 8B | HuggingFace | `Qwen/Qwen3-Embedding-8B` | FP16 | ~16GB |
| MedGemma 27B | Ollama | `alibayram/medgemma:27b` | Q4_K_M | ~17GB |

**Note**: MedGemma on Ollama is a **community upload**, NOT official Google.

### Gemma 3 27B Options (All Agents)

| Model | Backend | Tag/ID | Bits | Size | M1 Max | 4090 | Speed | Quality |
|-------|---------|--------|------|------|--------|------|-------|---------|
| Gemma 3 27B | HF | `google/gemma-3-27b-it` | 16-bit | 54GB | ✅ | ❌ | Slow | **Best** (BF16) |
| Gemma 3 27B | Ollama | `gemma3:27b-it-q8_0` | 8-bit | 29GB | ✅ | ❌ | Medium | Very Good |
| Gemma 3 27B | Ollama | `gemma3:27b-it-qat` | 4-bit | 17GB | ✅ | ✅ | Fast | Good (QAT-trained) |
| Gemma 3 27B | Ollama | `gemma3:27b` | 4-bit | 17GB | ✅ | ✅ | Fast | Good (Q4_K_M) |

**Paper reality check**: Paper text claims MacBook M3 Pro, but repo has A100 SLURM scripts.
Paper likely ran **BF16 on A100s** for the reported 0.619 MAE. Our Q4_K_M run got 0.778 MAE.

### Which Model Should We Use?

| Goal | Model | Why |
|------|-------|-----|
| **⭐ RECOMMENDED** | `gemma3:27b-it-qat` (4-bit) | QAT-trained, same speed as Q4, claims BF16 quality |
| Closer to paper's likely setup | `gemma3:27b-it-q8_0` (8-bit) | Paper likely used BF16 on A100s; Q8 is closest but slow |
| Old default (not recommended) | `gemma3:27b` (4-bit) | Post-hoc Q4_K_M, strictly worse than QAT |
| Maximum quality | HF `gemma-3-27b-it` (16-bit) | Full BF16, 54GB, very slow on M1 |

### Estimated Run Times (Full Pipeline, 41 Transcripts)

| Model | Est. Time | Notes |
|-------|-----------|-------|
| Ollama 4-bit (17GB) | ~2-4 hours | Current default |
| Ollama 8-bit (29GB) | ~6-12 hours | **Recommended for reproduction** |
| HF BF16 (54GB) | ~12-24+ hours | Memory-bound on M1 |

---

### MedGemma 27B (Quantitative Agent ONLY)

**NOT a general model option.** MedGemma is ONLY for the quantitative agent as a hard toggle.

| Model | Backend | Tag/ID | Size | Access | Notes |
|-------|---------|--------|------|--------|-------|
| MedGemma 27B | HF | `google/medgemma-27b-text-it` | 54GB | **Gated** | Official, medical fine-tuned |
| MedGemma 27B | Ollama | `alibayram/medgemma:27b` | 17GB | Open | **Community upload, NOT official** |

**Paper finding (Appendix F)**: MedGemma got **better MAE (0.505 vs 0.619)** but made **fewer predictions**.
The paper chose Gemma 3 for main results because MedGemma was too conservative.

To enable MedGemma:
```bash
LLM_BACKEND=huggingface
MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

---

## Current Configuration (Code Defaults)

### Backends

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_BACKEND` | `ollama` | Chat models (all agents) |
| `EMBEDDING_BACKEND` | `huggingface` | Embedding model only |

**No runtime fallback.** If configured backend fails → loud error with instructions.

### Chat Models (All Agents)

| Agent | Config Key | Default | Paper Reference |
|-------|------------|---------|-----------------|
| Qualitative | `MODEL_QUALITATIVE_MODEL` | `gemma3:27b-it-qat` | Section 2.2 (QAT for quality) |
| Judge | `MODEL_JUDGE_MODEL` | `gemma3:27b-it-qat` | Section 2.2 (QAT for quality) |
| Meta-Review | `MODEL_META_REVIEW_MODEL` | `gemma3:27b-it-qat` | Section 2.2 (QAT for quality) |
| Quantitative | `MODEL_QUANTITATIVE_MODEL` | `gemma3:27b-it-qat` | Section 2.2 (QAT for quality) |

**MedGemma** (`medgemma:27b`) is an ALTERNATIVE for quantitative agent only (Appendix F).
It requires `LLM_BACKEND=huggingface` for official weights. The Ollama community version may behave differently.

### Embedding Model

| Setting | Default | Backend | Precision |
|---------|---------|---------|-----------|
| `MODEL_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | HuggingFace | FP16 |

**Why HF default for embeddings?** FP16 embeddings produce better similarity scores than Q4_K_M.
To use Ollama instead: `EMBEDDING_BACKEND=ollama` (will use `qwen3-embedding:8b` Q4_K_M).

---

## When Embeddings Are Generated

### 1. Data Prep (Once)

Script: `scripts/generate_embeddings.py`
Output: `data/embeddings/paper_reference_embeddings.npz`

Generates reference embeddings for training set transcripts. Run once before few-shot mode.

### 2. Runtime (Every Assessment in Few-Shot Mode)

Location: `EmbeddingService.embed_text()` called from `QuantitativeAssessmentAgent`

Flow:
```
Transcript → Extract Evidence → Embed Evidence → Cosine Similarity → Reference Matches
                                     ↑                    ↑
                              (runtime embed)    (pre-computed refs)
```

**Consistency requirement**: Reference embeddings and runtime embeddings should use the same backend for precision consistency.

---

## Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                             PIPELINE                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Transcript                                                        │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ QUALITATIVE AGENT   │  Model: gemma3:27b-it-qat (chat)          │
│  │ (assess symptoms)   │  Backend: LLM_BACKEND (default: ollama)   │
│  └─────────────────────┘                                           │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ JUDGE AGENT         │  Model: gemma3:27b-it-qat (chat)          │
│  │ (evaluate + refine) │  Backend: LLM_BACKEND                     │
│  └─────────────────────┘                                           │
│      │  ↺ feedback loop (max 10 iterations)                        │
│      ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ QUANTITATIVE AGENT  │  Model: gemma3:27b-it-qat OR medgemma:27b │
│  │ (PHQ-8 scoring)     │  Backend: LLM_BACKEND (default: ollama)   │
│  │                     │                                           │
│  │  Few-shot mode:     │                                           │
│  │  - Embed evidence   │  Model: Qwen3-Embedding-8B                │
│  │  - Find references  │  Backend: EMBEDDING_BACKEND (default: hf) │
│  └─────────────────────┘                                           │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ META-REVIEW AGENT   │  Model: gemma3:27b-it-qat (chat)          │
│  │ (final severity)    │  Backend: LLM_BACKEND                     │
│  └─────────────────────┘                                           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Factory Logic (Current)

```python
# factory.py - NO FALLBACK
def create_llm_client(settings: Settings) -> LLMClient:
    backend = settings.backend.backend
    if backend == LLMBackend.OLLAMA:
        return OllamaClient(settings.ollama)
    if backend == LLMBackend.HUGGINGFACE:
        return HuggingFaceClient(...)  # Fails if deps missing
    raise ValueError(f"Unsupported backend: {backend}")
```

---

## Configuration Scenarios

### Scenario 1: Default (Recommended)

```bash
# .env (defaults - better embeddings, paper-parity chat)
LLM_BACKEND=ollama                # Chat: Ollama Q4_K_M
EMBEDDING_BACKEND=huggingface     # Embed: HuggingFace FP16
```

Requires: `pip install 'ai-psychiatrist[hf]'`

### Scenario 2: Pure Ollama (Paper Parity, Lower Quality)

```bash
LLM_BACKEND=ollama
EMBEDDING_BACKEND=ollama          # Opt-out of HF embeddings
```

All models Q4_K_M. Matches Paper Section 2.3.5 exactly.

### Scenario 3: MedGemma for Quant Agent (Appendix F)

```bash
LLM_BACKEND=huggingface           # Required for official MedGemma
MODEL_QUANTITATIVE_MODEL=medgemma:27b
EMBEDDING_BACKEND=huggingface     # Keep FP16 embeddings
```

Requires MedGemma access approved on HuggingFace.
Result: 18% better item MAE (0.505 vs 0.619) but fewer predictions.

### Scenario 4: Full HuggingFace (Maximum Precision)

```bash
LLM_BACKEND=huggingface           # Chat: FP16
EMBEDDING_BACKEND=huggingface     # Embed: FP16
```

Everything FP16. Requires ~54GB VRAM for chat + ~16GB for embeddings.

---

## What We Do NOT Support (By Design)

1. **Runtime fallback**: HF unavailable → silently use Ollama (breaks reproducibility)
2. **Model substitution**: medgemma → gemma3 (different clinical behavior)
3. **Mixed embedding precision**: FP16 refs + Q4_K_M runtime (breaks similarity scores)

---

## Final Architecture

### Default Configuration

```bash
# .env (defaults)
LLM_BACKEND=ollama                        # Chat: Ollama (paper parity)
EMBEDDING_BACKEND=huggingface             # Embedding: HuggingFace (better precision)
MODEL_QUANTITATIVE_MODEL=gemma3:27b-it-qat  # ⭐ QAT model (same speed, better quality)
```

### Startup Validation

1. **EMBEDDING_BACKEND=huggingface** (default):
   - Check HF deps installed
   - Missing → **ERROR**:
     ```
     ERROR: HuggingFace embedding backend requires dependencies.
     Install: pip install 'ai-psychiatrist[hf]'
     Or use: EMBEDDING_BACKEND=ollama
     ```

2. **Reference embedding validation**:
   - Load `paper_reference_embeddings.npz`
   - Check stored `backend` metadata
   - If metadata ≠ current EMBEDDING_BACKEND → **ERROR**:
     ```
     ERROR: Reference embeddings were generated with ollama (Q4_K_M).
     Current backend is huggingface (FP16). Precision mismatch.
     Regenerate: uv run python scripts/generate_embeddings.py
     ```

### MedGemma: Hard Toggle

```bash
LLM_BACKEND=huggingface
MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

If HF unavailable → **FAIL LOUDLY**. No silent substitution.

### Why This Design?

| Decision | Reason |
|----------|--------|
| HF embeddings default | FP16 > Q4_K_M for similarity quality |
| Ollama chat default | Paper parity (Section 2.3.5) |
| No runtime fallback | Reproducibility > convenience |
| Precision validation | Prevent silent embedding mismatch |
| Hard toggle for MedGemma | Different clinical behavior ≠ drop-in replacement |

---

## References

- Paper Section 2.2: Model specification (Gemma 3 27B, Qwen 3 8B Embedding)
- Paper Appendix F: MedGemma evaluation
- `src/ai_psychiatrist/config.py`: All defaults
- `src/ai_psychiatrist/infrastructure/llm/factory.py`: Client creation
- `src/ai_psychiatrist/infrastructure/llm/model_aliases.py`: Backend mapping
