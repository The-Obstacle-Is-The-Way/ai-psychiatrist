# BUG-044: HuggingFace Chunk Scores Missing (Wrong Embeddings Scored)

**Status**: RESOLVED (2026-01-02)
**Severity**: P1 (High; resolved)
**Discovered**: 2026-01-01

## Update (2026-01-02)

- Chunk scoring completed for participant-only HuggingFace embeddings:
  - `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json`
  - `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json`
- Reproduction runs now load these chunk scores at runtime (see `data/outputs/repro_post_preprocessing_20260101_183533.log`).
- Defaults updated to point at participant-only artifacts:
  - `.env.example`: `DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only`
  - `.env.example`: `EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train_participant_only`
  - `.env.example`: `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`

## Historical Update (2026-01-01)

- Post participant-only preprocessing, the active HuggingFace embeddings are `huggingface_qwen3_8b_paper_train_participant_only.*`.
- Chunk scoring for these embeddings is running and writes progress logs to `data/outputs/chunk_scoring_participant_only_20260101_183027.log`.

## The Issue

We ran chunk scoring (Spec 35) for hours on the **wrong embeddings**.

### What We Have

| Embeddings File | Precision | Quality | Chunk Scores? |
|-----------------|-----------|---------|---------------|
| `ollama_qwen3_8b_paper_train` | Q4_K_M | Lower | ✅ Yes (hours of compute) |
| `huggingface_qwen3_8b_paper_train_participant_only` | FP16 | **Higher** | ✅ Yes |

### Why This Matters

1. **Documentation says HuggingFace is better**: `.env.example` explicitly states:
   > "Default embedding backend is HuggingFace (FP16 precision) for better similarity scores."

2. **Chunk scoring is the Spec 35 fix** for the core few-shot design flaw (participant-level scores assigned to arbitrary chunks).

3. **We invested hours** generating chunk scores for the lower-quality embeddings.

4. **To use chunk-level scoring with HuggingFace embeddings**, we need to run chunk scoring on `huggingface_qwen3_8b_paper_train_participant_only`.

## Current Configuration State

### In `.env`:
```bash
EMBEDDING_BACKEND=huggingface
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train_participant_only
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
```

### Resolution Artifacts Present:
- `huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json`
- `huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json`

## Command to Generate HuggingFace Chunk Scores

**WARNING: This takes HOURS. Do not run casually.**

```bash
python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat \
  --allow-same-model
```

This will generate:
- `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json`
- `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json`

## Recommended Default Configuration

After generating HuggingFace chunk scores, `.env` should use:

```bash
# Use the BETTER embeddings (FP16)
EMBEDDING_BACKEND=huggingface  # or ollama if HF deps unavailable
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train_participant_only
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
```

## Configuration Philosophy Question

**Should HuggingFace be the documented default?**

Current state is confusing:
- `.env.example` comments say HuggingFace is the default
- But `EMBEDDING_BACKEND` is commented out (no explicit default shown)
- User's `.env` has Ollama set explicitly

Proposal:
1. Make HuggingFace the **explicit default** in `.env.example`
2. Document Ollama as the fallback for users without HuggingFace deps
3. Generate chunk scores for HuggingFace embeddings
4. Update `.env.example` to point to `huggingface_qwen3_8b_paper_train_participant_only`

## Immediate Decision Required

**Option A**: Run reproduction with current Ollama chunk scores (faster, use what we have)
- Pros: Hours already invested, can start now
- Cons: Using lower-quality embeddings

**Option B**: First generate HuggingFace chunk scores, then run reproduction
- Pros: Using the better embeddings
- Cons: More hours of compute before any results

**Option C**: Run both in parallel (if machine can handle it)
- Pros: Can compare results
- Cons: Double the compute

**Resolution**: Completed Option B and ran reproduction (see `docs/results/run-history.md` Run 8).

## Files Referenced

- Ollama embeddings: `data/embeddings/ollama_qwen3_8b_paper_train.*`
- HuggingFace embeddings: `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.*`
- Chunk scoring script: `scripts/score_reference_chunks.py`
- Configuration: `.env`, `.env.example`
