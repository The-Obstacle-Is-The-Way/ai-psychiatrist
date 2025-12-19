# AI Psychiatrist Model Registry

Last Updated: 2025-12-19
Purpose: Paper-aligned, reproducible model configuration for this repo.

## Paper-Optimal Models (Reproduction)

These models match the paper's methodology and are required for paper-accurate runs.

| Role | Model family | Params | Ollama tag (example) | Paper reference | Notes |
|------|--------------|--------|---------------------|----------------|-------|
| Qualitative Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for qualitative assessment |
| Judge Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for feedback loop |
| Meta-Review Agent | Gemma 3 | 27B | `gemma3:27b` | Section 2.2 | Used for final review |
| Quantitative Agent | MedGemma | 27B | `alibayram/medgemma:27b` | Appendix F | MAE 0.505 (fewer predictions) |
| Embedding | Qwen3 Embedding | 8B | `qwen3-embedding:8b` | Section 2.2 | 4096-dim embeddings (Appendix D) |

Approximate disk for paper-optimal pulls: ~38 GB (models share some layers).

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

Paper-optimal:

```bash
ollama pull gemma3:27b
ollama pull alibayram/medgemma:27b
ollama pull qwen3-embedding:8b
```

Development:

```bash
ollama pull gemma2:9b
ollama pull mxbai-embed-large
ollama pull nomic-embed-text
```

## Configuration (.env)

Paper-optimal values:

```bash
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIMENSION=4096
```

## Sources

- Paper: `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`
- Ollama library: `https://ollama.com/library/qwen3-embedding`
- Ollama library: `https://ollama.com/library/mxbai-embed-large`
- Ollama library: `https://ollama.com/library/nomic-embed-text`
- Ollama library: `https://ollama.com/library/gemma3`
