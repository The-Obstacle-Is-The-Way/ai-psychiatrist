# Next Steps

**Date**: 2026-01-01
**Status**: ACTIVE
**Hardware**: Apple M1 Max, 10 cores, 64 GB RAM

---

## Immediate Priority: Generate HuggingFace Chunk Scores

### The Problem

We have two sets of embeddings:

| Embeddings | Precision | Quality | Chunk Scores |
|------------|-----------|---------|--------------|
| `ollama_qwen3_8b_paper_train` | Q4_K_M | Lower | ✅ Generated |
| `huggingface_qwen3_8b_paper_train` | FP16 | **Higher** | ❌ **MISSING** |

We ran hours of chunk scoring on the **lower-quality** Ollama embeddings.
The better HuggingFace embeddings have no chunk scores.

See `PROBLEM-HUGGINGFACE-CHUNK-SCORES-MISSING.md` for full analysis.

### The Command (DO NOT RUN DURING ACTIVE REPRODUCTION)

```bash
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat \
  --allow-same-model
```

**Estimated time**: Several hours (58 participants, ~6800 chunks)

### When to Run

1. Wait for current reproduction run to finish (tmux `repro`)
2. Then run in a new tmux session:
   ```bash
   tmux new-session -d -s hf_chunk_scores "uv run python scripts/score_reference_chunks.py --embeddings-file huggingface_qwen3_8b_paper_train --scorer-backend ollama --scorer-model gemma3:27b-it-qat --allow-same-model 2>&1 | tee data/outputs/hf_chunk_scoring_$(date +%Y%m%d_%H%M%S).log"
   ```

---

## Priority 2: Clean Up `.env.example` (It's Confusing Slop)

### The Problem

`.env.example` is 146 lines of commented-out options, alternatives, and edge cases.
It violates our Configuration Philosophy: **correct behavior should be obvious**.

### Issues

1. **Too many comments about alternatives** - confusing
2. **Deprecated features still visible** - why show them at all?
3. **Spec-by-spec organization** - users don't think in specs
4. **Commented fallbacks mixed with active settings** - hard to parse

### The Fix: Restructure `.env.example`

Create TWO sections:
1. **Minimal config** - what you need to get started (10-15 lines)
2. **Advanced options** - everything else, clearly separated

Proposed new structure:

```bash
# ============================================================
# AI Psychiatrist - Minimal Configuration
# ============================================================
# Copy to .env. Most users only need to verify these values.

# Ollama server (required)
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434

# Embedding backend (HuggingFace recommended for quality)
EMBEDDING_BACKEND=huggingface
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train

# Models (Gemma 3 27B, paper-optimal)
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
MODEL_JUDGE_MODEL=gemma3:27b-it-qat
MODEL_META_REVIEW_MODEL=gemma3:27b-it-qat
MODEL_QUALITATIVE_MODEL=gemma3:27b-it-qat
MODEL_QUANTITATIVE_MODEL=gemma3:27b-it-qat

# ============================================================
# Advanced Options (most users don't need to change these)
# ============================================================
# See CONFIGURATION-PHILOSOPHY.md for guidance.

# [Rest of settings organized by category, not by spec]
```

### Action Items

- [ ] Rewrite `.env.example` with minimal/advanced split
- [ ] Remove deprecated features from visible config (they're in code for ablation only)
- [ ] Group by user intent, not by spec number
- [ ] Add cross-reference to `CONFIGURATION-PHILOSOPHY.md`

---

## Priority 3: Ensure HuggingFace is the Default Everywhere

### Checklist

- [x] `.env.example` shows HuggingFace as explicit default
- [ ] `CONFIGURATION-PHILOSOPHY.md` section 5a documents the quality difference
- [ ] `POST-ABLATION-DEFAULTS.md` updated to require HuggingFace chunk scores
- [ ] `CLAUDE.md` updated with HuggingFace as recommended

### Current State

| File | HuggingFace Default? | Notes |
|------|----------------------|-------|
| `.env.example` | ✅ Yes (just fixed) | Lines 23-24 |
| `config.py` | ✅ Yes (code default) | `embedding_backend: str = "huggingface"` |
| `CONFIGURATION-PHILOSOPHY.md` | ✅ Yes (section 5a) | Just added |
| `POST-ABLATION-DEFAULTS.md` | ⚠️ No mention | Needs update |
| `CLAUDE.md` | ❌ Not explicit | Needs update |

---

## Priority 4: Post-Ablation Default Consolidation

Once current reproduction run completes and we validate results:

### From `POST-ABLATION-DEFAULTS.md`:

| Setting | Current Default | Target Default |
|---------|-----------------|----------------|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `chunk` |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `true` |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | `0.3` |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | `500` |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | `true` |

### Validation Gates (Before Flipping Defaults)

- [ ] Spec 35 ablation complete (chunk vs participant scoring)
- [ ] `*.chunk_scores.json` artifacts exist for both Ollama AND HuggingFace
- [ ] No regressions in primary metrics
- [ ] CI passes with new defaults

---

## Execution Order

```
1. [RUNNING] Reproduction with Ollama chunk scores (tmux `repro`)
   └── Wait for completion, analyze results

2. [NEXT] Generate HuggingFace chunk scores
   └── Run overnight if needed

3. [NEXT] Clean up .env.example
   └── Minimal/advanced split

4. [NEXT] Run reproduction with HuggingFace chunk scores
   └── Compare with Ollama results

5. [AFTER VALIDATION] Flip defaults in config.py
   └── Per POST-ABLATION-DEFAULTS.md

6. [FINAL] Archive problem docs, simplify guidance
```

---

## Related Documents

- `PROBLEM-HUGGINGFACE-CHUNK-SCORES-MISSING.md` — Why we need HF chunk scores
- `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` — Scorer model selection
- `POST-ABLATION-DEFAULTS.md` — What defaults will change
- `CONFIGURATION-PHILOSOPHY.md` — How to think about config
- `SPEC-DOCUMENTATION-GAPS.md` — Doc migration status

---

## Hardware Notes

**Your machine (M1 Max, 64GB)** can:
- Run Gemma 3 27B via Ollama (fits in unified memory)
- Run chunk scoring (CPU-bound, ~10 cores available)
- Run reproduction while chunk scoring runs (if carefully managed)

**Recommendation**: Don't run chunk scoring and reproduction simultaneously.
The LLM calls will compete for the same Ollama instance.

---

*Last updated: 2026-01-01*
