# Paper Parity Guide: Reproducing the Original Research

**Audience**: Researchers wanting to compare results with the paper
**Related**: [Backfill Explained](../concepts/backfill-explained.md) | [Configuration Reference](../reference/configuration.md) | [Reproduction Notes](../results/reproduction-results.md)
**Last Updated**: 2026-01-01

---

## Overview

This guide explains how to run AI Psychiatrist in **paper parity mode** — configuration that matches the original paper’s methodology as closely as possible.

Important framing:
- “Paper parity” is about reproducing the **paper’s method**, even if that method is later shown to be flawed.
- This repo’s recommended `.env.example` may enable additional correctness/reliability features. For paper parity, set the flags explicitly (see below).

---

## Paper Parity Configuration (Explicit)

Paper parity means:
- keyword backfill OFF
- few-shot references use **participant-level** labels (paper design)
- retrieval uses no thresholds/budgets/tags/CRAG filters

Set:

```bash
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false

EMBEDDING_REFERENCE_SCORE_SOURCE=participant
EMBEDDING_ENABLE_ITEM_TAG_FILTER=false
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.0
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=0
EMBEDDING_ENABLE_REFERENCE_VALIDATION=false
```

Notes:
- Spec 37 (batch query embedding + query timeout) may still be enabled by default; it is a performance/stability change and does not change retrieval semantics.
- The prompt delimiter uses a proper XML closing tag (`</Reference Examples>`) instead of the notebook’s duplicate-open tag style; see `docs/concepts/few-shot-prompt-format.md`.

---

## Configuration Options

### Keyword Backfill OFF (Paper Parity)

```bash
# Code default: backfill OFF
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
```

### Higher Coverage Mode

Backfill is deprecated and is **not recommended** for new work. If you need a historical ablation
to match an older paper-code behavior, you can enable backfill explicitly:

```bash
# Enable backfill (deprecated; increases coverage but can inflate it incorrectly)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true \
  uv run python scripts/reproduce_results.py --split paper
```

---

## Running Ablation Studies

Compare backfill ON vs OFF:

```bash
# Mode 1: Paper parity (backfill OFF)
uv run python scripts/reproduce_results.py --split paper

# Mode 2: Higher coverage (backfill ON)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true \
  uv run python scripts/reproduce_results.py --split paper
```

Notes:
- `scripts/reproduce_results.py` prints the exact output path as: `Results saved to: ...`
- Output filename format: `{mode}_{split}_backfill-{on,off}_{YYYYMMDD_HHMMSS}.json` (see `generate_output_filename()` in `src/ai_psychiatrist/services/experiment_tracking.py`).

---

## N/A Reason Tracking

Each N/A result now includes a reason for debugging:

| Reason | Description |
|--------|-------------|
| `NO_MENTION` | Neither LLM nor keywords found any evidence |
| `LLM_ONLY_MISSED` | LLM missed it but keywords would have matched (backfill OFF) |
| `KEYWORDS_INSUFFICIENT` | Keywords matched but still insufficient for scoring |
| `SCORE_NA_WITH_EVIDENCE` | LLM extracted evidence but scoring still returned N/A |

---

## Full Paper Parity Configuration

```bash
# Backends (paper-parity / no HF deps)
LLM_BACKEND=ollama
EMBEDDING_BACKEND=ollama

# Select a reference embeddings artifact generated with the same backend.
# Recommended: use the modern, namespaced paper-train artifact:
EMBEDDING_EMBEDDINGS_FILE=ollama_qwen3_8b_paper_train
# Legacy alternative (if you generated/kept it):
# EMBEDDING_EMBEDDINGS_FILE=paper_reference_embeddings

# Model selection (Paper Section 2.2)
# Note: Paper likely used BF16 weights; both Ollama variants are quantized.
# Use gemma3:27b-it-qat for faster inference, or gemma3:27b for name parity.
MODEL_QUANTITATIVE_MODEL=gemma3:27b-it-qat
MODEL_QUALITATIVE_MODEL=gemma3:27b-it-qat
MODEL_JUDGE_MODEL=gemma3:27b-it-qat
MODEL_META_REVIEW_MODEL=gemma3:27b-it-qat
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b

# Embedding hyperparameters (Paper Appendix D)
EMBEDDING_DIMENSION=4096
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_CHUNK_STEP=2
EMBEDDING_TOP_K_REFERENCES=2
EMBEDDING_MIN_EVIDENCE_CHARS=8

# Few-shot retrieval must be set explicitly for paper parity (avoid non-paper filters):
EMBEDDING_REFERENCE_SCORE_SOURCE=participant
EMBEDDING_ENABLE_ITEM_TAG_FILTER=false
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.0
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=0
EMBEDDING_ENABLE_REFERENCE_VALIDATION=false
EMBEDDING_ENABLE_RETRIEVAL_AUDIT=false

# Sampling (Evidence-based clinical AI defaults)
MODEL_TEMPERATURE=0.0

# Feedback loop (Paper Section 2.3.1)
FEEDBACK_ENABLED=true
FEEDBACK_MAX_ITERATIONS=10
FEEDBACK_SCORE_THRESHOLD=3

# Quantitative assessment (SPEC-003)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false  # Default - paper parity
QUANTITATIVE_TRACK_NA_REASONS=true

# Pydantic AI (Spec 13 - enabled by default since 2025-12-26)
# Adds structured validation + automatic retries; no legacy fallback (fail-fast on errors)
PYDANTIC_AI_ENABLED=true
PYDANTIC_AI_RETRIES=3
```

---

## Reproduction Steps

```bash
# 1. Pull required models
ollama pull gemma3:27b-it-qat  # or gemma3:27b
ollama pull qwen3-embedding:8b

# 2. Create paper ground truth 58/43/41 split
uv run python scripts/create_paper_split.py --verify

# 3. Generate embeddings for training set
# Generate embeddings with the SAME backend you will use at runtime.
# For paper parity (Ollama embeddings):
EMBEDDING_BACKEND=ollama uv run python scripts/generate_embeddings.py --split paper-train
# Then ensure `EMBEDDING_EMBEDDINGS_FILE` points at the generated basename (e.g., `ollama_qwen3_8b_paper_train`).
#
# For higher-precision similarity (not paper-parity), you can generate HuggingFace embeddings instead:
# uv run python scripts/generate_embeddings.py --split paper-train

# Optional (Spec 34, not paper-parity): add `--write-item-tags` to generate a `.tags.json` sidecar for item-tag filtering, then set `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true` for runs.

# 4. Run reproduction (Pydantic AI enabled by default)
uv run python scripts/reproduce_results.py --split paper --few-shot-only
```

---

## Known Gaps

Even with SPEC-003 implemented, some differences remain:

### GAP-001: Unspecified Parameters

The paper says "fairly deterministic parameters" but doesn't specify exact values.
We use evidence-based clinical AI defaults: `temperature=0.0` (no top_k/top_p).

See [Agent Sampling Registry](../reference/agent-sampling-registry.md) for citations.

### GAP-002: Model Quantization

Ollama's `gemma3:27b` uses GGUF quantization, which may differ from paper's weights.

### GAP-003: Split Membership (Fixed)

The repo now uses the paper’s **ground truth** split membership reverse-engineered from the authors’ published output files.
See `docs/data/paper-split-registry.md` and `docs/data/paper-split-methodology.md`.

---

## Related Documentation

- **Preflight Checklists** (run before every reproduction):
  - [Zero-Shot Preflight](./preflight-checklist-zero-shot.md) - Pre-run verification for zero-shot
  - [Few-Shot Preflight](./preflight-checklist-few-shot.md) - Pre-run verification for few-shot
- [Backfill Explained](../concepts/backfill-explained.md) - How backfill works
- [Reproduction Notes](../results/reproduction-results.md) - Current results
- [Configuration Reference](../reference/configuration.md) - All settings
