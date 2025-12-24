# Paper Parity Guide: Reproducing the Original Research

**Audience**: Researchers wanting to compare results with the paper
**Related**: [SPEC-003](../specs/SPEC-003-backfill-toggle.md) | [Backfill Explained](../concepts/backfill-explained.md) | [Reproduction Notes](../results/reproduction-notes.md)
**Last Updated**: 2025-12-23

---

## Overview

This guide explains how to run AI Psychiatrist in **paper parity mode** - configuration that matches the original research methodology as closely as possible.

**Good news**: With SPEC-003 implemented, the **default configuration is now paper parity**.

---

## Default Behavior (Paper Parity)

Out of the box, keyword backfill is **OFF** to match the paper methodology:

```bash
# Default behavior - no configuration needed
uv run python scripts/reproduce_results.py --split paper
```

This produces results comparable to the paper:
- Coverage: ~50% (paper reports ~50% abstention rate)
- Pure LLM extraction (no keyword heuristics)

---

## Configuration Options

### Paper Parity Mode (Default)

```bash
# Already the default - backfill OFF
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false  # (this is the default)
```

### Higher Coverage Mode

For clinical utility (more items assessed), enable backfill:

```bash
# Enable backfill for ~74% coverage
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true \
  uv run python scripts/reproduce_results.py --split paper
```

---

## Running Ablation Studies

Compare backfill ON vs OFF:

```bash
# Mode 1: Paper parity (backfill OFF - default)
uv run python scripts/reproduce_results.py --split paper --output results_backfill_off.json

# Mode 2: Higher coverage (backfill ON)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true \
  uv run python scripts/reproduce_results.py --split paper --output results_backfill_on.json
```

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
# Model selection (Paper Section 2.2)
MODEL_QUANTITATIVE_MODEL=gemma3:27b
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b

# Embedding hyperparameters (Paper Appendix D)
EMBEDDING_DIMENSION=4096
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_CHUNK_STEP=2
EMBEDDING_TOP_K_REFERENCES=2

# Sampling (Paper says "fairly deterministic")
MODEL_TEMPERATURE=0.2
MODEL_TEMPERATURE_JUDGE=0.0

# Feedback loop (Paper Section 2.3.1)
FEEDBACK_ENABLED=true
FEEDBACK_MAX_ITERATIONS=10
FEEDBACK_SCORE_THRESHOLD=3

# Quantitative assessment (SPEC-003)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false  # Default - paper parity
QUANTITATIVE_TRACK_NA_REASONS=true
```

---

## Reproduction Steps

```bash
# 1. Create paper-style 58/43/41 split
uv run python scripts/create_paper_split.py --seed 42

# 2. Generate embeddings for training set
uv run python scripts/generate_embeddings.py --split paper-train

# 3. Run reproduction (default = paper parity mode)
uv run python scripts/reproduce_results.py --split paper --few-shot-only
```

---

## Known Gaps

Even with SPEC-003 implemented, some differences remain:

### GAP-001: Unspecified Parameters

The paper says "fairly deterministic parameters" but doesn't specify:
- Exact temperature value (we use 0.2)
- top_k / top_p sampling (we use Ollama defaults)
- Model quantization

See `docs/bugs/gap-001-paper-unspecified-parameters.md`.

### GAP-002: Model Quantization

Ollama's `gemma3:27b` uses GGUF quantization, which may differ from paper's weights.

### GAP-003: Split Membership

The paper doesn't publish exact participant IDs. Our split uses the same algorithm with a fixed seed, but membership may differ.

---

## Related Documentation

- [SPEC-003: Backfill Toggle](../specs/SPEC-003-backfill-toggle.md) - Implementation spec
- [Backfill Explained](../concepts/backfill-explained.md) - How backfill works
- [Coverage Investigation](../bugs/coverage-investigation.md) - Why metrics differ
- [Reproduction Notes](../results/reproduction-notes.md) - Current results
- [Configuration Reference](../reference/configuration.md) - All settings
