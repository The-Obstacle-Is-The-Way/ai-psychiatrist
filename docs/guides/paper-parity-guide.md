# Paper Parity Guide: Reproducing the Original Research

**Audience**: Researchers wanting to compare results with the paper
**Related**: [SPEC-003](../specs/SPEC-003-backfill-toggle.md) | [Backfill Explained](../concepts/backfill-explained.md) | [Reproduction Notes](../results/reproduction-notes.md)
**Last Updated**: 2025-12-23

---

> **⚠️ STATUS: BLOCKED ON SPEC-003**
>
> This guide documents the **target state** after [SPEC-003](../specs/SPEC-003-backfill-toggle.md) is implemented.
> Currently, keyword backfill **cannot be disabled** - it always runs.
>
> **Current workaround**: None. You must implement SPEC-003 first.

---

## Overview

This guide explains how to run AI Psychiatrist in **paper parity mode** - configuration that matches the original research methodology as closely as possible.

### Why Paper Parity Matters

Our implementation includes a keyword backfill mechanism that causes metrics to diverge from the paper:

| Metric | Paper | Current Implementation |
|--------|-------|------------------------|
| Coverage | ~50% | ~74% (backfill always ON) |
| Item MAE | 0.619 | ~0.75 |

The paper tests **pure LLM capability**. We currently test **LLM + rule-based heuristics**.

### What's Blocking Paper Parity?

The `_keyword_backfill()` function in `src/ai_psychiatrist/agents/quantitative.py:228-268` **always runs**. There is no configuration to disable it.

See [SPEC-003](../specs/SPEC-003-backfill-toggle.md) for the implementation plan.

---

## Current State (As of 2025-12-23)

### What EXISTS Now

These settings are implemented and can be configured:

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
```

### What DOES NOT Exist (Requires SPEC-003)

❌ `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` - Not implemented
❌ `QUANTITATIVE_TRACK_NA_REASONS` - Not implemented
❌ `settings.quantitative.*` - No `QuantitativeSettings` class exists

---

## After SPEC-003 Implementation

Once SPEC-003 is implemented, you will be able to:

### 1. Disable Keyword Backfill

```bash
# Paper parity mode (pure LLM) - FUTURE
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
```

### 2. Run Ablation Studies

```bash
# Mode 1: Default (backfill ON) - FUTURE
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true \
  uv run python scripts/reproduce_results.py --split paper --output results_backfill_on.json

# Mode 2: Paper parity (backfill OFF) - FUTURE
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false \
  uv run python scripts/reproduce_results.py --split paper --output results_backfill_off.json
```

### 3. Track N/A Reasons

After implementation, each N/A will include a reason:
- `NO_MENTION` - No evidence from LLM and no keyword matches
- `LLM_ONLY_MISSED` - LLM missed it but keywords matched (backfill OFF)
- `KEYWORDS_INSUFFICIENT` - Keywords matched but still insufficient
- `SCORING_REFUSED` - Evidence exists but LLM declined to score

---

## What You CAN Do Now

### Run Reproduction (With Backfill)

```bash
# 1. Create paper-style 58/43/41 split
uv run python scripts/create_paper_split.py --seed 42

# 2. Generate embeddings for training set
uv run python scripts/generate_embeddings.py --split paper-train

# 3. Run reproduction (few-shot mode)
# NOTE: This runs WITH backfill - results will differ from paper
uv run python scripts/reproduce_results.py --split paper --few-shot-only
```

### Understand the Divergence

Current results (with backfill always ON):

| Metric | Paper | Our Results | Reason |
|--------|-------|-------------|--------|
| Coverage | ~50% | 74.1% | Backfill adds evidence |
| Item MAE | 0.619 | 0.753 | More predictions = more errors |

This is expected behavior until SPEC-003 is implemented.

---

## Known Gaps (Even After SPEC-003)

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

## How to Report Current Results

Until SPEC-003 is implemented, be explicit about the backfill:

```markdown
## Results

We evaluated AI Psychiatrist with keyword backfill enabled (cannot be disabled in current version):

- Coverage: 74.1%
- Item MAE: 0.753
- Methodology: LLM + keyword backfill (always ON)

Paper reference (likely pure LLM):
- Coverage: ~50%
- Item MAE: 0.619

Note: Direct comparison is invalid due to backfill divergence.
See GitHub Issue #49 for paper parity implementation status.
```

---

## Related Documentation

- [SPEC-003: Backfill Toggle](../specs/SPEC-003-backfill-toggle.md) - Implementation spec (DRAFT)
- [Backfill Explained](../concepts/backfill-explained.md) - How backfill works
- [Coverage Investigation](../bugs/coverage-investigation.md) - Why metrics differ
- [Reproduction Notes](../results/reproduction-notes.md) - Current results
- [Configuration Reference](../reference/configuration.md) - Implemented settings
