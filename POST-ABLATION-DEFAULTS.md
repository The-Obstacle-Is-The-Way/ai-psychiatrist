# Post-Ablation Default Consolidation

**Date**: 2025-12-31
**Status**: Pending (requires ablation completion)
**Purpose**: After ablations complete, consolidate all fixes as defaults — no flags needed.

---

## Important: On "Paper Parity"

**The paper’s few-shot method (as described) introduces a fundamental label mismatch.**

We initially aimed for "paper parity" — matching the paper's exact methodology. Through
rigorous investigation, we discovered critical issues:

1. **Participant-level scores attached to retrieved chunks**: In our “paper-parity” pipeline,
   the score shown for a retrieved reference chunk is a **participant-level PHQ-8 item score**.
   This creates label noise: a chunk about “career goals” can be shown as `(PHQ8_Sleep Score: 2)`
   even if it contains no sleep evidence. This is not chunk-level ground truth.

2. **Keyword backfill is a heuristic**: Keyword triggers (“sleep”, “tired”, etc.) can increase
   apparent coverage but may introduce false positives and distort selective-prediction metrics.
   It must remain OFF by default and be clearly labeled when used.

3. **Reproducibility is ambiguous**: Despite extensive effort, we have not reproduced the
   paper’s headline improvements in our environment. This could be due to methodology gaps
   (under-specified prompts, artifacts, split details) and/or implementation differences.

**Our stance**: “Paper parity” is still useful as a **historical baseline**, but should not be
the default behavior. We aim for **research-honest behavior**: minimize label noise, avoid
silent heuristics, and fail fast when enabled features are broken.

See `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` for the full analysis.

---

## Executive Summary

We've implemented Specs 33-40 as gated features for ablation purposes. Once validated,
**the fixed behavior should be THE default** — not an opt-in flag.

**Principle**: Correct behavior is the default. Broken behavior requires explicit opt-in
(for historical reproduction only, not recommended).

---

## Current State: Flags Everywhere

| Spec | What It Fixes | Current Default | Should Be |
|------|---------------|-----------------|-----------|
| **35** | Chunk-level scoring | `participant` (broken) | `chunk` (fixed) |
| **34** | Item-tag filtering | `false` (disabled) | `true` (enabled) |
| **33** | Similarity threshold | `0.0` (disabled) | `0.3` (enabled) |
| **33** | Char limit per item | `0` (disabled) | `500` (enabled) |
| **37** | Batch query embedding | `true` | `true` (already correct) |
| **36** | CRAG validation | `false` (disabled) | `true` (enabled) |
| **38** | Skip-if-disabled, crash-if-broken | `true` | `true` (already correct) |
| **39** | Preserve exception types | `true` | `true` (already correct) |
| **40** | Fail-fast embedding generation | `true` | `true` (already correct) |

### Why CRAG (Spec 36) Should Be Default ON (Post-Ablation)

If our goal is **research-honest retrieval** (not paper-parity), reference validation is part of the
“correct” pipeline:

- Spec 34 (item tags) is a **static heuristic** and will miss symptom mentions that don’t match keywords.
- Spec 33 (similarity threshold/budget) is a **quality guardrail**, not a relevance proof.
- Spec 35 fixes **label correctness**, but does not prevent “semantically similar but clinically irrelevant” chunks.
- Spec 36 is the only layer that asks an LLM directly: “Is this reference actually about the target PHQ-8 item?”

In this repo’s research workflow (local Ollama, long-running ablations), **correctness outweighs latency**.
Spec 38 ensures we remain research-honest: if validation is enabled and broken, we crash rather than silently
continuing with unvalidated references.

Note: the code already supports a safe default when `EMBEDDING_VALIDATION_MODEL` is not set: it falls back to
`MODEL_JUDGE_MODEL` (see `server.py` and `scripts/reproduce_results.py`).

---

## The Problem We're Solving

### Why Paper-Parity Defaults Are Wrong

The paper's few-shot methodology has a **fundamental flaw** (see `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md`):

```
Participant 300 has PHQ8_Sleep = 2

Chunk 5 (about career goals):
  "Ellie: what's your dream job
   Participant: open a business"
  → Gets labeled: Sleep Score = 2  ← WRONG (nothing about sleep!)

Chunk 95 (about sleep):
  "Ellie: have you had trouble sleeping
   Participant: yes every night"
  → Gets labeled: Sleep Score = 2  ← CORRECT
```

**Every chunk from a participant gets the SAME score**, regardless of content.

### What We Fixed

| Spec | Fix |
|------|-----|
| **Spec 35** | Score each chunk individually (chunk-level PHQ-8 estimation) |
| **Spec 34** | Only retrieve Sleep-tagged chunks for Sleep queries |
| **Spec 33** | Drop low-similarity and overlength references |
| **Spec 37** | Batch embedding to prevent timeouts |

---

## Consolidated Defaults (Post-Ablation)

### config.py Changes Required

```python
# === EmbeddingSettings ===

# Spec 35: Chunk-level scoring (label-noise reduction)
reference_score_source: Literal["participant", "chunk"] = Field(
    default="chunk",  # CHANGED from "participant"
    description="Source of PHQ-8 scores for retrieved chunks.",
)

# Spec 34: Item-tag filtering
enable_item_tag_filter: bool = Field(
    default=True,  # CHANGED from False
    description="Enable filtering reference chunks by PHQ-8 item tags.",
)

# Spec 33: Retrieval quality guardrails
min_reference_similarity: float = Field(
    default=0.3,  # CHANGED from 0.0
    description="Drop retrieved references below this similarity.",
)

max_reference_chars_per_item: int = Field(
    default=500,  # CHANGED from 0
    description="Max total reference chunk chars per item.",
)

# Spec 36: CRAG-style runtime reference validation
enable_reference_validation: bool = Field(
    default=True,  # CHANGED from False
    description="Enable CRAG-style runtime validation of retrieved references (Spec 36).",
)
validation_model: str = Field(
    default="",
    description=(
        "Model to use for reference validation. If blank, orchestrators fall back to MODEL_JUDGE_MODEL "
        "(see server.py / scripts/reproduce_results.py)."
    ),
)
```

### Reduce the Flag Surface Area

After consolidation, the **core retrieval quality fixes** (Specs 33–35) should not require
users to remember multiple env vars for a “correct” run.

CRAG validation (Spec 36) is part of the “correctness” pipeline and should be default ON post-ablation.
Keep the flag only for targeted ablations / debugging (e.g., `EMBEDDING_ENABLE_REFERENCE_VALIDATION=false`).

### Deprecated Features (DO NOT ENABLE)

```python
# Keyword backfill (QuantitativeSettings) - DEPRECATED heuristic
# Retained for historical comparison only; must remain OFF by default.
enable_keyword_backfill: bool = Field(
    default=False,
    description="DEPRECATED: Do NOT enable. Flawed heuristic retained for ablation only.",
)
```

---

## .env.example Changes Required

### Before (Current)

```bash
# Spec 35: Chunk-level scoring (NOT paper-parity - use for ablation studies)
# Default: "participant" (paper-parity)
# Alternative: "chunk" (experimental - requires running score_reference_chunks.py first)
# EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
```

### After (Post-Ablation)

```bash
# Spec 35: Chunk-level scoring (DEFAULT)
# The "participant" mode is methodologically flawed (assigns participant-level
# scores to arbitrary chunks). Only use for paper-parity reproduction.
# Default: "chunk" (correct, requires chunk_scores.json artifact)
#
# To reproduce paper (broken) behavior:
# EMBEDDING_REFERENCE_SCORE_SOURCE=participant
```

---

## Validation Gates (Before Consolidation)

Do NOT consolidate until all of these are verified:

### Required Validations

- [ ] **Spec 35 ablation complete**: Run 7 (chunk scoring) vs Run 5/6 (participant scoring)
- [ ] **chunk_scores.json artifact exists**: Generated by `score_reference_chunks.py`
- [ ] **No regressions**: Primary metrics (AURC/AUGRC + MAE + coverage) meet or beat baseline
- [ ] **CI passes**: All tests green with new defaults

### Verification Commands

```bash
# Verify artifact exists
ls -la data/embeddings/*.chunk_scores.json

# Verify tests pass with new defaults
uv run pytest -m "unit or integration" -v

# Compare ablation results
cat data/outputs/run_*/metrics.json | jq '.total_score_mae'
```

---

## Migration Checklist

When ready to consolidate:

### 1. Update config.py Defaults

```bash
# File: src/ai_psychiatrist/config.py
# Lines: ~294-315 (EmbeddingSettings)
```

| Field | Old Default | New Default |
|-------|-------------|-------------|
| `reference_score_source` | `"participant"` | `"chunk"` |
| `enable_item_tag_filter` | `False` | `True` |
| `min_reference_similarity` | `0.0` | `0.3` |
| `max_reference_chars_per_item` | `0` | `500` |

### 2. Update .env.example Comments

Flip the framing:
- "chunk" is the default (correct)
- "participant" is opt-in (for paper reproduction only)

### 3. Update CLAUDE.md

Remove ablation-specific instructions. The system "just works" out of the box.

### 4. Update Tests

Tests should assume correct defaults. Add explicit `reference_score_source="participant"` only in paper-parity reproduction tests.

### 5. Archive Ablation Docs

Move to `docs/archive/`:
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md`
- `HYPOTHESIS-ZERO-SHOT-INFLATION.md`
- `PROBLEM-SPEC35-SCORER-MODEL-GAP.md`

These become historical context, not active guidance.

---

## Artifact Requirements

For the consolidated defaults to work, these artifacts MUST exist:

| Artifact | Required For | Generated By |
|----------|--------------|--------------|
| `*.npz` | All few-shot | `generate_embeddings.py` |
| `*.json` | All few-shot | `generate_embeddings.py` |
| `*.meta.json` | All few-shot | `generate_embeddings.py` |
| `*.tags.json` | Spec 34 | `generate_embeddings.py --write-item-tags` |
| `*.chunk_scores.json` | Spec 35 | `score_reference_chunks.py` |
| `*.chunk_scores.meta.json` | Spec 35 | `score_reference_chunks.py` |

**Current tmux session (`run6`)** is generating the chunk_scores artifacts.

---

## Why This Matters

### Before Consolidation
```bash
# User has to know about all these flags:
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500
```

### After Consolidation
```bash
# User just runs the system - it works correctly by default
# (no flags needed)
```

**Complexity is enemy of correctness.** Every flag is a potential misconfiguration.

---

## Related Documentation

- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` — Why participant-level scoring is broken
- `HYPOTHESIS-ZERO-SHOT-INFLATION.md` — Why zero-shot can "cheat"
- `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` — Scorer model selection for chunk scoring
- `docs/reference/features.md` — Feature index + defaults (non-archive canonical)
- `docs/reference/chunk-scoring.md` — Chunk scoring schema/workflow (Spec 35)
- `docs/guides/item-tagging-setup.md` — Tagging setup + schema (Spec 34)
- `docs/guides/debugging-retrieval-quality.md` — Diagnostics flow (Specs 32–36)

---

## Conclusion

After ablations complete:

1. **Flip the defaults** — correct behavior is the baseline
2. **Keep opt-in flags only for paper reproduction** — not the other way around
3. **Simplify the system** — no flags = less misconfiguration risk

The question isn't "should we enable the fixes?" — it's "why would we ever default to broken behavior?"

---

*"Make the right thing easy and the wrong thing hard."*
