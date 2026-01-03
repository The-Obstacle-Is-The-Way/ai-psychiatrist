# Configuration Philosophy

**Date**: 2026-01-02
**Purpose**: Define what should be configurable vs baked-in defaults.

---

## Core Principle

> **Correct behavior is the default. Broken behavior requires explicit opt-in.**

Not everything needs a flag. Flags add cognitive load and misconfiguration risk.

---

## SSOT + Terminology

- **SSOT for config names + code defaults**: `src/ai_psychiatrist/config.py`.
- **Recommended runtime baseline**: `.env.example` (what most runs use once copied to `.env`).
- When this doc says "default", it should be read as:
  - **Code default** = what happens with no `.env` overrides (or in tests where `.env` is ignored).
  - **Recommended `.env` baseline** = what we expect for normal research runs.

---

## On the Legacy Baseline (Paper-Derived)

**The paper's few-shot method (as described) introduces a fundamental label mismatch.**

We initially aimed to match the paper's reported methodology. Through
rigorous investigation, we discovered critical issues:

1. **Participant-level scores attached to retrieved chunks**: In the legacy baseline pipeline,
   the score shown for a retrieved reference chunk is a **participant-level PHQ-8 item score**.
   This creates label noise: a chunk about "career goals" can be shown as `(PHQ8_Sleep Score: 2)`
   even if it contains no sleep evidence. This is not chunk-level ground truth.

2. **Keyword backfill was a heuristic**: Keyword triggers ("sleep", "tired", etc.) can increase
   apparent coverage but may introduce false positives and distort selective-prediction metrics.
   This feature was removed in Spec 047; historical context is kept under `docs/_archive/`.

3. **Reproducibility is ambiguous**: Despite extensive effort, we have not reproduced the
   paper's headline improvements in our environment. This could be due to methodology gaps
   (under-specified prompts, artifacts, split details) and/or implementation differences.

**Our stance**: the legacy baseline is still useful as a **historical baseline**, but should not be
the default behavior. We aim for **research-honest behavior**: minimize label noise, avoid
silent heuristics, and fail fast when enabled features are broken.

See `_archive/misc/HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` for the full analysis.

---

## Configuration Categories

### 1. Always-On Correctness Invariants (Do Not "Tune")

These are **correctness behaviors**. Some have knobs in the codebase, but treating them as "tunable"
creates misconfiguration risk and can corrupt research runs.

| Behavior | Where Enforced | Config Knob? | Notes |
|----------|----------------|--------------|-------|
| **Skip-if-disabled, crash-if-broken** (Spec 38) | `ReferenceStore` + `ReferenceValidation` | No (automatic) | Disabled feature = no file I/O; enabled feature = strict load + validate |
| **Preserve exception types** (Spec 39) | Agents | No (automatic) | Log `error_type`, then `raise` to preserve the original exception |
| **Fail-fast embedding generation** (Spec 40) | `scripts/generate_embeddings.py` | CLI (`--allow-partial`) | Strict-by-default; partial is for debugging only |
| **Pydantic AI structured output** | Agents | Yes (`PYDANTIC_AI_ENABLED`) | Disabling is **not supported** (agents will raise; legacy fallback removed) |
| **Track N/A reasons** | Quantitative agent | Yes (`QUANTITATIVE_TRACK_NA_REASONS`) | Default ON; small runtime cost but improves run diagnostics |

**Rule**: if disabling a "correctness invariant" is possible, it must either:
- be clearly documented as **unsupported**, or
- be restricted to a **debug-only** escape hatch (explicit, noisy, and off by default).

---

### 2. Post-Ablation Defaults (Will Be Baked In)

After ablations complete, these become baked-in defaults:

| Setting | Code Default | `.env.example` Baseline | Post-Ablation Default | Why |
|---------|--------------|--------------------------|------------------------|-----|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `participant` | `chunk` | Fixes participant-score-on-chunk mismatch (Spec 35) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `true` | `true` | Improves item-level retrieval precision (Spec 34) |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | `0.3` | `0.3` | Drops low-similarity references (Spec 33) |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | `500` | `500` | Prevents context bloat (Spec 33) |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | `false` | `true` | CRAG validation to reject irrelevant references (Spec 36) |

**Post-ablation**: These become defaults. Flags remain ONLY for legacy baseline reproduction.

#### Why CRAG (Spec 36) Should Be Default ON

If our goal is **research-honest retrieval** (not the legacy baseline), reference validation is part of the
"correct" pipeline:

- Spec 34 (item tags) is a **static heuristic** and will miss symptom mentions that don't match keywords.
- Spec 33 (similarity threshold/budget) is a **quality guardrail**, not a relevance proof.
- Spec 35 fixes **label correctness**, but does not prevent "semantically similar but clinically irrelevant" chunks.
- Spec 36 is the only layer that asks an LLM directly: "Is this reference actually about the target PHQ-8 item?"

In this repo's research workflow (local Ollama, long-running ablations), **correctness outweighs latency**.

---

### 3. Tunable Hyperparameters (Keep Configurable)

Researchers should experiment with these. They affect results, not correctness.

**Important**: Some "hyperparameters" are **index-time** and require regenerating artifacts.
Changing them without regenerating embeddings/tags/chunk-scores will either crash or silently change
the retrieval universe.

| Setting | Code Default | `.env.example` Baseline | Runtime-Only? | Notes |
|---------|--------------|--------------------------|---------------|-------|
| `EMBEDDING_DIMENSION` | 4096 | 4096 | No | Must match embedding model + stored artifact dimension |
| `EMBEDDING_CHUNK_SIZE` | 8 | 8 | No | Requires regenerating `.npz` + sidecars |
| `EMBEDDING_CHUNK_STEP` | 2 | 2 | No | Requires regenerating `.npz` + sidecars |
| `EMBEDDING_TOP_K_REFERENCES` | 2 | 2 | Yes | Paper Appendix D chose `2`; can be tuned without reindex |
| `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING` | `true` | `true` | Yes | Spec 37 stability/perf default |
| `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS` | 300 | 300 | Yes | Stability knob (Spec 37) |
| `EMBEDDING_ENABLE_RETRIEVAL_AUDIT` | `false` | `true` | Yes | Diagnostics only (Spec 32); recommended ON for research runs |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | 0.0 | 0.3 | Yes | Retrieval-time filter; safe to tune |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | 0 | 500 | Yes | Retrieval-time budget; safe to tune |
| `FEEDBACK_MAX_ITERATIONS` | 10 | 10 | Yes | More iterations increases runtime and can change outputs |
| `FEEDBACK_SCORE_THRESHOLD` | 3 | 3 | Yes | Controls when refinement triggers |
| `EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM` | 2 | 2 | Yes | Bounds CRAG keep-set per item |

**These stay as env vars.** Researchers tune them for ablation studies.

---

### 4. Model Selection (Always Configurable)

Users must be able to swap models:

| Setting | Code Default | `.env.example` Baseline | Purpose |
|---------|--------------|--------------------------|---------|
| `MODEL_QUALITATIVE_MODEL` | `gemma3:27b` | `gemma3:27b-it-qat` | Qualitative agent |
| `MODEL_JUDGE_MODEL` | `gemma3:27b` | `gemma3:27b-it-qat` | Judge agent |
| `MODEL_META_REVIEW_MODEL` | `gemma3:27b` | `gemma3:27b-it-qat` | Meta-review agent |
| `MODEL_QUANTITATIVE_MODEL` | `gemma3:27b` | `gemma3:27b-it-qat` | Quantitative agent |
| `MODEL_EMBEDDING_MODEL` | `qwen3-embedding:8b` | `qwen3-embedding:8b` | Embedding model |
| `MODEL_TEMPERATURE` | `0.0` | `0.0` | Keep `0.0` for reproducibility |
| `EMBEDDING_VALIDATION_MODEL` | `""` (falls back) | (unset) | Effective default is `MODEL_JUDGE_MODEL` when validation is enabled |

**These are always configurable.** Different hardware = different models.

---

### 5. Infrastructure (Always Configurable)

Environment-specific setup:

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | Request timeout |
| `PYDANTIC_AI_TIMEOUT_SECONDS` | (unset) | Timeout for Pydantic AI calls |
| `LLM_BACKEND` | `ollama` | Chat backend |
| `EMBEDDING_BACKEND` | `huggingface` | Embedding backend |
| `HF_DEFAULT_CHAT_TIMEOUT` | `180` | HuggingFace chat timeout |
| `HF_DEFAULT_EMBED_TIMEOUT` | `120` | HuggingFace embed timeout |
| `DATA_*` paths | `data/...` | Data locations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `API_HOST`, `API_PORT` | `0.0.0.0:8000` | Server binding |

**These are always configurable.** Infrastructure varies by deployment.

---

### 6. Embedding Artifact Selection (Critical)

**Problem Identified**: Embedding artifacts and chunk scores must be generated separately for each backend.
HuggingFace embeddings (FP16) produce higher quality similarity scores than Ollama (Q4_K_M).

| Embeddings File | Backend | Precision | Quality |
|-----------------|---------|-----------|---------|
| `huggingface_qwen3_8b_paper_train_participant_only` | HuggingFace | FP16 | **Higher** |
| `ollama_qwen3_8b_paper_train_participant_only` | Ollama | Q4_K_M | Lower |

**Recommendation**:

1. **For best quality**: Use HuggingFace embeddings (`EMBEDDING_BACKEND=huggingface`)
2. **For accessibility**: Use Ollama if HuggingFace deps unavailable

**Important**: `EMBEDDING_EMBEDDINGS_FILE` and `EMBEDDING_BACKEND` should be coherent:
- HuggingFace backend → `huggingface_*` embeddings file
- Ollama backend → `ollama_*` embeddings file

**Chunk Scores Dependency**: If `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`, the corresponding
`.chunk_scores.json` file must exist for the selected embeddings file.

---

### 7. Removed Features

These are no longer present in the codebase (historical context is kept under `docs/_archive/`):

- Keyword backfill (Spec 047)

---

### 8. Safety Overrides (Danger Zone)

These bypass safety checks. Require explicit acknowledgment:

| Setting | Default | What It Bypasses |
|---------|---------|------------------|
| `EMBEDDING_ALLOW_CHUNK_SCORES_PROMPT_HASH_MISMATCH` | `false` | Prompt change detection |
| `--allow-same-model` (CLI) | N/A | Scorer circularity check |
| `--allow-partial` (CLI) | N/A | Fail-fast embedding generation |

**These should be OFF by default.** Explicit opt-in for known risks.

---

## Decision Framework

When adding a new setting, ask:

```text
1. Is this CORRECT BEHAVIOR vs BROKEN BEHAVIOR?
   → Correct = bake it in, no flag
   → Broken = require explicit opt-in (for legacy only)

2. Is this a RESEARCH HYPERPARAMETER?
   → Yes = make it tunable with env var
   → No = don't add a flag

3. Is this INFRASTRUCTURE-SPECIFIC?
   → Yes = make it configurable
   → No = use sensible default

4. Does this BYPASS SAFETY CHECKS?
   → Yes = default OFF, require explicit opt-in
   → No = default ON if it's correct behavior
```

---

## Anti-Patterns

### DON'T: Add flags for "flexibility"

```python
# BAD: Flag for something that should always be true
ENABLE_STRUCTURED_OUTPUT = True  # Why would you disable this?
```

### DON'T: Default broken behavior

```python
# BAD: Broken behavior as default
reference_score_source: str = "participant"  # Known to be wrong
```

### DON'T: Hide correctness behind opt-in

```python
# BAD: Correct behavior requires user action
enable_reference_validation: bool = False  # CRAG is "gold standard" but OFF?
```

### DO: Make correct behavior the default

```python
# GOOD: Correct behavior, opt-out for legacy
reference_score_source: str = "chunk"  # Correct default
# Legacy baseline (paper-derived, known flawed): EMBEDDING_REFERENCE_SCORE_SOURCE=participant
```

---

## Current State vs Target State

### Current (Pre-Ablation)

```bash
# To run the full "correct" retrieval pipeline today, ensure these are set.
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true
```

**Problem**: 5 flags to get correct behavior. Easy to misconfigure.

### Target (Post-Ablation)

```bash
# User runs system - it works correctly by default
# (no flags needed)

# ONLY if reproducing paper (broken) baseline:
EMBEDDING_REFERENCE_SCORE_SOURCE=participant
EMBEDDING_ENABLE_ITEM_TAG_FILTER=false
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.0
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=0
EMBEDDING_ENABLE_REFERENCE_VALIDATION=false
```

**Better**: Correct by default. Flags only for legacy reproduction.

---

## Post-Ablation Migration

### Validation Gates (Before Consolidation)

Do NOT consolidate defaults until all of these are verified:

- [ ] **Spec 35 ablation complete**: Chunk scoring vs participant scoring comparison
- [ ] **chunk_scores.json artifact exists**: Generated by `score_reference_chunks.py`
- [ ] **No regressions**: Primary metrics (AURC/AUGRC + MAE + coverage) meet or beat baseline
- [ ] **CI passes**: All tests green with new defaults

### config.py Changes Required

```python
# === EmbeddingSettings ===

# Spec 35: Chunk-level scoring (label-noise reduction)
reference_score_source: Literal["participant", "chunk"] = Field(
    default="chunk",  # CHANGED from "participant"
)

# Spec 34: Item-tag filtering
enable_item_tag_filter: bool = Field(
    default=True,  # CHANGED from False
)

# Spec 33: Retrieval quality guardrails
min_reference_similarity: float = Field(
    default=0.3,  # CHANGED from 0.0
)

max_reference_chars_per_item: int = Field(
    default=500,  # CHANGED from 0
)

# Spec 36: CRAG-style runtime reference validation
enable_reference_validation: bool = Field(
    default=True,  # CHANGED from False
)
```

### Artifact Requirements

For the consolidated defaults to work, these artifacts MUST exist:

| Artifact | Required For | Generated By |
|----------|--------------|--------------|
| `*.npz` | All few-shot | `generate_embeddings.py` |
| `*.json` | All few-shot | `generate_embeddings.py` |
| `*.meta.json` | All few-shot | `generate_embeddings.py` |
| `*.tags.json` | Spec 34 | `generate_embeddings.py --write-item-tags` |
| `*.chunk_scores.json` | Spec 35 | `score_reference_chunks.py` |
| `*.chunk_scores.meta.json` | Spec 35 | `score_reference_chunks.py` |

---

## Summary Table

| Category | Example | Configurable? | Default |
|----------|---------|---------------|---------|
| **Invariants** | Spec 38/39/40 semantics | No | Always ON |
| **Post-ablation retrieval** | CRAG, chunk scores, tag filter | Yes (for now) | Will be ON |
| **Performance/stability** | batch query embedding, timeouts | Yes | Default ON |
| **Hyperparameters** | top_k, thresholds, feedback | Yes | Baseline values |
| **Models** | quantitative_model | Yes | Code: `gemma3:27b` |
| **Infrastructure** | OLLAMA_HOST | Yes | localhost |
| **Removed** | keyword_backfill (Spec 047) | No | Removed |
| **Safety overrides** | allow_prompt_mismatch | Yes | Always OFF |

---

## Related Documentation

- [Configuration Reference](./configuration.md) — Full settings reference
- `.env.example` (repository root) — Example configuration
- `_archive/misc/HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` — Why participant-level scoring is broken

---

*"Make the right thing easy and the wrong thing hard."*
