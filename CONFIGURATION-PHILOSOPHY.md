# Configuration Philosophy

**Date**: 2025-12-31
**Purpose**: Define what should be configurable vs baked-in defaults.

---

## Core Principle

> **Correct behavior is the default. Broken behavior requires explicit opt-in.**

Not everything needs a flag. Flags add cognitive load and misconfiguration risk.

---

## Configuration Categories

### 1. BAKED-IN DEFAULTS (No Flag Needed)

These are **correct behaviors** that should just work. No user configuration required.

| Behavior | Why It's Baked In |
|----------|-------------------|
| **Fail-fast on errors** (Spec 38) | Research code should crash, not silently degrade |
| **Preserve exception types** (Spec 39) | Correct error handling, not a preference |
| **Batch query embedding** (Spec 37) | Pure optimization, no reason to disable |
| **Temperature = 0.0** | Clinical reproducibility standard (Med-PaLM) |
| **Pydantic AI validation** | Required for structured output |
| **Track N/A reasons** | Diagnostics, no downside |

**These have no env var or the env var is vestigial.** The code just does the right thing.

---

### 2. POST-ABLATION DEFAULTS (Will Be Baked In)

After ablations complete, these become baked-in defaults:

| Setting | Current | Post-Ablation | Why |
|---------|---------|---------------|-----|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `chunk` | Fixes label noise (Spec 35) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `true` | Fixes wrong-item retrieval (Spec 34) |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | `0.3` | Drops garbage references (Spec 33) |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | `500` | Prevents context overflow (Spec 33) |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | `true` | CRAG validation (Spec 36) |

**Post-ablation**: These become defaults. Flags remain ONLY for paper-parity reproduction.

---

### 3. TUNABLE HYPERPARAMETERS (Keep Configurable)

Researchers should experiment with these. They affect results, not correctness.

| Setting | Default | Range | Paper Reference |
|---------|---------|-------|-----------------|
| `EMBEDDING_DIMENSION` | 4096 | 512-8192 | Appendix D |
| `EMBEDDING_CHUNK_SIZE` | 8 | 2-20 | Appendix D |
| `EMBEDDING_CHUNK_STEP` | 2 | 1-8 | Appendix D |
| `EMBEDDING_TOP_K_REFERENCES` | 2 | 1-10 | Appendix D |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | 0.3 | 0.0-1.0 | Spec 33 |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | 500 | 0-2000 | Spec 33 |
| `FEEDBACK_MAX_ITERATIONS` | 10 | 1-20 | Section 2.3.1 |
| `FEEDBACK_SCORE_THRESHOLD` | 3 | 1-4 | Section 2.3.1 |
| `EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM` | 2 | 1-5 | Spec 36 |

**These stay as env vars.** Researchers tune them for ablation studies.

---

### 4. MODEL SELECTION (Always Configurable)

Users must be able to swap models:

| Setting | Default | Purpose |
|---------|---------|---------|
| `MODEL_QUALITATIVE_MODEL` | `gemma3:27b-it-qat` | Qualitative agent |
| `MODEL_JUDGE_MODEL` | `gemma3:27b-it-qat` | Judge agent |
| `MODEL_META_REVIEW_MODEL` | `gemma3:27b-it-qat` | Meta-review agent |
| `MODEL_QUANTITATIVE_MODEL` | `gemma3:27b-it-qat` | Quantitative agent |
| `MODEL_EMBEDDING_MODEL` | `qwen3-embedding:8b` | Embedding model |
| `EMBEDDING_VALIDATION_MODEL` | (judge_model) | CRAG validator |

**These are always configurable.** Different hardware = different models.

---

### 5. INFRASTRUCTURE (Always Configurable)

Environment-specific setup:

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | Request timeout |
| `LLM_BACKEND` | `ollama` | Chat backend |
| `EMBEDDING_BACKEND` | `huggingface` | Embedding backend |
| `DATA_*` paths | `data/...` | Data locations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `API_HOST`, `API_PORT` | `0.0.0.0:8000` | Server binding |

**These are always configurable.** Infrastructure varies by deployment.

---

### 6. DEPRECATED (Remove After Freeze)

These should NOT be used. Flags exist only for historical comparison:

| Setting | Status | Why Deprecated |
|---------|--------|----------------|
| `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` | **OFF permanently** | Flawed heuristic, inflates coverage |
| `QUANTITATIVE_KEYWORD_BACKFILL_CAP` | Irrelevant | Backfill should never be ON |

**Post-publication**: Delete these entirely from codebase.

---

### 7. SAFETY OVERRIDES (Danger Zone)

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

```
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
# To reproduce paper (broken): EMBEDDING_REFERENCE_SCORE_SOURCE=participant
```

---

## Current State vs Target State

### Current (Pre-Ablation)

```bash
# User must enable all these for "correct" behavior:
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

## Summary Table

| Category | Example | Configurable? | Default |
|----------|---------|---------------|---------|
| **Baked-in** | Fail-fast, batch embedding | No | Always ON |
| **Post-ablation** | CRAG, chunk scores | Yes (for now) | Will be ON |
| **Hyperparameters** | top_k, similarity threshold | Yes | Paper-optimal |
| **Models** | quantitative_model | Yes | gemma3:27b |
| **Infrastructure** | OLLAMA_HOST | Yes | localhost |
| **Deprecated** | keyword_backfill | No | Always OFF |
| **Safety overrides** | allow_prompt_mismatch | Yes | Always OFF |

---

## Related Documentation

- `POST-ABLATION-DEFAULTS.md` — When defaults will flip
- `FEATURES.md` — Feature status and configuration
- `docs/reference/configuration.md` — Full config reference
- `.env.example` — Example configuration

---

*"Make the right thing easy and the wrong thing hard."*
