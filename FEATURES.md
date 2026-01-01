# Feature Reference

Quick reference for all implemented features and how to enable them.

**Last Updated**: 2026-01-01

> **Post-Ablation Note**: After ablations complete, many "OFF" defaults will flip to "ON".
> See [`POST-ABLATION-DEFAULTS.md`](POST-ABLATION-DEFAULTS.md) for the consolidation plan.

---

## Feature Status Overview

| Feature | Spec | Status | Default | Config | Paper-Parity Impact |
|---------|------|--------|---------|--------|---------------------|
| Retrieval audit logging | 32 | Implemented | OFF | `EMBEDDING_ENABLE_RETRIEVAL_AUDIT` | **Neutral** (observability only) |
| Similarity threshold | 33 | Implemented | OFF (`0.0`) | `EMBEDDING_MIN_REFERENCE_SIMILARITY` | **Not paper-parity** (changes retrieval) |
| Context budget | 33 | Implemented | OFF (`0`) | `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | **Not paper-parity** (changes prompt content) |
| Item-tag filtering | 34 | Implemented | OFF | `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | **Not paper-parity** (changes retrieval candidate set) |
| Chunk-level scoring | 35 | Implemented | OFF | `EMBEDDING_REFERENCE_SCORE_SOURCE` | **Not paper-parity** (new labels) |
| CRAG reference validation | 36 | Implemented | OFF | `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | **Not paper-parity** (adds LLM filter) |
| Batch query embedding | 37 | Implemented | ON | `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING` | **Neutral** (performance-only) |
| Query embedding timeout | 37 | Implemented | ON (`300s`) | `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS` | **Neutral** (stability-only) |
| Conditional feature loading | 38 | Implemented | ON | (automatic) | **Neutral** (failure semantics only) |
| Preserve exception types | 39 | Implemented | ON | (automatic) | **Neutral** (error semantics only) |
| Fail-fast embedding generation | 40 | Implemented | ON | `scripts/generate_embeddings.py --allow-partial` | **Neutral** (artifact safety only) |

---

## Quick Start: Enable All Research Features

Add to your `.env`:

```bash
# === Spec 33: Retrieval Quality Guardrails ===
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3    # Drop low-similarity references
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500 # Limit context per PHQ-8 item

# === Spec 34: Item-Tag Filtering ===
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true     # Filter refs by PHQ-8 item tags

# === Spec 35: Chunk-Level Scoring (Experimental) ===
# WARNING: Not paper-parity! Use for ablation studies only.
# EMBEDDING_REFERENCE_SCORE_SOURCE=chunk  # Use per-chunk scores instead of participant-level

# === Spec 36: CRAG Reference Validation ===
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true # LLM validates each reference at runtime
# Optional: override validation model (default: MODEL_JUDGE_MODEL)
# EMBEDDING_VALIDATION_MODEL=gemma3:27b-it-qat
# Optional: cap accepted refs per item after validation
# EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM=2

# === Spec 32: Retrieval Audit Logging ===
EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true     # Log similarity scores and retrievals
```

---

## Feature Details

### Spec 33: Retrieval Quality Guardrails

**Problem**: Low-similarity references inject noise into few-shot prompts.

**Solution**: Two guardrails:
1. **Similarity threshold**: Drop references below a minimum similarity score
2. **Context budget**: Limit total characters per PHQ-8 item

**Configuration**:
```bash
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3    # 0.0 = disabled
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500 # 0 = disabled
```

**Docs**:
- [`docs/pipeline-internals/features.md`](docs/pipeline-internals/features.md) (feature index)
- [`docs/embeddings/debugging-retrieval-quality.md`](docs/embeddings/debugging-retrieval-quality.md) (how to interpret diagnostics)

---

### Spec 34: Item-Tagged Reference Embeddings

**Problem**: Retrieved chunks may be about the wrong PHQ-8 item (e.g., retrieving sleep content for appetite assessment).

**Solution**: Tag each chunk with relevant PHQ-8 items at index time, filter at retrieval time.

**Requirements**:
1. Generate embeddings with tags: `python scripts/generate_embeddings.py --write-item-tags`
2. Enable at runtime: `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`

**Configuration**:
```bash
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
```

**Docs**: [`docs/embeddings/item-tagging-setup.md`](docs/embeddings/item-tagging-setup.md)

---

### Spec 35: Offline Chunk-Level PHQ-8 Scoring

**Problem**: Participant-level scores attached to arbitrary chunks create label mismatch (chunk says "I'm fine" but label is 3).

**Solution**: Score each chunk individually using LLM, attach chunk-specific scores.

**Requirements**:
1. Generate chunk scores: `python scripts/score_reference_chunks.py`
2. Enable at runtime: `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`

**Configuration**:
```bash
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk  # default: "participant"
```

**WARNING**: This is NOT paper-parity. Label runs as "experimental" in results.

**Docs**: [`docs/embeddings/chunk-scoring.md`](docs/embeddings/chunk-scoring.md)

---

### Spec 36: CRAG Reference Validation

**Problem**: Some retrieved references are irrelevant despite high similarity scores.

**Solution**: LLM validates each reference at runtime (CRAG-style).

**Requirements**:
- Just enable: `EMBEDDING_ENABLE_REFERENCE_VALIDATION=true`

**Configuration**:
```bash
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true  # default: false
```

**Trade-off**: Adds ~1 LLM call per reference (latency cost).

**Docs**: [`docs/statistics/crag-validation-guide.md`](docs/statistics/crag-validation-guide.md)

---

### Spec 32: Few-Shot Retrieval Diagnostics

**Problem**: Hard to debug why few-shot retrieval is returning bad references.

**Solution**: Audit logging with similarity scores and chunk content.

**Configuration**:
```bash
EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true
```

**Output**: Logs each retrieval with similarity scores, chunk text, and participant ID.

**Docs**: [`docs/embeddings/debugging-retrieval-quality.md`](docs/embeddings/debugging-retrieval-quality.md)

---

### Spec 37: Batch Query Embedding

**Problem**: 8 separate embedding calls per participant (one per PHQ-8 item).

**Solution**: Batch all 8 items into a single embedding call.

**Configuration** (optional overrides):
```bash
# Default: true (batch all item queries into one embedding request per participant)
EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING=true

# Default: 300s (was previously hard-coded to 120s)
EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS=300
```

**Impact**: 8x reduction in embedding API calls.

**Docs**: [`docs/pipeline-internals/features.md`](docs/pipeline-internals/features.md)

---

### Spec 38: Conditional Feature Loading (Skip-If-Disabled, Crash-If-Broken)

**Problem**: Optional artifacts (e.g., `{name}.tags.json`) were being loaded/validated even when the feature was disabled, and some code paths silently fell back to “empty” data.

**Solution**:
- If a feature is **disabled**, do **no** file I/O for its artifacts.
- If a feature is **enabled**, missing/invalid artifacts **crash** with a clear error.

**Docs**: [`docs/developer/error-handling.md`](docs/developer/error-handling.md)

---

### Spec 39: Preserve Exception Types

**Problem**: Agent layers were catching broad exceptions and rethrowing `ValueError`, masking the true error types (timeouts vs parsing vs validation).

**Solution**: Log `error_type` and re-raise the original exception type.

**Docs**:
- [`docs/developer/error-handling.md`](docs/developer/error-handling.md)
- [`docs/developer/exceptions.md`](docs/developer/exceptions.md)

---

### Spec 40: Fail-Fast Embedding Generation

**Problem**: `generate_embeddings.py` silently skips failed participants/chunks.

**Solution**: Crash by default, opt-in to partial output with `--allow-partial`.

**Usage**:
```bash
# Strict mode (default) - any failure crashes
python scripts/generate_embeddings.py --split paper-train

# Partial mode - skips failures, writes manifest
python scripts/generate_embeddings.py --split paper-train --allow-partial
```

**Exit codes**:
- `0`: Success (all participants processed)
- `1`: Failure (any error in strict mode)
- `2`: Partial success (some skips in `--allow-partial` mode)

**Docs**: [`docs/embeddings/embedding-generation.md`](docs/embeddings/embedding-generation.md)

---

## Paper-Parity vs. Experimental Features

| Mode | Features Enabled | Use Case |
|------|------------------|----------|
| **Paper-Method Baseline** | Spec 32 (audit, optional) | Closest to paper method (no retrieval guardrails, no tags, no chunk scoring, no CRAG) |
| **Patched Reproduction Baseline** | + Spec 33 (guardrails), Spec 34 (tags) | Debugging + retrieval-quality improvements (NOT paper-parity) |
| **Experimental** | + Spec 35 (chunk scoring), Spec 36 (CRAG) | Ablation studies (explicitly non-paper) |

**Paper-method baseline config** (`.env`):
```bash
# Optional instrumentation only:
# EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true

# Paper-method (disable post-hoc retrieval changes):
EMBEDDING_ENABLE_ITEM_TAG_FILTER=false
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.0
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=0
```

**Patched reproduction baseline config** (`.env`):
```bash
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3
EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500
```

**Experimental config** (`.env`):
```bash
# All patched-baseline settings plus:
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true
```

---

## Recommended Research Workflow

1. **Baseline Run** (paper-method baseline):
   ```bash
   # Use the "Paper-method baseline config" above, then run:
   uv run python scripts/reproduce_results.py --split paper-test
   ```

2. **Patched Baseline** (retrieval-only improvements):
   ```bash
   # Enable Spec 33+34 in your .env (see "Patched reproduction baseline config" above), then:
   uv run python scripts/reproduce_results.py --split paper-test
   ```

3. **Ablation A** (add chunk scoring):
   ```bash
   EMBEDDING_REFERENCE_SCORE_SOURCE=chunk \
     uv run python scripts/reproduce_results.py --split paper-test
   ```

4. **Ablation B** (add CRAG validation):
   ```bash
   EMBEDDING_ENABLE_REFERENCE_VALIDATION=true \
     uv run python scripts/reproduce_results.py --split paper-test
   ```

5. **Full Stack** (all features):
   ```bash
   EMBEDDING_REFERENCE_SCORE_SOURCE=chunk \
   EMBEDDING_ENABLE_REFERENCE_VALIDATION=true \
     uv run python scripts/reproduce_results.py --split paper-test
   ```

---

## See Also

- [Configuration Reference](docs/configs/configuration.md) - All settings
- [Run History](docs/results/run-history.md) - Previous ablation results
- [Specs Index](docs/_specs/index.md) - All specifications
