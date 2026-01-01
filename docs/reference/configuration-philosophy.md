# Configuration Philosophy

**Audience**: Maintainers and researchers
**Last Updated**: 2026-01-01

This document defines what should be configurable vs baked in as defaults for a **research reproduction** codebase.

---

## Core Principle

> **Correct behavior is the default. Broken behavior requires explicit opt-in.**

Flags add cognitive load and increase the probability of accidental misconfiguration. In research, a “successful” run with the wrong behavior is worse than a crash.

---

## SSOT + Terminology

- **SSOT for config names + code defaults**: `src/ai_psychiatrist/config.py`
- **Recommended baseline for research runs**: `.env.example` (copy to `.env`)

When this doc says “default”, it refers to **code defaults** unless it explicitly says “`.env.example` baseline”.

---

## Configuration Categories

### 1) Correctness Invariants (Do Not “Tune”)

These are “always-on” correctness properties. If they have knobs, those knobs are either unsupported or debug-only.

| Invariant | Where | Knob | Notes |
|----------|-------|------|-------|
| Skip-if-disabled, crash-if-broken (Spec 38) | `ReferenceStore`, `ReferenceValidation` | None | Disabled optional features do no file I/O; enabled features must validate strictly |
| Preserve exception types (Spec 39) | Agents | None | Log `error_type`, re-raise original exception |
| Fail-fast embedding generation (Spec 40) | `scripts/generate_embeddings.py` | `--allow-partial` | Partial mode is debug-only and produces a skip manifest |
| Pydantic AI structured output | Agents | `PYDANTIC_AI_ENABLED` | Disabling is **not supported** (agents will raise; legacy fallback is intentionally removed) |

### 2) Post-Ablation Retrieval Defaults (Will Become “Baked In”)

These are retrieval-quality fixes that should become the default after ablations demonstrate net benefit:

| Setting | Code Default | `.env.example` Baseline | Post-Ablation Default | Why |
|---------|--------------|--------------------------|------------------------|-----|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `participant` | `chunk` | Avoids participant-score-on-chunk mismatch (Spec 35) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `true` | `true` | Filters wrong-item retrieval candidates (Spec 34) |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | `0.3` | `0.3` | Drops low-similarity references (Spec 33) |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | `500` | `500` | Bounds per-item reference context (Spec 33) |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | `false` | `true` | CRAG validation rejects irrelevant references (Spec 36) |

### 3) Tunable Hyperparameters

Researchers should tune these for ablations. Some are **index-time** and require regenerating artifacts.

| Setting | Runtime-Only? | Notes |
|---------|---------------|-------|
| `EMBEDDING_DIMENSION` | No | Must match embedding model + stored artifact dimension |
| `EMBEDDING_CHUNK_SIZE` / `EMBEDDING_CHUNK_STEP` | No | Requires regenerating embeddings and sidecars |
| `EMBEDDING_TOP_K_REFERENCES` | Yes | Paper Appendix D used `2`; can be tuned without reindex |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` / `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | Yes | Retrieval-time filters/budgets |
| `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING` | Yes | Spec 37 perf/stability; disable only for debugging older runs |
| `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS` | Yes | Spec 37 stability knob |
| `EMBEDDING_ENABLE_RETRIEVAL_AUDIT` | Yes | Diagnostics-only (Spec 32) |
| `EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM` | Yes | Bounds CRAG keep-set per item |
| `FEEDBACK_*` | Yes | Changes runtime and may change outputs |

### 4) Model Selection + Infrastructure

Always configurable:
- Models: `MODEL_*`
- Backends: `LLM_BACKEND`, `EMBEDDING_BACKEND`
- Timeouts: `OLLAMA_TIMEOUT_SECONDS`, `PYDANTIC_AI_TIMEOUT_SECONDS`, HF timeouts
- Paths: `DATA_*`

---

## Decision Framework

When adding a new setting:

1. Is this **correctness** or **preference**?
   - Correctness → make it default and hard to disable
   - Preference → keep it configurable
2. Is this **index-time**?
   - If yes, document required artifact regeneration
3. Does it bypass safety?
   - Default OFF, label it “unsafe”, and make it noisy

---

## Related Docs

- Configuration reference: `docs/reference/configuration.md`
- Feature index: `docs/reference/features.md`
- Error-handling philosophy: `docs/concepts/error-handling.md`
