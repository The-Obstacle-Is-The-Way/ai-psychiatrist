# Feature Reference (Non-Archive Canonical)

**Audience**: Researchers and maintainers
**Last Updated**: 2026-01-01

This page is the canonical, non-archive reference for implemented features that affect:
- **few-shot retrieval behavior**
- **artifact formats**
- **evaluation metrics**
- **fail-fast / reliability semantics**

If `docs/_archive/` disappeared tomorrow, this page (and the linked docs under `docs/`) should still be sufficient to run, debug, and interpret experiments.

---

## SSOT + Defaults

- **SSOT for config names + code defaults**: `src/ai_psychiatrist/config.py`
- **Recommended baseline for research runs**: `.env.example` (copy to `.env`)
- **Run provenance**: `scripts/reproduce_results.py` writes `run_metadata` (timestamp, git commit, run id, settings snapshot)

When this page says “default”, it refers to **code defaults** unless explicitly marked as “`.env.example` baseline”.

---

## Few-Shot Retrieval Features

| Feature | Spec | Config | Code Default | Artifact Requirement | What It Changes |
|---------|------|--------|--------------|----------------------|-----------------|
| Reference Examples prompt format | 31 (+33 XML) | *(none)* | ON | *(none)* | How references are formatted in the prompt |
| Retrieval audit logs | 32 | `EMBEDDING_ENABLE_RETRIEVAL_AUDIT` | `false` | *(none)* | Adds structured logs per retrieved reference |
| Similarity threshold | 33 | `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | *(none)* | Drops low-similarity references before top-k |
| Per-item context budget | 33 | `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | *(none)* | Caps total chars per item after top-k |
| Item-tag filtering | 34 (+38 semantics) | `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `{emb}.tags.json` | Filters candidate chunks by PHQ-8 item tags |
| Chunk-level score attachment | 35 | `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `{emb}.chunk_scores.json` + `{emb}.chunk_scores.meta.json` | Uses per-chunk estimated labels instead of participant-level labels |
| CRAG-style reference validation | 36 (+38 semantics) | `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | *(none)* | LLM validates each retrieved reference (`accept`/`reject`) |
| Batch query embedding | 37 | `EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING` | `true` | *(none)* | Uses 1 embedding call per participant (vs 8) |
| Query embedding timeout | 37 | `EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS` | `300` | *(none)* | Bounds embedding latency; replaces older hardcoded timeouts |
| Skip-if-disabled, crash-if-broken | 38 | *(automatic)* | ON | *(varies)* | Disabled optional features do no I/O; enabled features crash on invalid/missing artifacts |
| Preserve exception types | 39 | *(automatic)* | ON | *(none)* | Avoids masking errors as `ValueError` so failures are diagnosable |

**Notes:**
- “`{emb}`” means the resolved embeddings NPZ path: `resolve_reference_embeddings_path(...)` in `src/ai_psychiatrist/config.py`.
- Spec 31’s original notebook used an unusual “same open/close tag” (`<Reference Examples>` … `<Reference Examples>`). Spec 33 intentionally changed the closing delimiter to proper XML: `</Reference Examples>`.

---

## Embedding Artifact Safety

| Feature | Spec | Where | Behavior |
|---------|------|-------|----------|
| Fail-fast embedding generation | 40 | `scripts/generate_embeddings.py` | Default strict mode crashes on missing/corrupt transcripts or embedding failures; `--allow-partial` is debug-only and exits `2` with a `{output}.partial.json` skip manifest |

See: [Embedding generation](../embeddings/embedding-generation.md).

---

## Evaluation / Metrics

| Feature | Spec | Where | Why It Exists |
|---------|------|-------|---------------|
| Selective prediction metrics | 25 | `scripts/evaluate_selective_prediction.py`, `src/ai_psychiatrist/metrics/*` | Comparing MAE across different coverages is invalid; we report AURC/AUGRC + bootstrap CIs |

See:
- [Statistical methodology (AURC/AUGRC)](../statistics/statistical-methodology-aurc-augrc.md) (why AURC/AUGRC)
- [Metrics and evaluation](../statistics/metrics-and-evaluation.md) (exact definitions + output schema)

---

## Recommended Profiles (Research Workflow)

### Paper-Parity Baseline (Historical)

Goal: reproduce the paper’s *method as described*, even if it is noisy.

- `EMBEDDING_REFERENCE_SCORE_SOURCE=participant`
- `EMBEDDING_ENABLE_ITEM_TAG_FILTER=false`
- `EMBEDDING_MIN_REFERENCE_SIMILARITY=0.0`
- `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=0`
- `EMBEDDING_ENABLE_REFERENCE_VALIDATION=false`

### Research-Honest Retrieval (Post-Ablation Target)

Goal: minimize known failure modes (label mismatch, wrong-item retrieval, irrelevant references).

- `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`
- `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`
- `EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3`
- `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=500`
- `EMBEDDING_ENABLE_REFERENCE_VALIDATION=true`

See: [Preflight checklist (few-shot)](../preflight-checklist/preflight-checklist-few-shot.md).

---

## Where To Go Next

- [Configuration](../configs/configuration.md)
- [Run output schema](../results/run-output-schema.md)
- [Few-shot prompt format](../embeddings/few-shot-prompt-format.md)
- [Batch query embedding](../embeddings/batch-query-embedding.md)
- [Retrieval debugging](../embeddings/debugging-retrieval-quality.md)
- [Item tagging setup](../embeddings/item-tagging-setup.md)
- [Chunk scoring (Spec 35)](../embeddings/chunk-scoring.md)
- [CRAG validation guide](../statistics/crag-validation-guide.md)
- [Error-handling philosophy](../developer/error-handling.md)
- [Exception reference](../developer/exceptions.md)
