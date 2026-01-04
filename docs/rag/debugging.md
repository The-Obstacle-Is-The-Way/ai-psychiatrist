# RAG Debugging (Audit Logs + Guardrails)

**Audience**: Researchers debugging few-shot performance
**Last Updated**: 2026-01-03

This guide explains how to debug few-shot retrieval using the built-in diagnostic tools:
- retrieval audit logs (Spec 32)
- similarity threshold + per-item budgets (Spec 33)
- item tag filtering (Spec 34 + Spec 38 semantics)
- CRAG validation decisions (Spec 36)

---

## Step 0: Confirm What Method You Ran

Before debugging retrieval quality, confirm run configuration:
- `scripts/reproduce_results.py` prints the effective settings at startup
- your output JSON stores per-experiment settings in `experiments[*].provenance` (`run_metadata` is run-level environment info)

If you can’t explain exactly which features were enabled, do not interpret the results.

---

## Step 1: Enable Retrieval Audit Logs (Spec 32)

Set:

```bash
EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true
```

You should see `retrieved_reference` log events with fields:
- `item`, `evidence_key`
- `rank`, `similarity`
- `participant_id`, `reference_score`
- `chunk_preview`, `chunk_chars`

This is emitted after retrieval post-processing (threshold + top-k + budgets + CRAG filtering).

---

## Step 2: Triage Common Failure Modes

### A) Low Similarity References (Spec 33)

Symptom:
- top references have low similarity (e.g., < 0.3)

Mitigations:
- raise `EMBEDDING_MIN_REFERENCE_SIMILARITY`
- ensure embeddings backend is correct (`EMBEDDING_BACKEND=huggingface` is higher precision)

### B) Prompt Bloat / Drowning (Spec 33)

Symptom:
- many long chunks dominate the prompt, drowning out evidence

Mitigations:
- set `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` (e.g., 500–2000)

### C) Wrong-Item Retrieval (Spec 34)

Symptom:
- references talk about the wrong PHQ-8 item (“sleep” query pulls “failure” content)

Mitigations:
- regenerate embeddings with `--write-item-tags` to produce `{emb}.tags.json`
- set `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`

**Fail-fast note (Spec 38):**
- if filtering is enabled and `{emb}.tags.json` is missing/invalid → run should crash

### D) Semantically Irrelevant References (Spec 36)

Symptom:
- similarity is high but the chunk is clinically irrelevant or contradictory

Mitigations:
- enable CRAG validation:
  - `EMBEDDING_ENABLE_REFERENCE_VALIDATION=true`
  - optionally set `EMBEDDING_VALIDATION_MODEL` (if unset, runners fall back to `MODEL_JUDGE_MODEL`)

---

## Step 3: Check Artifact Preconditions

Few-shot retrieval requires:
- `{emb}.npz` and `{emb}.json`
- `{emb}.meta.json` (expected for modern artifacts; enables fail-fast mismatch detection)

Optional but required when features are enabled:
- `{emb}.tags.json` if `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`
- `{emb}.chunk_scores.json` + `{emb}.chunk_scores.meta.json` if `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`

See:
- [Artifact generation](artifact-generation.md)
- [Chunk scoring](chunk-scoring.md)

---

## Step 4: Interpret “No valid evidence found”

If the reference bundle contains:

```text
<Reference Examples>
No valid evidence found
</Reference Examples>
```

It can mean:
- evidence extraction returned no evidence for all items (no query embeddings)
- retrieval found no matches above threshold
- all matches had `reference_score=None` (common with chunk scores when the chunk is non-evidentiary)
- CRAG rejected all references

Use audit logs to disambiguate.

---

## Step 5: Check Failure Registry (Spec 056)

After each evaluation run, check `data/outputs/failures_{run_id}.json`:

```bash
cat data/outputs/failures_19b42478.json | jq '.summary'
```

The failure registry categorizes failures by:
- **Category**: `evidence_json_parse`, `embedding_nan`, `scoring_pydantic_retry_exhausted`, etc.
- **Severity**: `fatal`, `error`, `warning`, `info`
- **Stage**: `evidence_extraction`, `embedding_generation`, `scoring`
- **Participant**: Which participants failed most often

Use this to identify systematic issues (e.g., "participant 373 always fails on evidence extraction").

---

## Step 5b: Check Retry Telemetry (Spec 060)

Even when a run succeeds, the system can be “quietly brittle” (many retries, frequent JSON repair).

After each run, check `data/outputs/telemetry_{run_id}.json`:

```bash
cat data/outputs/telemetry_19b42478.json | jq '.summary'
```

This captures:
- PydanticAI retry triggers (`ModelRetry`) by extractor
- JSON repair usage (`tolerant_json_fixups`, python-literal fallback, `json-repair`)

If `dropped_events` is non-zero, the run hit the telemetry event cap (defaults to 5,000). Treat that as a sign of extreme brittleness.

If these counts spike, treat it as a regression risk even if MAE/AUGRC look good.

---

## Step 6: Diagnose Embedding Failures (Spec 055)

If you see `EmbeddingValidationError`:

| Error Pattern | Likely Cause | Fix |
|---------------|--------------|-----|
| `NaN detected` | Malformed input to embedding backend | Check transcript preprocessing |
| `Inf detected` | Numerical overflow | Check embedding model/backend |
| `All-zero vector` | Empty or whitespace-only input | Check chunking configuration |

**At generation time**: Regenerate artifacts with `scripts/generate_embeddings.py`

**At runtime**: Check query embedding input (evidence text may be empty or corrupted)

---

## Related Docs

- Feature index: `docs/pipeline-internals/features.md`
- Runtime features: [runtime-features.md](runtime-features.md)
- Error-handling philosophy: `docs/developer/error-handling.md`
- Failure registry: `docs/developer/error-handling.md#failure-pattern-observability-spec-056`
