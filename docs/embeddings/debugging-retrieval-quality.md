# Debugging Retrieval Quality (Audit Logs + Guardrails)

**Audience**: Researchers debugging few-shot performance
**Last Updated**: 2026-01-01

This guide explains how to debug few-shot retrieval using the built-in diagnostic tools:
- retrieval audit logs (Spec 32)
- similarity threshold + per-item budgets (Spec 33)
- item tag filtering (Spec 34 + Spec 38 semantics)
- CRAG validation decisions (Spec 36)

---

## Step 0: Confirm What Method You Ran

Before debugging retrieval quality, confirm run configuration:
- `scripts/reproduce_results.py` prints the effective settings at startup
- your output JSON contains a settings snapshot in `run_metadata`

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
- `docs/embeddings/embedding-generation.md`
- `docs/embeddings/chunk-scoring.md`

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

## Related Docs

- Feature index: `docs/pipeline-internals/features.md`
- Few-shot prompt format: `docs/embeddings/few-shot-prompt-format.md`
- Error-handling philosophy: `docs/developer/error-handling.md`
