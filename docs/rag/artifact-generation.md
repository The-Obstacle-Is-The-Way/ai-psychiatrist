# RAG Artifact Generation

**Audience**: Researchers generating few-shot reference artifacts
**Last Updated**: 2026-01-03

This guide describes how to generate embedding artifacts safely and reproducibly, including optional item tags (Spec 34).

SSOT implementation:
- `scripts/generate_embeddings.py`
- `src/ai_psychiatrist/services/reference_store.py` (loads/validates artifacts)

---

## Output Artifacts

Given an output basename `{name}`, embedding generation produces:

| File | Contents | Required |
|------|----------|----------|
| `{name}.npz` | Embedding vectors (per-participant keys like `emb_303`) | Yes |
| `{name}.json` | Chunk texts (participant id → list[str]) | Yes |
| `{name}.meta.json` | Provenance metadata for fail-fast mismatch detection | Yes |
| `{name}.tags.json` | PHQ-8 item tags per chunk (only if `--write-item-tags`) | Optional |

For chunk-level scoring artifacts, see [chunk-scoring.md](chunk-scoring.md).

---

## Basic Generation (Strict Mode)

Strict mode is fail-fast and recommended for production:
- transcript load failures → crash
- empty transcript → crash
- embedding failures → crash

```bash
uv run python scripts/generate_embeddings.py --split paper-train
```

---

## Generation With Item Tags (Spec 34)

Item tagging adds a `{name}.tags.json` sidecar so retrieval can filter candidate chunks to the target PHQ-8 item.

```bash
uv run python scripts/generate_embeddings.py \
  --split paper-train \
  --write-item-tags \
  --tagger keyword
```

This writes `{name}.tags.json` aligned with `{name}.json`.

### Enable Tag Filtering at Runtime

```bash
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
```

### Fail-Fast Semantics (Spec 38)

- If `EMBEDDING_ENABLE_ITEM_TAG_FILTER=false`:
  - `{name}.tags.json` is ignored (no load, no validation)
- If `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`:
  - missing `{name}.tags.json` → crash
  - invalid `{name}.tags.json` → crash

This is intentional: enabling a feature must not silently run a different method.

---

## Partial Mode (Debug Only)

Partial mode is opt-in for debugging:

```bash
uv run python scripts/generate_embeddings.py --split paper-train --allow-partial
```

Behavior:
- skips failed participants/chunks
- exits with code **2**
- writes a skip manifest `{output}.partial.json` if any skips occur

Manifest schema:

```json
{
  "output_npz": "data/embeddings/....npz",
  "skipped_participants": [487],
  "skipped_participant_count": 1,
  "skipped_chunks": 12
}
```

**Rule**: any artifact produced in partial mode with skips is **not valid** for final evaluation.

---

## Artifact Schemas

### Tags Sidecar (`{name}.tags.json`)

Top-level JSON object:
- keys: participant id strings (e.g., `"303"`)
- values: list of per-chunk tag lists aligned with `{name}.json`

Example:

```json
{
  "303": [
    ["PHQ8_Sleep", "PHQ8_Tired"],
    [],
    ["PHQ8_Depressed"]
  ]
}
```

Constraints:
- participant ids must match `{name}.json`
- per-participant list length must equal the chunk count in `{name}.json`
- each tag must be one of the 8 `PHQ8_*` strings

---

## Verification Checklist

After generation:

```bash
# Base artifacts
ls -la data/embeddings/{name}.npz data/embeddings/{name}.json data/embeddings/{name}.meta.json

# If using item tags
ls -la data/embeddings/{name}.tags.json

# If using chunk scores (see chunk-scoring.md)
ls -la data/embeddings/{name}.chunk_scores.json data/embeddings/{name}.chunk_scores.meta.json
```

---

## Related Docs

- Chunk-level scoring: [chunk-scoring.md](chunk-scoring.md)
- Few-shot preflight: `docs/preflight-checklist/preflight-checklist-few-shot.md`
- Feature index: `docs/pipeline-internals/features.md`
