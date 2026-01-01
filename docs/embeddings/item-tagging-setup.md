# Item Tagging Setup (Spec 34 + Spec 38 Semantics)

**Audience**: Researchers enabling item-tag filtering
**Last Updated**: 2026-01-01

Item tagging adds a `{emb}.tags.json` sidecar so retrieval can filter candidate chunks to the target PHQ-8 item.

SSOT implementations:
- `scripts/generate_embeddings.py` (`--write-item-tags`)
- `src/ai_psychiatrist/services/reference_store.py` (loads/validates tags)

---

## Step 1: Generate Embeddings With Tags

```bash
uv run python scripts/generate_embeddings.py \
  --split paper-train \
  --write-item-tags \
  --tagger keyword
```

This writes:
- `{name}.tags.json` aligned with `{name}.json`

---

## Step 2: Enable Tag Filtering at Runtime

```bash
EMBEDDING_ENABLE_ITEM_TAG_FILTER=true
```

---

## Fail-Fast Semantics (Spec 38)

- If `EMBEDDING_ENABLE_ITEM_TAG_FILTER=false`:
  - `{name}.tags.json` is ignored (no load, no validation)
- If `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`:
  - missing `{name}.tags.json` → crash
  - invalid `{name}.tags.json` → crash

This is intentional: enabling a feature must not silently run a different method.

---

## Tag Sidecar Schema (Exact)

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

## Related Docs

- Embedding generation: `docs/guides/embedding-generation.md`
- Retrieval debugging: `docs/guides/debugging-retrieval-quality.md`
