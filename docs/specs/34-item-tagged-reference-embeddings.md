# Spec 34: Item-Tagged Reference Embeddings (Index-Time Tagging + Retrieval-Time Filtering)

> **STATUS: IMPLEMENTED**
>
> This addresses "topic vs item" mismatch by restricting retrieval to chunks tagged as relevant to a PHQ-8 item.
>
> **Note**: Only `--tagger keyword` is implemented. LLM-based tagging (`--tagger llm`) is deferred to a future spec.

## Problem

Current reference chunks are generic transcript windows. A chunk can be topically similar to an evidence query while not actually containing item-relevant content for that specific PHQ-8 symptom. This produces irrelevant few-shot examples.

## Goals (Acceptance Criteria)

1. At embedding generation time, compute per-chunk **item tags** (`PHQ8_*`) indicating which PHQ-8 items the chunk actually discusses.
2. At retrieval time, when building references for item `X`, consider **only chunks tagged with `PHQ8_X`** (when enabled).
3. Default behavior must remain unchanged when tags are missing or filtering is disabled.

## Non-goals

- No chunk-level scoring (Spec 35).
- No CRAG/LLM judge (Spec 36).

## Artifact Format (Backwards Compatible)

Keep existing artifacts:

- `{name}.npz` embeddings
- `{name}.json` texts (participant_id → list[str])
- `{name}.meta.json` metadata

Add a new sidecar:

- `{name}.tags.json`

Schema:

```json
{
  "303": [
    ["PHQ8_Sleep", "PHQ8_Tired"],
    [],
    ["PHQ8_Depressed"],
    ...
  ],
  "304": [...]
}
```

Constraints:

- For each participant id, `len(tags[pid]) == len(texts[pid])`.
- Each tag is one of the 8 strings: `PHQ8_NoInterest`, `PHQ8_Depressed`, `PHQ8_Sleep`, `PHQ8_Tired`, `PHQ8_Appetite`, `PHQ8_Failure`, `PHQ8_Concentrating`, `PHQ8_Moving`.

## Implementation Plan

### Files to Change

- `scripts/generate_embeddings.py` (write `{name}.tags.json` when enabled)
- `src/ai_psychiatrist/services/reference_store.py` (load and validate tags sidecar)
- `src/ai_psychiatrist/services/embedding.py` (filter candidates by tags when enabled)
- `tests/unit/services/test_embedding.py` (new tests; reuse `_create_npz_embeddings`)

### 1) Generate Tags (Index Time)

Modify `scripts/generate_embeddings.py` to optionally write `{name}.tags.json` when:

- CLI flag: `--write-item-tags`
- AND tagging backend is configured:
  - `--tagger keyword` (deterministic baseline)
  - `--tagger llm` (semantic; must be mocked in tests)

#### Keyword Tagger (Deterministic Baseline)

- Reuse `src/ai_psychiatrist/resources/phq8_keywords.yaml` entries per item.
- Match using **case-insensitive substring** (the YAML is already collision-proofed for substring safety).
- If any keyword matches, add that item’s `PHQ8_*` tag.

#### Exact CLI Args (Add to `argparse`)

Add to `scripts/generate_embeddings.py`:

```python
parser.add_argument(
    "--write-item-tags",
    action="store_true",
    help="Write <output>.tags.json sidecar aligned with <output>.json texts",
)
parser.add_argument(
    "--tagger",
    choices=["keyword", "llm"],
    default="keyword",
    help="Chunk tagger backend (only used when --write-item-tags is set)",
)
```

#### Exact Output File

Write:

```python
tags_path = config.output_path.with_suffix(".tags.json")
```

with JSON:

```python
tags_json: dict[str, list[list[str]]] = {
    str(pid): [tags_for_chunk_0, tags_for_chunk_1, ...]
}
```

Constraints:
- `len(tags_json[str(pid)]) == len(json_texts[str(pid)])`
- `tags_for_chunk_i` is a list of `PHQ8_*` strings (may be empty)

### 2) Load Tags (ReferenceStore)

Extend `src/ai_psychiatrist/services/reference_store.py`:

- Load `{name}.tags.json` if present.
- Validate lengths and raise `EmbeddingArtifactMismatchError` on mismatch.
- Provide a method: `get_participant_tags(participant_id: int) -> list[list[str]]`.

Exact addition points:

- Add a lazy-loaded attribute:
  - `self._tags: dict[int, list[list[str]]] | None = None`
- Add helper:
  - `_get_tags_path(self) -> Path: return self._embeddings_path.with_suffix(".tags.json")`
- In `_load_embeddings`, after `texts_data = self._load_texts_json(...)`, if tags file exists:
  - Load tags JSON
  - Validate participant ids match
  - Validate per-participant list lengths match the texts list lengths
  - Store in `self._tags`

### 3) Filter Candidates (EmbeddingService)

Add setting:

- `EmbeddingSettings.enable_item_tag_filter: bool = False` (env: `EMBEDDING_ENABLE_ITEM_TAG_FILTER`)

In `EmbeddingService._compute_similarities(query_embedding, item=...)`:

- Enumerate chunks with index: `for idx, (chunk_text, embedding) in enumerate(chunks):`
- If `enable_item_tag_filter` and tags available:
  - only include candidate if `f"PHQ8_{item.value}" in tags_for_participant[idx]`

Exact behavior:
- If `item is None`: do not filter (keep current behavior).
- If tags sidecar missing or cannot be loaded: do not filter (warn once).

## TDD: Tests (Must Exist)

1. `test_reference_store_loads_tags_and_validates_length`
   - Create fake `{name}.json` and `{name}.tags.json` with mismatched lengths → must raise.
2. `test_item_tag_filter_excludes_untagged_chunks`
   - Two chunks: only first tagged with `PHQ8_Sleep`; retrieval for Sleep must only consider first.
3. `test_missing_tags_falls_back_to_unfiltered`
   - No tags sidecar → behavior unchanged even when flag enabled.

Copy/paste scaffolding guidance:

- Use existing `_create_npz_embeddings(...)` helper in `tests/unit/services/test_embedding.py` to write `{name}.npz` + `{name}.json`.
- Write `{name}.tags.json` alongside it in the test temp directory.

## Verification

- Regenerate embeddings with `--write-item-tags --tagger keyword`.
- Run reproduction with `EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`.
- Compare paired AURC deltas vs paper-parity baseline.

## Risks

- Keyword tagger may under-tag (false negatives), harming retrieval recall.
- LLM tagger may introduce variance; must run multiple seeds / multiple runs.
