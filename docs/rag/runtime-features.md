# RAG Runtime Features

**Audience**: Researchers configuring few-shot retrieval behavior
**Last Updated**: 2026-01-03

This document covers runtime features that affect how few-shot retrieval operates: prompt formatting, batch embedding, and CRAG validation.

SSOT implementations:
- `src/ai_psychiatrist/services/embedding.py` (retrieval + formatting)
- `src/ai_psychiatrist/services/reference_validation.py` (CRAG validation)

---

## Prompt Format (Reference Examples)

Few-shot mode retrieves reference chunks from a training split and inserts them into the scoring prompt as "reference examples".

### Reference Entry Format

Each included reference is formatted as:

```text
({EVIDENCE_KEY} Score: {SCORE})
{CHUNK_TEXT}
```

Where:
- `{EVIDENCE_KEY}` is `PHQ8_{item.value}` (e.g., `PHQ8_Sleep`)
- `{SCORE}` is an integer `0..3`
- `{CHUNK_TEXT}` is the raw chunk text (may contain internal newlines)

References with `reference_score=None` are omitted.

### Reference Bundle Format

All reference entries across all items are merged into a single block:

```text
<Reference Examples>

{entry_1}

{entry_2}

...

</Reference Examples>
```

If **no entries** survive filtering:

```text
<Reference Examples>
No valid evidence found
</Reference Examples>
```

### Ordering Rules

Ordering is deterministic:
1. Items are iterated in `PHQ8Item.all_items()` order.
2. Within an item, references are emitted in retrieval order (similarity-sorted).

### Paper Notebook vs Current Code

The paper notebook used an unusual delimiter style (`<Reference Examples>...<Reference Examples>`). Current code uses proper XML-style closing tags (`</Reference Examples>`). This was an intentional fix in Spec 33.

---

## Batch Query Embedding (Spec 37)

Spec 37 is a **performance + reliability** fix:
- **Before**: up to 8 sequential query embeddings per participant
- **After**: 1 batch query embedding per participant

This fixes timeout failures from repeated embedding calls.

### Configuration

```bash
# Enable batch embedding (default: true)
EMBEDDING_ENABLE_BATCH_QUERY_EMBEDDING=true

# Query embedding timeout in seconds (default: 300)
EMBEDDING_QUERY_EMBED_TIMEOUT_SECONDS=300
```

### Why This Exists

Few-shot retrieval embeds the **query evidence** to find similar reference chunks. Evidence is extracted per PHQ-8 item, so a participant can produce up to 8 evidence texts.

Historically, these were embedded one-by-one:
- 8 embeddings × 41 participants = 328 calls
- High timeout exposure

Spec 37 reduces this to 1 embedding operation per participant.

### Verification

Run few-shot on a small limit:

```bash
uv run python scripts/reproduce_results.py --split paper-test --few-shot-only --limit 3
```

If you see `LLM request timed out after 120s`, you're running older code or bypassing the timeout config.

---

## CRAG Reference Validation (Spec 36)

CRAG-style validation adds a second LLM step after retrieval:
1. Retrieve candidate reference chunks
2. Validate each reference against the item + evidence (`accept` / `reject`)
3. Include only accepted references in the few-shot prompt

### Enable CRAG Validation

```bash
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true

# Optional: specify validation model (defaults to MODEL_JUDGE_MODEL)
EMBEDDING_VALIDATION_MODEL=gemma3:27b-it-qat

# Optional: max accepted refs per item after validation (default: 2)
EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM=2
```

### Fail-Fast Semantics (Spec 38)

If validation is enabled, it must work or crash:
- invalid JSON responses raise `LLMResponseParseError`
- network/backend failures propagate (preserve exception type)

There is no "return unsure and continue" fallback. Silent fallbacks corrupt research.

### What CRAG Can and Cannot Fix

CRAG validation is a **filter**, not a relabeler:
- It can reject irrelevant or contradictory references
- It cannot correct a wrong `reference_score` label (that's Spec 35's job)

### Recommended Layering

1. Spec 35 (chunk scores) for label correctness
2. Spec 34 (item tags) for candidate set precision
3. Spec 33 (threshold/budget) for quality guardrails
4. Spec 36 (CRAG) for semantic validation

---

## Pipeline Flow Summary

```text
1. Extract evidence per PHQ-8 item from qualitative assessment
2. Batch embed all evidence texts (Spec 37)
3. For each item with evidence:
   a. Compute similarities against reference store
   b. Apply similarity threshold (Spec 33)
   c. Apply item-tag filter if enabled (Spec 34)
   d. Apply per-item char budget (Spec 33)
   e. Apply CRAG validation if enabled (Spec 36)
   f. Attach scores (participant-level or chunk-level per Spec 35)
4. Format unified <Reference Examples> block
5. Insert into quantitative scoring prompt
```

---

## Related Docs

- Artifact generation: [artifact-generation.md](artifact-generation.md)
- Chunk-level scoring: [chunk-scoring.md](chunk-scoring.md)
- Debugging: [debugging.md](debugging.md)
- Feature index: `docs/pipeline-internals/features.md`
