# Few-Shot Prompt Format (Reference Examples)

**Audience**: Researchers validating paper-parity and prompt invariants
**Last Updated**: 2026-01-01

This document is the canonical (non-archive) description of how few-shot reference examples are formatted and inserted into the quantitative scoring prompt.

SSOT implementation:
- `ReferenceBundle.format_for_prompt()` in `src/ai_psychiatrist/services/embedding.py`

---

## What This Format Is For

Few-shot mode retrieves reference chunks from a training split and inserts them into the scoring prompt as “reference examples”.

Each reference example is a `(chunk_text, reference_score)` pair.

---

## The Reference Entry Format (Exact)

Each included reference is formatted as:

```text
({EVIDENCE_KEY} Score: {SCORE})
{CHUNK_TEXT}
```

Where:
- `{EVIDENCE_KEY}` is `PHQ8_{item.value}` (e.g., `PHQ8_Sleep`)
- `{SCORE}` is an integer `0..3`
- `{CHUNK_TEXT}` is the raw chunk text (may contain internal newlines)

References with `reference_score=None` are omitted (not shown).

---

## The Reference Bundle Format (Unified Block)

All reference entries across all items are merged into a single block. Items with no included references are omitted entirely.

### Non-empty bundle

```text
<Reference Examples>

{entry_1}

{entry_2}

...

</Reference Examples>
```

### Empty bundle sentinel

If **no entries** survive filtering (e.g., no matches or all scores are `None`), the bundle is:

```text
<Reference Examples>
No valid evidence found
</Reference Examples>
```

---

## Ordering Rules

Ordering is deterministic:

1. Items are iterated in `PHQ8Item.all_items()` order.
2. Within an item, references are emitted in the retrieval order (already similarity-sorted).

---

## Paper Notebook vs Current Code

The paper notebook used an unusual delimiter style:

```text
<Reference Examples>
...
<Reference Examples>
```

Current code uses a proper XML-style closing tag:

```text
<Reference Examples>
...
</Reference Examples>
```

Rationale: Spec 33 intentionally adopted proper XML tags for clarity and compatibility with modern prompt-engineering guidance.

If you need strict notebook parity for historical comparison, treat that as a separate “paper-parity mode” with explicit labeling (do not mix with correctness-oriented retrieval features).

---

## Related Docs

- [Embeddings and few-shot retrieval](embeddings-explained.md)
- [Retrieval debugging](debugging-retrieval-quality.md)
- [Feature reference](../pipeline-internals/features.md)
