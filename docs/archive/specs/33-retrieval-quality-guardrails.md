# Spec 33: Retrieval Quality Guardrails (Similarity Threshold + Context Budget)

> **STATUS: ✅ IMPLEMENTED (2025-12-29)**
>
> **Do not enable by default**. This changes the method vs the paper.
>
> **Reproduction note**: This spec intentionally breaks Spec 31’s character-for-character
> notebook parity by switching the closing delimiter to proper XML (`</Reference Examples>`).

## Problem

Even with perfect paper-parity formatting (Spec 31), retrieval can still inject noise:

- Embedding similarity matches topic, not severity.
- Low-similarity “top-k” references can be irrelevant but still included.
- Too many/too-long references can dilute the prompt and increase overconfident scoring.

## Goals (Acceptance Criteria)

1. Add an optional **minimum similarity threshold** for including a retrieved reference.
2. Add an optional **context budget** for references to prevent prompt bloat.
3. **Fix XML closing tag** to use proper `</Reference Examples>` syntax (per [Anthropic best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags)).
4. Defaults must preserve current **retrieval selection** behavior (no filtering, no budget).

## Non-goals

- No item tagging (Spec 34).
- No chunk-level scoring (Spec 35).
- No LLM judge / CRAG (Spec 36).

## Design (Strategy + Chain-of-Responsibility)

Introduce a small post-retrieval pipeline:

1. Candidate retrieval (existing)
2. Filter by similarity threshold (new)
3. Enforce reference context budget (new)

This should be implemented as a composable chain so additional filters/rerankers can be added without branching logic.

**Implementation note (2025-12-29):** The current implementation uses simple sequential
post-processing (straight-line `if` blocks). If additional stages are added (e.g., Spec 34+),
consider refactoring into an explicit Strategy/CoR pipeline to avoid branching growth.

## Implementation

### Files to Change

- `src/ai_psychiatrist/config.py` (`EmbeddingSettings`)
- `src/ai_psychiatrist/services/embedding.py` (`EmbeddingService.__init__`, `EmbeddingService.build_reference_bundle`)
- `tests/unit/services/test_embedding.py` (new unit tests)

### Settings (All Optional)

Add to `EmbeddingSettings` (env prefix `EMBEDDING_`):

- `min_reference_similarity: float = 0.0`
  - Range: `0.0..1.0`
  - Meaning: drop matches with `match.similarity < min_reference_similarity`
- `max_reference_chars_per_item: int = 0`
  - `0` means unlimited
  - Meaning: after thresholding + sorting, keep highest-similarity matches until adding another entry would exceed the per-item char budget.

Concrete config fields to add in `src/ai_psychiatrist/config.py`:

```python
min_reference_similarity: float = Field(
    default=0.0,
    ge=0.0,
    le=1.0,
    description="Drop retrieved references below this similarity (0 disables)",
)
max_reference_chars_per_item: int = Field(
    default=0,
    ge=0,
    description="Max total reference chunk chars per item (0 disables)",
)
```

Expected env vars:

- `EMBEDDING_MIN_REFERENCE_SIMILARITY`
- `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM`

### Retrieval Post-Processing (Exact Algorithm)

For each item:

1. `matches` are already sorted desc by similarity.
2. Filter: `kept = [m for m in matches if m.similarity >= min_reference_similarity]`
3. Take up to `top_k` in order.
4. If `max_reference_chars_per_item > 0`, accumulate entries in order until budget exceeded.

**Budget accounting rule (exact)**:

- The “cost” of a reference is `len(match.chunk.text)` (do not include formatting overhead).

### Code Changes (Exact Place)

In `src/ai_psychiatrist/services/embedding.py`:

1. In `EmbeddingService.__init__`, store settings:

```python
self._min_reference_similarity = settings.min_reference_similarity
self._max_reference_chars_per_item = settings.max_reference_chars_per_item
```

2. In `EmbeddingService.build_reference_bundle`, after:

```python
matches = self._compute_similarities(query_emb, item=item)
matches.sort(key=lambda x: x.similarity, reverse=True)
```

Apply post-processing:

```python
if self._min_reference_similarity > 0.0:
    matches = [m for m in matches if m.similarity >= self._min_reference_similarity]

top_matches = matches[: self._top_k]

if self._max_reference_chars_per_item > 0:
    budgeted: list[SimilarityMatch] = []
    used = 0
    for m in top_matches:
        cost = len(m.chunk.text)
        if used + cost > self._max_reference_chars_per_item:
            break
        budgeted.append(m)
        used += cost
    top_matches = budgeted
```

### TDD: Unit Tests (Must Exist)

Add tests to `tests/unit/services/test_embedding.py`:

1. `test_reference_threshold_filters_low_similarity`
   - Given two matches at `0.9` and `0.1`, threshold `0.5` keeps only `0.9`.
2. `test_reference_budget_limits_included_matches`
   - Given three short matches, a small budget includes only first N matches.
3. `test_defaults_preserve_existing_selection`
   - Threshold `0.0` and budget `0` behave exactly as current selection.

✅ Copy/paste test code (self-safe): add these **methods** inside the existing
`class TestEmbeddingService` in `tests/unit/services/test_embedding.py`.

- These are **not** module-level functions (they use `self`).
- This uses `monkeypatch.setattr(...)` to avoid mypy `method-assign` ignores.
- No new imports are required beyond what `tests/unit/services/test_embedding.py` already has.

```python
    @pytest.mark.asyncio
    async def test_reference_threshold_filters_low_similarity(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = EmbeddingSettings(
            dimension=256,
            top_k_references=2,
            min_evidence_chars=1,
            min_reference_similarity=0.5,
            max_reference_chars_per_item=0,
        )
        service = EmbeddingService(mock_llm_client, mock_reference_store, settings)

        matches = [
            SimilarityMatch(
                chunk=TranscriptChunk(text="good", participant_id=1),
                similarity=0.9,
                reference_score=1,
            ),
            SimilarityMatch(
                chunk=TranscriptChunk(text="bad", participant_id=2),
                similarity=0.1,
                reference_score=1,
            ),
        ]
        monkeypatch.setattr(service, "_compute_similarities", MagicMock(return_value=matches))

        bundle = await service.build_reference_bundle({PHQ8Item.SLEEP: ["evidence"]})
        assert bundle.item_references[PHQ8Item.SLEEP] == [matches[0]]

    @pytest.mark.asyncio
    async def test_reference_budget_limits_included_matches(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = EmbeddingSettings(
            dimension=256,
            top_k_references=3,
            min_evidence_chars=1,
            min_reference_similarity=0.0,
            max_reference_chars_per_item=5,
        )
        service = EmbeddingService(mock_llm_client, mock_reference_store, settings)

        matches = [
            SimilarityMatch(
                chunk=TranscriptChunk(text="12345", participant_id=1),
                similarity=0.9,
                reference_score=1,
            ),
            SimilarityMatch(
                chunk=TranscriptChunk(text="12345", participant_id=2),
                similarity=0.8,
                reference_score=1,
            ),
        ]
        monkeypatch.setattr(service, "_compute_similarities", MagicMock(return_value=matches))

        bundle = await service.build_reference_bundle({PHQ8Item.SLEEP: ["evidence"]})
        # Budget=5 keeps only the first chunk of length 5.
        assert bundle.item_references[PHQ8Item.SLEEP] == [matches[0]]

    @pytest.mark.asyncio
    async def test_defaults_preserve_existing_selection(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = EmbeddingSettings(
            dimension=256,
            top_k_references=2,
            min_evidence_chars=1,
            min_reference_similarity=0.0,
            max_reference_chars_per_item=0,
        )
        service = EmbeddingService(mock_llm_client, mock_reference_store, settings)

        # Deliberately unsorted: build_reference_bundle must sort desc and take top_k.
        low = SimilarityMatch(
            chunk=TranscriptChunk(text="low", participant_id=1),
            similarity=0.2,
            reference_score=1,
        )
        high = SimilarityMatch(
            chunk=TranscriptChunk(text="high", participant_id=2),
            similarity=0.9,
            reference_score=1,
        )
        mid = SimilarityMatch(
            chunk=TranscriptChunk(text="mid", participant_id=3),
            similarity=0.5,
            reference_score=1,
        )
        monkeypatch.setattr(service, "_compute_similarities", MagicMock(return_value=[low, high, mid]))

        bundle = await service.build_reference_bundle({PHQ8Item.SLEEP: ["evidence"]})
        assert bundle.item_references[PHQ8Item.SLEEP] == [high, mid]
```

## Verification

- Run paper-parity baseline first (Spec 31 ablation).
- Then run with:
  - `EMBEDDING_MIN_REFERENCE_SIMILARITY=0.6`
  - `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM=800`
  - Compare paired AURC deltas.

### XML Closing Tag Fix (Trivial)

In `src/ai_psychiatrist/services/embedding.py`, in `ReferenceBundle.format_for_prompt()`:

**Before** (paper-parity, weird):
```python
return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n<Reference Examples>"
```

**After** (proper XML):
```python
return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n</Reference Examples>"
#                                                             ^ ADD SLASH
```

Also fix the empty case:
```python
return "<Reference Examples>\nNo valid evidence found\n</Reference Examples>"
#                                                        ^ ADD SLASH
```

This is a 2-character change. No config needed - just fix it.

## Risks / Failure Modes

- Thresholding may reduce coverage by removing references, potentially causing more N/A.
- Budgeting may bias towards earlier items if later items systematically have longer chunks (measure per-item effects).
- XML fix: theoretically could affect Gemma behavior, but proper XML is universally cleaner.
