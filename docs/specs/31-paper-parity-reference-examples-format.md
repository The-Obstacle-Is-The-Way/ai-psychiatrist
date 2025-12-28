# Spec 31: Paper-Parity Few-Shot Reference Examples Format

> **STATUS: READY (Implement Now)**
>
> **Scope**: Fix paper-parity divergences in `ReferenceBundle.format_for_prompt()` only.

## Problem

Our few-shot prompt formatting currently diverges from the paper’s notebook implementation (`_reference/ai_psychiatrist/quantitative_assessment/embedding_quantitative_analysis.ipynb`, cell id `49f51ff5`):

- We emit **8 separate sections** (one per PHQ-8 item) instead of **one unified** `<Reference Examples>` block.
- We label scores as `(Score: X)` instead of `({PHQ8_EVIDENCE_KEY} Score: X)` inline.
- We close with `</Reference Examples>` instead of the notebook’s (unusual) `<Reference Examples>` delimiter.
- We emit per-item `"No valid evidence found"` blocks; notebook **skips empty items** and only emits the sentinel if *all* references are empty.

This spec makes our formatting **character-for-character** match the notebook output.

## Goals (Acceptance Criteria)

1. **Exact string parity** with notebook for both:
   - Non-empty references
   - Fully-empty references
2. **Skip empty items** entirely (no per-item blocks).
3. **Inline domain labeling**: `(PHQ8_Sleep Score: 2)` not `(Score: 2)`.
4. **Open and close delimiter are identical**: `<Reference Examples>` (not XML-style closing tag).
5. **Deterministic ordering**: preserve notebook’s evidence-key order and similarity ranking order.

## Non-goals

- Do not change retrieval logic, scoring lookup, embedding generation, `top_k`, or similarity computation.
- Do not introduce similarity thresholds or relevance filtering (those belong in Spec 33+).

## Source of Truth (Notebook)

From cell `49f51ff5`:

```python
reference_entry = f"({evidence_key} Score: {score})\n{raw_text}"

if all_references:
    reference_evidence = "<Reference Examples>\n\n" + "\n\n".join(all_references) + "\n\n<Reference Examples>"
else:
    reference_evidence = "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

Evidence key order in the notebook:

```python
evidence_keys = [
  "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
  "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving"
]
```

## Implementation

### Files to Change

- `src/ai_psychiatrist/services/embedding.py:40` (`ReferenceBundle.format_for_prompt`)
- `tests/unit/services/test_embedding.py:55` (`TestReferenceBundle`)

### Exact Output Specification

Define `entries: list[str]` where each `entry` is exactly:

```text
({EVIDENCE_KEY} Score: {SCORE})
{CHUNK_TEXT}
```

Where:
- `{EVIDENCE_KEY}` is exactly `PHQ8_{item.value}` (e.g., `PHQ8_Sleep`).
- `{SCORE}` is integer `0..3`.
- `{CHUNK_TEXT}` is the retrieved chunk text (may contain internal newlines).

Then `format_for_prompt()` returns:

- If `entries` is non-empty:

```python
"<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n<Reference Examples>"
```

- If `entries` is empty:

```python
"<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

### Ordering Rules (Deterministic)

- Evidence-key order must be `PHQ8Item.all_items()` order (matches notebook list order).
- Within an item, references must appear in the order provided in `self.item_references[item]` (this list is already similarity-sorted by `EmbeddingService.build_reference_bundle`).

### “Before” (Current Behavior)

`src/ai_psychiatrist/services/embedding.py:40` currently emits:

- Per-item header: `[{item.value}]`
- Per-item `<Reference Examples>` block
- `(Score: X)` / `(Score: N/A)`
- XML-style closing: `</Reference Examples>`
- `"No valid evidence found"` inside each empty item block

### “After” (Copy/Paste Replacement)

Replace `ReferenceBundle.format_for_prompt()` with:

```python
def format_for_prompt(self) -> str:
    """Format references as prompt text (paper-parity).

    Paper notebook behavior (cell 49f51ff5):
    - Single unified <Reference Examples> block.
    - Each reference entry is labeled like: (PHQ8_Sleep Score: 2)
    - Items with no matches are omitted (no empty per-item blocks).
    - Uses the same literal tag to open and close: <Reference Examples>
    """
    entries: list[str] = []

    for item in PHQ8Item.all_items():
        evidence_key = f"PHQ8_{item.value}"
        for match in self.item_references.get(item, []):
            # Notebook behavior: only include references with available ground truth.
            if match.reference_score is None:
                continue
            entries.append(f"({evidence_key} Score: {match.reference_score})\n{match.chunk.text}")

    if entries:
        return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n<Reference Examples>"

    return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

## TDD: Unit Tests (Copy/Paste)

Replace the existing `TestReferenceBundle` class in `tests/unit/services/test_embedding.py` with:

```python
class TestReferenceBundle:
    """Tests for ReferenceBundle (paper-parity formatting)."""

    def test_format_empty_bundle(self) -> None:
        bundle = ReferenceBundle(item_references={})
        assert bundle.format_for_prompt() == "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"

    def test_format_with_single_match(self) -> None:
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="I can't enjoy anything anymore", participant_id=123),
            similarity=0.95,
            reference_score=2,
        )
        bundle = ReferenceBundle(item_references={PHQ8Item.NO_INTEREST: [match]})
        formatted = bundle.format_for_prompt()

        assert formatted.startswith("<Reference Examples>\n\n")
        assert formatted.endswith("\n\n<Reference Examples>")
        assert "(PHQ8_NoInterest Score: 2)\nI can't enjoy anything anymore" in formatted
        assert "[NoInterest]" not in formatted
        assert "</Reference Examples>" not in formatted
        assert "No valid evidence found" not in formatted

    def test_format_skips_none_score(self) -> None:
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="Some text", participant_id=123),
            similarity=0.8,
            reference_score=None,
        )
        bundle = ReferenceBundle(item_references={PHQ8Item.SLEEP: [match]})
        assert bundle.format_for_prompt() == "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"

    def test_format_multiple_items_preserves_order(self) -> None:
        # Order must follow PHQ8Item.all_items(): NoInterest, Depressed, Sleep, Tired, ...
        sleep_match = SimilarityMatch(
            chunk=TranscriptChunk(text="sleep ref", participant_id=100),
            similarity=0.9,
            reference_score=3,
        )
        tired_match = SimilarityMatch(
            chunk=TranscriptChunk(text="tired ref", participant_id=101),
            similarity=0.85,
            reference_score=1,
        )

        bundle = ReferenceBundle(
            item_references={
                PHQ8Item.TIRED: [tired_match],
                PHQ8Item.SLEEP: [sleep_match],
            }
        )
        formatted = bundle.format_for_prompt()

        sleep_idx = formatted.index("(PHQ8_Sleep Score: 3)\nsleep ref")
        tired_idx = formatted.index("(PHQ8_Tired Score: 1)\ntired ref")
        assert sleep_idx < tired_idx
```

## Edge Cases

- Item has no matches: **omit** (no empty block).
- `reference_score is None`: **omit** that match; if that results in zero total entries, emit sentinel.
- Empty chunk text: should be impossible (`TranscriptChunk` validates non-empty); treat as upstream data corruption.

## Verification

1. `uv run pytest tests/unit/services/test_embedding.py -q`
2. (Optional paper-parity ablation) `uv run python scripts/reproduce_results.py --split paper-test`

## Design Notes (SOLID / DRY / GoF)

- This is intentionally a **single-responsibility** change: prompt formatting only.
- Future format variants (if ever needed) should use a Strategy (`ReferenceFormatter`) rather than branching in `format_for_prompt` (out of scope here).
