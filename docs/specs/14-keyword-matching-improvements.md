# Spec 14: Keyword Matching Improvements

> **STATUS: DEFERRED**
>
> This spec is deferred. The collision-proofed YAML works well for current needs.
> Word-boundary matching and negation detection are precision refinements.
>
> **Tracked by**: [GitHub Issue #31](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/31)
>
> **Last Updated**: 2025-12-21

---

## Objective

Improve the PHQ-8 keyword backfill system from case-insensitive substring matching
to word-boundary regex matching with optional negation detection.

## Preconditions

- Spec 09 (Quantitative Agent) completed
- Current YAML collision-proofed (done in PR #30)

## Background

The keyword backfill system in `QuantitativeAssessmentAgent._keyword_backfill()` uses
case-insensitive substring matching. This has two limitations:

1. **No word boundaries**: Even with collision-proofed keywords, edge cases exist
2. **Negation blindness**: "I'm NOT depressed" still matches "depressed"

The current mitigation (collision-proofed YAML) works by expanding single tokens
into explicit phrases (e.g., "tired" → "feeling tired", "am tired", etc.). This
is effective but sacrifices some recall for precision.

## Goals

1. Implement **word-boundary regex matching** using `\b` anchors
2. Optionally implement **negation window detection** (filter matches preceded by negation words)
3. Allow restoration of high-sensitivity single-token keywords to the YAML

## Non-Goals

- Do not change the YAML format (keep it simple)
- Do not add complex NLP dependencies (spaCy, nltk, etc.)
- Do not break existing behavior (new code should be opt-in or backward compatible)

## Deliverables

1. `src/ai_psychiatrist/services/keyword_matching.py`
   - `word_boundary_match(keyword: str, text: str) -> bool`
   - `is_negated(text: str, match_start: int, window: int = 4) -> bool`
   - `find_keyword_matches(keywords: list[str], text: str, check_negation: bool = False) -> list[Match]`

2. Update `QuantitativeAssessmentAgent._keyword_backfill()` to use new matching

3. Configuration option in `QuantitativeSettings`:
   - `keyword_match_mode: Literal["substring", "word_boundary"]`
   - `check_negation: bool = False`

4. Unit tests for all matching scenarios

## Implementation Plan

### 1. Word-Boundary Regex Matching

```python
import re

def word_boundary_match(keyword: str, text: str) -> bool:
    """Match keyword at word boundaries only."""
    pattern = rf'\b{re.escape(keyword)}\b'
    return bool(re.search(pattern, text, re.IGNORECASE))
```

**Benefits:**
- "retired" won't match "tired"
- "sadly" won't match "sad"
- Restores ~15 high-sensitivity keywords

### 2. Negation Window Detection

```python
NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "dont", "can't", "cant",
    "won't", "wont", "didn't", "didnt", "isn't", "isnt",
    "aren't", "arent", "wasn't", "wasnt"
})

def is_negated(text: str, match_start: int, window: int = 4) -> bool:
    """Check if match is preceded by negation within window."""
    tokens_before = text[:match_start].lower().split()[-window:]
    return any(neg in tokens_before for neg in NEGATION_WORDS)
```

**Benefits:**
- "I'm not depressed" → no match
- "I haven't been sleeping well" → still matches (negation targets different phrase)

### 3. Configuration

```python
class QuantitativeSettings(BaseSettings):
    keyword_match_mode: Literal["substring", "word_boundary"] = "substring"
    check_negation: bool = False
```

Default remains `substring` for backward compatibility.

## Acceptance Criteria

- [ ] Word-boundary matching implemented with configurable toggle
- [ ] Negation window detection implemented with configurable toggle
- [ ] Default behavior unchanged (substring matching, no negation check)
- [ ] Unit tests cover edge cases:
  - "retired" vs "tired"
  - "sadly" vs "sad"
  - "I'm not depressed" (negated)
  - "I haven't slept well" (negated target is "slept", not "well")
- [ ] Performance: < 10ms overhead for typical transcript

## Testing

- Unit tests for `keyword_matching.py`
- Integration tests with `QuantitativeAssessmentAgent`
- Benchmark: measure precision/recall before/after on sample transcripts

## Priority

**Low** - The collision-proofed YAML is effective. This is a precision refinement
that can be done later if needed.

## References

- Current YAML: `src/ai_psychiatrist/resources/phq8_keywords.yaml`
- Keyword backfill: `QuantitativeAssessmentAgent._keyword_backfill()`
- PHQ-8 validation: Kroenke et al., 2009
- GitHub Issue: #31
