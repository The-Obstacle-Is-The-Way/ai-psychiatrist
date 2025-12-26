# Keyword Fallback Improvements (Deferred)

> **STATUS: DEFERRED — LOW DESIRABILITY**
>
> **Why this is deferred**: Improving keyword fallback would **negate the research
> question** this codebase exists to answer.
>
> The purpose of this codebase is to evaluate **pure LLM semantic understanding**
> of clinical interviews for depression assessment. Keyword fallback is **rule-based
> pattern matching** — the opposite of semantic understanding. Improving it would
> measure "LLM + better heuristics" rather than "LLM capability."
>
> From the paper (Section 2.3.2):
> > "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."
>
> **Additional reasons:**
> - Feature is OFF by default (`QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false`)
> - Paper methodology doesn't describe keyword backfill
> - Collision-proofed YAML (`phq8_keywords.yaml`) already handles major false positives
>
> **GitHub Issue**: #31 (closed as intentionally not implementing)
>
> **Last Updated**: 2025-12-26

---

## Context: What Is Keyword Fallback?

The keyword fallback is a **safety net** for when the LLM misses obvious evidence
during transcript analysis. It's **not part of the core pipeline**.

```text
Primary Pipeline (always runs):
  Transcript → LLM Evidence Extraction → LLM Scoring → PHQ-8 Assessment

Optional Fallback (OFF by default):
  If LLM misses evidence → Search transcript for keywords → Add to evidence pool
```

**Why it's OFF**: The paper text doesn't describe keyword backfill. For paper-text
parity, we disable it. The paper's *code* includes it, but we default to what the
paper *says*.

**Enable with**: `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true`

---

## Current Implementation

The fallback uses **case-insensitive substring matching** against
`src/ai_psychiatrist/resources/phq8_keywords.yaml`.

**Limitations:**
1. **Substring collisions**: "retired" matches "tired" (mitigated by collision-proofed YAML)
2. **Negation blindness**: "I'm NOT depressed" still matches "depressed"

**Current mitigation**: The YAML was collision-proofed (PR #30) by replacing
dangerous single-word keywords with explicit phrases:
- "tired" → "feeling tired", "am tired", "so tired", etc.
- "sad" → "feeling sad", "am sad", "so sad", etc.

This works well but sacrifices some recall for precision.

---

## What This Spec Would Add (If Implemented)

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
- Could restore ~15 high-sensitivity single-word keywords

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
- "I haven't been sleeping well" → still matches (negation targets "sleeping")

### 3. Configuration

```python
class QuantitativeSettings(BaseSettings):
    keyword_match_mode: Literal["substring", "word_boundary"] = "substring"
    check_negation: bool = False
```

---

## Deliverables (Planned, Not Implemented)

1. `src/ai_psychiatrist/services/keyword_matching.py`
   - `word_boundary_match(keyword: str, text: str) -> bool`
   - `is_negated(text: str, match_start: int, window: int = 4) -> bool`
   - `find_keyword_matches(keywords: list[str], text: str, check_negation: bool = False) -> list[Match]`

2. Update `QuantitativeAssessmentAgent._find_keyword_hits()` to use new matching

3. Configuration toggles in `QuantitativeSettings`

4. Unit tests for edge cases

---

## Acceptance Criteria

- [ ] Word-boundary matching with configurable toggle
- [ ] Negation window detection with configurable toggle
- [ ] Default behavior unchanged (substring, no negation check)
- [ ] Unit tests: "retired" vs "tired", "I'm not depressed", etc.
- [ ] Performance: < 10ms overhead for typical transcript

---

## References

- Keyword YAML: `src/ai_psychiatrist/resources/phq8_keywords.yaml`
- Backfill code: `QuantitativeAssessmentAgent._find_keyword_hits()` / `_merge_evidence()`
- Config: `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` (default: false)
- GitHub Issue: #31 (closed — intentionally not implementing)
