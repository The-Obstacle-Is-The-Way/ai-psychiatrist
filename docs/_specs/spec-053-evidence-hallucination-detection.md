# Spec 053: Evidence Hallucination Detection

**Status**: Ready to Implement
**Priority**: High
**Complexity**: Medium
**Related**: PIPELINE-BRITTLENESS.md, ANALYSIS-026

---

## Problem Statement

The LLM can return evidence quotes during extraction that do not exist in the source transcript. These hallucinated quotes:

1. Pollute the reference bundle with non-existent text
2. Lead to incorrect similarity computations
3. Produce misleading confidence signals
4. Cannot be detected post-hoc without the original transcript

This is a **silent corruption** - the pipeline succeeds but produces wrong results.

---

## Current Behavior

In `quantitative.py:_extract_evidence()`:

```python
# LLM returns evidence dict
obj = parse_llm_json(clean)

# We trust it completely - no validation
return {key: [str(q).strip() for q in obj.get(key, []) if str(q).strip()] for key in PHQ8_DOMAIN_KEYS}
```

**Example of hallucination**:
- Transcript: "I've been feeling okay lately"
- LLM returns: `{"PHQ8_Depressed": ["I feel hopeless and worthless every day"]}`
- This quote does not exist in the transcript

---

## Proposed Solution

Add fuzzy matching validation to verify each extracted quote exists in the source transcript.

### Matching Strategy

Use a two-tier approach:

1. **Exact substring match** (fast path): If quote is a direct substring of transcript
2. **Fuzzy match** (fallback): Use normalized Levenshtein ratio with threshold

### Threshold Selection

| Threshold | Effect |
|-----------|--------|
| 1.0 | Only exact matches (too strict - minor formatting differences rejected) |
| 0.9 | Very close matches (recommended - allows minor whitespace/punctuation differences) |
| 0.8 | Moderate similarity (may allow some paraphrasing) |
| <0.8 | Too permissive (defeats purpose) |

**Recommendation**: Start with 0.85 threshold, configurable via settings.

---

## Implementation

### New Configuration

```python
# config.py - EmbeddingSettings
evidence_hallucination_threshold: float = Field(
    default=0.85,
    ge=0.5,
    le=1.0,
    description="Minimum similarity ratio for evidence quote validation (0.85 = 85% match required)"
)
evidence_validation_enabled: bool = Field(
    default=True,
    description="Enable hallucination detection for extracted evidence"
)
```

### Validation Function

```python
# New file: src/ai_psychiatrist/services/evidence_validation.py

from rapidfuzz import fuzz
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, collapse whitespace, strip punctuation edges
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def find_best_match(quote: str, transcript: str) -> tuple[float, str | None]:
    """Find the best matching substring in transcript for a quote.

    Returns:
        (similarity_ratio, matched_substring or None)
    """
    quote_norm = normalize_text(quote)
    transcript_norm = normalize_text(transcript)

    # Fast path: exact substring
    if quote_norm in transcript_norm:
        return 1.0, quote

    # Fuzzy search: sliding window over transcript
    # Use partial_ratio for substring matching
    ratio = fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0

    return ratio, None if ratio < 0.5 else quote


def validate_evidence_quotes(
    evidence: dict[str, list[str]],
    transcript: str,
    threshold: float = 0.85,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, float]]]]:
    """Validate extracted evidence quotes against source transcript.

    Args:
        evidence: Dict mapping PHQ8 domain keys to lists of quote strings
        transcript: Source transcript text
        threshold: Minimum similarity ratio to accept quote

    Returns:
        (validated_evidence, rejected_quotes)
        - validated_evidence: Same structure with only validated quotes
        - rejected_quotes: Dict mapping domain to list of (quote, ratio) tuples
    """
    validated = {}
    rejected = {}

    for domain, quotes in evidence.items():
        validated[domain] = []
        rejected[domain] = []

        for quote in quotes:
            ratio, _ = find_best_match(quote, transcript)

            if ratio >= threshold:
                validated[domain].append(quote)
            else:
                rejected[domain].append((quote, ratio))
                logger.warning(
                    "evidence_quote_rejected",
                    domain=domain,
                    quote_preview=quote[:100],
                    similarity_ratio=round(ratio, 3),
                    threshold=threshold,
                )

    # Log summary
    total_validated = sum(len(v) for v in validated.values())
    total_rejected = sum(len(r) for r in rejected.values())

    if total_rejected > 0:
        logger.info(
            "evidence_validation_complete",
            validated_count=total_validated,
            rejected_count=total_rejected,
            rejection_rate=round(total_rejected / (total_validated + total_rejected), 3),
        )

    return validated, rejected
```

### Integration Point

```python
# quantitative.py - modify _extract_evidence()

async def _extract_evidence(self, transcript_text: str) -> dict[str, list[str]]:
    """Extract evidence quotes for each PHQ-8 domain."""
    # ... existing extraction code ...

    obj = parse_llm_json(clean)
    evidence = {key: [str(q).strip() for q in obj.get(key, []) if str(q).strip()]
                for key in PHQ8_DOMAIN_KEYS}

    # NEW: Validate evidence if enabled
    if self._embedding_settings.evidence_validation_enabled:
        from ai_psychiatrist.services.evidence_validation import validate_evidence_quotes

        validated, rejected = validate_evidence_quotes(
            evidence,
            transcript_text,
            threshold=self._embedding_settings.evidence_hallucination_threshold,
        )

        # Store rejection stats for diagnostics
        self._last_evidence_rejection_stats = {
            domain: len(quotes) for domain, quotes in rejected.items()
        }

        return validated

    return evidence
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
rapidfuzz = "^3.6.0"  # Fast fuzzy string matching
```

**Why rapidfuzz?**
- 10-100x faster than python-Levenshtein
- Pure Python fallback available
- MIT licensed
- `partial_ratio` is ideal for substring matching

---

## Testing

### Unit Tests

```python
# tests/unit/services/test_evidence_validation.py

def test_exact_match_accepted():
    evidence = {"PHQ8_Sleep": ["I can't sleep at night"]}
    transcript = "Patient said: I can't sleep at night. Very tired."
    validated, rejected = validate_evidence_quotes(evidence, transcript, threshold=0.85)
    assert validated["PHQ8_Sleep"] == ["I can't sleep at night"]
    assert rejected["PHQ8_Sleep"] == []


def test_hallucination_rejected():
    evidence = {"PHQ8_Depressed": ["I feel hopeless and worthless"]}
    transcript = "I've been feeling okay lately. Work is going well."
    validated, rejected = validate_evidence_quotes(evidence, transcript, threshold=0.85)
    assert validated["PHQ8_Depressed"] == []
    assert len(rejected["PHQ8_Depressed"]) == 1


def test_minor_whitespace_difference_accepted():
    evidence = {"PHQ8_Tired": ["I   feel  tired"]}  # Extra spaces
    transcript = "I feel tired all the time"
    validated, rejected = validate_evidence_quotes(evidence, transcript, threshold=0.85)
    assert len(validated["PHQ8_Tired"]) == 1


def test_case_insensitive_matching():
    evidence = {"PHQ8_Sleep": ["I CAN'T SLEEP"]}
    transcript = "i can't sleep at all"
    validated, rejected = validate_evidence_quotes(evidence, transcript, threshold=0.85)
    assert len(validated["PHQ8_Sleep"]) == 1


def test_threshold_boundary():
    # Quote with ~80% similarity
    evidence = {"PHQ8_Tired": ["I feel very tired today"]}
    transcript = "I feel somewhat tired lately"

    # Should fail at 0.9 threshold
    validated_strict, _ = validate_evidence_quotes(evidence, transcript, threshold=0.9)
    assert validated_strict["PHQ8_Tired"] == []

    # Should pass at 0.7 threshold
    validated_loose, _ = validate_evidence_quotes(evidence, transcript, threshold=0.7)
    assert len(validated_loose["PHQ8_Tired"]) == 1
```

### Integration Tests

```python
def test_full_pipeline_with_validation(mock_ollama):
    """Evidence validation integrates with full assessment flow."""
    # Configure mock to return a hallucinated quote
    mock_ollama.chat_response = json.dumps({
        "PHQ8_NoInterest": ["I have no interest in anything"],  # Real
        "PHQ8_Depressed": ["I want to end it all"],  # HALLUCINATED
        # ...
    })

    transcript = Transcript(participant_id=300, text="I have no interest in anything lately.")

    agent = QuantitativeAssessmentAgent(...)
    result = await agent.assess(transcript)

    # Hallucinated quote should not appear in final evidence
    assert "end it all" not in str(result.items[PHQ8Item.DEPRESSED].evidence)
```

---

## Metrics Impact

### Expected Effects

| Metric | Expected Change | Reason |
|--------|-----------------|--------|
| Evidence count per item | Slight decrease | Hallucinations removed |
| N/A rate | Possible slight increase | Items may have less evidence |
| Retrieval similarity | More accurate | Real quotes match better |
| AURC/AUGRC | Should improve | Cleaner confidence signals |

### Monitoring

Add to evaluation output:

```python
# Per-participant in results JSON
"evidence_validation_stats": {
    "total_extracted": 24,
    "total_validated": 22,
    "total_rejected": 2,
    "rejection_rate": 0.083,
    "rejected_by_domain": {"PHQ8_Depressed": 1, "PHQ8_Moving": 1}
}
```

---

## Rollout Plan

1. **Phase 1**: Implement with `evidence_validation_enabled=False` (off by default)
2. **Phase 2**: Run one evaluation with validation enabled, compare metrics
3. **Phase 3**: If metrics improve, enable by default
4. **Phase 4**: Remove the toggle after validation proves stable

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| False positives (valid quotes rejected) | Start with 0.85 threshold, tune based on data |
| Performance overhead | rapidfuzz is fast; add timing logs to monitor |
| Legitimate paraphrasing rejected | Consider higher threshold or semantic matching |

---

## Success Criteria

1. Zero hallucinated quotes in validated evidence (by definition)
2. <5% false positive rate (valid quotes rejected)
3. No regression in MAE or AURC metrics
4. <100ms additional latency per participant

---

## Open Questions

1. Should we use semantic similarity (embeddings) instead of string matching for validation?
   - Pro: Handles paraphrasing better
   - Con: Adds embedding call overhead, circular dependency with embedding service
   - **Decision**: Start with string matching, semantic matching as future enhancement

2. Should rejected quotes be logged to a separate file for analysis?
   - **Decision**: Yes, add optional `--dump-rejected-evidence` flag to evaluation script
