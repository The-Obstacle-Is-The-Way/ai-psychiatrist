# Spec 053: Evidence Hallucination Detection

**Status**: Implemented (PR #92, 2026-01-03)
**Priority**: High
**Complexity**: Medium
**Related**: PIPELINE-BRITTLENESS.md, ANALYSIS-026

---

## SSOT (Implemented)

- Code: `src/ai_psychiatrist/services/evidence_validation.py` (`validate_evidence_grounding()`)
- Wire-up: `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._extract_evidence()`)
- Config: `.env.example` (`QUANTITATIVE_EVIDENCE_QUOTE_VALIDATION_*`)
- Tests: `tests/unit/services/test_evidence_validation.py`, `tests/unit/agents/test_quantitative.py`

## Problem Statement

The LLM can return evidence quotes during extraction that do not exist in the source transcript. These hallucinated quotes:

1. Pollute the reference bundle with non-existent text
2. Lead to incorrect similarity computations
3. Produce misleading confidence signals
4. Cannot be detected post-hoc without the original transcript

This is a **silent corruption** - the pipeline succeeds but produces wrong results.

---

## Previous Behavior (Fixed)

In `src/ai_psychiatrist/agents/quantitative.py:QuantitativeAssessmentAgent._extract_evidence()`:

```python
# LLM returns a JSON object mapping item keys -> list[str] (supposedly)
obj = parse_llm_json(clean)

# Current behavior: best-effort coercion
evidence_dict: dict[str, list[str]] = {}
for key in PHQ8_DOMAIN_KEYS:
    arr = obj.get(key, []) if isinstance(obj, dict) else []
    if not isinstance(arr, list):
        arr = []
    cleaned = list({str(q).strip() for q in arr if str(q).strip()})
    evidence_dict[key] = cleaned
```

Notes:
- Evidence extraction currently runs in both modes (zero-shot and few-shot).
- In few-shot mode, the extracted evidence is embedded and used to retrieve references.
- In both modes, evidence counts are used to compute `llm_evidence_count` and N/A reasons.

**Example of hallucination**:
- Transcript: "I've been feeling okay lately"
- LLM returns: `{"PHQ8_Depressed": ["I feel hopeless and worthless every day"]}`
- This quote does not exist in the transcript

---

## Implemented Solution

Add deterministic **evidence grounding validation**: treat each extracted "quote" as valid only if it is actually present in the source transcript after conservative normalization.

Grounding rule (default):
- A quote is accepted iff `normalize(quote)` is a substring of `normalize(transcript)`.

Rationale:
- The evidence prompt explicitly requires verbatim excerpts ("do not reformat them").
- Substring matching is conservative and rejects paraphrases, which is desirable here.
- This prevents hallucinated evidence from contaminating retrieval and confidence signals.

### Matching Strategy

Use a conservative two-tier approach:

1. **Normalized substring match** (default): checks `normalize(quote) in normalize(transcript)`.
2. **Optional fuzzy substring match** (opt-in): `rapidfuzz.fuzz.partial_ratio` for whitespace/punctuation drift.

**Important**: fuzzy matching is *not* the default because it can accept paraphrases, defeating the purpose of hallucination detection.

### Threshold Selection

| Threshold | Effect |
|-----------|--------|
| 1.0 | Only exact matches (too strict - minor formatting differences rejected) |
| 0.9 | Very close matches (recommended - allows minor whitespace/punctuation differences) |
| 0.8 | Moderate similarity (may allow some paraphrasing) |
| <0.8 | Too permissive (defeats purpose) |

**Recommendation (if fuzzy enabled)**: start with 0.85, configurable via settings.

---

## Implementation

### New Configuration

```python
# config.py - QuantitativeSettings
evidence_quote_validation_enabled: bool = Field(
    default=True,
    description="Validate extracted evidence quotes against the transcript (Spec 053).",
)
evidence_quote_validation_mode: Literal["substring", "fuzzy"] = Field(
    default="substring",
    description=(
        "Evidence grounding mode. 'substring' is conservative and dependency-free; "
        "'fuzzy' requires rapidfuzz and uses partial_ratio."
    ),
)
evidence_quote_fuzzy_threshold: float = Field(
    default=0.85,
    ge=0.5,
    le=1.0,
    description="Only used when evidence_quote_validation_mode='fuzzy'.",
)
evidence_quote_fail_on_all_rejected: bool = Field(
    default=False,
    description=(
        "If true, fail when the LLM produced evidence but none of it can be grounded. "
        "When false (default), record a failure event and continue with empty evidence "
        "to avoid dropping participants while still preventing silent degradation."
    ),
)
evidence_quote_log_rejections: bool = Field(
    default=True,
    description="If true, log counts + hashes for rejected quotes (never raw transcript text).",
)
```

### Validation Function

```python
# New file: src/ai_psychiatrist/services/evidence_validation.py

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)

_WS_RE = re.compile(r"\\s+")
_SMART_QUOTES = str.maketrans(
    {
        "\\u2018": "'",
        "\\u2019": "'",
        "\\u201C": '"',
        "\\u201D": '"',
        "\\u00A0": " ",  # NBSP
    }
)
_ZERO_WIDTH = ("\\u200b", "\\u200c", "\\u200d", "\\ufeff")


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalize_for_quote_match(text: str) -> str:
    """Normalize text for conservative substring grounding checks.

    Properties:
    - Deterministic
    - Conservative (does not attempt semantic matching)
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_SMART_QUOTES)
    for ch in _ZERO_WIDTH:
        normalized = normalized.replace(ch, "")
    normalized = re.sub(r"<[^>]+>", " ", normalized)  # ignore nonverbal tags (<laughter>, <ma>, ...)
    normalized = _WS_RE.sub(" ", normalized).strip().lower()
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceGroundingStats:
    extracted_count: int
    validated_count: int
    rejected_count: int
    rejected_by_domain: dict[str, int]


class EvidenceGroundingError(ValueError):
    """Raised when extracted evidence cannot be grounded in the transcript."""


def validate_evidence_grounding(
    evidence: dict[str, list[str]],
    transcript_text: str,
    *,
    mode: str = "substring",
    fuzzy_threshold: float = 0.85,
    log_rejections: bool = True,
) -> tuple[dict[str, list[str]], EvidenceGroundingStats]:
    """Validate extracted evidence quotes against the source transcript.

    Args:
        evidence: Dict mapping PHQ8 domain keys to lists of quote strings.
        transcript_text: Source transcript text.
        mode: "substring" (default) or "fuzzy" (requires rapidfuzz).
        fuzzy_threshold: Similarity threshold in [0, 1] when mode="fuzzy".
        log_rejections: If true, emits privacy-safe logs for rejections.

    Returns:
        (validated_evidence, stats)
        - validated_evidence: Same structure with only grounded quotes.
        - stats: counts-only summary (no transcript text).
    """
    transcript_norm = normalize_for_quote_match(transcript_text)
    transcript_hash = _stable_hash(transcript_text)

    validated: dict[str, list[str]] = {}
    rejected_by_domain: dict[str, int] = {}
    extracted_count = 0
    validated_count = 0

    for domain, quotes in evidence.items():
        validated[domain] = []
        rejected_by_domain[domain] = 0

        for quote in quotes:
            extracted_count += 1
            quote_norm = normalize_for_quote_match(quote)
            grounded = bool(quote_norm) and (quote_norm in transcript_norm)

            if not grounded and mode == "fuzzy":
                from rapidfuzz import fuzz  # type: ignore[import-not-found]

                ratio = fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0
                grounded = ratio >= fuzzy_threshold

            if grounded:
                validated_count += 1
                validated[domain].append(quote)
            else:
                rejected_by_domain[domain] += 1
                if log_rejections:
                    logger.warning(
                        "evidence_quote_rejected",
                        domain=domain,
                        quote_hash=_stable_hash(quote),
                        quote_len=len(quote),
                        transcript_hash=transcript_hash,
                        transcript_len=len(transcript_text),
                        mode=mode,
                    )

    rejected_count = extracted_count - validated_count
    if rejected_count > 0 and log_rejections:
        logger.info(
            "evidence_grounding_complete",
            extracted_count=extracted_count,
            validated_count=validated_count,
            rejected_count=rejected_count,
            rejected_by_domain=rejected_by_domain,
            transcript_hash=transcript_hash,
        )

    return validated, EvidenceGroundingStats(
        extracted_count=extracted_count,
        validated_count=validated_count,
        rejected_count=rejected_count,
        rejected_by_domain=rejected_by_domain,
    )
```

### Integration Point

```python
# src/ai_psychiatrist/agents/quantitative.py - modify _extract_evidence()

async def _extract_evidence(self, transcript_text: str, *, participant_id: int) -> dict[str, list[str]]:
    """Extract evidence quotes for each PHQ-8 domain."""
    # ... existing extraction code ...

    obj = parse_llm_json(clean)

    # Spec 054 (schema): validate types first (list[str] only).
    evidence = validate_evidence_schema(obj)

    # Spec 053 (grounding): drop ungrounded quotes.
    if self._settings.evidence_quote_validation_enabled:
        evidence, stats = validate_evidence_grounding(
            evidence,
            transcript_text,
            mode=self._settings.evidence_quote_validation_mode,
            fuzzy_threshold=self._settings.evidence_quote_fuzzy_threshold,
            log_rejections=self._settings.evidence_quote_log_rejections,
        )

        if stats.validated_count == 0 and stats.extracted_count > 0:
            # Always record a privacy-safe failure event for post-run auditability.
            record_failure(
                FailureCategory.EVIDENCE_HALLUCINATION,
                FailureSeverity.ERROR,
                "LLM returned evidence quotes but none could be grounded in the transcript.",
                participant_id=participant_id,
                stage="evidence_extraction",
                mode=self._mode.value,
                extracted_count=stats.extracted_count,
                validation_mode=self._settings.evidence_quote_validation_mode,
            )

            # Strict mode: raise and mark participant as failed.
            if self._settings.evidence_quote_fail_on_all_rejected:
                raise EvidenceGroundingError(
                    "LLM returned evidence quotes but none could be grounded in the transcript."
                )

    return evidence
```

---

## Dependencies

None for the default `substring` grounding mode.

Optional (only if enabling `evidence_quote_validation_mode="fuzzy"`):

- `rapidfuzz` (substring Levenshtein; used for `partial_ratio`).

---

## Testing

### Unit Tests

```python
# tests/unit/services/test_evidence_validation.py

def test_exact_match_accepted():
    evidence = {"PHQ8_Sleep": ["I can't sleep at night"]}
    transcript = "Patient said: I can't sleep at night. Very tired."
    validated, stats = validate_evidence_grounding(evidence, transcript)
    assert validated["PHQ8_Sleep"] == ["I can't sleep at night"]
    assert stats.rejected_count == 0


def test_hallucination_rejected():
    evidence = {"PHQ8_Depressed": ["I feel hopeless and worthless"]}
    transcript = "I've been feeling okay lately. Work is going well."
    validated, stats = validate_evidence_grounding(evidence, transcript)
    assert validated["PHQ8_Depressed"] == []
    assert stats.rejected_by_domain["PHQ8_Depressed"] == 1


def test_minor_whitespace_difference_accepted():
    evidence = {"PHQ8_Tired": ["I   feel  tired"]}  # Extra spaces
    transcript = "I feel tired all the time"
    validated, _ = validate_evidence_grounding(evidence, transcript)
    assert len(validated["PHQ8_Tired"]) == 1


def test_case_insensitive_matching():
    evidence = {"PHQ8_Sleep": ["I CAN'T SLEEP"]}
    transcript = "i can't sleep at all"
    validated, _ = validate_evidence_grounding(evidence, transcript)
    assert len(validated["PHQ8_Sleep"]) == 1
```

### Integration Test (Suggested)

Prove the agent drops ungrounded quotes *before* retrieval (no transcript text persisted):
- mock `llm_client.simple_chat` to return JSON evidence with one ungrounded quote
- call `QuantitativeAssessmentAgent._extract_evidence(transcript.text)`
- assert returned evidence lists exclude the ungrounded quote

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

At minimum:
- emit a single `evidence_grounding_complete` log event per participant when rejections occur (counts + hashes only).

Optional (if you want this visible in run artifacts):
- extend `ItemAssessment` + `item_signals` with counts-only fields:
  - `llm_evidence_extracted_count`
  - `llm_evidence_rejected_count`

---

## Rollout Plan

1. **Phase 1**: Implement with `evidence_quote_validation_enabled=true` and `mode=substring`
2. **Phase 2**: Run one evaluation with validation enabled, compare metrics
3. **Phase 3**: If false rejections appear, tune normalization or enable fuzzy mode (explicit opt-in)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| False positives (valid excerpts rejected) | Keep normalization conservative; add fuzzy mode only if necessary |
| Legitimate paraphrasing rejected | Treat as a prompt-following failure (the prompt requires verbatim excerpts) |

---

## Success Criteria

1. Zero hallucinated quotes in validated evidence (by definition)
2. No transcript/quote text leaked in logs or artifacts (hashes + counts only)
3. No silent degradation of few-shot into zero-shot due to ungrounded evidence
4. No regression in MAE or AURC metrics attributable to this change

---

## Open Questions

1. Should we use semantic similarity (embeddings) instead of string matching for validation?
   - Pro: Handles paraphrasing better
   - Con: Adds embedding call overhead, circular dependency with embedding service
   - **Decision**: Start with string matching, semantic matching as future enhancement

2. Should rejected quotes be logged to a separate file for analysis?
   - **Decision**: Not by default (DAIC-WOZ licensing). If ever added, store only hashes + counts unless an explicit “unsafe debugging” flag is set.
