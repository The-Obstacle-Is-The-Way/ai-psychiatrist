# BUG-029: Evidence Grounding Validation Too Strict - 20% Participant Failure Rate

**Status**: Open (Critical)
**Severity**: P0 (Data Loss - Participants Skipped)
**Filed**: 2026-01-04
**Component**: `src/ai_psychiatrist/services/evidence_validation.py`
**Observed In**: Run 11 (run11_confidence_suite_20260103_215102.log)

---

## Summary

The evidence grounding validation (Spec 053) is rejecting LLM-generated quotes at a rate that causes **20% of participants to fail entirely**. When all quotes for a participant are rejected, the system throws `EvidenceGroundingError` and skips the participant, resulting in data loss.

---

## Impact

| Metric | Run 10 | Run 11 | Delta |
|--------|--------|--------|-------|
| Evidence Grounding Failures | 0 | 10 | +10 |
| Participants Skipped | 0 | ~10 | +10 |
| Failure Rate | 0% | ~20% | +20% |

**Failed Participants** (from logs):
- 386 (failed in both zero_shot AND few_shot)
- 409, 456, 487, 367 (failed in zero_shot)
- 386, and others (failed in few_shot)

---

## Root Cause Analysis

### 1. Substring Matching is Too Strict

The current validation uses **exact normalized substring matching**:

```python
# evidence_validation.py:193
grounded = bool(quote_norm) and (quote_norm in transcript_norm)
```

This fails when the LLM:
- **Paraphrases** slightly ("I feel tired" vs "I'm feeling tired")
- **Combines quotes** from multiple utterances
- **Truncates** or **extends** quotes with context
- **Reformats** speaker labels or punctuation

### 2. Fuzzy Mode Exists but Requires Optional Dependency

```python
# evidence_validation.py:195-200
if not grounded and mode == "fuzzy":
    if _rapidfuzz_fuzz is None:
        raise RuntimeError("evidence_quote_validation_mode='fuzzy' requires rapidfuzz")
    ratio = _rapidfuzz_fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0
    grounded = ratio >= fuzzy_threshold
```

But:
- Default mode is `"substring"` (strict)
- `rapidfuzz` is not in default dependencies
- Users must explicitly enable fuzzy mode

### 3. Fail-Fast on All-Rejected is Too Aggressive

```bash
# .env.example:130
QUANTITATIVE_EVIDENCE_QUOTE_FAIL_ON_ALL_REJECTED=true
```

When ALL quotes are rejected, the entire participant is skipped. This is the "loud failure" behavior, but with 20% failure rate, it's too aggressive.

---

## Evidence from Logs

```
2026-01-04T04:58:16.810798Z [error] failure_evidence_hallucination
  message='LLM returned evidence quotes but none could be grounded in the transcript.'
  mode=zero_shot
  participant_id=386

2026-01-04T11:05:50.398133Z [error] failure_evidence_hallucination
  message='LLM returned evidence quotes but none could be grounded in the transcript.'
  mode=few_shot
  participant_id=386
```

Same participant (386) failing in BOTH modes = systematic issue, not random.

---

## What Changed Between Run 10 and Run 11

Commits between runs:
```
5ad9d15 Implement pipeline robustness enhancements (Specs 053-057)
e983f91 chore(pipeline): implement brittleness suite (specs 053-057)
```

Spec 053 (evidence grounding) was implemented and **enabled by default**.

---

## Proposed Fixes (Based on 2025-2026 Best Practices)

### Fix 1: Enable RapidFuzz with `partial_ratio()` (MANDATORY)

**Research Finding**: Production hallucination detection systems use fuzzy matching with **threshold 0.85-0.90**.

> "To minimize errors in hallucination span detection, researchers use fuzzy matching with a similarity threshold of 0.9 (partial ratio)." — [SemEval 2025 Hallucination Detection](https://arxiv.org/html/2505.20880)

**Implementation**:

1. Add `rapidfuzz` to dependencies (already optional):
```toml
# pyproject.toml - move to required deps
dependencies = [
    ...
    "rapidfuzz>=3.0.0",
]
```

2. Change defaults:
```bash
# .env.example
QUANTITATIVE_EVIDENCE_QUOTE_VALIDATION_MODE="fuzzy"
QUANTITATIVE_EVIDENCE_QUOTE_FUZZY_THRESHOLD=0.85  # Research-backed threshold
```

3. Use `fuzz.partial_ratio()` for substring matching (already implemented, just enable):
```python
# evidence_validation.py:199 - current code is correct
ratio = _rapidfuzz_fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0
grounded = ratio >= fuzzy_threshold
```

### Fix 2: Add Case-Insensitive Preprocessing

**Research Finding**: RapidFuzz 3.0+ doesn't preprocess by default.

> "When comparing strings that have the same characters but different cases, their similarity score value might be different. For case-insensitive matching, use the `utils.default_process` processor." — [RapidFuzz GitHub](https://github.com/rapidfuzz/RapidFuzz)

**Implementation**:
```python
from rapidfuzz import utils as rapidfuzz_utils

def normalize_for_quote_match(text: str) -> str:
    # Existing normalization...
    normalized = unicodedata.normalize("NFKC", text).translate(_SMART_QUOTES)
    # Add RapidFuzz default processing for consistency
    normalized = rapidfuzz_utils.default_process(normalized)
    return normalized
```

### Fix 3: Soften Fail-Fast Behavior

Change from "fail if ALL rejected" to "fail if >90% rejected":

```python
# Instead of:
if stats.validated_count == 0 and stats.extracted_count > 0:
    raise EvidenceGroundingError(...)

# Do:
rejection_rate = stats.rejected_count / max(stats.extracted_count, 1)
if rejection_rate > 0.90:  # Only fail if 90%+ rejected
    raise EvidenceGroundingError(...)
```

### Fix 4: Multi-Level Validation (Hierarchical Semantic Piece)

**Research Finding**: Best systems use hierarchical matching.

> "Hierarchical semantic piece (HSP) uses hierarchical semantic piece extraction and evidence matching grounded in embedding cosine similarity." — [Springer 2025](https://link.springer.com/article/10.1007/s40747-025-01833-9)

**Implementation** (optional enhancement):
1. First: Exact substring match (fastest)
2. Second: RapidFuzz partial_ratio (fallback)
3. Third: Embedding cosine similarity (expensive, optional)

```python
def validate_evidence_grounding(...) -> tuple[...]:
    # Level 1: Exact substring
    if quote_norm in transcript_norm:
        grounded = True
    # Level 2: Fuzzy partial match
    elif mode == "fuzzy":
        grounded = partial_ratio(quote_norm, transcript_norm) >= threshold
    # Level 3: Embedding similarity (optional)
    elif mode == "semantic":
        grounded = cosine_similarity(embed(quote), embed(transcript)) >= threshold
```

---

## Observability Gaps

**Current**: We log `quote_hash` but NOT the actual quote text.

**Problem**: Can't debug WHY a quote was rejected without seeing it.

**Proposed Enhancement**:
```python
logger.warning(
    "evidence_quote_rejected",
    domain=domain,
    quote_preview=quote[:50],  # First 50 chars (privacy-safe if needed)
    quote_hash=_stable_hash(quote),
    transcript_sample=transcript_text[max(0, match_start-20):match_end+20],  # Context
    # ... existing fields
)
```

---

## Validation Required

Before fixing, run analysis:
1. Extract rejected quotes from failing participants
2. Check what normalization differences exist
3. Test with fuzzy matching at various thresholds

---

## Decision Points for Senior Review

- [ ] **Enable fuzzy mode by default** (add rapidfuzz to required deps) — RECOMMENDED
- [ ] **Set threshold to 0.85** (research-backed)
- [ ] **Add RapidFuzz preprocessing** for case-insensitive matching
- [ ] **Soften fail-on-all-rejected** to 90% threshold
- [ ] **Add quote preview to logs** for debugging
- [ ] **Consider hierarchical matching** (exact → fuzzy → semantic)

---

## References

### Internal
- Spec 053: Evidence Quote Grounding
- Run 11 Log: `data/outputs/run11_confidence_suite_20260103_215102.log`
- Code: `src/ai_psychiatrist/services/evidence_validation.py`

### 2025-2026 Research
1. [RapidFuzz GitHub](https://github.com/rapidfuzz/RapidFuzz) - MIT-licensed, 2-100x faster than FuzzyWuzzy
2. [SemEval 2025 Hallucination Detection](https://arxiv.org/html/2505.20880) - Uses 0.9 threshold with partial_ratio
3. [Hierarchical Semantic Piece (Springer 2025)](https://link.springer.com/article/10.1007/s40747-025-01833-9) - Multi-level matching approach
4. [HaluGate: Real-Time Hallucination Detection (vLLM 2025)](https://blog.vllm.ai/2025/12/14/halugate.html) - Production detection pipeline
5. [EdinburghNLP Awesome Hallucination Detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) - Curated paper list
