# BUG-032: Evidence Grounding “All Rejected” Should Not Drop Participants by Default

**Status**: ✅ Resolved (Implemented)
**Severity**: P0 (Participant loss / invalid comparisons)
**Filed**: 2026-01-04
**Resolved**: 2026-01-04
**Component**: `src/ai_psychiatrist/services/evidence_validation.py`, `src/ai_psychiatrist/agents/quantitative.py`, `src/ai_psychiatrist/config.py`
**Observed In**: Run 11 (`data/outputs/run11_confidence_suite_20260103_215102.log`)

---

## Summary

Run 11 produced **5/41** failures per mode (zero-shot and few-shot) with:

> `EvidenceGroundingError: LLM returned evidence quotes but none could be grounded in the transcript.`

This happened because:

1. Evidence grounding validation (Spec 053) rejected all extracted quotes for some participants.
2. The default policy at the time (`QUANTITATIVE_EVIDENCE_QUOTE_FAIL_ON_ALL_REJECTED=true`) treated “all rejected” as **fatal**, skipping the participant entirely.

Skipping participants is unacceptable for evaluation runs because it biases metrics and invalidates mode comparisons.

---

## Resolution

### 1) Make grounding more tolerant to transcript markup (still conservative)

Evidence grounding now ignores nonverbal/markup tags like `<laughter>` and `<ma>` during normalization, so a quote that omits these tags can still be grounded.

- Code: `src/ai_psychiatrist/services/evidence_validation.py` (`normalize_for_quote_match`)
- Regression: `tests/unit/services/test_evidence_validation.py::TestValidateEvidenceGrounding::test_nonverbal_tags_ignored_for_matching`

### 2) Change the default policy to “fail-open, record, continue”

Default behavior is now:

- If the LLM produced evidence but **none** could be grounded:
  - Record a privacy-safe failure event (Spec 056)
  - Continue with **empty evidence** (few-shot may degrade to transcript-only for that participant)

Strict mode is still available by setting:

```bash
QUANTITATIVE_EVIDENCE_QUOTE_FAIL_ON_ALL_REJECTED=true
```

- Code default: `src/ai_psychiatrist/config.py` (`QuantitativeSettings.evidence_quote_fail_on_all_rejected = False`)
- Run config: `.env.example` / `.env` updated to `false`
- Regression: `tests/unit/agents/test_quantitative.py::TestQuantitativeAssessmentAgent::test_extract_evidence_all_rejected_records_failure_and_continues`

---

## What This Prevents

- Participants being dropped due to evidence grounding failures
- “Run completes but comparisons are invalid” failure mode
- Silent degradation (we always record an observability event when all quotes are rejected)

---

## Notes

- This change does **not** weaken hallucination detection. Ungrounded evidence is still discarded.
- The system is now explicit about when it cannot ground evidence (failure registry + logs).
