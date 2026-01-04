# BUG-031: Observability Gaps - Cannot Debug Production Failures

**Status**: Open (High Priority)
**Severity**: P1 (Debugging Impediment)
**Filed**: 2026-01-04
**Component**: Multiple (logging, telemetry)
**Observed In**: Run 11 debugging session

---

## Summary

When investigating Run 11 failures (BUG-029, BUG-030), we discovered significant observability gaps that make it impossible to debug production failures without reproducing them locally.

---

## Gaps Identified

### Gap 1: No Raw Text in JSON Parse Failures

**Current Logging**:
```python
logger.warning(
    "Failed to parse LLM JSON after all repair attempts",
    json_error=str(json_error),
    text_hash=_stable_text_hash(text),
    text_length=len(text),
)
```

**Problem**: We only log the hash, not the actual malformed text. Cannot debug without reproducing.

**What We Need**:
```python
logger.warning(
    "Failed to parse LLM JSON after all repair attempts",
    json_error=str(json_error),
    error_position=json_error.pos,
    text_sample=text[max(0, json_error.pos-100):json_error.pos+100],
    fixups_applied=applied_fixes,
    # ... existing fields
)
```

### Gap 2: No Quote Preview in Evidence Rejection

**Current Logging**:
```python
logger.warning(
    "evidence_quote_rejected",
    domain=domain,
    quote_hash=_stable_hash(quote),
    quote_len=len(quote),
)
```

**Problem**: Cannot see WHY a quote was rejected (what text didn't match).

**What We Need**:
```python
logger.warning(
    "evidence_quote_rejected",
    domain=domain,
    quote_preview=quote[:80],  # Or use quote_hash if privacy-sensitive
    quote_normalized=quote_norm[:80],
    best_partial_match_score=0.72,  # If fuzzy enabled
    # ... existing fields
)
```

### Gap 3: No Telemetry for Failure Patterns

**Current State**: Failures are logged but not aggregated.

**What We Need**: Structured telemetry for post-run analysis:
```python
record_telemetry(
    TelemetryCategory.EVIDENCE_GROUNDING,
    participant_id=pid,
    domain=domain,
    outcome="rejected",
    quote_hash=quote_hash,
    similarity_score=0.72,
    threshold=0.85,
)
```

### Gap 4: No Retry Waste Detection

**Current State**: Same broken output triggers multiple retries.

**What We Need**:
```python
# Track text hashes across retries
if text_hash in seen_failure_hashes:
    logger.warning(
        "Deterministic retry - same output failing again",
        text_hash=text_hash,
        retry_attempt=attempt,
        recommendation="Consider temperature increase",
    )
```

### Gap 5: No Per-Participant Summary

**Current State**: Must grep logs to understand participant outcomes.

**What We Need**: Structured JSON output per participant:
```json
{
  "participant_id": 386,
  "mode": "zero_shot",
  "outcome": "failed",
  "failure_reason": "evidence_grounding",
  "quotes_extracted": 8,
  "quotes_rejected": 8,
  "json_parse_attempts": 1,
  "json_parse_failures": 0,
  "duration_seconds": 45.2
}
```

### Gap 6: No Preflight Validation Report

**Current State**: Run starts, then fails mid-way.

**What We Need**: Preflight check that:
1. Validates all participant transcripts exist
2. Checks embedding artifacts match config
3. Verifies LLM connectivity
4. Reports expected runtime

---

## Impact

| Activity | Time Without Observability | Time With Observability |
|----------|---------------------------|-------------------------|
| Identify failing participants | 10 min (grep logs) | 1 min (JSON summary) |
| Debug JSON parse failure | 30+ min (reproduce locally) | 5 min (log sample) |
| Debug evidence rejection | Can't (no quote logged) | 2 min (see preview) |
| Detect deterministic retries | Can't | Immediate (warning) |

---

## Proposed Implementation

### Phase 1: Critical Debugging Info (Immediate)

Add to `parse_llm_json()`:
```python
# On failure, log sample around error position
error_start = max(0, json_error.pos - 100)
error_end = min(len(text), json_error.pos + 100)
logger.error(
    "json_parse_failure_context",
    error_position=json_error.pos,
    text_sample=text[error_start:error_end],
    char_at_error=repr(text[json_error.pos]) if json_error.pos < len(text) else "EOF",
)
```

Add to `validate_evidence_grounding()`:
```python
# On rejection, log quote preview (first 80 chars)
logger.warning(
    "evidence_quote_rejected",
    quote_preview=quote[:80] + "..." if len(quote) > 80 else quote,
    # ... existing fields
)
```

### Phase 2: Structured Telemetry (Short-term)

Create `telemetry_participant.json` per run with per-participant outcomes.

### Phase 3: Retry Optimization (Medium-term)

Track failure hashes to avoid deterministic retry waste.

---

## Privacy Considerations

**Concern**: Logging raw text may expose PII from transcripts.

**Mitigations**:
1. Log only first N characters (preview)
2. Hash sensitive fields
3. Use separate debug log level that's not enabled in production
4. Redact speaker names if needed

---

## Code Locations Requiring Changes

| File | Function | Change |
|------|----------|--------|
| `responses.py` | `parse_llm_json()` | Add error context logging |
| `evidence_validation.py` | `validate_evidence_grounding()` | Add quote preview |
| `reproduce_results.py` | `evaluate_participant()` | Add per-participant JSON output |
| NEW | `telemetry.py` | Add structured telemetry recording |

---

## Decision Points for Senior Review

- [ ] **Add raw text samples to JSON parse failures** (with length limits)
- [ ] **Add quote previews to evidence rejections**
- [ ] **Create per-participant JSON summary**
- [ ] **Add deterministic retry detection**
- [ ] **Define privacy guidelines** for log content

---

## References

- BUG-029: Evidence Grounding Too Strict
- BUG-030: JSON Control Char Sanitization Incomplete
- Run 11 Log: `data/outputs/run11_confidence_suite_20260103_215102.log`
