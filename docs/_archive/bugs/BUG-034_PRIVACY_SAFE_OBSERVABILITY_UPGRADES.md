# BUG-034: Observability Gaps (Privacy-Safe Debugging)

**Status**: ✅ Resolved (Implemented)
**Severity**: P1 (Debugging impediment)
**Filed**: 2026-01-04
**Resolved**: 2026-01-04
**Component**: `src/ai_psychiatrist/infrastructure/observability.py`, `src/ai_psychiatrist/infrastructure/telemetry.py`

---

## Summary

Several run-debugging requests asked for raw transcript / raw LLM output snippets to be logged on failures.

This is **not acceptable** in this repository:

- DAIC-WOZ transcripts are licensed and must not leak into logs/artifacts.
- LLM outputs can contain transcript text (evidence quotes) and are treated as sensitive.

The correct approach is **privacy-safe observability**:

- stable hashes
- lengths
- error types and positions
- per-run JSON registries

---

## Resolution

### 1) Failure Registry (Spec 056)

Evaluation runs initialize and persist:

- `failures_{run_id}.json` — per-failure taxonomy + counts (no raw text)

SSOT:
- `src/ai_psychiatrist/infrastructure/observability.py`
- `scripts/reproduce_results.py` (`init_run_observability`, `finalize_run_observability`)

### 2) Telemetry Registry (Spec 060)

Evaluation runs also persist:

- `telemetry_{run_id}.json` — JSON repair path usage + PydanticAI retry reasons

SSOT:
- `src/ai_psychiatrist/infrastructure/telemetry.py`

### 3) Evidence grounding summaries (Spec 053)

When evidence grounding rejects quotes, we now emit a single counts-only summary event:

- `evidence_grounding_complete`: extracted/validated/rejected counts + per-domain counts + transcript hash/len

SSOT:
- `src/ai_psychiatrist/services/evidence_validation.py`

### 4) JSON parse failure metadata (Spec 059)

When `parse_llm_json()` fails after all repair attempts, we log:

- JSON decode error type + lineno/colno/pos
- repair error type
- text hash + length

SSOT:
- `src/ai_psychiatrist/infrastructure/llm/responses.py`

---

## Design Principle

If you need raw failure payloads for research forensics, introduce an **explicit opt-in unsafe flag**
that writes quarantined files locally and is excluded from VCS. It must never be enabled by default.
