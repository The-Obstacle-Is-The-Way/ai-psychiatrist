# BUG-011: Evidence Extraction JSON Parsing is Fragile

**Severity**: LOW (P3)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/09_QUANTITATIVE_AGENT.md`

---

## Executive Summary

Evidence extraction in the QuantitativeAssessmentAgent parses JSON using `_strip_json_block` (handles `<answer>` tags and markdown fences) but **does not apply `_tolerant_fixups` or any repair path**. When the LLM output contains minor formatting issues (trailing commas, smart quotes, extra text), parsing fails and the agent proceeds with **empty evidence**, reducing few-shot quality without clear failure signals.

---

## Evidence

- Evidence extraction uses `_strip_json_block` then `json.loads` without `_tolerant_fixups` or repair. (`src/ai_psychiatrist/agents/quantitative.py` in `_extract_evidence`)
- On parse failure, it logs a warning and uses `{}` (empty evidence) with no recovery. (`src/ai_psychiatrist/agents/quantitative.py` in `_extract_evidence`)
- Robust JSON repair utilities exist for scoring but are not reused here. (`src/ai_psychiatrist/agents/quantitative.py` in `_parse_response` and `_tolerant_fixups`)

---

## Impact

- A single minor formatting issue can drop all evidence and reduce few-shot prompting to near zero-shot.
- This creates silent degradation and makes debugging difficult.

---

## Scope & Disposition

- **Code Path**: Current implementation (`src/ai_psychiatrist/agents/quantitative.py`).
- **Fix Category**: Robustness / error handling.
- **Recommended Action**: Fix soon (prefer now); reuse tolerant parsing to avoid silent degradation.

---

## Recommended Fix

- Reuse `_tolerant_fixups` for evidence parsing.
- Optionally add a lightweight LLM repair path (similar to scoring repair) for evidence JSON.
- At minimum, include response preview in the warning log to aid debugging.

---

## Files Involved

- `src/ai_psychiatrist/agents/quantitative.py`
