# BUG-011: Evidence Extraction JSON Parsing is Fragile

**Severity**: LOW (P3)
**Status**: OPEN
**Date Identified**: 2025-12-19
**Spec Reference**: `docs/specs/09_QUANTITATIVE_AGENT.md`

---

## Executive Summary

Evidence extraction in the QuantitativeAssessmentAgent uses `json.loads` on the raw response with **no tolerant fixups or repair**. When the LLM output contains minor formatting issues (trailing commas, smart quotes, extra text), parsing fails and the agent silently proceeds with **empty evidence**, reducing few-shot quality without clear failure signals.

---

## Evidence

- Evidence extraction parses JSON directly with no tolerant fixups or repair. (`src/ai_psychiatrist/agents/quantitative.py:173-180`)
- On parse failure, it logs a warning and uses `{}` (empty evidence) with no recovery. (`src/ai_psychiatrist/agents/quantitative.py:179-181`)
- Robust JSON repair utilities exist for scoring but are not reused here. (`src/ai_psychiatrist/agents/quantitative.py:252-276` and `_tolerant_fixups` at `src/ai_psychiatrist/agents/quantitative.py:314-335`)

---

## Impact

- A single minor formatting issue can drop all evidence and reduce few-shot prompting to near zero-shot.
- This creates silent degradation and makes debugging difficult.

---

## Recommended Fix

- Reuse `_tolerant_fixups` for evidence parsing.
- Optionally add a lightweight LLM repair path (similar to scoring repair) for evidence JSON.
- At minimum, include response preview in the warning log to aid debugging.

---

## Files Involved

- `src/ai_psychiatrist/agents/quantitative.py`
