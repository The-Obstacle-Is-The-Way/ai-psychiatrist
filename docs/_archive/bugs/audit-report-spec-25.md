# Audit Report: Spec 25 & AURC/AUGRC Implementation

**Date**: 2025-12-26 (Original) / 2025-12-27 (Resolved)
**Auditor**: Gemini (Original) / Claude (Resolution)
**Status**: ✅ RESOLVED - All recommendations implemented

---

## Resolution Summary

All items from the original audit have been addressed:

| Original Issue | Resolution |
|----------------|------------|
| Phase 1 listed as "To Do" | Spec 25 now marked **IMPLEMENTED** and archived |
| `item_signals` claimed missing | Present in `scripts/reproduce_results.py` (lines 108-110, 287-310) |
| Phase 2-4 pending | All phases implemented (see below) |

---

## Implementation Status (Verified 2025-12-27)

| Phase | Status | Evidence |
|-------|--------|----------|
| **Phase 1**: Signal persistence | ✅ Complete | `EvaluationResult.item_signals` in `reproduce_results.py` |
| **Phase 2**: Metrics module | ✅ Complete | `src/ai_psychiatrist/metrics/selective_prediction.py` (257 lines) |
| **Phase 3**: Bootstrap utilities | ✅ Complete | `src/ai_psychiatrist/metrics/bootstrap.py` (223 lines) |
| **Phase 4**: Evaluation script | ✅ Complete | `scripts/evaluate_selective_prediction.py` (690 lines) |

---

## Test Coverage

- 19 passing tests in `tests/unit/metrics/` + `tests/integration/`
- Canonical numeric test vector from Spec 25 Section 9.1 verified
- Bootstrap edge cases (single participant, empty data) covered

---

## Spec Location

Spec 25 has been archived to: `docs/archive/specs/25-aurc-augrc-implementation.md`

---

## Original Audit (Historical)

The original audit (below) is preserved for reference. All issues have been resolved.

---

**Date**: 2025-12-26
**Auditor**: Gemini
**Target**: `docs/specs/25-aurc-augrc-implementation.md` (Spec 25)

## Summary
The audit confirms that **Spec 25 is mathematically sound** and aligned with the reference implementations (`fd-shifts`, `AsymptoticAURC`), including the adapted definitions for selective prediction with abstention. However, the spec's implementation status is **outdated**: Phase 1 (signal persistence) is already implemented in the codebase, but the spec lists it as a future task.

## 1. Confirmed Correct
| Item | Verification Method | Notes |
| :--- | :--- | :--- |
| **Numeric Test Vector** | Manual Derivation | Section 9.1 vector produces exact matches for all working points, risks, and areas (AURC, AUGRC, truncated). |
| **Integration Convention** | Reference Check | Matches `fd-shifts` magnitude; Spec 25 correctly uses increasing coverage (dx > 0) vs `fd-shifts` decreasing. |
| **Denominator Logic** | First Principles | `N=P*8` correctly penalizes abstentions in "generalized risk", adapting `fd-shifts` logic to our use case. |
| **Signal Semantics** | Code Review | `quantitative.py` correctly implements `llm_evidence_count` and `keyword_evidence_count` as defined in spec. |
| **Ties/Plateaus** | Reference Check | "Unique confidence thresholds" rule aligns with `fd-shifts` working point logic. |

## 2. ~~Incorrect / Misleading~~ (RESOLVED)
| File:Line | Issue | Resolution |
| :--- | :--- | :--- |
| ~~`Spec 25:15-18` (Status)~~ | ~~Lists Phase 1 as "To Do"~~ | ✅ Spec marked IMPLEMENTED and archived |
| ~~`Spec 25:208` (Section 5.1)~~ | ~~Claims `item_signals` missing~~ | ✅ Spec archived; code has `item_signals` |
| ~~`Spec 25:440` (Phase 1)~~ | ~~Lists Phase 1 as future work~~ | ✅ All phases complete |

## 3. ~~Implementation Gaps~~ (RESOLVED)
| Gap | Resolution |
| :--- | :--- |
| ~~`src/ai_psychiatrist/metrics/` does not exist~~ | ✅ Module created with `selective_prediction.py` + `bootstrap.py` |
| ~~`scripts/evaluate_selective_prediction.py` does not exist~~ | ✅ Script created (690 lines, full CLI) |
