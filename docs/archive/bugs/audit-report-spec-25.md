# Audit Report: Spec 25 & AURC/AUGRC Implementation

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

## 2. Incorrect / Misleading
| File:Line | Issue | Proposed Fix |
| :--- | :--- | :--- |
| `Spec 25:15-18` (Status) | Lists Phase 1 as "To Do"; Status "Proposed". | Update Status to "In Progress"; Mark Phase 1 as "[Completed]". |
| `Spec 25:208` (Section 5.1) | Claims `item_signals` is missing from current output. | Update Section 5.1 to show `item_signals` is present in current JSON schema. |
| `Spec 25:440` (Phase 1) | Implementation Plan lists Phase 1 as future work. | Mark as Completed. |

## 3. Open Questions / Cannot Verify
*   **None**. The math is solid and the code state is unambiguous.

## 4. Implementation Gaps (Verified)
*   `src/ai_psychiatrist/metrics/` does not exist (Phase 2 pending).
*   `scripts/evaluate_selective_prediction.py` does not exist (Phase 4 pending).

## Action Plan
1.  Update `docs/specs/25-aurc-augrc-implementation.md` to reflect the actual state of the repo (Phase 1 done).
2.  No code changes required (code is ahead of spec).
