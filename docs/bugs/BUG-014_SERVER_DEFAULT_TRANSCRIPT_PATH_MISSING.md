# BUG-014: Server Default Transcript Path Missing

**Severity**: HIGH (P1)  
**Status**: OPEN  
**Date Identified**: 2025-12-19  
**Spec Reference**: `docs/specs/11_FULL_PIPELINE.md`, `docs/specs/12.5_FINAL_CLEANUP_LEGACY_REMOVAL.md`

---

## Executive Summary

`server.py` uses `InterviewSimulator` from the legacy `agents/` package. The simulator defaults to loading `agents/transcript.txt`, but that file does **not** exist in the repo. As a result, the `/full_pipeline` endpoint fails with `FileNotFoundError` on every request unless an environment variable is manually set.

---

## Evidence

- `InterviewSimulator` defaults to `agents/transcript.txt` if `TRANSCRIPT_PATH` is not set. (`agents/interview_simulator.py:11-19`)
- The file does not exist in the repository:
  - `ls agents/transcript.txt` → `No such file or directory`
- `server.py` always calls `interview_loader.load()` with no request-provided transcript. (`server.py:24-39`)

---

## Impact

- The legacy API fails at runtime with a 500 error by default.
- Makes manual testing and demos impossible without undocumented setup steps.
- Masks deeper issues in server pipeline (see BUG-012).

---

## Recommended Fix

1. Replace `InterviewSimulator` with the new `TranscriptService` or accept transcript input in the API request.
2. If the legacy server must remain temporarily, include a default transcript file or validate `TRANSCRIPT_PATH` at startup and fail fast.
3. Remove legacy server usage once Spec 11 API is implemented.

---

## Files Involved

- `server.py`
- `agents/interview_simulator.py`
