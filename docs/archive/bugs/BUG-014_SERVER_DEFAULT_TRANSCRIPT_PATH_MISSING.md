# BUG-014: Server Default Transcript Path Missing

**Severity**: HIGH (P1)
**Status**: RESOLVED
**Date Identified**: 2025-12-19
**Date Resolved**: 2025-12-19
**Spec Reference**: `docs/specs/11_FULL_PIPELINE.md`, `docs/specs/12.5_FINAL_CLEANUP_LEGACY_REMOVAL.md`

---

## Executive Summary

`server.py` uses `InterviewSimulator` from the legacy `agents/` package. The simulator defaults to loading `agents/transcript.txt`, but that file does **not** exist in the repo. As a result, the `/full_pipeline` endpoint fails with `FileNotFoundError` on every request unless an environment variable is manually set.

---

## Evidence (Pre-Fix)

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

## Scope & Disposition

- **Code Path**: Legacy server path (`server.py` + `agents/interview_simulator.py`) but currently active runtime path.
- **Fix Category**: Production stability.
- **Recommended Action**: Resolved; server.py now uses TranscriptService from modern codebase.

---

## Resolution

1. Replaced `InterviewSimulator` with `TranscriptService` from `ai_psychiatrist.services`.
2. API now accepts either:
   - `participant_id`: Loads transcript from DAIC-WOZ dataset via `TranscriptService.load_transcript()`
   - `transcript_text`: Uses raw text input via `TranscriptService.load_transcript_from_text()`
3. Removed dependency on hardcoded file paths.
4. Clear error messages when transcript cannot be resolved.

---

## Verification

```bash
ruff check server.py
pytest tests/ -v --no-cov
```

---

## Files Changed

- `server.py`
