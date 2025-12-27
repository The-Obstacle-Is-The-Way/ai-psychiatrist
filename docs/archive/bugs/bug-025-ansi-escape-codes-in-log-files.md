# BUG-025: ANSI Escape Codes in Log Files

**Status**: ✅ RESOLVED
**Severity**: LOW (cosmetic, doesn't affect functionality)
**Found**: 2025-12-23 (during few-shot reproduction run)
**Resolved**: 2025-12-26 (Spec 16 implementation)
**Related**: Issue #53 (experiment tracking/provenance)

---

## Resolution

**Fixed in**: Spec 16 (Log Output Improvements), commit `b72e45d`

The fix implemented the recommended approach:
- Added `LOG_FORCE_COLORS` config setting (`LoggingSettings.force_colors`)
- Auto-detects TTY via `sys.stdout.isatty()`
- Respects `NO_COLOR` environment variable

**Code location**: `src/ai_psychiatrist/infrastructure/logging.py:33-39`

```python
def _should_use_colors(settings: LoggingSettings) -> bool:
    if settings.force_colors is not None:
        return settings.force_colors
    if os.environ.get("NO_COLOR"):
        return False
    return _stdout_isatty()
```

**Test coverage**: `tests/unit/infrastructure/test_logging.py`
- `test_setup_logging_console_disables_colors_when_not_tty`
- `test_setup_logging_console_force_colors_overrides_tty`
- `test_setup_logging_console_no_color_env_disables_colors`

---

## Summary

When reproduction runs are piped to log files via `tee`, ANSI color escape codes are written as raw text, making logs difficult to read and search.

**Example** (from `reproduction_run_20251223_224516.log`):
```log
[2m2025-12-24T03:45:17.652082Z[0m [[32m[1minfo     [0m] [1mLoading transcript[0m
```

**Expected output**:
```text
2025-12-24T03:45:17.652082Z [info] Loading transcript
```

---

## Root Cause

In `src/ai_psychiatrist/infrastructure/logging.py:72-75`:

```python
final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=True,  # <-- HARDCODED
        exception_formatter=structlog.dev.plain_traceback,
    )
]
```

The `colors=True` is hardcoded, so even when stdout is redirected to a file, ANSI codes are emitted.

---

## Impact

1. **Log file readability**: Raw escape codes clutter the output
2. **Grep unfriendly**: Searching logs requires escaping or stripping codes
3. **Reproduction traceability**: Harder to review historical runs

---

## Recommended Fix

Auto-detect TTY and disable colors when output is redirected:

```python
import sys

final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=sys.stdout.isatty(),  # <-- Auto-detect
        exception_formatter=structlog.dev.plain_traceback,
    )
]
```

**Alternative**: Add `LOG_COLORS` setting to config for explicit control.

---

## Workaround

For now, strip ANSI codes from existing logs:

```bash
# Using sed
sed 's/\x1b\[[0-9;]*m//g' reproduction_run.log > clean.log

# Using perl
perl -pe 's/\e\[[0-9;]*m//g' reproduction_run.log > clean.log

# Or use LOG_FORMAT=json in .env (no colors in JSON output)
```

---

## Files Involved

- `src/ai_psychiatrist/infrastructure/logging.py`
- `src/ai_psychiatrist/config.py` (if adding LOG_COLORS setting)

---

## Related

- Issue #53: Experiment tracking - could include clean log format requirements
