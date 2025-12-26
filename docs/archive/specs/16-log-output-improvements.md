# Spec 16: Log Output Improvements

> **STATUS: IMPLEMENTED**
>
> This spec addresses ANSI escape code pollution in log files when using `tee`.
> One-line fix that significantly improves log readability and grep-ability.
>
> **Tracked by**: [GitHub Issue #54](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/54)
>
> **Last Updated**: 2025-12-26

---

## Objective

Automatically disable ANSI color codes when stdout is not a TTY, making log files
readable when piped through `tee` or redirected to files.

## Problem Statement

### Current Behavior

When running reproduction with `tee`:

```bash
uv run python scripts/reproduce_results.py 2>&1 | tee reproduction.log
```

Log files contain raw ANSI escape codes:

```
[2m2025-12-24T03:45:17.652082Z[0m [[32m[1minfo     [0m] [1mLoading transcript[0m
```

This makes logs:
- Hard to read in editors
- Impossible to grep effectively
- Ugly when viewed in non-terminal contexts

### Root Cause

In `src/ai_psychiatrist/infrastructure/logging.py` (in `setup_logging()`, console format branch):

```python
final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=True,  # <-- HARDCODED
        exception_formatter=structlog.dev.plain_traceback,
    )
]
```

---

## Solution

### TTY Auto-Detection

Replace hardcoded `colors=True` with automatic TTY detection:

```python
import sys

final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=sys.stdout.isatty(),  # <-- Auto-detect
    )
]
```

This follows the Unix convention:
- **TTY present**: Show colors for human readability
- **TTY absent (pipe/redirect)**: No colors for machine processing

Also follow the de-facto ecosystem convention:
- If `NO_COLOR` is set (non-empty), disable colors even if stdout is a TTY.
  - Reference: https://no-color.org/

---

## Deliverables

### 1. Update Logging Configuration

**File**: `src/ai_psychiatrist/infrastructure/logging.py`

```python
# Before
final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=True,
    )
]

# After
import sys

final_processors = [
    structlog.dev.ConsoleRenderer(
        colors=sys.stdout.isatty(),
    )
]
```

### 2. Optional: Force Colors Setting

Add configuration option for cases where user wants to force colors on/off:

**File**: `src/ai_psychiatrist/config.py`

```python
class LoggingSettings(BaseSettings):
    # ... existing fields ...

    force_colors: bool | None = Field(
        default=None,
        description="Force colors on/off. None = auto-detect TTY.",
    )
```

**File**: `src/ai_psychiatrist/infrastructure/logging.py`

```python
import os
import sys

def _should_use_colors(settings: LoggingSettings) -> bool:
    """Determine if colors should be used."""
    if settings.force_colors is not None:
        return settings.force_colors
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()
```

---

## Implementation

### Minimal Fix (Recommended)

Single line change in `logging.py`:

```diff
- colors=True,
+ colors=sys.stdout.isatty(),
```

**Effort**: 5 minutes
**Risk**: None

### Extended Fix (Optional)

Add `LOG_FORCE_COLORS` env var for explicit control:

```bash
# Force colors even when piped
LOG_FORCE_COLORS=true uv run python scripts/reproduce_results.py | tee log.txt

# Force no colors even in terminal
LOG_FORCE_COLORS=false uv run python scripts/reproduce_results.py
```

**Effort**: 15 minutes
**Risk**: Very low

---

## Acceptance Criteria

- [ ] `python script.py | tee file.log` produces clean logs without ANSI codes
- [ ] `python script.py` (interactive) still shows colors
- [ ] Existing tests pass
- [ ] Optional: `LOG_FORCE_COLORS` env var works
- [ ] Optional: `NO_COLOR` disables colors even in TTY

---

## Testing

### Manual Test

```bash
# Should show colors in terminal
uv run python -c "from ai_psychiatrist.infrastructure.logging import get_logger; get_logger('test').info('test')"

# Should NOT show colors when piped
uv run python -c "from ai_psychiatrist.infrastructure.logging import get_logger; get_logger('test').info('test')" | cat

# Should NOT show colors in log file
uv run python -c "from ai_psychiatrist.infrastructure.logging import get_logger; get_logger('test').info('test')" > /tmp/test.log && cat /tmp/test.log
```

### Unit Test

```python
def test_logger_no_colors_when_not_tty(monkeypatch):
    """Logger should not use colors when stdout is not a TTY."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    # Reinitialize logging
    # Check ConsoleRenderer has colors=False
```

---

## Workarounds (Historical)

These workarounds were useful before the fix landed; they may still help if you're running an older commit or capturing logs from third-party tools.

### Option A: Use JSON Format (Only If Script Respects `LOG_FORMAT`)

```bash
LOG_FORMAT=json uv run python scripts/reproduce_results.py | tee log.json
```

JSON output never has colors, but note: `scripts/reproduce_results.py` currently forces
`format="console"` when calling `setup_logging(...)`, so this workaround only applies
after we stop hardcoding the log format in scripts (or when running other entrypoints
that call `setup_logging(get_settings().logging)`).

### Option B: Strip ANSI Codes

```bash
uv run python scripts/reproduce_results.py 2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' > clean.log)
```

### Option C: Use `script` Command

```bash
script -q /dev/null uv run python scripts/reproduce_results.py > log.txt
```

---

## Priority

**LOW** - Cosmetic issue that doesn't affect functionality. However, it's a
one-line fix that significantly improves developer experience.

---

## References

- GitHub Issue #54: ANSI escape codes in log files
- BUG-025 doc (if created): docs/bugs/bug-025-ansi-escape-codes-in-log-files.md
- structlog ConsoleRenderer: https://www.structlog.org/en/stable/api.html#structlog.dev.ConsoleRenderer
