# Bug Reports Index

This directory tracks active bug investigations. Resolved bugs are archived in `docs/_archive/bugs/`.

## Active Investigations

_No active investigations at this time._

## Active Bugs (Pending Senior Review)

| ID | Title | Status | Impact |
|----|-------|--------|--------|

_No bugs pending senior review at this time._

## Archive

All resolved bugs are in `docs/_archive/bugs/`:

- **BUG-001 to BUG-020**: Legacy bugs from initial development
- **BUG-021 to BUG-025**: Spec 048-052 implementation audit (2026-01-03)
- **ANALYSIS-026**: JSON parsing architecture audit (2026-01-03)
- **BUG-027 to BUG-034**: Various fixes (2026-01-04)
- **BUG-035**: Few-shot prompt confound fix (2026-01-06)
- **BUG-036**: CLI arg validation bypass (2026-01-07)
- **BUG-037**: Non-archive doc link drift fix (2026-01-07)

## Filing New Bugs

1. Check if bug already exists in archive
2. Create file: `docs/_bugs/BUG-XXX-short-title.md`
3. Use next available number (currently: BUG-038)
4. Include: Severity, Status, File, Description, Impact, Fix

When resolved, move to `docs/_archive/bugs/` with naming convention:
`BUG-XXX_UPPER_SNAKE_CASE_TITLE.md`
