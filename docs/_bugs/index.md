# Bug Reports Index

This directory tracks active bug investigations. Resolved bugs are archived in `docs/_archive/bugs/`.

## Active Investigations

_No active investigations at this time._

## Active Bugs (Pending Senior Review)

| ID | Title | Severity | Status | Key Fix |
|----|-------|----------|--------|---------|
| [BUG-029](./BUG-029-evidence-grounding-too-strict.md) | Evidence Grounding Too Strict (20% Failure Rate) | P0 | Open | Enable RapidFuzz fuzzy matching @ 0.85 threshold |
| [BUG-030](./BUG-030-json-control-char-sanitization-incomplete.md) | JSON Control Char Sanitization Incomplete | P0 | Open | Reorder fixups, add invalid escape handler |
| [BUG-031](./BUG-031-observability-gaps-debugging-failures.md) | Observability Gaps for Debugging | P1 | Open | Add raw text logging, per-participant summary |

## Recently Resolved (2026-01-04)

| ID | Title | Status | Resolution |
|----|-------|--------|------------|
| [BUG-027](../_archive/bugs/BUG-027_CONSISTENCY_TEMPERATURE_OPTIMIZATION.md) | Consistency Temperature Optimization | ✅ Resolved | Baseline set to `CONSISTENCY_TEMPERATURE=0.2` (docs + code + tests) |

## Recently Resolved (2026-01-03)

| ID | Title | Status | Resolution |
|----|-------|--------|------------|
| [ANALYSIS-026](./ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md) | JSON Parsing Architecture Audit | ✅ Resolved | Canonical parser, no silent fallbacks, Ollama format:json |
| [BUG-021](../_archive/bugs/BUG-021_FLOAT_EQUALITY_DOMINANT_POINTS.md) | Float Equality in Dominant Points | Resolved | Already fixed with epsilon |
| [BUG-022](../_archive/bugs/BUG-022_DRY_VIOLATION_ORACLE_ITEMS.md) | DRY Violation in Oracle Items | Resolved | Helper already exists |
| [BUG-023](../_archive/bugs/BUG-023_HARDCODED_CALIBRATION_LOGIC.md) | Hardcoded Calibration Logic | Won't Fix | Tech debt, not a bug |
| [BUG-024](../_archive/bugs/BUG-024_EMBEDDING_LINEAR_SCAN.md) | O(N*M) Linear Scan | Resolved | Already vectorized |
| [BUG-025](../_archive/bugs/BUG-025_PYDANTIC_AI_JSON_PYTHON_LITERAL_FALLBACK.md) | PydanticAI TextOutput Python-Literal JSON | Resolved | Added tolerant parse fallback + consistency sampling resilience |

## Archive

All resolved bugs are in `docs/_archive/bugs/`:

- **BUG-001 to BUG-020**: Legacy bugs from initial development
- **BUG-021 to BUG-025**: Spec 048-052 implementation audit (2026-01-03)
- **ANALYSIS-026**: JSON parsing architecture audit (2026-01-03)

## Filing New Bugs

1. Check if bug already exists in archive
2. Create file: `docs/_bugs/BUG-XXX-short-title.md`
3. Use next available number (currently: BUG-032)
4. Include: Severity, Status, File, Description, Impact, Fix

When resolved, move to `docs/_archive/bugs/` with naming convention:
`BUG-XXX_UPPER_SNAKE_CASE_TITLE.md`
