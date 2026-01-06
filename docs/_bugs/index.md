# Bug Reports Index

This directory tracks active bug investigations. Resolved bugs are archived in `docs/_archive/bugs/`.

## Active Investigations

_No active investigations at this time._

## Active Bugs (Pending Senior Review)

| ID | Title | Status | Impact |
|----|-------|--------|--------|

_No bugs pending senior review at this time._

## Recently Resolved (2026-01-06)

| ID | Title | Status | Resolution |
|----|-------|--------|------------|
| [BUG-035](./BUG-035-FEW-SHOT-PROMPT-CONFOUND.md) | Few-Shot vs Zero-Shot Prompt Confound | ✅ Resolved | Empty retrieval now produces identical prompt to zero-shot; regression tests added |

## Recently Resolved (2026-01-04)

| ID | Title | Status | Resolution |
|----|-------|--------|------------|
| [BUG-032](../_archive/bugs/BUG-032_EVIDENCE_GROUNDING_ALL_REJECTED_POLICY.md) | Evidence grounding “all rejected” policy | ✅ Resolved | Default is fail-open + failure registry event; strict mode remains available |
| [BUG-033](../_archive/bugs/BUG-033_JSON_PARSE_FAILURES_POST_SPEC059.md) | JSON parse failures (post Spec 059) | ✅ Resolved | json-repair fallback + improved failure metadata + regression tests |
| [BUG-034](../_archive/bugs/BUG-034_PRIVACY_SAFE_OBSERVABILITY_UPGRADES.md) | Privacy-safe observability upgrades | ✅ Resolved | Counts-only summaries + failure/telemetry registries; no raw transcript text |
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
3. Use next available number (currently: BUG-036)
4. Include: Severity, Status, File, Description, Impact, Fix

When resolved, move to `docs/_archive/bugs/` with naming convention:
`BUG-XXX_UPPER_SNAKE_CASE_TITLE.md`
