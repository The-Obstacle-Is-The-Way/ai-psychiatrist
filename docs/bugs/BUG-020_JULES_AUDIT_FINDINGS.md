# BUG-020: Jules Audit Findings (Validated)

**Severity**: MIXED (P2-P4)
**Status**: RESOLVED (audit closed; remaining items deferred)
**Date Identified**: 2025-12-20
**Date Resolved**: 2025-12-21
**Source**: Jules async agent audit, validated by Claude
**Original Branch**: `origin/dev_jules_audit-3010918533045583642`

---

## Overview

This document captures **validated findings** from the Jules async agent audit.
The original audit contained some false/outdated claims which have been corrected.

**Validation Summary**:
- Original claims: 12 bugs
- Valid findings: 8 bugs
- False/outdated: 2 bugs (BUG-007, BUG-008)
- Overstated severity: 1 bug (BUG-001)

---

## Valid Findings

### P2: Pickle Usage (Not P0)

**Original Claim**: "P0 RCE vulnerability via pickle"

**Validated Status**: ⚠️ P2 - Technical debt, not critical vulnerability

**Location**: `src/ai_psychiatrist/services/reference_store.py:93`

```python
with self._embeddings_path.open("rb") as f:
    raw_data = pickle.load(f)
```

**Reality Check**:
- File is generated internally by `scripts/generate_embeddings.py`
- Path comes from config, not user input
- Not exposed to untrusted data in normal operation
- Would require attacker to have filesystem write access

**Recommendation**: Consider migrating to SafeTensors/NPZ for embeddings in future.
**Priority**: P2 (technical debt, not security emergency)

---

### P2: Brittle JSON Parsing

**Location**: `src/ai_psychiatrist/agents/quantitative.py`

**Validated**: ✅ TRUE - Multi-level repair chain exists:
- `_strip_json_block()` - Tag/code-fence stripping (string-based)
- `_tolerant_fixups()` - Syntax repair
- `_llm_repair()` - Recursive LLM call

**Impact**: Adds latency, masks prompt issues, non-deterministic

**Recommendation**: Keep (Spec 09 requires robust parsing); optionally evaluate Ollama JSON mode later (`format: json`)
**Priority**: P3 (acceptable by-spec defensive coding)

---

### P2: Naive Sentence Splitting

**Location**: `src/ai_psychiatrist/agents/quantitative.py:245`

**Validated**: ✅ TRUE

```python
parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
```

**Impact**: Incorrectly splits "Dr. Smith" or "e.g. example"

**Recommendation**: Keep for paper fidelity (used only for keyword backfill); consider improving later if it measurably impacts MAE
**Priority**: P4 (low impact)

---

### P3: Global State in server.py

**Location**: `server.py` (lifespan + dependency injection)

**Validated Status**: ✅ FIXED - migrated to `app.state` (FastAPI best practice)

**Evidence**:
- Resources are initialized on `app.state` in `lifespan()` (e.g., `server.py:41`)
- Dependencies read from `request.app.state` (e.g., `server.py:83`)

**Priority**: RESOLVED

---

### P4: Missing py.typed Marker

**Location**: `src/ai_psychiatrist/`

**Validated Status**: ✅ FIXED - marker added and included in wheel builds

**Impact**: Package consumers can't verify types with mypy

**Evidence**:
- `src/ai_psychiatrist/py.typed` (added)
- `pyproject.toml:71` force-includes marker for hatchling builds

**Priority**: RESOLVED

---

### P4: Magic Number 999_999

**Location**: `server.py:32`

**Validated Status**: ✅ FIXED - named constant introduced

```python
AD_HOC_PARTICIPANT_ID = 999_999
```

**Priority**: RESOLVED

---

### P3: Hardcoded DOMAIN_KEYWORDS

**Location**: `src/ai_psychiatrist/agents/prompts/quantitative.py`

**Validated**: ✅ TRUE - Keyword lists are hardcoded in source (small, spec-aligned)

**Recommendation**: Keep for paper fidelity; consider externalizing only if clinical review requires non-code editing
**Priority**: P4 (low impact)

---

## Invalid/Outdated Claims

### ❌ BUG-007: "Over-mocking / Testing the Mock"

**Original Claim**: "Integration tests rely exclusively on MockLLMClient"

**Status**: **ALREADY FIXED** in PR #21

We added real Ollama E2E tests:
- `tests/e2e/test_ollama_smoke.py`
- `tests/e2e/test_agents_real_ollama.py`
- `tests/e2e/test_server_real_ollama.py`

---

### ❌ BUG-008: "Low Coverage 30-50%"

**Original Claim**: "~30-50% coverage"

**Status**: **FALSE**

Actual coverage: **96.52%** (603 tests passing)

---

## Action Items

| Item | Priority | Effort | Status |
|------|----------|--------|--------|
| Add `py.typed` marker | P4 | 1 min | DONE |
| Replace ad-hoc participant magic number | P4 | 1 min | DONE |
| Migrate `server.py` to `app.state` | P3 | 30 min | DONE |
| Evaluate JSON mode for Ollama | P3 | Research | Deferred |
| Replace pickle embeddings format | P2 | 2 hrs | Deferred |

---

## Original Source

The original Jules audit is preserved in the archived branch for reference.
Code changes from that branch were **NOT merged** due to:
- Broken test imports
- Missing spacy model installation
- Incomplete implementation

Only this validated documentation was extracted.
