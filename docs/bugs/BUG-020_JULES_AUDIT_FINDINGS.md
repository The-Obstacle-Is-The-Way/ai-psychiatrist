# BUG-020: Jules Audit Findings (Validated)

**Severity**: MIXED (P2-P4)
**Status**: OPEN (for review)
**Date Identified**: 2025-12-20
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

**Location**: `src/ai_psychiatrist/services/reference_store.py:100`

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
- `_strip_json_block()` - Regex extraction
- `_tolerant_fixups()` - Syntax repair
- `_llm_repair()` - Recursive LLM call

**Impact**: Adds latency, masks prompt issues, non-deterministic

**Recommendation**: Consider Ollama's native JSON mode (`format: json`)
**Priority**: P2

---

### P2: Naive Sentence Splitting

**Location**: `src/ai_psychiatrist/agents/quantitative.py:245`

**Validated**: ✅ TRUE

```python
parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
```

**Impact**: Incorrectly splits "Dr. Smith" or "e.g. example"

**Recommendation**: Consider nltk.sent_tokenize or spacy (lighter than full spacy model)
**Priority**: P3 (minor impact on evidence extraction)

---

### P3: Global State in server.py

**Location**: `server.py:31-36`

**Validated**: ✅ TRUE - 6 module-level globals

```python
_ollama_client: OllamaClient | None = None
_transcript_service: TranscriptService | None = None
# ... 4 more
```

**Recommendation**: Migrate to `app.state` (FastAPI standard)
**Priority**: P3 (works fine, but not idiomatic)

---

### P4: Missing py.typed Marker

**Location**: `src/ai_psychiatrist/`

**Validated**: ✅ TRUE - File does not exist on main

**Impact**: Package consumers can't verify types with mypy

**Recommendation**: Add empty `src/ai_psychiatrist/py.typed`
**Priority**: P4 (easy fix)

---

### P4: Magic Number 999_999

**Location**: `server.py:437`

**Validated**: ✅ TRUE

```python
participant_id=999_999,  # Ad-hoc transcript
```

**Recommendation**: Define `AD_HOC_PARTICIPANT_ID = 999_999` constant
**Priority**: P4 (cosmetic)

---

### P3: Hardcoded DOMAIN_KEYWORDS

**Location**: `src/ai_psychiatrist/agents/prompts/quantitative.py`

**Validated**: ✅ TRUE - Large dict hardcoded in source

**Recommendation**: Consider externalizing to YAML/JSON config
**Priority**: P3 (works fine, but harder to update without code change)

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
| Add py.typed marker | P4 | 1 min | TODO |
| Define AD_HOC_PARTICIPANT_ID constant | P4 | 1 min | TODO |
| Migrate server.py to app.state | P3 | 30 min | TODO |
| Evaluate JSON mode for Ollama | P2 | Research | TODO |
| Consider SafeTensors for embeddings | P2 | 2 hrs | Future |

---

## Original Source

The original Jules audit is preserved in the archived branch for reference.
Code changes from that branch were **NOT merged** due to:
- Broken test imports
- Missing spacy model installation
- Incomplete implementation

Only this validated documentation was extracted.
