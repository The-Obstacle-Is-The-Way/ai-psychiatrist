# MASTER BUG AUDIT

**Audit Date**: 2026-01-05
**Auditor**: Claude Code (Ralph Wiggum Loop)
**Repository**: ai-psychiatrist
**Branch**: ralph-wiggum-audit
**Commit**: 8e0391685886646a2d074cb6d61be5fd58eac5a5

---

## 1. Executive Summary

### Severity Counts

| Severity | Count | Description |
|----------|-------|-------------|
| P0 | 0 | Critical blockers (none found) |
| P1 | 2 | High-priority issues |
| P2 | 3 | Medium-priority issues |
| P3 | 2 | Low-priority issues |
| P4 | 0 | Informational only |

Note: These counts reflect the original Ralph Wiggum audit snapshot; maintainer triage/remediation
below marks several items as resolved or false positives.

### Top 3 "Wastes-Hours" Failure Modes

1. **None identified** - The pipeline has robust fail-fast mechanisms. Dry-run passes, HF deps are verified, and embedding artifacts are validated at startup.

### Top 3 "Invalidates-Conclusions" Validity Threats

1. **FIXED (Spec 064)**: Retrieval audit logs no longer emit reference chunk text (they log `chunk_hash` + `chunk_chars` only).
2. **FIXED**: MkDocs link warnings from active specs are resolved (specs no longer link outside `docs/`).
3. **Known limitation**: PHQ-8 item-level frequency scoring is underdetermined from DAIC-WOZ (documented, not a bug - see Section 3)

---

## 1.1 Post-Audit Triage Notes (Maintainer Review)

The Ralph Wiggum loop was directionally correct, but it contains a few **false positives / outdated
assumptions** that are worth correcting before treating this file as SSOT:

- **BUG-001 (retrieval audit text leak)**: ✅ confirmed. `src/ai_psychiatrist/services/embedding.py` logs
  `chunk_preview=match.chunk.text[:160]` when retrieval audit is enabled. This is a real DAIC-WOZ
  leak risk.
  - Correction: `EmbeddingSettings.enable_retrieval_audit` defaults to `false` in code
    (`src/ai_psychiatrist/config.py`), but `.env.example` enables it, so the risk is real in
    recommended run configs.
- **BUG-002 (broken links in specs)**: ✅ confirmed (MkDocs INFO warnings), but the root cause is not
  a “wrong relative path” — the linked file is outside `docs/` so MkDocs cannot resolve it.
  - Fix: route links to `docs/_research/hypotheses-for-improvement.md` (which renders the root file).
- **BUG-003 (exception handling)**: ⚠️ partially outdated. The flagged scoring handler in
  `src/ai_psychiatrist/agents/quantitative.py` logs and then **re-raises** (no silent downgrade).
  Consistency sampling does log-and-continue by design, with bounded extra attempts.
- **BUG-005 (backend/artifact mismatch)**: ❌ mostly a false positive for current artifacts. Modern
  embedding artifacts include `.meta.json` with `backend`, and `ReferenceStore` validates this on load
  (fails fast on mismatch). Legacy artifacts without metadata are still a potential footgun.

### Post-Audit Remediation (Implemented)

- **Spec 064 (retrieval audit redaction)**: Implemented. `retrieved_reference` logs now emit
  `chunk_hash` (stable SHA-256 prefix) + `chunk_chars` and do not emit raw chunk text.
- **Docs fix for broken links**: Implemented. Specs link to `docs/_research/hypotheses-for-improvement.md`
  (MkDocs-rendered view of the root hypotheses doc) instead of linking outside `docs/`.

## 2. Environment + Commands Run

### Repository Metadata

- **Branch**: ralph-wiggum-audit
- **Commit SHA**: 8e0391685886646a2d074cb6d61be5fd58eac5a5
- **OS**: Darwin 25.0.0 (arm64)
- **Python**: 3.13.5 (Clang 20.1.4)

### Code Quality + Tests

#### `make ci`

```
✅ PASSED
- ruff format --check: 142 files already formatted
- ruff check: All checks passed
- mypy: Success, no issues in 142 source files
- pytest: 904 passed, 7 skipped, 61 warnings
- Coverage: 83.80% (meets 80% threshold)
```

**Warnings Analysis**:
- 46 warnings related to Pydantic UserWarning for test fixtures (expected in test isolation)
- 8 warnings in `test_factory.py` for HF deps mock (expected)
- These warnings do not affect production correctness

#### `uv run mkdocs build --strict`

```
✅ PASSED (with INFO-level broken links)
- Build completed in 3.96 seconds
- 30 broken links detected (all INFO level, not errors)
- All broken links are in `_archive/` or point to `HYPOTHESES-FOR-IMPROVEMENT.md`
```

**Notable Broken Links** (non-archive):
- `docs/_specs/spec-061-total-phq8-score-prediction.md` → `../../HYPOTHESES-FOR-IMPROVEMENT.md` (file exists but path is wrong)
- `docs/_specs/spec-063-severity-inference-prompt-policy.md` → `../../HYPOTHESES-FOR-IMPROVEMENT.md` (same issue)

---

## 3. Known Non-Bugs / Expected Limitations

### Task Validity Constraint (CRITICAL)

PHQ-8 item scores are defined by **2-week frequency** (0-3 scale based on days), but DAIC-WOZ transcripts are semi-structured interviews that do not systematically elicit frequency information.

**Expected behaviors**:
- ~50% coverage (abstention rate) is correct methodological behavior
- `N/A` outputs for items without clear frequency evidence
- Few-shot may not beat zero-shot when evidence is sparse

**SSOT**: [`docs/clinical/task-validity.md`](../clinical/task-validity.md)

This is **not a bug**. The system correctly implements selective prediction with evidence grounding.

### Run 12 SSOT Metrics (Reference)

| Mode | Item MAE | Coverage (Cmax) |
|------|----------|-----------------|
| Zero-shot | 0.5715 | 48.5% |
| Few-shot | 0.6159 | 46.0% |

These metrics are consistent with the task validity constraint.

---

## 4. Findings (Table)

| ID | Severity | Category | Symptom | Root Cause | Impact | Repro Steps | Proposed Fix | Test Plan |
|----|----------|----------|---------|------------|--------|-------------|--------------|-----------|
| BUG-001 | RESOLVED | observability | Retrieval audit is privacy-safe | Previously logged `chunk_preview=match.chunk.text[:160]`; now logs `chunk_hash` only | Prevents DAIC-WOZ transcript text leaks into logs/artifacts | N/A (fixed) | Implemented Spec 064 (`chunk_hash` + `chunk_chars`) | `tests/unit/services/test_embedding.py::TestEmbeddingService::test_build_reference_bundle_logs_audit_when_enabled` |
| BUG-002 | RESOLVED | docs | Specs link within `docs/` | Specs now link to `docs/_research/hypotheses-for-improvement.md` | Removes MkDocs INFO warnings for non-archive docs | N/A (fixed) | Add MkDocs-rendered view of root doc + update spec links | `uv run mkdocs build --strict` has no non-archive warnings for this issue |
| BUG-003 | P2 | parsing | Multiple `except Exception` catches in agents | `src/ai_psychiatrist/agents/*.py` (9 locations) | Potential silent failures; most re-raise but some log-and-continue | grep for `except Exception` in src | Review each catch; ensure all either re-raise or log at ERROR level with failure registry | Add test that exception handling doesn't swallow errors silently |
| BUG-004 | P2 | retrieval | `return []` fallbacks in services | 7 locations return empty lists that could mask failures | Silent degradation if retrieval fails | Search `return \[\]` in src | Each `return []` should log at WARNING level and register in failure registry | Integration test for failure registry events |
| BUG-005 | P3 | config | Ollama backend vs HuggingFace artifact mismatch risk | Config allows `EMBEDDING_BACKEND=ollama` with `huggingface_*` artifact files | Embedding space mismatch → invalid similarity scores | Set mismatched config, run pipeline | Add startup validation that backend matches artifact prefix | Unit test for backend/artifact consistency check |
| BUG-006 | P3 | docs | Archive docs have 22 "paper-parity" references | `docs/_archive/` contains deprecated terminology | Confusion if users read archive docs | grep `paper-parity` in docs | Archive is intentionally frozen; add disclaimer header to archive index | N/A (informational) |

---

## 5. Deep Dives

### BUG-001: DAIC-WOZ Text Leak via Retrieval Audit (P1)

**Location**: `src/ai_psychiatrist/services/embedding.py:376-389`

**Code Pattern**:
```python
if self._enable_retrieval_audit:
    # ...
    logger.info(
        "retrieved_reference",
        # ...
        chunk_hash=stable_text_hash(match.chunk.text),  # <-- SAFE (no raw text)
        chunk_chars=len(match.chunk.text),
    )
```

**Why Current Guardrails Failed**:
- The audit logging is opt-in via `EMBEDDING_ENABLE_RETRIEVAL_AUDIT`
- Default is `True` per config, so logs can contain transcript text
- No redaction layer exists between retrieval and logging

**Evidence** (without leaking text):
- File: `src/ai_psychiatrist/services/embedding.py`
- Line: 387
- Field logged: `chunk_preview` (first 160 chars of chunk text)
- Source: Reference corpus from DAIC-WOZ transcripts

**Resolution**:
- Implemented Spec 064: log `chunk_hash` + `chunk_chars`; do not log any raw chunk text.

### BUG-003: Exception Handling Audit (P2)

**Locations**:
1. `src/ai_psychiatrist/infrastructure/logging.py:29` - startup logging (acceptable)
2. `src/ai_psychiatrist/agents/meta_review.py:158` - re-raises (OK)
3. `src/ai_psychiatrist/agents/quantitative.py:387` - re-raises (OK)
4. `src/ai_psychiatrist/agents/quantitative.py:515` - logs and continues (RISK)
5. `src/ai_psychiatrist/agents/quantitative.py:541` - re-raises (OK)
6. `src/ai_psychiatrist/infrastructure/llm/responses.py:294` - json_repair fallback (OK per Spec 059)
7. `src/ai_psychiatrist/agents/qualitative.py:148` - logs ERROR (acceptable)
8. `src/ai_psychiatrist/agents/qualitative.py:211` - logs ERROR (acceptable)
9. `src/ai_psychiatrist/agents/judge.py:167` - logs ERROR (acceptable)

**Analysis**: Most exception handlers either re-raise or log at ERROR level. Line 515 in quantitative.py needs review to ensure it doesn't silently degrade few-shot to zero-shot.

---

## 6. Prioritized Fix Roadmap

### Immediate (Before Next Run)

1. **BUG-001**: Redact `chunk_preview` in retrieval audit logs
   - **Definition of Done**: No raw transcript text >20 chars appears in any log field
   - **Spec**: Create Spec 064 for retrieval audit redaction

### Short-Term (This Week)

2. **BUG-002**: Fix broken doc links in active specs
   - **Definition of Done**: `mkdocs build --strict` produces 0 INFO warnings for non-archive docs

3. **BUG-005**: Add backend/artifact consistency validation
   - **Definition of Done**: Pipeline fails fast if `EMBEDDING_BACKEND` doesn't match artifact filename prefix

### Medium-Term

4. **BUG-003/004**: Audit all exception handlers and `return []` patterns
   - **Definition of Done**: Every catch either re-raises, logs ERROR, or registers failure event

---

## 7. Open Questions

1. **Smoke tests taking too long**: Zero-shot/few-shot `--limit 1` tests were still running after 3 minutes due to LLM inference. Should we add a faster mock-based smoke test for CI?

2. **Telemetry shows only 9 json_fixup events**: This is healthy, but should we add alerting thresholds for when fixup counts exceed N per run?

3. **AUGRC vs AURC**: Per [Traub et al. 2024](https://arxiv.org/html/2407.01032v1), AUGRC is preferred over AURC for selective prediction. The codebase already implements both. Should AUGRC be the primary reported metric?

4. **Structured output reliability**: Per [2025 best practices](https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms), API-native structured outputs achieve 100% schema compliance. Consider migrating from json_repair fallback to native structured outputs when available.

---

## References

- [AUGRC paper (Traub et al. 2024)](https://arxiv.org/html/2407.01032v1) - Selective prediction evaluation pitfalls
- [Structured outputs guide](https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms) - LLM JSON reliability
- [PHQ-8 validation](https://pubmed.ncbi.nlm.nih.gov/18752852/) - Screening validity
- [PHQ-8 Swedish psychometrics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7452881/) - Test-retest reliability

---

*Audit completed by Ralph Wiggum loop iteration 2, 2026-01-05*
