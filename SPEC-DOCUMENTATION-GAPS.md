# Spec Documentation Gaps Analysis

**Date**: 2025-12-31
**Purpose**: Identify gaps between archived specs and canonical docs to enable archive deletion.

---

## Executive Summary

Audited 11 archived specs (25, 31-40) against canonical documentation. **None can be safely deleted yet** — all require content migration before archive removal.

| Spec | Status | Primary Gap | Can Delete? |
|------|--------|-------------|-------------|
| **25** | Partial | No metric formulas in canonical docs | **NO** |
| **31** | Partial | No format specification guide | **NO** |
| **32** | Partial | No troubleshooting/debugging guide | **NO** |
| **33** | Partial | No algorithm specification | **NO** |
| **34** | Partial | No generation workflow guide | **NO** |
| **35** | Partial | No artifact schema documentation | **NO** |
| **36** | Partial | No CRAG concepts in embeddings-explained.md | **NO** |
| **37** | Partial | No backend-specific behavior docs | **NO** |
| **38** | Partial | No fail-fast philosophy doc | **YES** (delete SUPERSEDED) |
| **39** | Partial | No error handling guide | **NO** |
| **40** | Partial | No embedding generation guide | **NO** |

**Immediate action**: Delete `38-embedding-graceful-degradation-SUPERSEDED.md` (explicitly rejected).

---

## Detailed Gap Analysis

### Spec 25: AURC/AUGRC Implementation

**Archived Location**: `docs/archive/specs/25-aurc-augrc-implementation.md`

**What's in canonical docs**:
- `docs/reference/statistical-methodology-aurc-augrc.md` — motivation, interpretation, running evaluation
- `docs/concepts/coverage-explained.md` — clinical context

**MISSING from canonical docs**:
1. Exact metric formulas (risk-coverage, AURC, AUGRC, truncated areas)
2. Bootstrap methodology (participant-level clustering)
3. Output schema specification
4. Test vectors and edge cases
5. Implementation phases
6. No mention in `FEATURES.md`

**Action Required**:
- [ ] Create `docs/reference/metrics-and-evaluation.md` with formulas + schema
- [ ] Add Spec 25 to `FEATURES.md`

---

### Spec 31: Paper-Parity Reference Examples Format

**Archived Location**: `docs/archive/specs/31-paper-parity-reference-examples-format.md`

**What's in canonical docs**:
- `docs/specs/index.md` — one-line reference
- `docs/results/run-history.md` — metric impact only
- `docs/concepts/embeddings-explained.md` — visual example (but not exact format)

**MISSING from canonical docs**:
1. Exact format specification (`<Reference Examples>` symmetric tags)
2. Edge case handling (None scores, empty items)
3. Why per-item headers were removed
4. Spec 31 → Spec 33 tag change explanation

**Action Required**:
- [ ] Create `docs/concepts/few-shot-prompt-format.md`
- [ ] Update `docs/guides/paper-parity-guide.md` with format section

---

### Spec 32: Few-Shot Retrieval Diagnostics

**Archived Location**: `docs/archive/specs/32-few-shot-retrieval-diagnostics.md`

**What's in canonical docs**:
- `FEATURES.md` — brief description + config
- `docs/reference/configuration.md` — config setting listed

**MISSING from canonical docs**:
1. Troubleshooting guide for using audit logs
2. Log interpretation examples
3. Integration with Specs 33-36 diagnostic flow
4. No mention in preflight checklist

**Action Required**:
- [ ] Create `docs/guides/debugging-retrieval-quality.md`
- [ ] Add diagnostic phase to `docs/guides/preflight-checklist-few-shot.md`

---

### Spec 33: Retrieval Quality Guardrails

**Archived Location**: `docs/archive/specs/33-retrieval-quality-guardrails.md`

**What's in canonical docs**:
- `FEATURES.md` — problem statement + config
- `docs/reference/configuration.md` — config settings

**MISSING from canonical docs**:
1. Exact post-processing algorithm (pseudocode)
2. Budget accounting semantics (`cost = len(match.chunk.text)`)
3. Top-k interaction order
4. Unit test code
5. XML closing tag fix details

**Action Required**:
- [ ] Add algorithm section to `docs/concepts/embeddings-explained.md`
- [ ] Expand `docs/reference/configuration.md` with algorithm details

---

### Spec 34: Item-Tagged Reference Embeddings

**Archived Location**: `docs/archive/specs/34-item-tagged-reference-embeddings.md`

**What's in canonical docs**:
- `FEATURES.md` — problem + solution summary
- `docs/concepts/embeddings-explained.md` — conceptual explanation
- `docs/reference/configuration.md` — config settings
- `docs/data/artifact-namespace-registry.md` — `.tags.json` mentioned

**MISSING from canonical docs**:
1. Exact CLI syntax (`--write-item-tags --tagger keyword`)
2. Sidecar format constraints (8 PHQ8_* strings, length parity)
3. Keyword tagger behavior
4. Filter logic details (fallback when tags missing)
5. Test requirements

**Action Required**:
- [ ] Add generation workflow to `docs/concepts/embeddings-explained.md`
- [ ] Create `docs/guides/item-tagging-setup.md`

---

### Spec 35: Offline Chunk-Level PHQ-8 Scoring

**Archived Location**: `docs/archive/specs/35-offline-chunk-level-phq8-scoring.md`

**What's in canonical docs**:
- `FEATURES.md` — brief description
- `docs/reference/configuration.md` — config settings
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` — motivation
- `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` — scorer selection

**MISSING from canonical docs**:
1. Artifact schema (JSON structure)
2. Scoring prompt template
3. Scorer model selection guide (consolidated)
4. Implementation checklist
5. Verification strategy

**Action Required**:
- [ ] Create `docs/reference/chunk-scoring.md` with schema + prompt + scorer guide
- [ ] Consolidate `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` into canonical docs

---

### Spec 36: CRAG Reference Validation

**Archived Location**: `docs/archive/specs/36-crag-reference-validation.md`

**What's in canonical docs**:
- `FEATURES.md` — brief description + config
- `docs/reference/configuration.md` — config settings

**MISSING from canonical docs**:
1. No mention in `docs/concepts/embeddings-explained.md`
2. No usage guide (when/how to enable CRAG)
3. Protocol/interface documentation
4. Trade-offs vs Specs 33/34

**Action Required**:
- [ ] Add "Part 9: Reference Validation" to `docs/concepts/embeddings-explained.md`
- [ ] Create `docs/guides/spec-36-crag-validation-guide.md`

---

### Spec 37: Batch Query Embedding

**Archived Location**: `docs/archive/specs/37-batch-query-embedding.md`

**What's in canonical docs**:
- `FEATURES.md` — status + config
- `docs/reference/configuration.md` — config settings
- `.env.example` — documented

**MISSING from canonical docs**:
1. Root cause analysis (why 120s timeout occurred)
2. Performance impact (8x reduction) in config docs
3. Backend-specific behavior (HuggingFace batching vs Ollama sequential)
4. Edge cases and fallback behavior

**Action Required**:
- [ ] Add batching section to `docs/concepts/embeddings-explained.md`
- [ ] Expand `docs/reference/configuration.md` with performance notes

---

### Spec 38: Conditional Feature Loading

**Archived Location**: `docs/archive/specs/38-conditional-feature-loading.md`

**What's in canonical docs**:
- `FEATURES.md` — brief summary

**MISSING from canonical docs**:
1. Fail-fast philosophy ("Skip If Disabled, Crash If Broken")
2. Why graceful degradation is wrong for research
3. Error handling specifics
4. Reference validation crash semantics

**Action Required**:
- [ ] Expand `FEATURES.md` Spec 38 section with rationale
- [ ] Add "Error Handling" section to `docs/reference/configuration.md`
- [ ] **DELETE** `38-embedding-graceful-degradation-SUPERSEDED.md` immediately

---

### Spec 39: Preserve Exception Types

**Archived Location**: `docs/archive/specs/39-preserve-exception-types.md`

**What's in canonical docs**:
- `FEATURES.md` — brief summary
- `docs/archive/bugs/bug-039-exception-handlers-mask-error-types.md` — companion bug

**MISSING from canonical docs**:
1. Error handling design philosophy
2. Exception hierarchy documentation
3. Fail-fast principle explanation
4. Targeted exception handling patterns

**Action Required**:
- [ ] Create `docs/concepts/error-handling.md`
- [ ] Create `docs/reference/exceptions.md`

---

### Spec 40: Fail-Fast Embedding Generation

**Archived Location**: `docs/archive/specs/40-fail-fast-embedding-generation.md`

**What's in canonical docs**:
- `FEATURES.md` — good coverage (exit codes, usage)
- `docs/archive/bugs/bug-042-generate-embeddings-silent-skips.md` — companion bug

**MISSING from canonical docs**:
1. `.partial.json` manifest format
2. When to use partial mode (debugging only)
3. Step-by-step embedding generation guide
4. Troubleshooting section

**Action Required**:
- [ ] Create `docs/guides/embedding-generation.md`
- [ ] Update `docs/guides/paper-parity-guide.md` with failure handling

---

## New Canonical Docs Required

Based on the gap analysis, these new docs need to be created:

| New Doc | Content Source | Priority |
|---------|---------------|----------|
| `docs/reference/metrics-and-evaluation.md` | Spec 25 | High |
| `docs/reference/chunk-scoring.md` | Spec 35 + PROBLEM-SPEC35 | High |
| `docs/concepts/few-shot-prompt-format.md` | Spec 31 | Medium |
| `docs/concepts/error-handling.md` | Specs 38, 39 | Medium |
| `docs/guides/debugging-retrieval-quality.md` | Spec 32 | Medium |
| `docs/guides/embedding-generation.md` | Spec 40 | Medium |
| `docs/guides/item-tagging-setup.md` | Spec 34 | Low |
| `docs/guides/spec-36-crag-validation-guide.md` | Spec 36 | Low |
| `docs/reference/exceptions.md` | Spec 39 | Low |

---

## Existing Docs Requiring Updates

| Doc | Updates Needed | Source Specs |
|-----|---------------|--------------|
| `docs/concepts/embeddings-explained.md` | Add Parts 9-10 (CRAG, batching, guardrails algorithm) | 33, 36, 37 |
| `docs/reference/configuration.md` | Add error handling section, expand descriptions | 38, 39, 37 |
| `docs/guides/paper-parity-guide.md` | Add format section, failure handling | 31, 40 |
| `docs/guides/preflight-checklist-few-shot.md` | Add diagnostics phase | 32 |
| `FEATURES.md` | Add Spec 25, expand Specs 38-39 sections | 25, 38, 39 |
| `CLAUDE.md` | Add evaluation workflow reference | 25, 35 |

---

## Immediate Actions

1. **Delete now**: `docs/archive/specs/38-embedding-graceful-degradation-SUPERSEDED.md`
2. **Prioritize**: Create `docs/reference/chunk-scoring.md` (Spec 35 is currently running)
3. **Consolidate**: Merge `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` into new chunk-scoring doc

---

## Consolidation Workflow

For each spec, follow this process:

1. **Read archived spec** — extract essential content
2. **Create/update canonical doc** — migrate content appropriately
3. **Add cross-reference** — link archived spec for historical context
4. **Verify coverage** — ensure all key info is in canonical docs
5. **Delete archived spec** — only after steps 1-4 complete

**Timeline**: Complete consolidation before next major release. Archived specs can remain during consolidation — they're not blocking development.

---

## Related Documents

- `POST-ABLATION-DEFAULTS.md` — Post-ablation configuration changes
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` — Few-shot methodology issues
- `PROBLEM-SPEC35-SCORER-MODEL-GAP.md` — Scorer model selection

---

*This document tracks documentation gaps. Update as consolidation progresses.*
