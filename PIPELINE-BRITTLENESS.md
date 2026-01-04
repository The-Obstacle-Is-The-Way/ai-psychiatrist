# Pipeline Brittleness Analysis: Input → Output Chain

**Created**: 2026-01-03
**Purpose**: Complete mapping of all failure points from raw transcript to final PHQ-8 assessment
**Status**: ✅ RESOLVED - All actionable items implemented in PR #92 (2026-01-03)

> **Note**: This analysis document led to Specs 053-057, which are now implemented and archived.
> Canonical documentation for these features is in `docs/` (not this file).
> See: `docs/pipeline-internals/features.md`, `docs/developer/error-handling.md`, `docs/rag/debugging.md`

---

## Why This Document Exists

We've had recurring issues with JSON parsing failures, silent fallbacks, and mode contamination (ANALYSIS-026). Instead of patching symptoms, this document maps the **entire chain** to identify:

1. Where failures can occur
2. Which failures are silent (dangerous) vs. loud (safe)
3. What 2025 best practices can fix them
4. What we should implement

---

## Privacy / Licensing Boundary (Non-Negotiable)

DAIC-WOZ transcripts are licensed and must not leak into logs or artifacts. Any robustness or observability work proposed here MUST:
- Never write raw transcript text to disk outside `data/` (which is already local-only).
- Never include transcript text or evidence quotes in logs or “failure summaries”.
- Prefer counts, lengths, stable hashes, model ids, and error codes.

## The Complete Pipeline Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: RAW TRANSCRIPT LOADING                                            │
│  File: src/ai_psychiatrist/services/transcript.py                           │
│  Input: data/transcripts_participant_only/{pid}_P/{pid}_TRANSCRIPT.csv      │
│  Output: Transcript entity (participant_id, text)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: EVIDENCE EXTRACTION (All Modes)                                  │
│  File: src/ai_psychiatrist/agents/quantitative.py                           │
│  LLM Call: Evidence extraction → JSON dict of quotes per PHQ-8 domain       │
│  Parser: parse_llm_json() in infrastructure/llm/responses.py                │
│  Notes: Counts feed N/A reasons; few-shot uses evidence for retrieval       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: EMBEDDING GENERATION                                              │
│  File: src/ai_psychiatrist/services/embedding.py                            │
│  Input: Evidence dict + pre-loaded reference embeddings                     │
│  Output: ReferenceBundle (formatted refs + similarity stats)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: REFERENCE LOADING & NORMALIZATION                                 │
│  File: src/ai_psychiatrist/services/reference_store.py                      │
│  Input: .npz embeddings + .json text chunks + .tags.json + .chunk_scores    │
│  Output: Vectorized reference matrix with L2-normalized embeddings          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: SCORING (LLM JSON Call)                                           │
│  File: src/ai_psychiatrist/agents/quantitative.py                           │
│  LLM Call: Transcript + refs → JSON with 8 PHQ-8 item assessments           │
│  Parser: Pydantic AI extractors → parse_llm_json() → QuantitativeOutput     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: FINAL AGGREGATION                                                 │
│  File: src/ai_psychiatrist/agents/quantitative.py                           │
│  Input: Parsed item assessments + reference bundle stats                    │
│  Output: PHQ8Assessment entity with scores, confidence, severity            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: EVALUATION SERIALIZATION                                          │
│  File: scripts/reproduce_results.py                                         │
│  Input: PHQ8Assessment + ground truth from CSV                              │
│  Output: JSON results file → data/outputs/run_*.json                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Brittleness Classification

### TIER 1: HARD FAILURES (Loud - We See Them)

These block entire participants but are visible:

| Stage | Failure Mode | Current Handling | Status |
|-------|--------------|------------------|--------|
| 1 | Transcript file not found | Raises `TranscriptError` | ✅ Good |
| 1 | Invalid CSV format | Catches pandas errors → `TranscriptError` | ✅ Good |
| 2 | LLM timeout on evidence extraction | Exception propagates | ✅ Good (fixed in ANALYSIS-026) |
| 2 | JSON parse failure after all fixups | Raises `json.JSONDecodeError` | ✅ Good (fixed in ANALYSIS-026) |
| 3 | Embedding artifact missing | Raises `FileNotFoundError` | ✅ Good |
| 4 | Tag file structure invalid | Raises validation error | ✅ Good |
| 5 | Pydantic validation failure (missing items) | Retries then fails | ✅ Good |
| 7 | Ground truth CSV mismatch | Raises `ValueError` (BUG-025 fix) | ✅ Good |

### TIER 2: SILENT CORRUPTIONS (Dangerous - We Don't See Them)

These produce wrong results without any indication:

| Stage | Failure Mode | What Happens | Severity | Fix Status |
|-------|--------------|--------------|----------|------------|
| 2 | Non-list value in evidence JSON | Wrong types silently become `[]` | High | ✅ Fixed (Spec 054, PR #92) |
| 2 | LLM hallucinated evidence quotes | Ungrounded quotes contaminate retrieval | High | ✅ Fixed (Spec 053, PR #92) |
| 3 | Insufficient embedding dimension | Some reference chunks skipped; may reduce retrieval quality | High | ✅ Fixed (Spec 057, PR #92) |
| 3 | NaN/Inf/zero embeddings | Propagates through cosine similarity | Medium | ✅ Fixed (Spec 055, PR #92) |
| 4 | Text/embedding count mismatch | Skips participant unless strict alignment required | Medium | ⚠️ Logged, fatal when alignment required |
| 6 | Reference chunk duplication | Same chunk appears multiple times | Low | ❌ Not deduplicated |

### TIER 3: GRACEFUL DEGRADATIONS (Acceptable)

These reduce quality but results remain valid:

| Stage | Failure Mode | What Happens | Status |
|-------|--------------|--------------|--------|
| 2 | Insufficient evidence for item | Returns N/A (intentional) | ✅ By design |
| 3 | Reference retrieval fails | Falls back to zero-shot for that item | ✅ Acceptable |
| 5 | Token logprobs unavailable | Confidence signals = None | ✅ Acceptable |
| 4 | Chunk scores file missing | Uses participant-level scores | ✅ Fallback documented |

---

## Root Causes of JSON Failures

Based on code analysis and 2025 research:

### Where JSON Gets Generated

| Stage | LLM Call | JSON Mode | Failure Risk |
|-------|----------|-----------|--------------|
| Evidence Extraction | `format="json"` via Ollama | Grammar-constrained | Low (Ollama enforces) |
| Scoring | Pydantic AI with `<answer>` tags | Prompt-only constraint | Medium (model may deviate) |

### Why Models Still Fail JSON

Even with `format="json"`, models can produce:

1. **Schema violations**: Valid JSON, wrong structure (missing keys, wrong types)
2. **Semantic errors**: Valid structure, wrong content (hallucinated evidence)
3. **Token-level issues**: Rare with grammar constraints, but possible edge cases

### What 2025 Best Practices Say

From web research ([Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs), [Constrained Decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)):

| Practice | Status in Our Codebase | Notes |
|----------|------------------------|-------|
| Use `format="json"` for Ollama | ✅ Evidence extraction uses it | Grammar-level enforcement |
| Temperature = 0 for schema adherence | ✅ We use 0.0 | Deterministic output |
| Start with simple schemas | ✅ Our schemas are reasonable | 8 items × 4 fields |
| Validate with Pydantic after parse | ✅ Used in extractors | Catches structure issues |
| Retry on validation failure | ✅ PydanticAI retries 3x | Good error recovery |
| Monitor success rates | ❌ Not implemented | Should log failure patterns |

---

## Preprocessing Analysis

### Current State

Preprocessing script: `scripts/preprocess_daic_woz_transcripts.py`

**Robust Features:**
- Deterministic variant selection (participant_only, both_speakers_clean, participant_qa)
- Known interruption windows handled (participants 373, 444)
- Sync markers removed
- Preamble removal (pre-interview content)
- Speaker normalization (case-insensitive "Ellie", "Participant")
- Missing Ellie sessions documented (451, 458, 480)
- Manifest written with full stats

**Potential Brittleness:**
- No Unicode normalization beyond pandas defaults
- No special character sanitization
- No validation of value field content (could contain control characters)
- No length limits on individual utterances

### Sample Participant-Only Transcript

```
start_time	stop_time	speaker	value
62.328	63.178	Participant	good
68.978	70.288	Participant	atlanta georgia
...
```

This format is clean. The brittleness is **not in preprocessing** - it's in:
1. The LLM's ability to generate valid JSON from arbitrary text
2. Our ability to detect when it fails silently

---

## Actionable Recommendations

### HIGH PRIORITY (Silent Corruption Risks)

#### 1. Add Hallucination Detection for Evidence

**Problem**: LLM can return evidence quotes not present in transcript.

**Solution**: Validate that each extracted quote is grounded in the source transcript after conservative normalization (substring match). (Spec 053)

```python
# Implemented in:
# - src/ai_psychiatrist/services/evidence_validation.py: validate_evidence_grounding()
# - src/ai_psychiatrist/agents/quantitative.py: QuantitativeAssessmentAgent._extract_evidence()
```

**Status**: ✅ Implemented (PR #92).

#### 2. Add Schema Validation for Evidence JSON

**Problem**: Non-list values silently become empty arrays.

**Solution**: Strict validation immediately after parse.

```python
# Implemented in:
# - src/ai_psychiatrist/services/evidence_validation.py: validate_evidence_schema()
# - src/ai_psychiatrist/agents/quantitative.py: QuantitativeAssessmentAgent._extract_evidence()
```

**Status**: ✅ Implemented (PR #92).

#### 3. Add NaN Detection in Embeddings

**Problem**: NaN vectors propagate through cosine similarity.

**Solution**: Validate embeddings after generation.

```python
# Implemented in:
# - src/ai_psychiatrist/infrastructure/validation.py: validate_embedding(), validate_embedding_matrix()
# - scripts/generate_embeddings.py (generation-time)
# - src/ai_psychiatrist/services/reference_store.py + embedding.py (load/runtime)
```

**Status**: ✅ Implemented (PR #92).

### MEDIUM PRIORITY (Observability)

#### 4. Add Failure Pattern Logging

**Problem**: We don't know how often each failure mode occurs.

**Solution**: Structured failure registry + privacy-safe logging (Spec 056).

```python
# Implemented in:
# - src/ai_psychiatrist/infrastructure/observability.py: FailureRegistry
# - scripts/reproduce_results.py: per-participant recording + failures_{run_id}.json artifact
```

**Status**: ✅ Implemented (PR #92).

#### 5. Add Embedding Dimension Strict Mode Default

**Problem**: If any reference embedding vector has fewer dims than `EMBEDDING_DIMENSION`, reference chunks can be skipped, reducing the corpus.

**Solution**: Enforce dimension invariants (generation + load) and fail fast by default (Spec 057).

```python
# Proposed env/config escape hatch (default false):
# EMBEDDING_ALLOW_INSUFFICIENT_DIMENSION_EMBEDDINGS=false
#
# Also enforce at generation time (scripts/generate_embeddings.py):
# if len(embedding) != config.dimension: fail (or skip only in --allow-partial mode)
```

**Status**: ✅ Implemented (PR #92). Default is fail-fast; escape hatch is
`EMBEDDING_ALLOW_INSUFFICIENT_DIMENSION_EMBEDDINGS=true` for forensics only.

### LOW PRIORITY (Nice to Have)

#### 6. Deduplicate Reference Chunks

**Problem**: Same chunk can appear multiple times in bundle.

**Solution**: Dedupe by chunk text before formatting.

**Status**: Minor issue. Low impact on results.

#### 7. Add Prompt Size Validation

**Problem**: Very long transcripts + refs can exceed model context.

**Solution**: Warn when prompt approaches context limit.

**Status**: Not implemented. Rare failure mode.

---

## What We've Already Fixed (ANALYSIS-026)

The following are now **correctly handled**:

| Issue | Old Behavior | New Behavior |
|-------|--------------|--------------|
| JSON parse failure in evidence extraction | Silent return `{}` | Raises `json.JSONDecodeError` |
| Few-shot mode with empty evidence | Silently becomes zero-shot | Fails loudly |
| Non-canonical JSON parsing | Multiple parsers with different behaviors | Single `parse_llm_json()` SSOT |
| Ollama evidence extraction | Prompt-only JSON constraint | `format="json"` grammar constraint |
| Evidence JSON schema violations | Silent coercion to `[]` | Raises `EvidenceSchemaError` (Spec 054) |
| Evidence quote hallucinations | Ungrounded quotes contaminate retrieval | Grounding validation (Spec 053) |
| NaN/Inf/zero embeddings | Propagate into similarity | Validation fails loudly (Spec 055) |
| Dimension mismatch (partial) | Warn + skip | Fail fast by default; escape hatch available (Spec 057) |
| Failure pattern visibility | Ad-hoc logs | FailureRegistry + failures JSON artifact (Spec 056) |

---

## Summary: The Real Problem

**The pipeline is now LOUD about JSON failures.** The remaining risks are:

1. **Semantic corruption** (hallucinated evidence, wrong item mapping)
2. **Numeric corruption** (NaN propagation, dimension truncation)
3. **Observability gaps** (we don't log failure patterns systematically)

The preprocessing is solid. The JSON parsing is fixed. The remaining work is **validation and observability**, not fundamental architecture changes.

---

## Related Documentation

- `docs/_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md` - JSON parsing fix details
- `docs/pipeline-internals/evidence-extraction.md` - How evidence extraction works
- `docs/results/run-history.md` - Run integrity warnings
- `scripts/preprocess_daic_woz_transcripts.py` - Preprocessing implementation

---

## Sources

- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs) - JSON mode / constrained output
- [Constrained Decoding Guide](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output) - GBNF/FSM approaches
- [Taming LLMs: Structured Output](https://dev.to/shrsv/taming-llms-how-to-get-structured-output-every-time-even-for-big-responses-445c) - Best practices
- [Ollama + Qwen3 Structured Output](https://medium.com/@rosgluk/constraining-llms-with-structured-output-ollama-qwen3-python-or-go-2f56ff41d720) - Python implementation patterns
- [Gemma 3 Function Calling](https://medium.com/google-cloud/function-calling-with-gemma3-using-ollama-120194577fa6) - Gemma-specific guidance
