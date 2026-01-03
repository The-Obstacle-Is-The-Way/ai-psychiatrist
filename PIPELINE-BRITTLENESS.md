# Pipeline Brittleness Analysis: Input → Output Chain

**Created**: 2026-01-03
**Purpose**: Complete mapping of all failure points from raw transcript to final PHQ-8 assessment
**Status**: Actionable analysis for eliminating silent failures

---

## Why This Document Exists

We've had recurring issues with JSON parsing failures, silent fallbacks, and mode contamination (ANALYSIS-026). Instead of patching symptoms, this document maps the **entire chain** to identify:

1. Where failures can occur
2. Which failures are silent (dangerous) vs. loud (safe)
3. What 2025 best practices can fix them
4. What we should implement

---

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
│  STAGE 2: EVIDENCE EXTRACTION (Few-Shot Only)                               │
│  File: src/ai_psychiatrist/agents/quantitative.py                           │
│  LLM Call: Evidence extraction → JSON dict of quotes per PHQ-8 domain       │
│  Parser: parse_llm_json() in infrastructure/llm/responses.py                │
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
| 1 | Null values in text fields | Becomes "None: None" in dialogue | High | ❌ Not fixed |
| 2 | Non-list value in evidence JSON | Coerced to empty array silently | High | ❌ Not fixed |
| 2 | LLM hallucinated evidence | Included in bundle with no validation | High | ❌ Not fixed |
| 3 | Embedding dimension mismatch | Silent truncation loses data | High | ⚠️ Configurable strict mode |
| 3 | NaN in embedding vectors | Propagates through cosine similarity | Medium | ❌ Not fixed |
| 4 | Text/embedding count mismatch | Silently skips if not strict mode | Medium | ⚠️ Configurable strict mode |
| 5 | Wrong enum key mapping | Item assessments misaligned | High | ❌ Not validated |
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

From web research ([Ollama Structured Outputs](https://ollama.com/blog/structured-outputs), [Constrained Decoding](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output)):

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

**Solution**: Validate that each extracted quote exists (fuzzy match) in the source transcript.

```python
# In quantitative.py after evidence extraction
def _validate_evidence_quotes(evidence: dict, transcript: str) -> dict:
    """Remove hallucinated quotes not found in transcript."""
    validated = {}
    for key, quotes in evidence.items():
        validated[key] = [q for q in quotes if _fuzzy_match(q, transcript)]
    return validated
```

**Status**: Not implemented. Would catch hallucinated evidence.

#### 2. Add Schema Validation for Evidence JSON

**Problem**: Non-list values silently become empty arrays.

**Solution**: Strict validation immediately after parse.

```python
# Add to _extract_evidence()
for key in PHQ8_DOMAIN_KEYS:
    value = obj.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {key}, got {type(value)}")
```

**Status**: Not implemented. Would catch type coercion issues.

#### 3. Add NaN Detection in Embeddings

**Problem**: NaN vectors propagate through cosine similarity.

**Solution**: Validate embeddings after generation.

```python
if np.isnan(embedding).any():
    raise ValueError(f"NaN detected in embedding for {text[:50]}")
```

**Status**: Not implemented. Would catch embedding issues early.

### MEDIUM PRIORITY (Observability)

#### 4. Add Failure Pattern Logging

**Problem**: We don't know how often each failure mode occurs.

**Solution**: Structured logging for all failure types.

```python
logger.warning(
    "evidence_extraction_failure",
    participant_id=pid,
    failure_type="json_parse",
    raw_response_preview=raw[:200],
    error=str(e),
)
```

**Status**: Partially implemented. Need consistent taxonomy.

#### 5. Add Embedding Dimension Strict Mode Default

**Problem**: Dimension mismatch silently truncates.

**Solution**: Make strict mode the default.

```python
# In config.py
embedding_dimension_strict: bool = True  # Was False
```

**Status**: Configurable but not default. Should flip.

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

- [Ollama Structured Outputs](https://ollama.com/blog/structured-outputs) - Grammar-level JSON constraints
- [Constrained Decoding Guide](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output) - GBNF/FSM approaches
- [Taming LLMs: Structured Output](https://dev.to/shrsv/taming-llms-how-to-get-structured-output-every-time-even-for-big-responses-445c) - Best practices
- [Ollama + Qwen3 Structured Output](https://medium.com/@rosgluk/constraining-llms-with-structured-output-ollama-qwen3-python-or-go-2f56ff41d720) - Python implementation patterns
- [Gemma 3 Function Calling](https://medium.com/google-cloud/function-calling-with-gemma3-using-ollama-120194577fa6) - Gemma-specific guidance
