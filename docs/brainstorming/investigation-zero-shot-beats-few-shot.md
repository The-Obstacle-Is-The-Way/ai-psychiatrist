# Investigation: Why Zero-Shot Beats Few-Shot

**Date**: 2025-12-28
**Status**: HYPOTHESIS - Awaiting Ablation + Senior Review
**Severity**: High - Core Paper Claim Inversion (in our runs)

---

## UPDATE (2025-12-28): Paper-Parity Divergences Found

Investigation identified divergences between our implementation and the paper's notebook.

Current (non-archive) references:
- Few-shot prompt format: `docs/concepts/few-shot-prompt-format.md`
- Retrieval debugging workflow: `docs/guides/debugging-retrieval-quality.md`

### Verified Divergences

| Issue | Type | Evidence Source |
|-------|------|-----------------|
| **Score-Chunk Mismatch** | Paper methodology (correctly implemented) | Paper Section 2.4.2 + Notebook cell `49f51ff5` |
| **Format Mismatch** | OUR DIVERGENCE | Notebook uses single `<Reference Examples>` block |
| **Missing Domain Labels** | OUR DIVERGENCE | Notebook: `f"({evidence_key} Score: {score})"` |
| **Closing Tag** | OUR DIVERGENCE | Notebook uses `<Reference Examples>` not `</Reference Examples>` |

### Hypothesis (Not Proven)

The format divergences **may** cause few-shot underperformance. We hypothesize:
- Fragmented 8-block structure disrupts holistic reasoning
- Missing domain labels reduce score-context association

**Caveat**: Causality not proven. Need ablation (fix format, re-run) to verify.

### Paper's Metrics vs Ours

| Approach | Metric | Notes |
|----------|--------|-------|
| Paper | MAE at Cmax | Valid conditional metric (error on non-N/A) |
| Ours | AURC/AUGRC | Better for system comparison when coverages differ |

MAE is not "invalid" - it's incomplete when coverages differ significantly.

### Next Steps

1. **Fix format divergences** - Achieve paper parity
2. **Run ablation** - Does fixing format improve few-shot?
3. **Add retrieval diagnostics** - Empirically verify chunk alignment
4. **Senior review** after ablation results

---

## ✅ ADDED (Senior Review): Implementation Checklist (Paper-Parity Ablation)

This is the minimum, **implementation-ready** sequence to test whether the paper-parity formatting divergences matter.

### Step 1 — Apply Fix 1 (paper-parity reference formatting)

Source of truth: `docs/concepts/few-shot-prompt-format.md` (canonical, non-archive).

Required edits:
- `src/ai_psychiatrist/services/embedding.py` (`ReferenceBundle.format_for_prompt`)
- `tests/unit/services/test_embedding.py` (`TestReferenceBundle` expectations)

Paper notebook string format (cell `49f51ff5`) must be matched exactly:
- Non-empty: `"<Reference Examples>\\n\\n" + "\\n\\n".join(entries) + "\\n\\n<Reference Examples>"`
- Empty: `"<Reference Examples>\\nNo valid evidence found\\n<Reference Examples>"`
- Skip items with no evidence/matches (no per-item empty blocks).

### Step 2 — Run unit tests (format regression guard)

```bash
uv run pytest tests/unit/services/test_embedding.py -q
```

### Step 3 — Re-run reproduction (same split, same model)

Run full paper-test reproduction (writes a new `data/outputs/*.json`):

```bash
uv run python scripts/reproduce_results.py --split paper-test
```

### Step 4 — Compute paired selective-prediction deltas (rigorous comparison)

Given the output file from Step 3 (call it `data/outputs/<RUN>.json`), compute paired deltas on the overlapping successful participants:

```bash
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<RUN>.json --mode zero_shot \
  --input data/outputs/<RUN>.json --mode few_shot \
  --intersection-only
```

Record:
- `daurc_full` CI (few − zero): does it cross 0?
- `dcmax` CI: did coverage change?

### Step 5 — (Optional) Re-run paper-style MAE@Cmax for comparability

The reproduction artifact also includes paper-style item MAE excluding N/A (conditional MAE). Extract these from the output JSON under each experiment:
- `results.item_mae_by_subject` (mean per-subject MAE on available items)
- `results.item_mae_weighted` (mean error over all predicted items; count-weighted)
- `results.prediction_coverage`

This is useful for paper comparability, but not sufficient for “system-level” comparison when coverages differ.

## Data Structure Analysis

The current data structure only supports participant-level scoring.

| File | Contents | Chunk-Level Scores? |
|------|----------|---------------------|
| `data/embeddings/paper_reference_embeddings.json` | Plain text chunks only | **NO** |
| `data/train_split_Depression_AVEC2017.csv` | One row per participant | **NO** |
| Score lookup in `reference_store.py` | `get_score(participant_id, item)` | **NO** |

### Concrete Evidence: Participant 321

**Ground truth**: PHQ8_Sleep = 3 (severe, nearly every day)

**Their 115+ chunks include**:
- ~7% sleep-related: "I haven't had a good night's sleep in a year... I sleep in 1-3 hour intervals"
- ~93% other topics: work, family, PTSD history, hobbies, grandchildren

**Problem**: ALL 115 chunks get attached `(PHQ8_Sleep Score: 3)` when retrieved.

A chunk like:
> "I'm proud of my children and grandchildren"

Gets attached: `(PHQ8_Sleep Score: 3)` ← **Semantically meaningless**

### Data Flow

```text
Chunks (JSON)     →  Just text, no scores
                      ↓
Score lookup      →  get_score(participant_id, item)
                      ↓
Ground truth CSV  →  One row per participant
```

**To implement chunk-level scoring would require**:
1. LLM annotation of each chunk during embedding generation
2. New data structure with chunk IDs and per-chunk scores

This is the paper's methodology - participant-level scores attached to chunks.

---

## 2025 STATE-OF-THE-ART SOLUTIONS

This is a **known problem in RAG literature** with established solutions.

### The Core Problem

The paper's methodology:
1. Retrieves chunks by **topic similarity** (embedding cosine)
2. Attaches **participant-level scores** (not chunk-level)
3. Chunk content may not match attached score severity
4. Creates noisy/contradictory few-shot calibration examples

### Why Retrieval Isn't Smart Enough

**Embedding similarity = Topic matching, NOT severity matching**

A chunk saying "I sleep fine" and "I can't sleep at all" are BOTH about sleep. Both might be retrieved. But they describe vastly different severities.

### Solution Options (2025 Best Practices)

| Solution | Stage | Cost | Our Use Case |
|----------|-------|------|--------------|
| **CRAG** | Post-retrieval | Runtime (every query) | Good - validates chunks |
| **Contextual Retrieval** | Pre-embedding | Index time | Partial - better embeddings |
| **Pre-compute Chunk Scores** | Index time | One-time | **Best** - fixes at source |
| **Hybrid** | Index + Runtime | Both | **Ideal** - double-checked |

✅ ADDED (Senior Review): Implementation readiness note

- The above are **research directions**, not paper-parity fixes.
- Paper-parity formatting + retrieval diagnostics are now documented in canonical pages:
  - `docs/concepts/few-shot-prompt-format.md`
  - `docs/guides/debugging-retrieval-quality.md`
- CRAG and chunk scoring are no longer “future work” here; see:
  - `docs/guides/crag-validation-guide.md`
  - `docs/reference/chunk-scoring.md`

### Recommended Architecture

```
[Index Time - One-time]
1. Chunk transcripts (existing)
2. Embed chunks (existing)
3. NEW: Score each chunk with LLM → chunk_scores
4. Store: {chunk, embedding, chunk_scores}

[Query Time - Per Assessment]
1. Extract evidence (existing)
2. Embed evidence (existing)
3. Retrieve similar chunks (existing)
4. NEW: Use chunk_scores instead of participant_scores
5. OPTIONAL: CRAG validation as safety net
6. Show to LLM as few-shot examples (existing)
```

### Why Pre-Computed Chunk Scores Are Valid

**Concern**: "Chunk scores are LLM-estimated, not ground truth!"

**Reality**:
- Participant-level ground truth = human assessment of WHOLE interview
- Chunk-level ground truth **doesn't exist and can't exist**
- LLM-estimated chunk scores = best approximation of chunk severity
- More semantically correct than participant-level scores on misaligned chunks

### References

- [CRAG (LangChain)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval)
- [RAG Architectures 2025](https://medium.com/data-science-collective/rag-architectures-a-complete-guide-for-2025-daf98a2ede8c)
- [Google Sufficient Context (ICLR 2025)](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)

---

## Executive Summary

Our reproduction with proper AURC/AUGRC methodology shows **zero-shot statistically significantly outperforms few-shot**, directly contradicting the paper's central claim.

| Mode | AURC | 95% CI | AUGRC | Coverage |
|------|------|--------|-------|----------|
| **Zero-shot** | **0.134** | [0.094, 0.176] | **0.037** | 55.5% |
| Few-shot | 0.214 | [0.160, 0.278] | 0.074 | 71.9% |

**Key observation**: Few-shot predicts more (71.9% vs 55.5%) but is LESS accurate. This pattern suggests **overconfidence** - the model is being influenced by examples to make predictions it shouldn't.

---

## 1. Literature Review: When Few-Shot Fails

### 1.1 The Over-Prompting Problem (2024-2025 Research)

> "Incorporating excessive domain-specific examples into prompts can paradoxically degrade performance in certain LLMs."
> — [The Few-shot Dilemma: Over-prompting Large Language Models](https://arxiv.org/html/2509.13196v1)

**Key findings**:
- Performance degrades after ~5-20 examples in some models
- Smaller models (< 8B params) show weakening from the start
- Long contexts hurt more than they help

**Relevance to our system**: We inject up to 16 reference chunks (2 per PHQ-8 item). This may exceed optimal example count.

### 1.2 Zero-Shot Can Be Stronger Than Few-Shot (2025)

> "Recent strong models already exhibit strong reasoning capabilities under the Zero-shot CoT setting, and the primary role of Few-shot CoT exemplars is to align the output format with human expectations."
> — [Revisiting Chain-of-Thought Prompting](https://arxiv.org/html/2506.14641)

**Key findings**:
- RLLMs (reasoning LLMs) work better zero-shot
- Few-shot primarily helps with output FORMAT, not reasoning
- DeepSeek-R1 reports performance DEGRADATION with few-shot

**Relevance**: Gemma3:27b may have strong enough reasoning that few-shot adds noise rather than signal.

### 1.3 Pre-Training Bias Conflicts

> "ICL may have difficulty unlearning biases derived from pre-training data... This is not random variation, but a systematic conflict: each additional inverted example strengthens the contradiction between demonstrated and pre-trained semantics."
> — [Semantic Anchors in In-Context Learning](https://www.researchgate.net/publication/398026142)

**Relevance**: If reference examples conflict with the model's pre-trained understanding of depression symptoms, they may hurt rather than help.

### 1.4 RAG Retrieval Quality Issues

> "Traditional RAG retrieves a fixed number of documents, often introducing irrelevant or conflicting data."
> — [Enhancing Retrieval-Augmented Generation: A Study of Best Practices](https://arxiv.org/html/2501.07391v1)

**Relevance**: Our retrieval is purely semantic similarity based - no relevance filtering or quality assessment.

---

## 2. Architecture Analysis: How Our Few-Shot Works

### 2.1 Reference Embedding Generation (`generate_embeddings.py`)

```
For each training participant:
    transcript → sliding_window(8 lines, step 2) → chunks[]
    For each chunk:
        embedding = embed(chunk)
    Store: {participant_id: [(chunk_text, embedding), ...]}
```

**Issue #1: Chunks are NOT item-tagged**. A chunk about sleep problems is stored as a generic transcript chunk, not as "PHQ8_Sleep evidence".

### 2.2 Few-Shot Retrieval (`embedding.py:build_reference_bundle`)

```
For each PHQ-8 item (e.g., PHQ8_Sleep):
    evidence_quotes = extracted_evidence[item]  # "I can't sleep at night"
    query_embedding = embed(evidence_quotes)
    similar_chunks = find_top_k_similar(query_embedding, all_chunks)

    For each similar_chunk:
        participant_score = get_score(chunk.participant_id, item)  # PHQ8_Sleep score
        format: "(Score: {score})\n{chunk_text}"
```

**Issue #2: Semantic mismatch**. We find chunks semantically similar to "I can't sleep", but:
- The matched chunk might be 8 random lines of conversation
- The score attached is the participant's overall PHQ8_Sleep score
- That score might come from OTHER parts of their transcript, not this chunk

### 2.3 Original Authors' Implementation

**Source of truth**: `_reference/ai_psychiatrist/quantitative_assessment/embedding_quantitative_analysis.ipynb`

The notebook shows pure LLM evidence extraction without keyword backfill:
- `evidence_extraction_prompt` → LLM call
- `process_evidence_for_references` → embedding similarity
- `run_phq8_analysis` → final scoring

**Note**: The `.py` files in `_reference/agents/` are dead code (slop). They were never executed. The `_keyword_backfill` function visible in `quantitative_assessor_f.py` was NOT used by the authors. See `_reference/README.md` for details.

---

## 3. Hypotheses

### Hypothesis A: Over-Prompting / Context Overload

**Theory**: 16 reference chunks create excessive context that degrades performance.

**Evidence**:
- Research shows performance drops after 5-20 examples
- Our prompts become very long with reference bundles
- Gemma3:27b may struggle with long contexts

**Test**: Run with `top_k=1` instead of `top_k=2`.

✅ Implementation detail:

- `top_k` is controlled by `EmbeddingSettings.top_k_references` (`src/ai_psychiatrist/config.py:250`).
- Override via env var: `EMBEDDING_TOP_K_REFERENCES=1`

Concrete command (few-shot only for speed):

```bash
EMBEDDING_TOP_K_REFERENCES=1 uv run python scripts/reproduce_results.py --split paper-test --few-shot-only
```

Or run both modes in one artifact (recommended for paired evaluation convenience):

```bash
EMBEDDING_TOP_K_REFERENCES=1 uv run python scripts/reproduce_results.py --split paper-test
```

### Hypothesis B: Semantic Mismatch in References - **PLAUSIBLE (Not Proven)**

**Theory**: Retrieved chunks are semantically similar but not PHQ-8-item-aligned.

**Status**: Divergences documented in BUG-031. Causality not proven.

**Divergences Found**:
1. **Score-Chunk Mismatch** (`embedding.py:199`): Paper methodology - correctly implemented
2. **Format Mismatch**: 8 separate sections vs paper's 1 unified block
3. **Missing Domain Labels**: `(Score: 2)` vs paper's `(PHQ8_Sleep Score: 2)`
4. **Closing Tag**: We use `</Reference Examples>`, paper uses `<Reference Examples>`

**Example**:
```
Evidence: "I haven't been sleeping well"
Retrieved chunk: "I've been really tired lately, can't focus on anything"
Attached score: PHQ8_Sleep = 3 for that participant

But this score is from the PARTICIPANT'S overall PHQ8_Sleep,
NOT from this chunk's content!
```

**Root Cause**: Chunks are generic transcript windows without item tagging. Scores are looked up from participant-level ground truth at retrieval time, not analyzed from chunk content.

### Hypothesis C: Overconfidence from Examples

**Theory**: Few-shot examples make the model more willing to predict, but less accurate.

**Evidence**:
- Coverage: 71.9% (few-shot) vs 55.5% (zero-shot)
- MAE: 0.795 (few-shot) vs 0.640 (zero-shot)
- Pattern: "I see examples, so I'll predict too" → wrong predictions

**Test**: Analyze N/A rate by item and correlation with reference quality.

### Hypothesis D: ~~Keyword Backfill Missing~~ **RULED OUT**

~~**Theory**: Original implementation used keyword backfill; we didn't.~~

**Status**: INCORRECT. This hypothesis was based on reading `quantitative_assessor_f.py` line 478, which is **dead code**.

**Verification**: The actual notebook (`embedding_quantitative_analysis.ipynb`) does NOT call `_keyword_backfill`. The paper authors ran pure LLM extraction without keyword backfill.

**Note**: The `.py` files in `_reference/` are slop - the notebooks are the source of truth. See `_reference/README.md`.

### Hypothesis E: Model Capacity / Quantization

**Theory**: Gemma3:27b-it-qat (4-bit) may not have capacity for complex ICL.

**Evidence**:
- Research shows smaller models struggle with ICL
- QAT quantization may reduce reasoning capacity
- Paper likely used BF16 (54GB)

**Test**: Run with gemma3:27b-it-fp16 if VRAM available.

---

## 4. Action Items

### Immediate Investigations

1. [x] **Log retrieved references**: Investigated via code analysis - found Score-Chunk Mismatch bug.

2. [x] **Inspect reference quality manually**: Code review confirmed chunks are generic windows with participant-level scores.

3. [x] ~~**Re-run with keyword backfill ON**~~ - **RULED OUT**: Original authors did NOT use keyword backfill. Not relevant to investigation.

4. [ ] **Test with top_k=1**: Reduce reference examples to see if fewer helps.

### Deeper Analysis

5. [ ] **Participant-level comparison**: For participants where zero-shot wins big, what do few-shot references look like?

6. [ ] **Item-level retrieval analysis**: Are some PHQ-8 items getting better references than others?

7. [ ] **Similarity score distribution**: What are the actual similarity scores? Are they low (poor matches)?

### Code Improvements (Future)

8. [ ] **Add retrieval quality filter**: Only use references above similarity threshold.

9. [ ] **Item-tagged embeddings**: Generate embeddings specifically for each PHQ-8 item, not generic transcript chunks.

10. [ ] **CRAG-style retrieval**: Add retrieval evaluator to assess reference quality before using.

---

## 5. Raw Data References

### Output Files
- Zero-shot metrics: `data/outputs/selective_prediction_metrics_20251228T133513Z.json`
- Few-shot metrics: `data/outputs/selective_prediction_metrics_20251228T133532Z.json`
- Combined run: `data/outputs/few_shot_paper_backfill-off_20251228_024244.json`

### Key Logs to Check
This repo does not reliably write logs to `logs/` by default during reproduction runs. Prefer console logs.

If you implement retrieval diagnostics (Spec 32), enable it explicitly:
- `EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true` (audit logs are emitted at INFO level)

Then you can filter console output with:

```bash
rg -n \"retrieved_reference|Found references for item|bundle_length|top_similarity\"
```

### Reference Implementation
- Original few-shot: `_reference/ai_psychiatrist/agents/quantitative_assessor_f.py`
- Our implementation: `src/ai_psychiatrist/agents/quantitative.py`
- Embedding service: `src/ai_psychiatrist/services/embedding.py`

---

## 6. Conclusion

### Divergences Found

Investigation identified paper-parity divergences that may contribute to the performance gap:

1. **Score-Chunk Mismatch** (`embedding.py:199`): Paper methodology - correctly implemented
2. **Format Mismatch** (`embedding.py:40-70`): 8 separate sections vs paper's 1 unified block
3. **Missing Domain Labels** (`embedding.py:58-62`): `(Score: 2)` vs paper's `(PHQ8_Sleep Score: 2)`
4. **Closing Tag**: `</Reference Examples>` vs paper's `<Reference Examples>`

### Hypotheses Status

| Hypothesis | Status | Notes |
|------------|--------|-------|
| A: Over-prompting | UNLIKELY | Paper tested top_k in Appendix D; 16 within range |
| **B: Semantic mismatch** | **PLAUSIBLE** | Divergences documented; causality unproven |
| C: Overconfidence | POSSIBLE | Pattern consistent with hypothesis, not proven |
| D: Keyword backfill | RULED OUT | Authors did NOT use it (notebook verified) |
| E: Model capacity | UNKNOWN | Would need BF16 testing |
| F: Context dilution | NOT TESTED | Few-shot adds many tokens |
| G: LLM stochasticity | NOT TESTED | Single run, no variance measured |

### What We Know vs What We Hypothesize

| Known (Verified) | Hypothesized (Unproven) |
|------------------|-------------------------|
| Zero-shot AURC < few-shot AURC in our runs | Format divergences caused the gap |
| Our format differs from paper's notebook | Score-chunk mismatch harms performance |
| Paper uses participant-level scores | Fixing format will improve few-shot |

### Next Steps

1. **Fix format divergences** - Match paper's exact format
2. **Run ablation** - Re-evaluate to see if few-shot improves
3. **Add retrieval diagnostics** - Log and audit retrieved chunks
4. **Assess LLM variance** - Run multiple times
5. **Senior review** after ablation results

---

## Appendix: Research Sources

- [The Few-shot Dilemma: Over-prompting Large Language Models](https://arxiv.org/html/2509.13196v1)
- [Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger](https://arxiv.org/html/2506.14641)
- [Semantic Anchors in In-Context Learning](https://www.researchgate.net/publication/398026142)
- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://openreview.net/forum?id=ZS4m74kZpH)
- [Enhancing Retrieval-Augmented Generation: A Study of Best Practices](https://arxiv.org/html/2501.07391v1)
