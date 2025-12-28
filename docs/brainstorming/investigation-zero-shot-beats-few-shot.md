# Investigation: Why Zero-Shot Beats Few-Shot

**Date**: 2025-12-28
**Status**: OPEN - Requires Further Investigation
**Severity**: High - Core Paper Claim Inversion

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

### Hypothesis B: Semantic Mismatch in References

**Theory**: Retrieved chunks are semantically similar but not PHQ-8-item-aligned.

**Evidence**:
- Chunks are stored as generic transcript windows
- No guarantee that "similar" means "relevant for this PHQ-8 item"
- Score attached may not reflect chunk content

**Example**:
```
Evidence: "I haven't been sleeping well"
Retrieved chunk: "I've been really tired lately, can't focus on anything"
Attached score: PHQ8_Sleep = 3 for that participant

But this chunk is about FATIGUE/CONCENTRATION, not sleep!
```

**Test**: Log retrieved chunks and manually inspect relevance.

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

1. [ ] **Log retrieved references**: Add debug logging to see what chunks are actually being retrieved and their similarity scores.

2. [ ] **Inspect reference quality manually**: For 5 participants, manually review if retrieved chunks are relevant.

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
```bash
# Grep for retrieval logs
grep -r "Found references" logs/
grep -r "bundle_length" logs/
grep -r "top_similarity" logs/
```

### Reference Implementation
- Original few-shot: `_reference/ai_psychiatrist/agents/quantitative_assessor_f.py`
- Our implementation: `src/ai_psychiatrist/agents/quantitative.py`
- Embedding service: `src/ai_psychiatrist/services/embedding.py`

---

## 6. Conclusion

This investigation reveals multiple potential causes for the counterintuitive result. The most likely explanations are:

1. **Over-prompting**: 16 examples may be too many for this model/task
2. **Semantic mismatch**: Generic transcript chunks don't align with PHQ-8 items
3. **Overconfidence**: Few-shot makes model predict more, but with lower accuracy

~~Keyword backfill was initially suspected but ruled out - the original authors did NOT use it.~~

**Next step**: Launch parallel investigation agents to test each hypothesis systematically.

---

## Appendix: Research Sources

- [The Few-shot Dilemma: Over-prompting Large Language Models](https://arxiv.org/html/2509.13196v1)
- [Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger](https://arxiv.org/html/2506.14641)
- [Semantic Anchors in In-Context Learning](https://www.researchgate.net/publication/398026142)
- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://openreview.net/forum?id=ZS4m74kZpH)
- [Enhancing Retrieval-Augmented Generation: A Study of Best Practices](https://arxiv.org/html/2501.07391v1)
