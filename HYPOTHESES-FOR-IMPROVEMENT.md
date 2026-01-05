# Hypotheses for Improvement: First-Principles Analysis

**Status**: Research findings document
**Created**: 2026-01-05
**Analysis Scope**: Full pipeline from DAIC-WOZ → Evidence Extraction → Embedding → Scoring

---

## Executive Summary

A first-principles audit of the PHQ-8 scoring pipeline reveals several fundamental mismatches between the dataset, the task, and our implementation. These are not bugs in the traditional sense—the code executes correctly—but rather **methodological constraints** that limit what is achievable with this approach on this dataset.

**Key Finding**: DAIC-WOZ was designed to detect behavioral indicators of depression, not to elicit explicit PHQ-8 frequency information. Our prompts require frequency evidence that the dataset was never designed to provide.

---

## 1. Dataset-Task Mismatch (Critical)

### What DAIC-WOZ Was Designed For

Per the [DAIC-WOZ documentation](https://dcapswoz.ict.usc.edu/):

> "These interviews were collected as part of a larger effort to create a computer agent that interviews people and **identifies verbal and nonverbal indicators of mental illness**."

The virtual interviewer "Ellie" conducts semi-structured interviews designed to:
- Create interactional situations favorable to assessing distress indicators
- Capture behavioral markers correlated with depression
- Collect multimodal data (audio, video, text)

### What PHQ-8 Scoring Requires

PHQ-8 is a **frequency-based** instrument asking "Over the last 2 weeks, how often have you been bothered by [symptom]?":
- 0 = Not at all (0-1 days)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

### The Mismatch

**The interview doesn't ask about frequency. Participants don't state frequency.** They say things like:
- "I've been feeling tired" (no frequency)
- "I have trouble sleeping sometimes" (vague)
- "I've been stressed lately" (qualitative)

This explains why:
- Only 30.4% of chunks have any scoreable evidence
- ~50% of extracted quotes fail evidence grounding
- Coverage ceiling is ~46-49%

---

## 2. Evidence Extraction Paradox

### Current Prompt Logic

Our prompts (see `src/ai_psychiatrist/agents/prompts/quantitative.py:116-117`) say:
```
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency
```

### The Paradox

This is **methodologically correct but practically limiting**:
- Most transcripts don't contain explicit frequency statements
- Correct behavior: output N/A for most items
- Result: ~50% abstention rate

### Hypothesis 2A: Frequency Can Be Inferred

A skilled psychiatrist doesn't require patients to say "I felt tired 8 out of 14 days." They infer frequency from:
- Temporal markers ("lately", "recently", "since [event]")
- Intensity qualifiers ("always", "sometimes", "occasionally")
- Impact statements ("I can't function", "it's been hard")
- Context patterns (multiple mentions across the interview)

**Current Status**: Our prompts discourage inference. They demand explicit frequency.

**Improvement Hypothesis**: Update prompts to allow clinical inference while maintaining transparency:
```
When explicit frequency is not stated, you may infer approximate frequency from:
- Temporal language ("lately" → several days, "always" → nearly every day)
- Intensity markers ("sometimes" → several days)
- Functional impact ("can't work" → more than half the days)
Document your inference in the reason field.
```

**Trade-off**: Higher coverage, potentially lower precision. Needs ablation.

---

## 3. Chunk Scoring Validity Issues

### Observation from Scored Chunks

Examining chunks scored for PHQ8_Sleep (see `data/embeddings/*.chunk_scores.json`):

**Chunk 303:37 scored Sleep=3**:
```
"i need my rest because i'm out there driving that bus..."
"what am i like irritated tired um lazy"
"feel like i wanna lay down probably go to sleep"
```

**Problem**: This participant is expressing:
- Value for rest ("I need my rest")
- Desire to sleep ("feel like i wanna lay down")
- General tiredness

**NOT**: Trouble falling/staying asleep or sleeping too much (the actual PHQ-8 Sleep item)

### Hypothesis 3A: Semantic Confusion in Chunk Scoring

The LLM scorer is confusing:
| What participant said | What LLM inferred | Actual PHQ-8 construct |
|-----------------------|-------------------|------------------------|
| "I need rest" | Sleep problems | Not a symptom |
| "I feel tired" | Sleep issues | Different item (Tired) |
| "I want to nap" | Sleeping too much | Maybe, context-dependent |

**Improvement Hypothesis**: Add explicit symptom definitions to chunk scoring prompt:
```
PHQ8_Sleep asks about: "Trouble falling or staying asleep, OR sleeping too much"
- Wanting rest is NOT a sleep problem
- Feeling tired belongs to PHQ8_Tired, not PHQ8_Sleep
- "Sleeping too much" means actually sleeping excessive hours, not wanting to
```

---

## 4. Embedding Space Limitations

### Current Approach

1. Extract evidence text from test transcript
2. Embed evidence text
3. Find similar chunks from reference corpus
4. Use reference chunk scores as anchors

### Hypothesis 4A: Semantic Similarity ≠ Severity Similarity

Embedding captures **topic similarity**, not **severity similarity**:
- "I can't sleep at night" (severe) ≈ "I value good rest" (not a symptom)
- Both are "about sleep" in embedding space
- One is PHQ8_Sleep=3, one is PHQ8_Sleep=0

**Evidence**: Item-tag filtering helps (Spec 34), but doesn't solve the severity confusion within a topic.

### Hypothesis 4B: Score Reranking

**Improvement Hypothesis**: After semantic retrieval, rerank by:
1. Presence of severity markers in reference chunk
2. Score distribution (prefer balanced exemplars)
3. Exclude chunks that are topic-adjacent but not symptom-indicative

---

## 5. N/A Criteria Analysis

### Current Behavior

Two paths to N/A:
1. `NO_MENTION`: LLM evidence count = 0 (no relevant quotes found)
2. `SCORE_NA_WITH_EVIDENCE`: LLM found evidence but explicitly said N/A

### Hypothesis 5A: Over-Abstention on Implicit Evidence

Run 12 data: 51.5% abstention (zero-shot), 54% abstention (few-shot).

Many participants may have depression symptoms visible in their language patterns (word choice, response length, topic avoidance) without explicit symptom mentions.

**Question**: Should we abstain on items where behavioral indicators suggest pathology but explicit frequency is missing?

**Trade-off**:
- Abstaining is methodologically conservative (no hallucinated scores)
- But may miss clinically meaningful signals
- Psychiatrists use holistic assessment, not just verbal frequency statements

---

## 6. Frequency Inference Hierarchy

### Proposed Inference Rules (Hypothesis)

| Language Pattern | Inferred Frequency | PHQ-8 Score |
|------------------|-------------------|-------------|
| "every day", "constantly", "all the time" | 12-14 days | 3 |
| "most days", "usually" | 7-11 days | 2 |
| "sometimes", "a few times", "lately" | 2-6 days | 1 |
| "once", "rarely", "not really" | 0-1 days | 0 |
| No temporal marker, only symptom mention | Ambiguous | N/A or 1? |

**Current behavior**: Ambiguous → N/A
**Alternative**: Ambiguous → 1 (conservative non-zero) with low confidence

---

## 7. Pipeline Architecture Questions

### Question 7A: Evidence Extraction as Bottleneck

The current pipeline:
```
Transcript → Evidence Extraction → Embedding → Reference Retrieval → Scoring
```

Evidence extraction is a **filter**:
- Grounded quotes only (substring match)
- Rejects ~50% of extracted quotes as "hallucinated"

**Hypothesis**: Evidence grounding is too strict. "Hallucinated" quotes may be:
- Paraphrases (valid signal, wrong words)
- Composite statements (synthesized from multiple utterances)
- Reasonable inferences (not literal but implied)

**Improvement Hypothesis**: Fuzzy grounding with semantic similarity instead of substring match.

### Question 7B: Direct Scoring vs. Evidence-Mediated

Alternative architecture:
```
Transcript → Direct Scoring (no evidence extraction)
```

Let the LLM see the full transcript and score directly. Trade-offs:
- Pro: No evidence extraction bottleneck
- Con: Less interpretable, harder to ground
- Con: May increase hallucination

---

## 8. Ground Truth Reliability

### The Meta-Question

How reliable is the PHQ-8 ground truth?
- Patients self-report their symptoms
- Self-report has known biases (social desirability, recall error)
- The same patient might score differently on different days

**Implication**: Even perfect prediction can't exceed ground truth reliability. MAE floor may be ~0.5 due to label noise, not model error.

---

## 9. Summary of Hypotheses

| ID | Hypothesis | Type | Effort |
|----|-----------|------|--------|
| 2A | Allow frequency inference from temporal/intensity markers | Prompt change | Low |
| 3A | Add explicit symptom definitions to chunk scorer | Prompt change | Low |
| 4A | Embedding captures topic, not severity | Architecture | High |
| 4B | Rerank by severity markers, not just similarity | Code change | Medium |
| 5A | Consider behavioral indicators beyond verbal frequency | Research | High |
| 7A | Fuzzy evidence grounding (semantic similarity) | Config + code | Medium |
| 7B | Direct scoring without evidence extraction | Architecture | High |

---

## 10. Recommended Next Steps

### Immediate (Low-effort, testable)

1. **Hypothesis 3A**: Update chunk scoring prompt with explicit symptom definitions
2. **Hypothesis 2A**: Create a "frequency inference" prompt variant and ablate

### Research (Higher effort)

3. **Hypothesis 7A**: Implement fuzzy grounding and compare to substring match
4. **Hypothesis 4B**: Implement severity-aware reranking

### Fundamental Re-evaluation

5. Consider whether PHQ-8 frequency scoring is the right task for this dataset
6. Explore alternative tasks: binary depression detection, severity classification (none/mild/moderate/severe)

---

## 11. Related Documentation

- [Few-Shot Analysis](docs/results/few-shot-analysis.md) — Why few-shot may not beat zero-shot
- [RAG Design Rationale](docs/rag/design-rationale.md) — Original design decisions
- [Metrics and Evaluation](docs/statistics/metrics-and-evaluation.md) — AURC/AUGRC definitions

---

## Sources

- [DAIC-WOZ Database](https://dcapswoz.ict.usc.edu/)
- [DAIC-WOZ Documentation](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)
- [DAIC-WOZ: On the Validity of Using the Therapist's prompts](https://arxiv.org/abs/2404.14463)
- [The Distress Analysis Interview Corpus](https://www.researchgate.net/publication/311643727_The_Distress_Analysis_Interview_Corpus_of_human_and_computer_interviews)
