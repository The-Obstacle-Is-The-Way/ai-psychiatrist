# Three Questions: Few-Shot Design Analysis

**Date**: 2025-12-30
**Status**: Critical Design Investigation
**Origin**: First-principles analysis of few-shot methodology

---

## Question 1: Can Zero-Shot "Cheat"?

### Answer: YES

**Evidence from code** (`quantitative.py:202`):
```python
reference_text = ""  # Empty for zero-shot
prompt = make_scoring_prompt(transcript.text, reference_text)
```

The LLM receives the **full DAIC-WOZ transcript** including:
- Ellie's structured interview questions (probing PHQ-8 symptoms)
- Participant's direct responses

**Example of "shortcut" available:**
```
Ellie: have you been diagnosed with depression
Participant: yes i was diagnosed last year
Ellie: can you tell me more about that    ← PROBING = SYMPTOM DETECTED
Participant: i was feeling really down...
```

**External Validation**: Burdisso et al. (2024) proved that models using Ellie's prompts achieve 0.88 F1 vs 0.85 F1 for participant-only. They achieved **0.90 F1 by intentionally exploiting this bias**.

**Conclusion**: Zero-shot isn't "cheating" - it's using the interview as designed. But Ellie's structured questions provide discriminative shortcuts that make PHQ-8 prediction easier.

---

## Question 2: Is Few-Shot Done Incorrectly?

### Answer: YES - There's a Fundamental Design Flaw

### How Our Few-Shot Actually Works

1. **Extract evidence** from target participant's transcript
2. **Embed evidence** for each PHQ-8 item
3. **Retrieve similar CHUNKS** from reference embeddings
4. **Assign participant-level scores** to those chunks
5. **Format as reference examples**

### The Critical Flaw

**What gets retrieved**: 8-line sliding window chunks from OTHER participants

**What score gets assigned**: The OTHER participant's OVERALL PHQ-8 item score

**Result**: Completely irrelevant chunks get labeled with scores:

```xml
<Reference Examples>

(PHQ8_Sleep Score: 2)
Ellie: how are you doing today
Participant: good
Ellie: what's your dream job
Participant: to open a business
Ellie: do you travel a lot        ← NOTHING ABOUT SLEEP!
Participant: no
Ellie: why
Participant: no specific reason

</Reference Examples>
```

This chunk has **NOTHING to do with sleep**, but gets labeled "Score: 2" because that participant happened to have a Sleep score of 2!

### What FEW-SHOT SHOULD Look Like

You correctly intuited this: examples should show **what different PHQ scores look like**:

```xml
<Reference Examples>

(PHQ8_Sleep Score: 3 - Nearly Every Day)
Participant: i can never sleep, maybe 2-3 hours a night
Participant: i lie awake thinking about everything
Participant: i'm exhausted all the time from not sleeping

(PHQ8_Sleep Score: 0 - Not At All)
Participant: i sleep great actually
Participant: usually get 8 hours no problem
Participant: sleep is the one thing i don't have issues with

</Reference Examples>
```

### Is This The Paper's Design?

**Yes** - this is exactly what Section 2.4.2 describes:
- Sliding window chunks (N_chunk=8, step=2)
- Cosine similarity retrieval
- Participant-level score assignment

But **the paper's design is flawed**. The chunks are:
1. NOT symptom-specific (random 8-line windows)
2. NOT score-representative (assigned participant-level scores)
3. Retrieved by embedding similarity, not clinical relevance

### Can This Be Fixed?

**Spec 35** attempted to fix this with chunk-level scoring, but:
- It scores ALL chunks with the participant's overall PHQ-8 score (per item)
- The scorer prompt asks: "Would this evidence support score X for PHQ8_Sleep?"
- This is backwards - we're asking if random chunks justify a predetermined score

**What Would Actually Fix It**:

1. **Symptom-Aware Chunking**: Chunk by question-answer pairs, not sliding windows
2. **Clinical Relevance Filtering**: Only keep chunks that actually discuss the symptom
3. **Full Transcript Examples**: Show "this is what a PHQ8 total 20 looks like" vs "this is what a 5 looks like"
4. **Expert-Curated References**: Hand-pick clinically relevant examples per symptom

---

## Question 3: Is Few-Shot Even The Right Approach?

### Answer: 2025 Research Says NO - RAG is Better

**Web search results (December 2025) show clear trends:**

### RED: RAG for Depression Detection (arxiv 2503.01315)

| Method | Macro F1 |
|--------|----------|
| Direct Prompting (naive LLM) | 78.90% |
| Naive RAG | 84.39% |
| **RED (with filtering + knowledge base)** | **90.00%** |

Key innovations:
- **Adaptive Judge Module**: LLM decides when to stop retrieving
- **Symptom-Aligned Retrieval**: Retrieve only snippets relevant to PHQ-8 aspects
- **Knowledge Base Grounding**: Use psychological knowledge to filter noise

### Adaptive RAG for Mental Health (arxiv 2501.00982)

Key insight:
> "Reframe prediction from direct text-to-diagnosis as correlating text and questionnaire items"

Instead of few-shot examples, they:
- Retrieve relevant user posts for EACH questionnaire item
- Generate item-level responses in zero-shot
- Aggregate into final diagnosis

**Results**: Outperformed benchmarks (55% DCHR vs 45% best baseline) in completely unsupervised setting.

### GPT-4 Clinical Depression Study (arxiv 2501.00199)

- GPT-4 achieved F1 0.73 for depression classification
- PHQ-8 estimates correlated strongly (r = 0.71) with true scores
- **Zero-shot GPT-4 outperformed few-shot GPT-3.5**

### Key Takeaway from 2025 Literature

**Few-shot with random chunks is OUTDATED**. Modern approaches use:

1. **Structured RAG**: Retrieve symptom-specific evidence, not random chunks
2. **Filtering/Validation**: LLM judges to reject irrelevant retrievals (like our Spec 36!)
3. **Questionnaire Grounding**: Predict item-by-item, not overall diagnosis
4. **Zero-shot with good prompts**: Often beats naive few-shot

---

## Recommendations

### Immediate Actions

1. **Enable Spec 36** (CRAG validation) - it implements the "Judge Module" pattern
2. **Consider disabling few-shot entirely** - zero-shot may be optimal for DAIC-WOZ
3. **Run A/B test**: Zero-shot vs Few-shot with Spec 36 enabled

### Architectural Changes (Future)

1. **Symptom-Aware Chunking**: Replace sliding windows with QA-pair extraction
2. **Full Transcript RAG**: Instead of chunks, use complete interviews as examples
3. **Expert-Curated Examples**: Hand-pick references per PHQ-8 item
4. **Move to RED-style Architecture**: Adopt the filtering + knowledge base approach

### Experimental Validation

1. Strip Ellie from transcripts - measure zero-shot performance drop
2. Use hand-picked few-shot examples - measure if quality references help
3. Analyze retrieval audit logs - measure clinical relevance of retrieved chunks

---

## Conclusion

**The few-shot implementation is not a code bug - it's a design flaw inherited from the paper.**

The paper's methodology:
- Uses random 8-line chunks
- Assigns participant-level scores to irrelevant text
- Retrieves by embedding similarity, not clinical relevance

2025 research shows this is outdated. Modern approaches use:
- Symptom-aligned retrieval
- LLM-based filtering
- Questionnaire grounding
- Or just zero-shot with good prompts

**Our zero-shot may actually be near-optimal because DAIC-WOZ's structured interview design gives the LLM direct access to PHQ-8-relevant evidence through Ellie's questions.**

---

## Sources

- [Explainable Depression Detection with Personalized RAG (RED)](https://arxiv.org/html/2503.01315)
- [Adaptive RAG for Mental Health Screening](https://arxiv.org/html/2501.00982v1)
- [GPT-4 on Clinical Depression Assessment](https://arxiv.org/html/2501.00199)
- [Zero-Shot Strike: LLM Depression Detection](https://www.sciencedirect.com/science/article/abs/pii/S0885230824000469)
- [DAIC-WOZ Prompt Bias (Burdisso et al.)](../_literature/markdown/daic-woz-prompts/daic-woz-prompts.md)

---

*"The code may be right, but the behavior may be wrong."* - The insight that led to this investigation.
