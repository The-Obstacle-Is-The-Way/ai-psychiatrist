# Hypothesis: Zero-Shot Performance May Be Artificially Inflated

**Date**: 2025-12-30
**Status**: Under Investigation
**Origin**: Shower thought + code audit + literature review

## Executive Summary

The persistent observation that **few-shot underperforms zero-shot** in our PHQ-8 scoring pipeline may be backwards. The real issue may be that **zero-shot is artificially inflated** due to Ellie's structured interview questions providing discriminative shortcuts.

## The Paradox

Our experiments consistently show:
- Zero-shot MAE: ~0.796 (reported in paper)
- Few-shot MAE: ~0.619 (reported in paper)
- Our reproductions: Few-shot often performs *worse* than zero-shot

This contradicts the paper's claim and general intuition that examples should help.

## What Gets Fed to the LLM?

### Zero-Shot Pathway
```
┌─────────────────────────────────────────────────────────────────┐
│ make_scoring_prompt(transcript.text, reference_text="")        │
│                                                                 │
│ The LLM receives:                                               │
│ 1. Full transcript wrapped in <transcript> tags                 │
│    - Ellie's questions (structured, standardized)               │
│    - Participant's responses                                    │
│                                                                 │
│ 2. No reference examples                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Few-Shot Pathway
```
┌─────────────────────────────────────────────────────────────────┐
│ make_scoring_prompt(transcript.text, reference_text=bundle)    │
│                                                                 │
│ The LLM receives:                                               │
│ 1. Full transcript (SAME as zero-shot)                          │
│ 2. PLUS: Reference chunks from OTHER participants               │
│    - <Reference Examples> block                                 │
│    - (PHQ8_Sleep Score: 2)\nChunk text...                       │
│    - May include 2+ references per PHQ-8 item                   │
└─────────────────────────────────────────────────────────────────┘
```

## The Transcript Structure

DAIC-WOZ transcripts contain **both** Ellie's questions AND participant responses:

```
Ellie: how are you doing today
Participant: good
Ellie: do you consider yourself an introvert
Participant: yes definitely
...
Ellie: have you been diagnosed with depression
Participant: yes i was diagnosed last year
Ellie: can you tell me more about that
Participant: i was feeling really down and couldn't sleep
```

**Key Insight**: Ellie asks DIRECT questions about PHQ-8 symptoms:
- "What are you like when you don't get enough sleep?" → PHQ8_Sleep
- "Do you have trouble concentrating?" → PHQ8_Concentrating
- "Have you been diagnosed with depression?" → PHQ8_Depressed

## External Validation: The Burdisso Paper

**Paper**: "DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic Depression Detection" (Burdisso et al., 2024)

**Critical Finding**: Models using Ellie's prompts achieve **0.88 F1** vs **0.85 F1** for participant-only models. The authors proved:

> "Models using interviewer's prompts learn to focus on a specific region of the interviews, where questions about past experiences with mental health issues are asked, and use them as **discriminative shortcuts** to detect depressed participants."

**Their Figure 1** shows that Ellie-based models focus on the **second half of interviews** where mental health probing occurs. They achieved **0.90 F1** by intentionally exploiting this bias.

**Implication for Us**: Our zero-shot prompt gives the LLM access to these same shortcuts.

## The Hypothesis

### Primary Hypothesis
Zero-shot performance is inflated because the LLM can extract PHQ-8 scores by:
1. Reading Ellie's direct questions about symptoms
2. Reading participant's direct answers
3. Using Ellie's probing patterns as evidence (deeper probing = more symptoms)

This is NOT "cheating" - it's using the interview as designed. But it means zero-shot has a structural advantage that few-shot disrupts.

### Secondary Hypothesis
Few-shot underperforms because the reference chunks:
1. **Add noise** - chunks from other participants may be semantically similar but clinically irrelevant
2. **Create confusion** - participant-level scores applied to individual chunks mislead the LLM
3. **Interfere with pattern matching** - the LLM's natural ability to read Ellie's questions is disrupted by conflicting examples

## Evidence From Our Codebase

### 1. The Transcript is Passed Whole
```python
# quantitative.py:202
prompt = make_scoring_prompt(transcript.text, reference_text)
```
The full transcript (Ellie + Participant) is always passed to the LLM.

### 2. Reference Chunks Are From Different Participants
```python
# embedding.py:247-253
matches.append(
    SimilarityMatch(
        chunk=TranscriptChunk(
            text=chunk_text,
            participant_id=participant_id,  # DIFFERENT participant!
        ),
        ...
    )
)
```

### 3. Scores Come From Participant-Level Ground Truth
```python
# embedding.py:243
score = self._reference_store.get_score(participant_id, lookup_item)
```
The score is the **participant's total PHQ-8 item score**, not a chunk-specific score.

## The Retrieval Problem

When we retrieve "similar" chunks for PHQ8_Sleep, we might get:

**Query Evidence**: "I can't sleep, I lie awake all night"
**Retrieved Chunk**: "Ellie: what are you like when you don't sleep\nParticipant: i get cranky"
**Chunk Score**: 2 (participant's overall Sleep score)

But is this chunk *actually* about sleep problems? Or is it about personality when tired? The embedding similarity found a match, but the clinical relevance may be low.

## Testing The Hypothesis

### Experiment 1: Participant-Only Transcripts
Strip Ellie's utterances from transcripts. Compare zero-shot performance:
- If performance drops significantly → Ellie's questions provide shortcuts
- If performance stays similar → participant responses are sufficient

### Experiment 2: Question-Only Analysis
Create prompts with ONLY Ellie's questions (no participant responses):
- High performance → Ellie's probing patterns are discriminative
- Low performance → Participant responses are essential

### Experiment 3: Reference Quality Audit
For each retrieved reference, manually evaluate:
- Is it clinically relevant to the PHQ-8 item?
- Does the assigned score make sense for this specific chunk?
- Are we retrieving noise?

### Experiment 4: Oracle Few-Shot
Instead of embedding-based retrieval, use hand-picked, clinically relevant examples:
- If this improves over zero-shot → retrieval is the problem
- If this matches zero-shot → few-shot isn't helping for this task

## Implications

### If Hypothesis is True:
1. The original paper's few-shot advantage may be an artifact of their specific retrieval implementation
2. Zero-shot may be near-optimal for DAIC-WOZ because of its structured interview format
3. Spec 35/36 (chunk scoring, reference validation) may help by filtering noise, not by improving retrieval

### For Our Reproduction:
1. Focus on **reference quality** rather than quantity
2. Consider **Ellie-aware chunking** (chunk by question-answer pairs, not sliding windows)
3. The "few-shot paradox" may be fundamental to DAIC-WOZ, not a bug in our code

## Next Steps

1. [ ] Run Experiment 1 (Participant-only transcripts)
2. [ ] Analyze retrieval audit logs for clinical relevance
3. [ ] Compare our retrieval quality to paper's methodology (if available)
4. [ ] Document findings in bug report or spec

## Related Documentation

- `docs/bugs/bug-032-spec34-visibility-gap.md` - Coverage issues
- `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md` - Burdisso paper
- `docs/archive/specs/35-offline-chunk-level-phq8-scoring.md` - Chunk scoring attempt
- `FEATURES.md` - Current feature flags

## Conclusion

We may have been debugging the wrong problem. The question isn't "why is few-shot worse?" but "is few-shot appropriate for structured interviews?" The DAIC-WOZ dataset's design gives LLMs direct access to clinical evidence through Ellie's questions. Adding noisy reference examples may hurt more than help.

---

*"Maybe we're thinking about it in reverse."* - The shower thought that started this investigation.
