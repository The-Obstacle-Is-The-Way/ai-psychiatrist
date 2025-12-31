# Hypothesis: Zero-Shot Performance May Be Artificially Inflated

**Date**: 2025-12-30
**Status**: Under Investigation
**Origin**: Shower thought + code audit + literature review

---

## Executive Summary

The persistent observation that **few-shot underperforms zero-shot** in our PHQ-8 scoring pipeline may be backwards. The real issue may be that **zero-shot is artificially inflated** due to Ellie's structured interview questions providing discriminative shortcuts.

**However**: This doesn't mean few-shot is worthless. For small parameter models (like Gemma 27B), few-shot provides necessary calibration. The issue is that our current few-shot implementation is flawed, not that few-shot as a concept is wrong.

---

## The Paradox

Our experiments consistently show:
- Zero-shot MAE: ~0.796 (reported in paper)
- Few-shot MAE: ~0.619 (reported in paper)
- Our reproductions: Few-shot often performs *worse* than zero-shot

This contradicts the paper's claim and general intuition that examples should help.

---

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

---

## The Transcript Structure (The "Cheating" Mechanism)

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

---

## External Validation: The Burdisso Paper

**Paper**: "DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic Depression Detection" (Burdisso et al., 2024)

**Location**: `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md`

**Critical Finding**: Models using Ellie's prompts achieve **0.88 F1** vs **0.85 F1** for participant-only models. The authors proved:

> "Models using interviewer's prompts learn to focus on a specific region of the interviews, where questions about past experiences with mental health issues are asked, and use them as **discriminative shortcuts** to detect depressed participants."

**Their Figure 1** shows that Ellie-based models focus on the **second half of interviews** where mental health probing occurs. They achieved **0.90 F1** by intentionally exploiting this bias.

**Implication for Us**: Our zero-shot prompt gives the LLM access to these same shortcuts.

---

## What We Already Documented

See: `docs/brainstorming/daic-woz-preprocessing.md`

**Section 12 Verdict**:
> "From first principles + research evidence + clinical perspective: **participant-only is the correct approach.**"

**Section 5 (First Principles)**:
> "**Goal**: Predict PHQ-8 scores from what the *patient* says about their symptoms."
> "**Ellie's questions** = interviewer protocol, NOT patient data."

**The TRUE task hierarchy**:

| Mode | What It Tests | Validity |
|------|---------------|----------|
| Zero-shot (participant-only) | Can LLM assess patient's words? | **HIGH - The real test** |
| Zero-shot (full transcript) | Can LLM read Ellie's shortcuts? | LOW - Inflated |
| Few-shot (current) | Noisy chunks + wrong scores | LOW - Broken |
| Few-shot + Spec 35/36 | Filtered chunks + correct scores | Medium-High |

---

## The Hypothesis

### Primary Hypothesis
Zero-shot performance with full transcripts is inflated because the LLM can extract PHQ-8 scores by:
1. Reading Ellie's direct questions about symptoms
2. Reading participant's direct answers
3. Using Ellie's probing patterns as evidence (deeper probing = more symptoms)

This is NOT "cheating" in the sense of a bug - it's using the interview as designed. But it means **zero-shot with full transcripts has a structural advantage** that:
1. Doesn't generalize to other interview formats
2. Measures "can you read structured questions" not "can you assess depression"

### Secondary Hypothesis
The TRUE baseline should be **participant-only zero-shot**. This removes the Ellie shortcut and tests what we actually care about: can the LLM understand depression signals from patient speech?

---

## Testing The Hypothesis

### Experiment 1: Participant-Only Transcripts
Strip Ellie's utterances from transcripts. Compare zero-shot performance:
- If performance drops significantly → Ellie's questions provide shortcuts (hypothesis confirmed)
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

---

## Implications

### If Hypothesis is True:
1. The original paper's results may be inflated by Ellie's shortcuts
2. Zero-shot with participant-only is the TRUE baseline
3. Few-shot with proper implementation (Spec 35+36) may then show improvement

### For Our Reproduction:
1. Run participant-only as the valid baseline
2. Compare against few-shot with Spec 35+36 enabled
3. Report both full-transcript and participant-only results

---

## Related Documentation

- `docs/brainstorming/daic-woz-preprocessing.md` - Full analysis of Ellie inclusion
- `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md` - Burdisso paper
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` - Companion hypothesis on few-shot issues
- GitHub Issue #79 - Interviewer utterance bias
- GitHub Issue #69 - Few-shot chunk/score mismatch

---

## Conclusion

We may have been measuring the wrong thing. The question isn't "why is few-shot worse?" but:
1. "Is zero-shot artificially inflated by Ellie's shortcuts?"
2. "What is the TRUE baseline (participant-only)?"

**The DAIC-WOZ dataset's design gives LLMs direct access to PHQ-8-relevant evidence through Ellie's questions. This should be treated as a confounder, not a feature.**

---

*"Maybe we're thinking about it in reverse."* - The shower thought that started this investigation.
