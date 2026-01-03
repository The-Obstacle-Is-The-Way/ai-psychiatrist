# Investigation: DAIC-WOZ Preprocessing Gap

**Date**: 2025-12-30
**Status**: BRAINSTORMING - Needs First-Principles Analysis
**Severity**: Medium-High - Potential Methodological Oversight in Paper

---

## Executive Summary

The codebase includes **both interviewer (Ellie) AND participant utterances** in transcripts with **no speaker filtering**. Recent research (Burdisso et al., 2024) urges caution with interviewer prompts, showing models can exploit interviewer patterns rather than genuine clinical signals.

**Key Question**: Should we filter to participant-only utterances, or does LLM semantic understanding make this unnecessary?

---

## 1. Current Implementation

**File**: `src/ai_psychiatrist/services/transcript.py:136-153`

```python
def _parse_daic_woz_transcript(self, path: Path) -> str:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["speaker", "value"])  # Only removes NaN
    df["dialogue"] = df["speaker"] + ": " + df["value"]
    return "\n".join(df["dialogue"].tolist())
```

**Result**: Most transcripts include **both** Ellie + Participant utterances (proportions vary). In the current local dataset, **3 sessions** (451, 458, 480) contain **only participant** utterances due to missing virtual agent transcriptions. Nothing is speaker-filtered today.

---

## 2. Research Evidence (2024-2025)

### 2.1 The Interviewer Bias Problem

**Source**: [Burdisso et al., ACL ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/)

| Model | Input | Macro F1 (Avg) |
|-------|-------|----------|
| P-longBERT | Participant only | 0.72 |
| E-longBERT | Ellie only | 0.84 |
| P-GCN | Participant only | 0.85 |
| E-GCN | Ellie only | 0.88 |
| Ensemble (P-GCN + E-GCN) | Both models combined | 0.90 |

**Critical finding**: Interviewer-only models *outperform* participant-only because they exploit targeted follow-up probing questions as shortcuts. Notably, the paper explicitly shows (Fig. 2) that "Have you been diagnosed with depression?" is **not** part of the highlighted shortcut region—the model disregards it in that example. Instead, the discriminative region includes **follow-up questions** that probe deeper into mental health history:
- "what got you to seek help"
- "do you still go to therapy now"
- "why did you stop"
- "how has seeing a therapist affected you"
- "do you have disturbing thoughts"

> *"models using interviewer's prompts learn to focus on a specific region of the interviews, where questions about past experiences with mental health issues are asked, and use them as discriminative shortcuts to detect depressed participants."* (Abstract)

### 2.2 Why Participant-Only Scores Lower (But Is More Valid)

The paper's heatmap analysis reveals:

- **E-GCN (Ellie only)**: Focuses on **a single segment** appearing "after halfway through interviews" - where Ellie asks about mental health history
- **P-GCN (Participant only)**: Distributes attention **across the entire interview**

**Interpretation**: Participant-only models learn **genuine clinical signals** distributed throughout conversation. Ellie-only models learn a **shortcut** - one specific region where targeted mental health questions are asked.

The 0.90 F1 for "both" is an **ensemble of two models**, not proof that including Ellie helps a single model. It's combining a biased shortcut learner with a genuine signal learner.

**Conclusion**: Lower F1 for participant-only reflects **harder, more valid learning** - not worse performance. The model must learn actual depression markers rather than exploiting interviewer protocol.

### 2.3 Prevalence of This Error

**Source**: [MDPI Applied Sciences 2025](https://www.mdpi.com/2076-3417/16/1/422)

**42.4% of DAIC-WOZ studies** include interviewer utterances without justification - a documented methodological pitfall.

### 2.4 Known Dataset Issues

These issues are documented in the Bailey/Plumbley preprocessing tool (`_reference/daic_woz_process/config_files/config_process.py`) and are also present in our current `data/transcripts/`.

| Interview ID | Issue | Recommended Action |
|--------------|-------|-------------------|
| 373 | Human enters room to fix Ellie (interruption) | Remove the interruption window (Bailey/Plumbley: 395–428s) |
| 444 | Phone alarm interruption | Remove the interruption window (Bailey/Plumbley: 286–387s) |
| 451, 458, 480 | Missing Ellie transcripts | Handle gracefully |
| 318, 321, 341, 362 | Transcript timing misaligned with audio | Only relevant if using audio; apply time shift correction |
| 409 | Wrong `PHQ8_Binary` label (score=10 but binary=0) | Deterministic fix: set `PHQ8_Binary=1` when `PHQ8_Score>=10` |
| All | Pre-interview chatter | Remove (not clinical) |
| 342, 394, 398, 460 | Missing participant IDs | Expected gaps (absent from `data/transcripts/`) |

---

## 3. What the Original Paper Says

The paper states:

> *"We prompted Gemma 3 27B to assume the role of a psychiatrist tasked with generating an objective and concise assessment based on the **participant's interview transcript**"*

**Ambiguity**: Does "participant's interview transcript" mean:
1. The transcript from the participant's interview (entire conversation) ← current implementation
2. Only the participant's utterances ← research recommendation

The prompts DO explicitly identify speaker roles, suggesting intentional inclusion.

---

## 4. LLM vs Traditional ML: Why This Is Nuanced

| Aspect | Traditional ML | LLM-Based (This System) |
|--------|---------------|------------------------|
| Speaker understanding | Encodes both equally | Semantically understands speaker roles |
| Context | Loses meaning without dialogue flow | Can use Q&A pairs for understanding |
| Retrieval | Embeddings can't distinguish speakers | Item-tagged filtering helps |
| Risk | High shortcut exploitation | Lower, but still present in embeddings |

### Arguments FOR Keeping Both Speakers

1. LLMs understand speaker roles semantically
2. Interviewer questions provide context (what prompted the answer?)
3. Few-shot retrieval benefits from full dialogue context

### Arguments FOR Participant-Only (Research Consensus)

1. 50% of content is non-clinical noise
2. Chunk efficiency: 8-line chunks have diluted clinical signal
3. Embeddings may encode interviewer patterns as shortcuts
4. Research strongly recommends against inclusion
5. **Beats the purpose**: We want to assess the *patient*, not the interview structure

---

## 5. First-Principles Analysis

### What Are We Actually Trying to Do?

**Goal**: Predict PHQ-8 scores from what the *patient* says about their symptoms.

**The patient's clinical signal** = their words about:
- Sleep disturbances
- Mood states
- Energy levels
- Appetite changes
- Self-perception
- Concentration
- Psychomotor changes
- Suicidal ideation

**Ellie's questions** = interviewer protocol, NOT patient data.

### The Fundamental Problem (Leaky Signal)

When we embed chunks like:

```text
Ellie: what got you to seek help
Participant: uh i was just missing a lot of school
Ellie: do you still go to therapy now
Participant: no i haven't gone to therapy
Ellie: why did you stop
Participant: i didn't feel it was helping me at all
```

The embedding captures Ellie's **follow-up probing questions** about therapy history—questions that only appear when participants indicate prior mental health treatment. This is a **leaky signal**.

**Note**: The paper explicitly shows (Fig. 2) that "Have you been diagnosed with depression?" is not part of the highlighted shortcut region—the model disregards it in that example. The shortcut region is the **follow-up probing questions** that come after, which indicate Ellie detected something worth exploring.

**The logic models learn**:

```text
IF "do you still go to therapy now" or "why did you stop" appears in transcript
THEN predict depression (because these questions only get asked to certain participants)
```

That's not clinical assessment—that's pattern matching on interviewer protocol.

### Why This Matters for Embeddings Specifically

When we do similarity search, the embedding vector captures Ellie's probing questions ("therapy", "seek help", "why did you stop") semantically. We may retrieve chunks dominated by **interviewer follow-up patterns** rather than chunks where the patient *expresses* depressive symptoms.

**The embedding layer cannot distinguish**:
- "Ellie asked follow-up questions about therapy" (interviewer protocol revealing prior assessment)
- "Patient discussed their therapy experience" (clinical signal)

Both encode similarly in vector space. The leaky signal exists at the retrieval layer, even if the LLM can distinguish speaker roles in the prompt.

### Practical Implications

If we remove Ellie:

1. **Chunks get denser** - 8 lines of pure patient signal, not 4
2. **Embeddings become purer** - No interviewer protocol patterns
3. **Retrieval improves** - Forced to find actual symptom expressions
4. **Coverage might improve** - Less noise, clearer evidence extraction

### Psychiatrist Perspective

A psychiatrist assessing a patient focuses on:
- **What the patient says** (chief complaint, symptoms)
- **How they say it** (affect, speech patterns)
- **What they don't say** (avoidance, minimization)

NOT on what questions the interviewer asked.

---

## 6. Recommended Preprocessing

**Reference implementation**: Bailey/Plumbley preprocessing tool (mirrored in `_reference/daic_woz_process/`), primarily:

- `_reference/daic_woz_process/utils/utilities.py::transcript_text_processing`
- `_reference/daic_woz_process/config_files/config_process.py`

```python
# Sketch of the Bailey/Plumbley behavior (not exact code):
# 1) Drop anything before Ellie starts (or, for missing-Ellie sessions 451/458/480, drop sync markers)
# 2) Filter to participant-only utterances for text features
# 3) Remove known interruption windows (drop affected participant utterances):
#    - 373: [395, 428] seconds
#    - 444: [286, 387] seconds
# 4) Remove sync markers (e.g., <sync>, <synch>) and strip tokens containing < > [ ]
#    Also removes placeholder/unknown tokens (e.g., 'xxx', 'xxxx') for classical feature extraction
# 5) Fix known label bug: participant 409 has PHQ8_Score=10 but PHQ8_Binary=0 → should be 1
#
# NOTE: The tool does NOT explicitly remove disfluencies like "um"/"uh" by default.
# If we want that, it should be an explicit, separately ablated option.
```

---

## 7. Impact on Current System

### Current Chunks (Mixed)

```
chunk_preview="Ellie: that's good \nEllie: how are you at controlling...
Participant: mm\nParticipant: i try not to get angry um..."
```

### If Participant-Only

```
chunk_preview="mm\ni try not to get angry um\nbecause i have a really bad temper..."
```

**Density of clinical signal increases 2x**.

### Embedding Quality

- Current: May retrieve chunks dominated by Ellie's targeted questions
- Participant-only: Forced to find genuine symptom expressions

---

## 8. Decision Matrix

| Option | Pros | Cons | Risk |
|--------|------|------|------|
| **Keep Both (Status Quo)** | Legacy baseline, LLM context | Research warns against, 50% noise | Medium |
| **Participant Only** | Research best practice, denser signal | Loses question context | Low |
| **Hybrid (Tag Speakers)** | Best of both, LLM can filter | More complex, untested | Medium |

---

## 9. Proposed Experiment

**Spec 35+ Candidate**: Run ablation study

1. **Baseline**: Current implementation (both speakers)
2. **Variant A**: Participant-only transcripts
3. **Variant B**: Participant-only + disfluency removal

**Metrics**:
- MAE on PHQ-8 scores
- Coverage (% non-N/A predictions)
- Retrieval quality (similarity scores)
- Qualitative: Do retrieved chunks contain actual symptoms?

---

## 10. Sources

- [DAIC-WOZ: On the Validity of Using Therapist Prompts (ACL 2024)](https://aclanthology.org/2024.clinicalnlp-1.8/)
- [Common Pitfalls in DAIC-WOZ (MDPI 2025)](https://www.mdpi.com/2076-3417/16/1/422)
- [DAIC-WOZ Preprocessing Tool](https://github.com/adbailey1/daic_woz_process)
- [Bias in DAIC-WOZ Research Code](https://github.com/idiap/bias_in_daic-woz)
- [DAIC-WOZ Official Database](https://dcapswoz.ict.usc.edu/)

---

## 11. Counter-Arguments (Why LLMs Might Be Different)

The one argument FOR keeping Ellie: LLMs understand "Ellie: are you depressed?" is a question, not a patient statement. The prompts explicitly say who's who.

**BUT**: This doesn't help at the **embedding/retrieval layer**. Embeddings don't semantically distinguish "Ellie asked about X" from "Patient said X". The leaky signal exists in the vector space before the LLM ever sees the retrieved examples.

This is the key insight: even if the LLM can distinguish speakers in the final prompt, the retrieval step has already been corrupted by interviewer content.

---

## 12. Verdict: Remove Ellie

**Source of truth**: The 2024 ACL ClinicalNLP paper ([Burdisso et al.](https://aclanthology.org/2024.clinicalnlp-1.8/)) is the most rigorous recent analysis. Their conclusion is unambiguous:

> *"More broadly, our findings underline the need for caution when incorporating interviewers' prompts into mental health diagnostic models. Interviewers often strategically adapt their questioning to probe for potential symptoms. As a result, models may learn to exploit these targeted prompts as discriminative shortcuts, rather than learning to characterize the language and behavior that are truly indicative of mental health conditions."* (Section 6: Conclusions)

The paper authors likely didn't address this explicitly because:

1. They may not have known about the 2024 bias research (it's recent)
2. They may have assumed LLM semantic understanding handles it
3. It wasn't their research focus (they focused on multi-agent architecture)

**From first principles + research evidence + clinical perspective: participant-only is the correct approach.**

---

## 13. Open Questions

1. Did the original paper authors intentionally include Ellie, or was this an oversight?
2. Does LLM semantic understanding mitigate the shortcut problem at inference time?
3. Would regenerating embeddings with participant-only chunks improve retrieval?
4. Should we also remove disfluencies (um, uh, mhm)?
5. Do the known problematic interviews (373, 444, 451, 458, 480) affect current results?

---

## 14. Recommendation (Research-Aligned Default)

If the goal is **validity and generalization** (not just maximizing metrics on DAIC-WOZ), the safest default is:

1. **Participant-only for embeddings/retrieval/indexing** (at minimum), since the retrieval layer can be biased by interviewer follow-up patterns before the LLM ever sees the prompt.
2. Treat **both-speakers transcripts as a legacy baseline**, and explicitly label any gains as potentially influenced by interviewer-protocol leakage.
3. If participant-only harms interpretability for short answers (e.g., "yes/no"), test a compromise variant:
   - **Participant-only utterances + minimal question context**, e.g., include the immediately preceding Ellie prompt line for each participant response (Q/A context), while still excluding other interviewer content from indexing.

This should be handled as an **ablation** and reported transparently:
- Baseline: both speakers (status quo)
- Variant A: participant-only
- Variant B: participant-only + minimal question context

---

## Notes

*This investigation was prompted by noticing the paper doesn't explicitly address speaker filtering, and discovering 2024 research that strongly recommends against including interviewer utterances.*

*Related GitHub Issue: [#79 - Interviewer utterance inclusion may introduce retrieval bias](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/79)*
