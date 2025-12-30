# Investigation: DAIC-WOZ Preprocessing Gap

**Date**: 2025-12-30
**Status**: BRAINSTORMING - Needs First-Principles Analysis
**Severity**: Medium-High - Potential Methodological Oversight in Paper

---

## Executive Summary

The codebase includes **both interviewer (Ellie) AND participant utterances** in transcripts with **no speaker filtering**. Recent research (2024) strongly recommends against this for depression detection, as models learn to exploit interviewer patterns rather than genuine clinical signals.

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

**Result**: Each transcript is ~50% Ellie, ~50% Participant. Nothing filtered.

---

## 2. Research Evidence (2024-2025)

### 2.1 The Interviewer Bias Problem

**Source**: [Burdisso et al., ACL ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/)

| Model Input | F1 Score |
|-------------|----------|
| **Interviewer prompts ONLY** | 0.88 |
| **Participant responses ONLY** | 0.85 |

**Critical finding**: Interviewer-only models *outperform* participant-only because they exploit targeted questions like "have you been diagnosed with depression?" as shortcuts.

> *"Models using interviewer's prompts learn to focus on a specific region of the interviews... rather than learning to characterize the language and behavior that are genuinely indicative of the patient's mental health condition."*

### 2.2 Prevalence of This Error

**Source**: [MDPI Applied Sciences 2025](https://www.mdpi.com/2076-3417/16/1/422)

**42.4% of DAIC-WOZ studies** include interviewer utterances without justification - a documented methodological pitfall.

### 2.3 Known Dataset Issues

| Interview ID | Issue | Recommended Action |
|--------------|-------|-------------------|
| 373, 444 | Long interruptions | Remove or segment |
| 451, 458, 480 | Missing Ellie transcripts | Handle gracefully |
| All | Pre-interview chatter | Remove (not clinical) |

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

### The Fundamental Problem

When we embed chunks like:
```
Ellie: have you been diagnosed with depression
Participant: yes i was diagnosed three years ago
Ellie: i see
Ellie: how long ago was that
```

The embedding captures "diagnosed with depression" from Ellie's question - NOT from the patient's clinical presentation. This is a **leaky signal**.

### Psychiatrist Perspective

A psychiatrist assessing a patient focuses on:
- **What the patient says** (chief complaint, symptoms)
- **How they say it** (affect, speech patterns)
- **What they don't say** (avoidance, minimization)

NOT on what questions the interviewer asked.

---

## 6. Recommended Preprocessing

**Standard practice** per [adbailey1/daic_woz_process](https://github.com/adbailey1/daic_woz_process):

```python
# Filter to participant only
df = df[df['speaker'] == 'Participant']

# Remove disfluencies (optional)
DISFLUENCIES = {'um', 'uh', 'mhm', 'yeah', 'okay', 'mm'}
df = df[~df['value'].str.strip().isin(DISFLUENCIES)]

# Remove known problematic interviews
EXCLUDE_IDS = {373, 444}  # Long interruptions
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
| **Keep Both (Status Quo)** | Paper parity, LLM context | Research warns against, 50% noise | Medium |
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

## 11. Open Questions

1. Did the original paper authors intentionally include Ellie, or was this an oversight?
2. Does LLM semantic understanding mitigate the shortcut problem?
3. Would regenerating embeddings with participant-only chunks improve retrieval?
4. Should we also remove disfluencies (um, uh, mhm)?

---

## Notes

*This investigation was prompted by noticing the paper doesn't explicitly address speaker filtering, and discovering 2024 research that strongly recommends against including interviewer utterances.*
