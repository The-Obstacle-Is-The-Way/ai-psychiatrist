# Hypothesis: Few-Shot Design Flaw (And How To Fix It)

**Date**: 2025-12-30
**Status**: Under Investigation
**Origin**: First-principles analysis of few-shot methodology

---

## Executive Summary

The paper's few-shot implementation has a **fundamental design flaw**: participant-level PHQ-8 scores are assigned to individual chunks regardless of chunk content. This creates noisy, misleading examples.

**However**: Few-shot/RAG is NOT worthless. It provides:
1. **Calibration for small models** (Gemma 27B needs examples; GPT-4 may not)
2. **Explainability** (grounded, auditable reasoning vs. hallucinated CoT)
3. **Reproducibility** (same retrieval = same explanation)

**Status check (codebase)**:
- Spec 34 (item-tag filtering), Spec 35 (chunk-level scoring), and Spec 36 (CRAG-style validation) are **implemented**.
- Spec 35 requires a one-time preprocessing step to generate the `<embeddings>.chunk_scores.json` sidecar
  (see `PROBLEM-SPEC35-SCORER-MODEL-GAP.md`).
  - The spec defaults to a disjoint scorer model for defensibility.
  - The script supports `--allow-same-model` so we can ablate “same vs disjoint” empirically.

---

## Three Questions Answered

### Q1: Can Zero-Shot "Cheat"?

**YES.** See `HYPOTHESIS-ZERO-SHOT-INFLATION.md` for full analysis.

The LLM can read Ellie's direct symptom questions as shortcuts. Per the Burdisso paper:
> "Models using interviewer's prompts learn to focus on a specific region of the interviews... and use them as discriminative shortcuts."

**Location**: `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md`

---

### Q2: Is Few-Shot Done Incorrectly?

**YES - Design Flaw, Not Code Bug.**

#### How PHQ-8 Works
```text
PHQ-8 = 8 items (domains), each scored 0-3
Total score = sum of all 8 items = 0-24

Items: NoInterest, Depressed, Sleep, Tired, Appetite, Failure, Concentrating, Moving
```

#### How Chunks Are Created
Transcripts are split into **8-line sliding windows** (step=2):
```text
Participant 300's transcript:
Line 1-8:   CHUNK 0 (may be about anything)
Line 3-10:  CHUNK 1 (may be about anything)
...
Line 195-202: CHUNK 95 (maybe discusses sleep)
```

**Result**: ~100 chunks per participant, but only a FEW actually discuss any specific symptom.

#### The Flaw: Score Assignment

From `src/ai_psychiatrist/services/reference_store.py:976` (paper-parity `reference_score_source="participant"`):
```python
def get_score(self, participant_id: int, item: PHQ8Item) -> int | None:
    df = self._load_scores()
    if df.empty:
        return None

    row = df[df["Participant_ID"] == participant_id]

    if row.empty:
        return None

    col_name = PHQ8_COLUMN_MAP.get(item)  # e.g., "PHQ8_Sleep"
    if col_name is None or col_name not in row.columns:
        return None

    try:
        return int(row[col_name].iloc[0])  # Participant-level item score
    except (ValueError, TypeError):
        return None
```

**EVERY chunk from a participant gets the SAME score**, regardless of content.

#### Visual Example

```text
Chunk 5 (about career goals):
"Ellie: what's your dream job
Participant: open a business
Ellie: do you travel
Participant: no"

→ Gets assigned: "PHQ8_Sleep Score: 2"  ← NOTHING ABOUT SLEEP!
```

```text
Chunk 95 (actually about sleep):
"Ellie: have you had trouble sleeping
Participant: yes every night i lie awake"

→ Gets assigned: "PHQ8_Sleep Score: 2"  ← CORRECT
```

Both chunks get the SAME score because it's the participant's overall score, not the chunk's content.

---

### Q3: Is There A Better Approach?

**YES - And We've Already Designed It.**

#### The 2025 Research Landscape

| Approach | Result | Source |
|----------|--------|--------|
| Naive RAG | 78.90% F1 | RED paper (depression detection) |
| RAG + Judge/Filtering | **90.00% F1** | RED paper (depression detection) |

Key 2025 papers:
- [RED: Personalized RAG for Depression](https://arxiv.org/html/2503.01315) - Symptom-aligned retrieval + Judge module
- [Adaptive RAG for Mental Health](https://arxiv.org/html/2501.00982v1) - Questionnaire-grounded retrieval
- [GPT-4 Clinical Depression](https://arxiv.org/html/2501.00199) - Zero-shot GPT-4 beats few-shot GPT-3.5

#### Our Solution: Specs 34 + 35 + 36

| Spec | What It Does | Fixes |
|------|--------------|-------|
| **Spec 34** | Tag chunks with relevant PHQ-8 items at index time | Only retrieve Sleep-tagged chunks for Sleep queries |
| **Spec 35** | Score each chunk individually via LLM | Chunks get accurate, content-based scores |
| **Spec 36** | Validate references at query time (CRAG-style) | Reject irrelevant/contradictory chunks before use (does not create new scores) |

##### Together = CRAG-Style RAG Pipeline

Important nuance:
- Spec 36 is a *filter* (relevance/contradiction checking) and cannot magically make a participant-level
  `reference_score` correct for a chunk.
- The only automated fix for the score/label mismatch is Spec 35 (or human-curated chunk scores).

```text
Naive Few-Shot (paper)           = Naive RAG
   ↓ add Spec 34 (tag filter)    = Better RAG
   ↓ add Spec 35 (chunk scoring) = Even Better RAG
   ↓ add Spec 36 (validation)    = CRAG (2025 gold standard)
```

---

## Why Few-Shot Still Matters

### Model Size Dependency

| Model Size | Few-Shot Value | Reason |
|------------|----------------|--------|
| Small (Gemma 27B, local) | **HIGH** | Needs calibration examples |
| Large (GPT-4, frontier) | Lower | Has already learned patterns |

**From GitHub Issue #40**:
> "Small models + RAG becomes a genuine value proposition... Consumer hardware deployment is essential [for resource-limited settings]."

### Explainability Value

**From GitHub Issue #39**:

| Property | Chain-of-Thought | RAG/CRAG |
|----------|------------------|----------|
| Reproducibility | Varies between runs | Fixed with same index |
| Grounding | Generated rationalization | Anchored to real examples |
| Verifiability | Cannot verify reasoning | Can examine retrieved examples |
| Auditability | May change on re-run | Citable, stable |

> "RAG-based explainability provides something that chain-of-thought prompting fundamentally cannot — **grounded, verifiable clinical reasoning**."

---

## What CORRECT Few-Shot Would Look Like

### Option A: Full Transcript Examples
```xml
<Reference Examples>
This is a transcript of someone who scored PHQ8_Sleep = 3:
[Full transcript showing clear severe sleep issues]

This is a transcript of someone who scored PHQ8_Sleep = 0:
[Full transcript with no sleep issues mentioned]
</Reference Examples>
```

### Option B: Curated Symptom-Specific Examples
```xml
<Reference Examples for PHQ8_Sleep>
Score 3 (Nearly every day):
"I haven't slept in days, maybe 2 hours a night if I'm lucky"

Score 0 (Not at all):
"I sleep great, 8 hours every night, no problems"
</Reference Examples>
```

### Option C: Spec 35 + 36 (Automated)
- **Spec 35**: LLM scores each chunk ("What does THIS chunk suggest for Sleep?")
- **Spec 36**: LLM validates ("Is this chunk about Sleep?")

This matches what the RED paper calls "Judge module + symptom-aligned retrieval."

**Trade-off**: More LLM calls, but fully automated and scalable.

---

## The Core Insight

### Embedding Similarity ≠ Clinical Relevance

A chunk saying "I sleep fine" and "I can't sleep" are both about sleep. Both might be retrieved for a sleep query. But they describe opposite severities.

The paper's methodology retrieves by **topic similarity** but labels by **participant severity**. These don't match at the chunk level.

**Spec 35 + 36 fixes this** by:
1. Scoring chunks based on their actual content (Spec 35)
2. Validating relevance before use (Spec 36)

---

## Recommendations

### Immediate
1. **Enable Spec 36** as an ablation if runtime budget allows (it’s a filter, not a relabeler)
2. **Run participant-only** as the true baseline
3. **Compare**: Zero-shot (participant-only) vs Few-shot + Spec 36

### Future
1. **Enable Spec 35** for chunk-level scoring (requires scorer model + sidecar generation)
2. **Full CRAG pipeline**: Spec 34 + 35 + 36
3. **Consider**: Is the LLM overhead worth it vs zero-shot?

### Experimental Matrix

| Configuration | What It Tests |
|---------------|---------------|
| Zero-shot (full transcript) | Current baseline (inflated) |
| Zero-shot (participant-only) | TRUE baseline |
| Few-shot (current) | Broken implementation |
| Few-shot + Spec 36 | CRAG (filtered) |
| Few-shot + Spec 35 + 36 | Full CRAG (proper scores + filtering) |

---

## Related Documentation

- `HYPOTHESIS-ZERO-SHOT-INFLATION.md` - Zero-shot analysis
- `docs/brainstorming/daic-woz-preprocessing.md` - Ellie inclusion analysis
- `docs/reference/chunk-scoring.md` - Chunk-level scoring (Spec 35)
- `docs/guides/crag-validation-guide.md` - CRAG reference validation (Spec 36)
- `_literature/markdown/daic-woz-prompts/daic-woz-prompts.md` - Burdisso paper
- GitHub Issue #69 - Few-shot chunk/score mismatch
- GitHub Issue #40 - Model size and local inference value
- GitHub Issue #39 - RAG explainability value

---

## Conclusion

1. **The paper's few-shot is flawed** - chunks get wrong scores
2. **We've already designed the fix** - Spec 35 + 36 = CRAG
3. **Few-shot still has value** for small models and explainability
4. **Zero-shot baseline is inflated** by Ellie's shortcuts
5. **TRUE baseline** = participant-only zero-shot

### The Real Question

The question isn't "few-shot vs zero-shot" — it's "broken few-shot vs proper CRAG vs participant-only zero-shot."

---

*"The code may be right, but the behavior may be wrong."* - The insight that started this investigation.
