# Evidence Extraction Mechanism: How It Actually Works

**Audience**: Anyone wanting to understand the core engineering behind PHQ-8 scoring
**Last Updated**: 2025-12-23

---

## Overview

This document explains how evidence extraction works, why it succeeds or fails, and how that leads to coverage.

---

## The Pipeline (High Level)

```
┌─────────────────────────────────────────────────────────────┐
│                     INTERVIEW TRANSCRIPT                    │
│  "I've been feeling really down lately. Can't sleep at all. │
│   Work is stressful but I still enjoy my hobbies..."        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: EVIDENCE EXTRACTION (LLM)              │
│                                                             │
│  LLM reads entire transcript and extracts quotes for each   │
│  of the 8 PHQ-8 items.                                      │
│                                                             │
│  Output: JSON with arrays of evidence per item              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: KEYWORD BACKFILL (Rule-Based)          │
│                                                             │
│  For any item with insufficient evidence:                   │
│  Search transcript for keywords like "can't sleep",         │
│  "exhausted", "hopeless" and add matching sentences.        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: SCORING (LLM)                          │
│                                                             │
│  For each item WITH evidence:                               │
│    → Score 0-3 based on frequency/severity                  │
│                                                             │
│  For each item WITHOUT evidence:                            │
│    → Return "N/A" (cannot assess)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL PHQ-8 ASSESSMENT                   │
│                                                             │
│  NoInterest: 2    Depressed: 1    Sleep: 2    Tired: N/A    │
│  Appetite: N/A    Failure: 1      Concentrating: 0          │
│  Moving: N/A                                                │
│                                                             │
│  Coverage: 5/8 = 62.5%                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Evidence Extraction (The LLM Part)

### What Happens

The LLM receives a prompt containing:
1. The full interview transcript
2. Instructions to find quotes for each PHQ-8 domain
3. Expected JSON output format

### The Actual Prompt

From `src/ai_psychiatrist/agents/prompts/quantitative.py`:

```
Analyze the following therapy transcript and extract specific text chunks
that provide evidence for each PHQ-8 domain.

PHQ-8 domains:
- nointerest: little interest or pleasure in activities
- depressed: feeling down, depressed, or hopeless
- sleep: sleep problems (trouble falling/staying asleep or sleeping too much)
- tired: feeling tired or having little energy
- appetite: appetite changes (poor appetite or overeating)
- failure: negative self-perception or feeling like a failure
- concentrating: trouble concentrating on tasks
- moving: psychomotor changes (moving/speaking slowly or restlessness)

Return a JSON object with arrays of relevant transcript quotes for each domain.
```

### What the LLM Does Internally

The LLM **semantically analyzes** the transcript:

1. Reads the entire text
2. For each sentence, determines which PHQ-8 domain it relates to (if any)
3. Groups quotes by domain
4. Returns structured JSON

**Example Analysis:**

| Transcript Quote | LLM's Semantic Understanding | Assigned Domain |
|-----------------|------------------------------|-----------------|
| "I can't sleep at night" | Mentions sleep difficulty | PHQ8_Sleep |
| "I feel worthless" | Negative self-perception | PHQ8_Failure |
| "I love playing guitar" | Positive interest mention | (none - positive) |
| "My job is stressful" | Work stress, not PHQ symptom | (none) |

### Why Extraction Can Fail

| Failure Type | What Happens | Example |
|--------------|--------------|---------|
| **Not discussed** | Patient never mentioned that symptom | No mention of appetite → no Appetite evidence |
| **LLM misses it** | LLM doesn't recognize the relevance | "I'm so drained" not mapped to Tired |
| **Ambiguous language** | Could be interpreted multiple ways | "I'm fine" - denial or truth? |
| **JSON parsing error** | LLM returns malformed output | Missing quote, bad escaping |

---

## Step 2: Keyword Backfill (The Rule-Based Safety Net)

### Why We Need It

LLM extraction isn't perfect. Sometimes it misses obvious evidence like "I'm exhausted" for the Tired domain.

### How It Works

From `src/ai_psychiatrist/agents/quantitative.py` (`QuantitativeAssessmentAgent._keyword_backfill`):

```python
parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
sentences = [p.strip() for p in parts if p and len(p.strip()) > 0]

out = {k: list(v) for k, v in current.items()}

for key, keywords in DOMAIN_KEYWORDS.items():
    need = max(0, cap - len(out.get(key, [])))
    if need == 0:
        continue

    hits: list[str] = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in keywords):
            hits.append(sent)
        if len(hits) >= need:
            break

    if hits:
        existing = set(out.get(key, []))
        merged = out.get(key, []) + [h for h in hits if h not in existing]
        out[key] = merged[:cap]

return out
```

The real implementation uses a simple sentence splitter and caps the number of
backfilled sentences per domain to keep prompts bounded.

### The Keyword Lists

From `src/ai_psychiatrist/resources/phq8_keywords.yaml`:

| Domain | Example Keywords |
|--------|------------------|
| Sleep | "can't sleep", "insomnia", "wake up tired" |
| Tired | "exhausted", "no energy", "feeling tired" |
| Depressed | "hopeless", "crying", "feeling down" |
| Appetite | "no appetite", "overeating", "lost weight" |

### Why This Increases Coverage

Without backfill:
- LLM misses "I'm so tired" → Tired gets N/A

With backfill:
- LLM misses it, but "tired" is a keyword
- Backfill finds "I'm so tired" → Tired gets evidence → Tired gets scored

Keyword backfill can increase coverage relative to a pure LLM-only evidence
extraction approach. The paper reports that **in ~50% of cases** the model was
unable to provide a prediction due to insufficient evidence (Section 3.2). In our
example reproduction run, overall item prediction coverage was **74.1%** (see
`docs/results/reproduction-notes.md`). Attribution requires an ablation run with
keyword backfill disabled.

---

## Step 3: Scoring (Back to the LLM)

### What Happens

For items WITH evidence, the LLM is asked:
> "Based on this evidence, what score (0-3) should this symptom receive?"

For items WITHOUT evidence:
> LLM returns "N/A" (cannot assess without evidence)

### Scoring Criteria

| Score | Meaning | Frequency |
|-------|---------|-----------|
| 0 | Not at all | 0-1 days in past 2 weeks |
| 1 | Several days | 2-6 days |
| 2 | More than half the days | 7-11 days |
| 3 | Nearly every day | 12-14 days |
| N/A | Cannot assess | No evidence found |

### What Determines Score vs N/A

The decision tree:

```
Has evidence for this item?
├── YES → Attempt scoring (0-3)
│         └── Does evidence indicate frequency?
│             ├── YES → Assign 0, 1, 2, or 3
│             └── NO  → Conservative: likely 0 or 1
└── NO  → Return N/A
```

---

## How Coverage is Calculated

### Per-Item Coverage

For each PHQ-8 item, across all participants:

```
Item Coverage = (Number of participants with a score) / (Total participants)
```

**Example**: Sleep item
- 40 participants got a score (0, 1, 2, or 3)
- 1 participant got N/A
- Sleep coverage = 40/41 = 97.6%

### Per-Participant Coverage

For each participant, across all 8 items:

```
Participant Coverage = (Items with scores) / 8
```

**Example**: Participant 303
- 4 items scored: Depressed, Sleep, Tired, Failure
- 4 items N/A: NoInterest, Appetite, Concentrating, Moving
- Participant coverage = 4/8 = 50%

### Overall Coverage

Total scored items across all participants:

```
Overall Coverage = (Total items with scores) / (Total participants × 8)
```

For a concrete example run (including per-item counts and coverage), see
`docs/results/reproduction-notes.md` and the corresponding JSON artifact under
`data/outputs/`.

---

## What Parameters Affect Extraction?

### Temperature

| Value | Effect on Extraction |
|-------|---------------------|
| 0.0 | Very conservative, may miss subtle evidence |
| 0.2 (default) | Slightly creative, catches more evidence |
| 0.7+ | Too creative, may hallucinate evidence |

### Model Choice

| Model | Extraction Quality |
|-------|-------------------|
| gemma3:27b | Good balance of coverage and accuracy |
| MedGemma 27B | Lower coverage (more N/A); Appendix F reports lower MAE on the subset with evidence |
| Smaller models | May miss nuanced evidence |

### Keyword List Quality

Better keywords → more backfill matches → higher coverage

Our keyword list is hand-curated and includes collision-avoidance heuristics (for
substring matching). It is not a substitute for clinical validation; treat it as a
best-effort fallback mechanism. See `src/ai_psychiatrist/resources/phq8_keywords.yaml`.

---

## Summary: The Complete Picture

1. **LLM reads transcript** and extracts quotes per symptom (semantic analysis)
2. **Keyword backfill** catches what LLM missed (rule-based)
3. **Evidence exists?**
   - Yes → LLM scores it (0-3)
   - No → N/A
4. **Coverage** = percentage of items that got scores instead of N/A

**The key insight**: Extraction depends on:
- Whether the symptom was discussed in the interview
- How well the LLM recognizes relevant language
- How comprehensive our keyword list is
- Model parameters (temperature, model size)

---

## Code References

| File | What It Does |
|------|--------------|
| `src/ai_psychiatrist/agents/quantitative.py` | Evidence extraction, keyword backfill, and scoring |
| `src/ai_psychiatrist/agents/prompts/quantitative.py` | Prompt templates + keyword resource loader |
| `src/ai_psychiatrist/resources/phq8_keywords.yaml` | Keyword lists (packaged resource) |
| `src/ai_psychiatrist/services/embedding.py` | Few-shot retrieval + similarity computation |

---

## Related Documentation

- [coverage-explained.md](./coverage-explained.md) - Plain-language coverage explanation
- [clinical-understanding.md](./clinical-understanding.md) - Clinical context
- [phq8.md](./phq8.md) - PHQ-8 questionnaire details
