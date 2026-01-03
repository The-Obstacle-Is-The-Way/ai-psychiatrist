# Evidence Extraction Mechanism: How It Actually Works

**Audience**: Anyone wanting to understand the core engineering behind PHQ-8 scoring
**Last Updated**: 2026-01-03

---

## Overview

This document explains how evidence extraction works, why it succeeds or fails, and how that leads to coverage.

---

## The Pipeline (High Level)

```text
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
│              STEP 2: SCORING (LLM)                          │
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

```text
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

### JSON Parsing Robustness (CRITICAL)

**Problem**: LLMs sometimes output malformed JSON (Python-style `True` instead of `true`, missing commas, etc.). This was causing silent data corruption where few-shot mode would degrade to zero-shot without indication.

**Solution (as of 2026-01-03)**:

1. **Ollama `format:"json"`**: Evidence extraction now uses Ollama's grammar-level JSON constraint, which guarantees well-formed JSON at token generation time. See [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs).

2. **Canonical Parser**: All JSON parsing uses `parse_llm_json()` in `responses.py`:
   - Applies tolerant fixups (smart quotes, trailing commas)
   - Falls back to Python literal parsing for `True`/`False`/`None`
   - **NO SILENT FALLBACKS** - raises on failure

3. **No Silent Degradation**: If JSON parsing fails, the system raises an exception instead of silently returning empty evidence. This prevents corrupted research results.

**Code Location**: `src/ai_psychiatrist/infrastructure/llm/responses.py:parse_llm_json()`

**Related**: [ANALYSIS-026](../_bugs/ANALYSIS-026-JSON-PARSING-ARCHITECTURE-AUDIT.md) - Full audit of JSON parsing architecture

---

## Step 2: Scoring (Back to the LLM)

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

> **Note**: The "N/A" response is specific to this evidence-based extraction method. In standard clinical PHQ-8 administration, all items receive a score from 0 to 3 based on patient self-report. Our system returns N/A when insufficient evidence exists in the transcript to make an informed prediction.

### What Determines Score vs N/A

The decision tree:

```text
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

```text
Item Coverage = (Number of participants with a score) / (Total participants)
```

**Example**: Sleep item
- 40 participants got a score (0, 1, 2, or 3)
- 1 participant got N/A
- Sleep coverage = 40/41 = 97.6%

### Per-Participant Coverage

For each participant, across all 8 items:

```text
Participant Coverage = (Items with scores) / 8
```

**Example**: Participant 303
- 4 items scored: Depressed, Sleep, Tired, Failure
- 4 items N/A: NoInterest, Appetite, Concentrating, Moving
- Participant coverage = 4/8 = 50%

### Overall Coverage

Total scored items across all participants:

```text
Overall Coverage = (Total items with scores) / (Total participants × 8)
```

For concrete example runs (including per-item counts and coverage), see:
- `docs/results/run-history.md`
- `docs/results/reproduction-results.md`

Output artifacts are stored locally under `data/outputs/` (gitignored due to DAIC-WOZ licensing; not committed to repo).

---

## What Parameters Affect Extraction?

### Temperature

Note: The paper text does not specify exact sampling settings; the effects below are heuristics and can vary by model/backend. See [Agent Sampling Registry](../configs/agent-sampling-registry.md).

| Value | Effect on Extraction |
|-------|---------------------|
| 0.0 (default) | Conservative and reproducible (greedy decoding); may miss subtle evidence |
| 0.2 | Slightly more permissive; may catch more evidence but increases variability |
| 0.7+ | Too creative, may hallucinate evidence |

### Model Choice

| Model | Extraction Quality |
|-------|-------------------|
| gemma3:27b | Paper’s main baseline model family; exact behavior depends on build/quantization/backend |
| MedGemma 27B | Appendix F: lower MAE on the subset with available evidence, but fewer predictions overall (more abstention) |
| Smaller models | Often less robust on nuance (heuristic) |

---

## Summary: The Complete Picture

1. **LLM reads transcript** and extracts quotes per symptom (semantic analysis)
2. **Evidence exists?**
   - Yes → LLM scores it (0-3)
   - No → N/A
3. **Coverage** = percentage of items that got scores instead of N/A

**The key insight**: Extraction depends on:
- Whether the symptom was discussed in the interview
- How well the LLM recognizes relevant language
- Model parameters (temperature, model size)

---

## Code References

| File | What It Does |
|------|--------------|
| `src/ai_psychiatrist/agents/quantitative.py` | Evidence extraction and scoring |
| `src/ai_psychiatrist/agents/prompts/quantitative.py` | Prompt templates |
| `src/ai_psychiatrist/services/embedding.py` | Few-shot retrieval + similarity computation |

---

## Related Documentation

- [Coverage explained](../statistics/coverage.md) - Plain-language coverage explanation
- [Clinical understanding](../clinical/clinical-understanding.md) - Clinical context
- [PHQ-8](../clinical/phq8.md) - PHQ-8 questionnaire details
