# Hypotheses Explained: Current State vs Future Improvements

**Status**: Research roadmap document
**Created**: 2026-01-06
**Purpose**: Explain what we have now, what specs 061-063 will add, and what each remaining hypothesis would change

---

## 1. What We Have Now (Current Pipeline)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Evidence Extraction (LLM)
┌────────────────────────────────────────────────────────────────────────┐
│  INPUT: Full transcript                                                │
│  PROMPT: "Extract quotes that support PHQ-8 scoring"                   │
│  OUTPUT: {"PHQ8_Sleep": ["quote1", "quote2"], "PHQ8_Tired": [...]}     │
│                                                                        │
│  PROBLEM: LLM may paraphrase, merge, or synthesize quotes              │
│           (not always verbatim)                                        │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 2: Evidence Grounding (SUBSTRING MATCH)
┌────────────────────────────────────────────────────────────────────────┐
│  FOR EACH extracted quote:                                             │
│    normalize(quote) in normalize(transcript)?                          │
│      YES → Keep quote                                                  │
│      NO  → REJECT as "hallucination"                                   │
│                                                                        │
│  CURRENT RESULT: ~49.5% of quotes REJECTED                             │
│  REASON: LLM paraphrases, doesn't copy verbatim                        │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 3: Query Embedding (Few-shot only)
┌────────────────────────────────────────────────────────────────────────┐
│  FOR EACH PHQ-8 item with surviving evidence:                          │
│    Embed the evidence text → query_vector                              │
│    Find similar chunks in reference corpus                             │
│    Filter by: item tag, similarity > 0.3, char budget                  │
│                                                                        │
│  PROBLEM: Embedding captures TOPIC similarity, not SEVERITY            │
│  "I can't sleep at night" ≈ "I value good rest" (same topic)           │
│  But one is PHQ8_Sleep=3, other is PHQ8_Sleep=0                        │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
STEP 4: LLM Scoring
┌────────────────────────────────────────────────────────────────────────┐
│  PROMPT includes:                                                      │
│    - Full transcript                                                   │
│    - (Few-shot) Reference examples with scores                         │
│    - Instructions: "Only score if FREQUENCY is clear"                  │
│                                                                        │
│  CURRENT BEHAVIOR:                                                     │
│    - If participant says "I've been tired" (no frequency) → N/A        │
│    - If participant says "always tired" → still often N/A              │
│      (prompt is STRICT about explicit frequency)                       │
│                                                                        │
│  RESULT: ~50% abstention (N/A) rate                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Current Results (Run 12 - Valid)

| Metric | Zero-shot | Few-shot |
|--------|-----------|----------|
| MAE | 0.572 | 0.616 |
| Coverage | 48.5% | 46.0% |
| Items with evidence | 32% | 32% |
| Items with references | N/A | 15.2% |

**Key observation**: Few-shot is *worse* than zero-shot. Why? Evidence grounding starves retrieval of data.

> **BUG-035 Note (2026-01-06)**: Run 12 was affected by a prompt confound where few-shot prompts
> differed from zero-shot even when retrieval returned nothing. This has been fixed. Post-fix
> runs are needed to validate true retrieval effects. See [BUG-035](docs/_bugs/BUG-035-FEW-SHOT-PROMPT-CONFOUND.md).

---

## 2. What We'll Have After Specs 061-063

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     AFTER SPECS 061-063                                 │
└─────────────────────────────────────────────────────────────────────────┘

SPEC 063: Severity Inference Prompts
┌────────────────────────────────────────────────────────────────────────┐
│  BEFORE: "Only score if EXPLICIT frequency (e.g., '7 days')"           │
│  AFTER:  "Infer frequency from markers:                                │
│           'always' → 3, 'usually' → 2, 'sometimes' → 1"                │
│                                                                        │
│  EXPECTED: Coverage 48% → 70-80%                                       │
│  RISK: May introduce inference errors (needs ablation)                 │
└────────────────────────────────────────────────────────────────────────┘

SPEC 061: Total Score Prediction
┌────────────────────────────────────────────────────────────────────────┐
│  BEFORE: Predict 8 items (0-3 each), many N/A                          │
│  AFTER:  Option to predict total (0-24) directly                       │
│           - Phase 1: Sum of items (errors average out)                 │
│           - Phase 2: Direct prediction prompt                          │
│                                                                        │
│  EXPECTED: Coverage ~90%+ (one prediction per participant)             │
│  TRADE-OFF: Less interpretable (no item breakdown)                     │
└────────────────────────────────────────────────────────────────────────┘

SPEC 062: Binary Classification
┌────────────────────────────────────────────────────────────────────────┐
│  BEFORE: 8 items × (0-3) = complex output                              │
│  AFTER:  "Depressed" vs "Not depressed" (PHQ-8 ≥ 10)                   │
│                                                                        │
│  EXPECTED: Coverage ~95%+, Paper reports 78% accuracy                  │
│  TRADE-OFF: Least interpretable, but most actionable clinically        │
└────────────────────────────────────────────────────────────────────────┘
```

### What 061-063 FIX

The **output task** problem. They sidestep the frequency issue by:
- Allowing inference (063)
- Aggregating errors (061)
- Simplifying the task (062)

### What 061-063 DON'T FIX

The **pipeline internals** (evidence extraction, grounding, embedding).

---

## 3. The Remaining Hypotheses - Deep Dive

### Hypothesis 7C: Verbatim Quote Finder

**CURRENT STATE**:

```python
# Evidence extraction prompt (simplified)
"""
Extract quotes from this transcript that support PHQ-8 scoring.
For each item, identify relevant evidence and determine the appropriate score.
"""
```

The prompt asks the LLM to **both** extract quotes **and** think about scoring. This creates a mixed objective that incentivizes the model to "clean up" or synthesize quotes to make them more scoreable.

**WHAT 7C WOULD CHANGE**:

```python
# Proposed verbatim-only prompt
"""
Copy EXACT substrings from this transcript that mention:
- Sleep problems or tiredness
- Interest or pleasure in activities
- Mood or feelings
...

RULES:
- Do NOT paraphrase
- Do NOT merge multiple utterances
- Do NOT clean up grammar
- Copy character-for-character
"""
```

**IMPLICATION**:
- **Current**: LLM extracts `"I've been having trouble sleeping lately"` when transcript says `"yeah um i've been having um trouble sleeping you know lately"`
- **After 7C**: LLM copies verbatim `"yeah um i've been having um trouble sleeping you know lately"`

**WOULD IT HELP?**: Probably yes for grounding rate. The ~49.5% rejection rate might drop significantly because quotes would actually substring-match. But the quotes would be messier/less readable.

**EFFORT**: Medium (prompt rewrite + evaluation)

---

### Hypothesis 7A: Fuzzy Evidence Grounding

**CURRENT STATE**:

```python
# evidence_validation.py (simplified)
def is_grounded(quote, transcript):
    return normalize(quote) in normalize(transcript)  # EXACT substring
```

If the LLM extracts `"I have trouble sleeping"` but the transcript says `"I've been having trouble sleeping"`, this **FAILS** because `"have"` ≠ `"having"`.

**WHAT 7A WOULD CHANGE**:

```python
# Fuzzy matching with semantic similarity
def is_grounded(quote, transcript):
    # Try exact first
    if normalize(quote) in normalize(transcript):
        return True
    # Fallback to fuzzy
    similarity = rapidfuzz.ratio(quote, best_matching_segment(transcript))
    return similarity >= 0.85  # or use embedding similarity
```

**IMPLICATION**:
- **Current**: Rejects valid paraphrases as "hallucinations"
- **After 7A**: Accepts semantically equivalent text even if not verbatim

**WOULD IT HELP?**: Yes, would reduce rejection rate. But introduces risk: might accept actual hallucinations (quotes the person never said anything like).

**EFFORT**: Medium (config + code change, needs threshold tuning)

**RELATIONSHIP TO 7C**: These are **alternatives**:
- 7C says "make LLM output verbatim so substring works"
- 7A says "make grounding accept non-verbatim"

You'd implement **ONE**, not both.

---

### Hypothesis 4A/4B: Embedding Captures Topic, Not Severity

**CURRENT STATE**:

```text
Query: "I can't sleep at night, it's terrible"
Reference corpus search finds:
  - "I need my rest because I'm out there driving that bus" (score=3)
  - "I sleep pretty well actually" (score=0)

Both are "about sleep" in embedding space!
```

The embedding model (`qwen3-embedding:8b`) is a **general-purpose** encoder. It clusters by **topic** (sleep, energy, mood) not by **clinical severity**.

**WHAT 4A/4B WOULD CHANGE**:

**4A (Research insight)**: Acknowledge this limitation. Don't expect embeddings to distinguish severity.

**4B (Severity reranking)**:

```python
def rerank_by_severity(matches, query_text):
    for match in matches:
        # Check for severity markers in reference
        severity_score = 0
        if any(w in match.text for w in ["always", "every day", "constantly"]):
            severity_score += 2
        if any(w in match.text for w in ["can't", "unable", "terrible"]):
            severity_score += 1
        # Combine with similarity
        match.adjusted_score = match.similarity * 0.7 + severity_score * 0.3
    return sorted(matches, key=lambda m: m.adjusted_score, reverse=True)
```

**IMPLICATION**:
- **Current**: Retrieved references may be topically similar but severity-mismatched
- **After 4B**: References prioritize severity alignment, not just topic

**WOULD IT HELP?**: Unclear - needs ablation. The paper doesn't report this, and it's not clear if heuristic reranking would improve over pure similarity.

**EFFORT**: Medium (code change + evaluation)

**ALTERNATIVE (4C)**: Use clinical-domain embeddings (ClinicalBERT, PubMedBERT) that might better represent symptom severity. High effort (new embedding generation, full re-evaluation).

---

### Hypothesis 5A: Behavioral Indicators Beyond Verbal Frequency

**CURRENT STATE**:

```text
Prompt: "Only assign scores when evidence clearly indicates FREQUENCY"

Participant transcript shows:
- Very short responses (behavioral withdrawal)
- Long pauses (psychomotor retardation)
- Topic avoidance on pleasure/interest questions
- Flat affect in word choice

Current system: N/A (no explicit frequency mention)
```

A psychiatrist watching this interview would likely score depression symptoms based on **behavioral patterns**, not just what the person explicitly says.

**WHAT 5A WOULD CHANGE**:

```python
# Hypothetical behavioral scoring
"""
In addition to explicit statements, consider:
- Response length patterns (very short answers may indicate withdrawal)
- Topic engagement (avoidance of certain topics)
- Linguistic markers of depression (first-person singular overuse, negative emotion words)
- Interview dynamics (requires interviewer questions for context)
"""
```

**IMPLICATION**:
- **Current**: Only scores what people explicitly say about symptoms
- **After 5A**: Also considers *how* they say it (behavioral/linguistic patterns)

**WOULD IT HELP?**: Theoretically yes, but **HIGH RISK**:
- Requires access to interviewer questions (currently stripped in participant-only mode)
- Linguistic pattern → depression scoring is its own research area
- Much harder to ground/validate
- Could introduce systematic biases

**EFFORT**: High (research project, not a code change)

---

### Hypothesis 7B: Direct Scoring Without Evidence Extraction

**CURRENT STATE**:

```text
Transcript → Extract Evidence → Ground Evidence → Embed → Retrieve → Score
              ↑                    ↑
              50% lost here         50% lost here
```

The evidence extraction step is a **bottleneck** that loses information.

**WHAT 7B WOULD CHANGE**:

```text
Transcript → Direct LLM Scoring (see full text, score directly)
```

Skip evidence extraction entirely. Let the LLM read the whole transcript and score.

**IMPLICATION**:
- **Current**: Evidence extraction acts as interpretability + grounding layer
- **After 7B**: Faster, no bottleneck, but less interpretable

**WOULD IT HELP?**: Maybe, but with trade-offs:
- **PRO**: No evidence bottleneck
- **PRO**: LLM sees full context
- **CON**: Can't explain *why* it scored something (no evidence quotes)
- **CON**: Harder to detect hallucination (no grounding step)
- **CON**: Few-shot becomes harder (what do you retrieve on?)

**EFFORT**: High (architecture change, loses interpretability features)

---

## 4. The Big Picture: Is It Fundamentally Incorrect?

### What's CORRECT About the Current System

| Aspect | Assessment |
|--------|------------|
| Methodological rigor | ✅ Conservative, evidence-grounded |
| Hallucination prevention | ✅ Strict grounding catches fabricated quotes |
| N/A behavior | ✅ Abstaining when uncertain is scientifically correct |
| Selective prediction framing | ✅ Reports coverage + AURC/AUGRC |
| Reproducibility | ✅ Temperature=0, deterministic splits |

### What's LIMITING (Not Incorrect)

| Limitation | Cause | Fix |
|------------|-------|-----|
| ~50% coverage | PHQ-8 requires frequency; transcripts lack it | Spec 063 (inference) |
| Evidence grounding rejects valid paraphrases | Substring matching is strict | Hypothesis 7A or 7C |
| Few-shot ≤ zero-shot | Evidence bottleneck starves retrieval | Hypotheses 7A/7C first |
| Embedding finds topic, not severity | General-purpose embeddings | Hypothesis 4B or 4C |

### The Key Insight

> **The current system is NOT fundamentally incorrect—it's conservative by design.**

It was designed to:
1. Never hallucinate evidence
2. Never assign scores without clear frequency
3. Abstain rather than guess

This is **methodologically sound** but **practically limiting** for a dataset (DAIC-WOZ) that doesn't elicit frequency information.

---

## 5. If We Implemented Everything

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYPOTHETICAL "EVERYTHING FIXED" PIPELINE             │
└─────────────────────────────────────────────────────────────────────────┘

OPTION A: Fix the bottlenecks (7A + 7C + 4B + 063)
┌────────────────────────────────────────────────────────────────────────┐
│  1. Evidence extraction with VERBATIM-ONLY prompt (7C)                 │
│  2. Fuzzy grounding as fallback (7A) - if 7C doesn't fully work        │
│  3. Severity-aware reranking for few-shot (4B)                         │
│  4. Inference-enabled scoring prompts (063)                            │
│                                                                        │
│  Expected: Coverage 70-85%, MAE similar or better                      │
│  Effort: Medium-High                                                   │
└────────────────────────────────────────────────────────────────────────┘

OPTION B: Bypass the pipeline (7B + 061/062)
┌────────────────────────────────────────────────────────────────────────┐
│  1. Direct scoring without evidence extraction (7B)                    │
│  2. Total score or binary output (061/062)                             │
│                                                                        │
│  Expected: Coverage 90%+, interpretability lost                        │
│  Effort: High (architecture change)                                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Recommended Implementation Path

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDED PATH                                 │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: Specs 061-063 (Low risk, high value)
├─ Spec 063 first (prompt-only change, may get 70-80% coverage)
├─ Spec 061 (total score aggregation)
└─ Spec 062 (binary classification)

PHASE 2: Evidence bottleneck (If few-shot still underperforms)
├─ Hypothesis 7C (verbatim prompt) OR
└─ Hypothesis 7A (fuzzy grounding)

PHASE 3: Embedding improvements (Research/ablation)
├─ Hypothesis 4B (severity reranking)
└─ Hypothesis 4C (clinical embeddings) - if 4B doesn't help

SKIP (Unless research focus):
├─ Hypothesis 5A (behavioral indicators) - too speculative
└─ Hypothesis 7B (direct scoring) - loses interpretability
```

---

## 7. Summary

**Bottom line**: The current system is correct but conservative. Specs 061-063 are the right first step because they're low-risk, additive (CLI flags), and address the biggest practical limitation (coverage). The other hypotheses are research directions for if few-shot still underperforms after 063.

---

## Related Documentation

- [Specs Index](docs/_specs/index.md) — Implementation specs (061-063)
- [Hypotheses for Improvement](HYPOTHESES-FOR-IMPROVEMENT.md) — Original hypothesis list
- [Task Validity](docs/clinical/task-validity.md) — Why ~50% coverage is expected
- [Few-Shot Analysis](docs/results/few-shot-analysis.md) — Why few-shot may not beat zero-shot
