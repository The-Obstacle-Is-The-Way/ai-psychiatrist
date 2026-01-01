# Embeddings and Few-Shot Learning: A Plain-Language Guide

**Audience**: Clinicians and non-CS folks who want to understand the "magic"
**Last Updated**: 2025-12-29

---

## The Question This Answers

"How does the system find similar patients to help score new ones?"

This document explains **embeddings** and **few-shot learning** without requiring any computer science background.

---

## The Core Idea

When you read a patient's interview, you might think:

> "This reminds me of Patient X from last year who also couldn't sleep and felt hopeless. That patient had moderate depression."

The system does the same thing, but mathematically.

---

## Part 1: What is an Embedding?

### The Analogy: GPS Coordinates

Imagine every sentence in the world has a "location" in a giant map of meaning.

- "I can't sleep at night" → Location A
- "I have insomnia" → Location B (very close to A - similar meaning)
- "I love pizza" → Location C (far from A and B - different meaning)

An **embedding** is like GPS coordinates for a sentence's meaning.

### The Technical Reality

Instead of 2D coordinates (latitude, longitude), embeddings use 4096 dimensions. But the principle is the same: **similar meanings have similar coordinates**.

| Sentence | "Meaning Location" (simplified) |
|----------|--------------------------------|
| "I can't sleep" | [0.8, 0.2, 0.9, ...4096 numbers...] |
| "I have insomnia" | [0.79, 0.21, 0.88, ...very similar...] |
| "I love pizza" | [0.1, 0.7, 0.3, ...very different...] |

### Why 4096 Dimensions?

More dimensions = more nuance captured. The paper's Appendix D confirms 4096 dimensions is optimal for this task among the tested values (64, 256, 1024, 4096).

Think of it like describing a patient:
- 2 dimensions: "depressed" and "anxious"
- 10 dimensions: add "sleep quality", "energy", "appetite", etc.
- 4096 dimensions: captures extremely subtle differences in meaning

---

## Part 2: How Similarity is Measured

### The Analogy: Distance on a Map

If two places have similar GPS coordinates, they're close together.

Same with embeddings: if two sentences have similar "meaning coordinates," they're **semantically similar**.

### Cosine Similarity

**Raw cosine similarity** ranges from **-1 to 1**:
- **1.0** = identical direction (very similar meaning)
- **0.0** = orthogonal (no directional similarity)
- **-1.0** = opposite direction (very dissimilar)

In this codebase, we store similarity in a **0 to 1** range by applying a simple,
monotonic transform:

```text
similarity = (1 + raw_cosine) / 2
```

So in the stored similarity scale:
- **1.0** = identical (`raw_cosine = 1.0`)
- **0.5** = neutral / orthogonal (`raw_cosine = 0.0`)
- **0.0** = opposite (`raw_cosine = -1.0`)

| Comparison | Similarity |
|------------|------------|
| "I can't sleep" vs "I have insomnia" | 0.92 |
| "I can't sleep" vs "I feel tired" | 0.75 |
| "I can't sleep" vs "I love hiking" | 0.15 |

These numbers are illustrative; exact values depend on the embedding model.

---

## Part 3: The Reference Store (Knowledge Base)

### What It Contains

Before running on new patients, we processed all **training patients**:

1. Split each transcript into chunks (8 lines each)
2. Computed embeddings for each chunk
3. Stored them with their ground-truth PHQ-8 scores

**Result**: A database of thousands of chunks from the training split. In the paper-style split, the
training set is 58 participants; in the AVEC2017 split, it is 107 participants. The exact chunk
count depends on the chosen split and chunking parameters, but the contents are always:
- The text itself
- Its embedding (4096 numbers)
- The patient's actual PHQ-8 scores

### Visualized

```
REFERENCE STORE
┌──────────────────────────────────────────────────────────────┐
│ Patient 101, Chunk 3                                         │
│ Text: "I haven't been able to sleep... I'm so exhausted"     │
│ Embedding: [0.45, 0.82, 0.31, ... 4096 numbers ...]          │
│ PHQ8_Sleep score: 2                                          │
├──────────────────────────────────────────────────────────────┤
│ Patient 142, Chunk 7                                         │
│ Text: "Nothing brings me joy anymore, I don't care"          │
│ Embedding: [0.71, 0.23, 0.88, ... 4096 numbers ...]          │
│ PHQ8_NoInterest score: 3                                     │
├──────────────────────────────────────────────────────────────┤
│ ... ~7,000 more chunks ...                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Part 4: Few-Shot Learning

### The Analogy: "Show, Don't Tell"

Imagine training a new resident to score PHQ-8. You could:

**Option A (Zero-Shot)**: Give them the PHQ-8 manual and say "score this patient."

**Option B (Few-Shot)**: Show them 2-3 examples first:
> "Here's Patient A who said 'I can't sleep' and had a score of 2.
> Here's Patient B who said 'I sleep too much' and had a score of 2.
> Now, this new patient says 'I wake up every night.' What's your score?"

Option B is better because examples calibrate their judgment.

### How the System Does This

For each PHQ-8 item in a new patient:

1. **Extract evidence**: "The patient said: 'I wake up at 3am every night'"
2. **Embed the evidence**: Convert to 4096-dimension coordinates
3. **Find similar chunks**: Search reference store for closest matches
4. **Retrieve examples**: Get the 2 most similar chunks with their scores
5. **Score with examples**: LLM sees the new evidence PLUS similar examples

### Visual Example

```
NEW PATIENT'S EVIDENCE (Sleep):
"I wake up at 3am every night and can't get back to sleep"
                    │
                    ▼ Compute embedding + search reference store
                    │
    ┌───────────────┴───────────────┐
    │                               │
    ▼                               ▼
REFERENCE 1 (similarity: 0.89)   REFERENCE 2 (similarity: 0.85)
"I keep waking up at night"      "Can't stay asleep, up at 4am"
Score: 2                         Score: 2
    │                               │
    └───────────────┬───────────────┘
                    │
                    ▼
LLM PROMPT:
"Here are similar examples:
 - 'I keep waking up at night' → Score 2
 - 'Can't stay asleep, up at 4am' → Score 2

 Now score: 'I wake up at 3am every night and can't get back to sleep'"

LLM OUTPUT: Score 2
```

---

## Part 5: Why This Works

### The Calibration Effect

Without examples, the LLM must guess what "2" means on the PHQ-8 scale.

With examples, the LLM learns:
> "Oh, 'waking up at night' is a 2, not a 3. Got it."

### The Paper's Results

| Mode | MAE | Explanation |
|------|-----|-------------|
| Zero-shot | 0.796 | No examples, LLM guesses |
| Few-shot | 0.619 | 2 examples per item, calibrated |

That is a **22% lower item-level MAE** vs zero-shot (paper-reported). In this repository, few-shot performance is sensitive to retrieval quality and can underperform zero-shot; see `docs/results/reproduction-results.md` and `docs/results/run-history.md`.

---

## Part 6: Per-Item Retrieval

### Each Symptom Gets Its Own Examples

The system doesn't find "similar patients overall." It finds similar evidence **per PHQ-8 item**:

| Item | Evidence Extracted | Similar References Found |
|------|-------------------|-------------------------|
| Sleep | "I wake up at 3am" | 2 sleep-related chunks |
| Tired | "I have no energy" | 2 fatigue-related chunks |
| Appetite | (none found) | (none) → N/A |

### Why Per-Item?

A patient might have severe sleep problems but mild appetite issues. Using overall similarity would miss this nuance.

---

## Part 7: The Item Tagging Problem (Spec 34)

### The Problem: Topic vs. Item Mismatch

Embedding similarity finds chunks that are **semantically similar** overall, but similarity doesn't guarantee the chunk is about the **same PHQ-8 item**.

**Example of the problem:**

You're scoring Sleep for a new patient. Your extracted evidence is:
> "I can't sleep, I'm up all night worrying"

The embedding search might return:
> Reference 1: "I worry constantly about money" (high similarity - both mention worry)
> Reference 2: "I toss and turn at night" (moderate similarity - about sleep)

The first reference is semantically similar (both express anxiety/worry), but it's tagged with PHQ8_Failure or PHQ8_Concentrating—not PHQ8_Sleep. Using it as a few-shot example for Sleep could confuse the model.

### The Solution: Item Tagging

We now tag each reference chunk with which PHQ-8 items it actually discusses:

```
BEFORE (untagged):
┌─────────────────────────────────────────────┐
│ Chunk: "I worry constantly about money"     │
│ Embedding: [0.45, 0.82, ...]                │
│ PHQ8 scores: (participant-level only)       │
└─────────────────────────────────────────────┘

AFTER (tagged):
┌─────────────────────────────────────────────┐
│ Chunk: "I worry constantly about money"     │
│ Embedding: [0.45, 0.82, ...]                │
│ PHQ8 scores: (participant-level only)       │
│ Tags: ["PHQ8_Failure", "PHQ8_Concentrating"]│  ← NEW
└─────────────────────────────────────────────┘
```

### How Tagging Works

At **index time** (when embeddings are generated):
1. Each chunk is analyzed for PHQ-8-related keywords
2. Keywords are matched against a curated keyword list (`phq8_keywords.yaml`)
3. Matching items are stored in a `.tags.json` sidecar file

At **retrieval time** (when scoring a new patient):
1. If item tag filtering is enabled (`EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`)
2. When retrieving references for PHQ8_Sleep, only chunks tagged with `PHQ8_Sleep` are considered
3. This eliminates semantically-similar-but-wrong-item references

### Visual Example

```
RETRIEVING REFERENCES FOR PHQ8_Sleep (with filtering)

New Evidence: "I can't sleep, I'm up all night"
                    │
                    ▼ Search with item filter
                    │
    ┌───────────────┼───────────────────────────────┐
    │               │                               │
    ▼               ▼                               ▼
Chunk A          Chunk B                         Chunk C
"I worry about   "I toss and turn               "Up every night
 money"           at night"                      can't sleep"
Tags: [Failure]  Tags: [Sleep]                  Tags: [Sleep]
                    │                               │
    ✗ FILTERED      ▼                               ▼
    (no Sleep tag)  ✓ INCLUDED                      ✓ INCLUDED
```

### The Artifacts

Item tagging creates a new sidecar file alongside embeddings:

| File | Contents |
|------|----------|
| `{name}.npz` | Embedding vectors (unchanged) |
| `{name}.json` | Chunk text (unchanged) |
| `{name}.meta.json` | Generation metadata (unchanged) |
| `{name}.tags.json` | **NEW**: Per-chunk PHQ-8 item tags |

The `.tags.json` format:
```json
{
  "303": [
    ["PHQ8_Sleep", "PHQ8_Tired"],
    [],
    ["PHQ8_Depressed"]
  ],
  "304": [...]
}
```

### Why This Matters

Without item tagging, few-shot retrieval can inject noise:
- High-similarity chunks about the wrong symptom
- Calibration examples that confuse rather than help

With item tagging, references are both:
1. **Semantically similar** (embedding-based)
2. **Topically relevant** (item-tagged)

Goal: reduce semantically-similar-but-wrong-item references. Whether this improves metrics depends on the model/run.

---

## Part 8: When It Doesn't Help

### The Appetite Problem

The paper (Appendix E) found:
> "PHQ-8-Appetite had no successfully retrieved reference chunks"

Important: this statement is about **few-shot reference retrieval** (“no retrieved reference chunks”),
not prediction coverage directly.

The paper continues (Appendix E) that Gemma 3 27B *“did not identify any evidence related to appetite
issues in the available transcripts, resulting in no reference for that symptom.”* In our pipeline,
reference retrieval is driven by embedding the **extracted evidence** per item. If the evidence
extraction step returns no appetite evidence, there’s nothing to embed/query, so reference retrieval
returns no appetite examples.

This often correlates with low appetite coverage (more N/A), but the two are not identical metrics.
Appetite coverage varies by run/model; see `docs/results/run-history.md` for concrete runs.

---

## Summary: The Complete Picture

1. **Embeddings** = Mathematical representation of meaning (like GPS for sentences)
2. **Reference Store** = Database of training chunks with known scores
3. **Similarity Search** = Find chunks with similar meaning to new evidence
4. **Few-Shot** = Show the LLM similar examples before asking it to score

**The key insight**: Instead of telling the LLM "here's what a 2 means," we SHOW it examples of labeled chunks. The paper reports a large few-shot improvement, but in this repo few-shot performance depends heavily on retrieval quality (and can underperform zero-shot in some runs). See `docs/results/reproduction-results.md` and `docs/results/run-history.md`.

---

## Glossary

| Term | Plain Definition |
|------|------------------|
| **Embedding** | A list of ~4000 numbers representing a sentence's meaning |
| **Similarity (transformed cosine)** | A 0–1 score derived from cosine similarity: 1=identical, 0.5=neutral, 0=opposite |
| **Reference Store** | Database of training examples with known scores |
| **Few-Shot** | Showing examples before asking for a prediction |
| **Zero-Shot** | Predicting without any examples |
| **Chunk** | A small section of a transcript (~8 lines) |

---

## Related Documentation

- [extraction-mechanism.md](./extraction-mechanism.md) - How evidence is found
- [coverage-explained.md](./coverage-explained.md) - Why some items get N/A
- [clinical-understanding.md](./clinical-understanding.md) - Clinical context
