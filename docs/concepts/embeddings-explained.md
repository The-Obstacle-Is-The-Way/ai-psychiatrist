# Embeddings and Few-Shot Learning: A Plain-Language Guide

**Audience**: Clinicians and non-CS folks who want to understand the "magic"
**Last Updated**: 2025-12-23

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

More dimensions = more nuance captured. The paper found 4096 is optimal for this task.

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

That is a **22% lower item-level MAE** vs zero-shot (0.796 → 0.619 in the paper).

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

## Part 7: When It Doesn't Help

### The Appetite Problem

The paper (Appendix E) found:
> "PHQ-8-Appetite had no successfully retrieved reference chunks"

Why? Because:
1. Few patients discuss eating in interviews
2. Few training chunks mention appetite
3. Nothing in the reference store to match against

**Result**: Appetite has low coverage (34%) — not enough data to learn from.
In our example reproduction run, appetite coverage was **34%** (see
`docs/results/reproduction-notes.md`), but the exact value varies by run.

---

## Summary: The Complete Picture

1. **Embeddings** = Mathematical representation of meaning (like GPS for sentences)
2. **Reference Store** = Database of training chunks with known scores
3. **Similarity Search** = Find chunks with similar meaning to new evidence
4. **Few-Shot** = Show the LLM similar examples before asking it to score

**The key insight**: Instead of telling the LLM "here's what a 2 means," we SHOW it examples of 2s. This calibrates its judgment and improves accuracy by 22%.

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
