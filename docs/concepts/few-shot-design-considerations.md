# Few-Shot Design Considerations

**Audience**: Researchers evaluating few-shot vs zero-shot approaches
**Last Updated**: 2026-01-01

---

## Overview

This document covers critical design considerations for few-shot PHQ-8 scoring, including known limitations and their fixes.

---

## The Participant-Level Score Problem

### The Issue

The paper's few-shot implementation has a **fundamental limitation**: participant-level PHQ-8 scores are assigned to individual chunks regardless of chunk content.

**How PHQ-8 works**:
- 8 items (Sleep, Tired, Appetite, etc.), each scored 0-3
- Total score = sum of all 8 items = 0-24

**How chunks are created**:
- Transcripts split into 8-line sliding windows (step=2)
- Result: ~100 chunks per participant
- Only a FEW chunks actually discuss any specific symptom

**The flaw**:
```text
Participant 300 has PHQ8_Sleep = 2

Chunk 5 (about career goals):
  "Ellie: what's your dream job
   Participant: open a business"
  → Gets labeled: Sleep Score = 2  ← WRONG (nothing about sleep!)

Chunk 95 (about sleep):
  "Ellie: have you had trouble sleeping
   Participant: yes every night"
  → Gets labeled: Sleep Score = 2  ← CORRECT
```

**Every chunk from a participant gets the SAME score**, regardless of content.

### The Fix: Spec 35 (Chunk-Level Scoring)

Instead of assigning participant-level scores to chunks, we score each chunk individually:

```bash
# Generate per-chunk scores
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat

# Enable at runtime
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
```

See `docs/data/chunk-scoring.md` for full details.

---

## Zero-Shot Inflation Hypothesis

### The Issue

The DAIC-WOZ transcripts include **Ellie's questions**, which directly probe PHQ-8 symptoms:

```text
Ellie: have you been diagnosed with depression
Participant: yes i was diagnosed last year
Ellie: can you tell me more about that
Participant: i was feeling really down and couldn't sleep
```

**Key insight**: Ellie asks DIRECT questions about PHQ-8 symptoms:
- "What are you like when you don't get enough sleep?" → PHQ8_Sleep
- "Do you have trouble concentrating?" → PHQ8_Concentrating

### External Validation: The Burdisso Paper

**Paper**: "DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic Depression Detection" (Burdisso et al., 2024)

**Critical Finding**: Models using Ellie's prompts achieve **0.88 F1** vs **0.85 F1** for participant-only models.

> "Models using interviewer's prompts learn to focus on a specific region of the interviews, where questions about past experiences with mental health issues are asked, and use them as **discriminative shortcuts** to detect depressed participants."

### Implications

| Mode | What It Tests | Validity |
|------|---------------|----------|
| Zero-shot (participant-only) | Can LLM assess patient's words? | **HIGH - The real test** |
| Zero-shot (full transcript) | Can LLM read Ellie's shortcuts? | Lower - potentially inflated |
| Few-shot (paper method) | Noisy chunks + wrong scores | Lower - label noise |
| Few-shot + Spec 35 | Filtered chunks + correct scores | Higher |

### The TRUE Baseline

The question isn't "why is few-shot worse?" but:
1. Is zero-shot artificially inflated by Ellie's shortcuts?
2. What is the TRUE baseline (participant-only)?

---

## Few-Shot Still Has Value

Despite the design flaws, few-shot is valuable for:

### Model Size Dependency

| Model Size | Few-Shot Value | Reason |
|------------|----------------|--------|
| Small (Gemma 27B, local) | **HIGH** | Needs calibration examples |
| Large (GPT-4, frontier) | Lower | Has already learned patterns |

### Explainability

| Property | Chain-of-Thought | RAG/CRAG |
|----------|------------------|----------|
| Reproducibility | Varies between runs | Fixed with same index |
| Grounding | Generated rationalization | Anchored to real examples |
| Verifiability | Cannot verify reasoning | Can examine retrieved examples |
| Auditability | May change on re-run | Citable, stable |

> "RAG-based explainability provides something that chain-of-thought prompting fundamentally cannot — **grounded, verifiable clinical reasoning**."

---

## The CRAG Pipeline (Specs 34 + 35 + 36)

| Spec | What It Does | Fixes |
|------|--------------|-------|
| **Spec 34** | Tag chunks with relevant PHQ-8 items at index time | Only retrieve Sleep-tagged chunks for Sleep queries |
| **Spec 35** | Score each chunk individually via LLM | Chunks get accurate, content-based scores |
| **Spec 36** | Validate references at query time (CRAG-style) | Reject irrelevant/contradictory chunks |

Together:
```text
Naive Few-Shot (paper)           = Naive RAG
   ↓ add Spec 34 (tag filter)    = Better RAG
   ↓ add Spec 35 (chunk scoring) = Even Better RAG
   ↓ add Spec 36 (validation)    = CRAG (2025 gold standard)
```

---

## Scorer Model Selection (Spec 35)

### Circularity Risk

If the scorer and assessor are the same model:
- **Correlated bias**: assessor is more likely to agree with scorer's labeling style
- **Metric inflation**: few-shot can look "better" because examples match model's priors

### Recommendation

| Priority | Scorer Choice | Notes |
|----------|---------------|-------|
| 1 (ideal) | MedGemma via HuggingFace | Medical tuning, most defensible |
| 2 (practical) | Different model family (qwen2.5, llama3.1) | Truly disjoint |
| 3 (baseline) | Same model with `--allow-same-model` | Explicit opt-in, ablate against disjoint |

See `docs/data/chunk-scoring.md` for generation commands.

---

## Summary

1. **Participant-level scoring is flawed** - chunks get wrong labels (Spec 35 fixes)
2. **Zero-shot may be inflated** - Ellie's questions provide shortcuts
3. **Few-shot has value** - for small models and explainability
4. **CRAG pipeline** (Specs 34+35+36) is the current best practice

---

## Related Documentation

- [Chunk Scoring Reference](../data/chunk-scoring.md) - Spec 35 details
- [Embeddings Explained](./embeddings-explained.md) - How retrieval works
- [CRAG Validation Guide](../guides/crag-validation-guide.md) - Spec 36 details
- [Item Tagging Setup](../guides/item-tagging-setup.md) - Spec 34 details
