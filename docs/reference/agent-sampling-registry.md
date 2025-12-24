# Agent Sampling Parameter Registry

**Purpose**: Single Source of Truth (SSOT) for all agent sampling parameters.
**Last Updated**: 2025-12-24
**Related**: [Configuration Reference](./configuration.md) | [GAP-001](../bugs/gap-001-paper-unspecified-parameters.md)

---

## Pipeline Architecture

```
Transcript
    │
    ▼
┌─────────────────────┐
│ 1. QUALITATIVE      │  ← Analyzes transcript, outputs narrative
└─────────────────────┘     (social, biological, risk factors)
    │
    ▼
┌─────────────────────┐
│ 2. JUDGE            │  ← Scores qualitative output (4 metrics)
└─────────────────────┘     (coherence, completeness, specificity, accuracy)
    │                       ↺ loops back if any score < 4 (max 10 iterations)
    ▼
┌─────────────────────┐
│ 3. QUANTITATIVE     │  ← Predicts PHQ-8 item scores (0-3 each)
└─────────────────────┘     (zero-shot OR few-shot with embeddings)
    │
    ▼
┌─────────────────────┐
│ 4. META-REVIEW      │  ← Integrates all assessments
└─────────────────────┘     → Final severity (0-4) + MDD indicator
```

---

## Sampling Parameters: Our Defaults

### First-Principles Rationale

| Agent | Needs Creativity? | Has Grounding? | Recommended temp |
|-------|-------------------|----------------|------------------|
| Qualitative | Some (narrative) | No (just transcript) | Low (0.2) |
| Judge | None (scoring) | Yes (rubric) | Zero (0.0) |
| Quantitative (zero-shot) | None | No | Zero (0.0) |
| Quantitative (few-shot) | Slight | Yes (examples) | Low (0.2) |
| Meta-Review | None (integration) | Yes (prior assessments) | Zero (0.0) |

**Key insight**: Agents with grounding (examples, rubrics, prior context) can tolerate
slight temperature. Agents without grounding need full determinism.

### What We Settled On (Our Implementation)

| Agent | temp | top_k | top_p | Config Key | Rationale |
|-------|------|-------|-------|------------|-----------|
| **Qualitative** | 0.2 | 20 | 0.8 | `temperature` | Narrative needs some flexibility |
| **Judge** | 0.0 | 20 | 0.8 | `temperature_judge` | Scoring must be deterministic |
| **Quantitative (zero-shot)** | 0.0 | 1 | 1.0 | `*_zero_shot` | Matches notebook exactly |
| **Quantitative (few-shot)** | 0.2 | 20 | 0.8 | `temperature` | Matches notebook exactly |
| **Meta-Review** | 0.0 | 20 | 0.8 | `temperature_judge`* | Integration, not generation |

*Meta-Review currently uses judge temp (0.0). Could add dedicated setting if needed.

### Decision Summary

1. **Quantitative**: Match notebooks exactly (verified source)
   - Zero-shot: `temp=0, top_k=1, top_p=1.0` (greedy)
   - Few-shot: `temp=0.2, top_k=20, top_p=0.8` (slight flexibility)

2. **Other agents**: Use first-principles defaults (no reliable source)
   - Judge: `temp=0` (deterministic scoring)
   - Qualitative: `temp=0.2` (narrative flexibility)
   - Meta-Review: `temp=0` (integration task)
   - All share `top_k=20, top_p=0.8`

---

## What the Paper Specifies

| Parameter | Paper Says | Section |
|-----------|-----------|---------|
| Model | Gemma 3 27B | Section 2.2 |
| Temperature | "fairly deterministic" | Section 4 |
| top_k / top_p | **NOT SPECIFIED** | - |
| Per-agent sampling | **NOT SPECIFIED** | - |

**The paper only tuned embedding hyperparameters (Appendix D):**
- chunk_size = 8
- top_k_references = 2
- dimension = 4096

Sampling parameters were NOT tuned or specified.

---

## What Their Code Does (Reference Only)

### Critical: Notebooks Only Exist for Quantitative Agent

| Agent | Has Notebook? | Source Type |
|-------|---------------|-------------|
| Qualitative | NO | Python only (`qual_assessment.py`) |
| Judge | NO | Python only (`qualitive_evaluator.py`) |
| Quantitative | YES | Notebooks are authoritative |
| Meta-Review | NO | Python only (`meta_review.py`) |

**This means**: We can only trust notebook values for Quantitative. Other agents'
sampling params are from Python files (which have wrong model defaults like `llama3`).

### Verified Notebook Values (Quantitative Only)

| Mode | temp | top_k | top_p | Source | Line |
|------|------|-------|-------|--------|------|
| Zero-shot | 0 | 1 | 1.0 | `basic_quantitative_analysis.ipynb` | 207 |
| Few-shot | 0.2 | 20 | 0.8 | `embedding_quantitative_analysis.ipynb` | 1174 |

Comment in zero-shot notebook: `"# Most deterministic temp, top_k, and top_p"`

### Python File Values (Less Trustworthy)

| Agent | temp | top_k | top_p | Source |
|-------|------|-------|-------|--------|
| Qualitative | 0 | 20 | 0.9 | `qual_assessment.py:103-105` |
| Judge | 0 | 20 | 0.9 | `qualitive_evaluator.py:146` |
| Meta-Review | 0 | 20 | 1.0 | `meta_review.py:132-134` |

**Note**: These Python files also have wrong model defaults (`llama3`), so their
sampling params may also be arbitrary/experimental. The differences (top_p=0.9 vs 1.0)
appear to be inconsistent copy-paste, not intentional design.

---

## Environment Variables

```bash
# Few-shot / default (used by Qualitative, Quantitative few-shot)
MODEL_TEMPERATURE=0.2
MODEL_TOP_K=20
MODEL_TOP_P=0.8

# Judge (deterministic scoring)
MODEL_TEMPERATURE_JUDGE=0.0

# Zero-shot Quantitative (fully greedy)
MODEL_TEMPERATURE_ZERO_SHOT=0.0
MODEL_TOP_K_ZERO_SHOT=1
MODEL_TOP_P_ZERO_SHOT=1.0
```

---

## Future Considerations

If per-agent control is needed, we could add:

```bash
MODEL_TEMPERATURE_QUALITATIVE=0.2
MODEL_TEMPERATURE_META_REVIEW=0.0
MODEL_TOP_K_JUDGE=20
MODEL_TOP_P_JUDGE=0.9
# etc.
```

Currently not implemented because:
1. Paper doesn't specify per-agent params
2. Their code's per-agent differences appear arbitrary
3. Simpler config is easier to maintain

---

## References

- [GAP-001: Paper Unspecified Parameters](../bugs/gap-001-paper-unspecified-parameters.md)
- [Configuration Reference](./configuration.md)
- [Pipeline Concepts](../concepts/pipeline.md)
- Paper Section 2.2: Model specification
- Paper Section 4: "fairly deterministic" mention
- Paper Appendix D: Hyperparameter optimization (embedding only)
