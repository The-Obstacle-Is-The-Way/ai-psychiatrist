# Agent Sampling Parameter Registry

**Purpose**: Single Source of Truth (SSOT) for all agent sampling parameters.
**Last Updated**: 2025-12-26
**Related**: [Configuration Reference](./configuration.md) | [GAP-001](../archive/bugs/gap-001-paper-unspecified-parameters.md) | [GitHub Issue #46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46)

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

## Our Implementation (Evidence-Based)

### The Config

```python
# All agents: temperature=0.0, nothing else
temperature = 0.0
```

That's it.

### Parameter Table

| Agent | temp | top_k | top_p | Rationale |
|-------|------|-------|-------|-----------|
| **Qualitative** | 0.0 | — | — | Clinical extraction, not creative writing |
| **Judge** | 0.0 | — | — | Classification (1-5 scoring) |
| **Quantitative (zero-shot)** | 0.0 | — | — | Classification (0-3 scoring) |
| **Quantitative (few-shot)** | 0.0 | — | — | Classification (0-3 scoring) |
| **Meta-Review** | 0.0 | — | — | Classification (severity 0-4) |

**Note on "—"**: Don't set these. At temp=0 they're irrelevant (greedy decoding), and best practice is "use temperature only."

---

## Why These Settings (With Citations)

| Source | What It Says | Link |
|--------|--------------|------|
| **Med-PaLM** | "sampled with temperature 0.0" for clinical answers | [Nature Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/) |
| **medRxiv 2025** | "Lower temperatures promote diagnostic accuracy and consistency" | [Study](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full) |
| **Anthropic** | "Use temperature closer to 0.0 for analytical / multiple choice" | [PromptHub](https://www.prompthub.us/blog/using-anthropic-best-practices-parameters-and-large-context-windows) |
| **Anthropic** | "top_k is recommended for advanced use cases only. You usually only need to use temperature" | [PromptHub](https://www.prompthub.us/blog/using-anthropic-best-practices-parameters-and-large-context-windows) |
| **Anthropic/AWS** | "You should alter either temperature or top_p, but not both" | [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html) |
| **Claude 4.x APIs** | Returns ERROR: "temperature and top_p cannot both be specified" | [GitHub](https://github.com/valentinfrlch/ha-llmvision/issues/562) |
| **IBM** | Low temperature reduces randomness but should be combined with RAG, calibration, and human oversight for clinical safety | [IBM Think](https://www.ibm.com/think/topics/llm-temperature) |

---

## Best Practice: Use Temperature Only (2025)

### Why Not Both top_k AND top_p?

> "OpenAI and most AI companies recommend changing one or the other, not both."
> — [OpenAI Community](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542)

> "Newer Claude models return an error: 'temperature and top_p cannot both be specified'"
> — [GitHub Issue](https://github.com/valentinfrlch/ha-llmvision/issues/562)

### Why Not top_k At All?

> "Top K is not a terribly useful parameter... not as well-supported, notably missing from OpenAI's API."
> — [Vellum](https://www.vellum.ai/llm-parameters/temperature)

**2025 Reality:**
- Use **temperature only** (preferred)
- OR top_p only (alternative)
- **Never both** — newer Claude models literally error
- **top_k is obsolete** — not even in OpenAI's API

### At temp=0, Do top_k/top_p Matter?

**No.** At temperature=0, you get **greedy decoding** (argmax). There's no sampling, so sampling filters (top_k, top_p) have nothing to filter.

---

## What the Paper Says

| Parameter | Paper Says | Section |
|-----------|-----------|---------|
| Model | Gemma 3 27B | Section 2.2 |
| Temperature | "fairly deterministic" | Section 4 |
| top_k / top_p | **NOT SPECIFIED** | — |
| Per-agent sampling | **NOT SPECIFIED** | — |

**The paper only tuned embedding hyperparameters (Appendix D):**
- chunk_size = 8
- top_k_references = 2
- dimension = 4096

Sampling parameters were NOT tuned or specified.

---

## What Their Code Does (Reference Only)

### Their Codebase is Internally Contradictory

| Source | temp | top_k | top_p | Notes |
|--------|------|-------|-------|-------|
| `basic_quantitative_analysis.ipynb:207` | 0 | 1 | 1.0 | Zero-shot |
| `basic_quantitative_analysis.ipynb:599` | 0.1 | 10 | — | Experimental? |
| `embedding_quantitative_analysis.ipynb:1174` | 0.2 | 20 | 0.8 | Few-shot |
| `qual_assessment.py` | 0 | 20 | 0.9 | Wrong model default |
| `meta_review.py` | 0 | 20 | 1.0 | Wrong model default |

Python files also have wrong model defaults (`llama3` instead of `gemma3`). Cannot be trusted as SSOT.

**Our decision**: Use evidence-based clinical AI defaults, not reverse-engineered contradictory code.

---

## Environment Variables

```bash
# Clinical AI default: temp=0 only
MODEL_TEMPERATURE=0.0

# top_k and top_p: DO NOT SET
# - At temp=0 they're irrelevant (greedy decoding)
# - Best practice: "use temperature only, not both"
# - Claude 4.x APIs error if you set both temp and top_p
```

---

## References

- [GitHub Issue #46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46) - Full investigation
- [GAP-001: Paper Unspecified Parameters](../archive/bugs/gap-001-paper-unspecified-parameters.md)
- [Configuration Reference](./configuration.md)
- [Med-PaLM - Nature Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/)
- [Temperature in Clinical AI - medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full)
- [Anthropic Best Practices - PromptHub](https://www.prompthub.us/blog/using-anthropic-best-practices-parameters-and-large-context-windows)
