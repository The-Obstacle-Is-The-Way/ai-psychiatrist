# Problem: Spec 35 Scorer Model Not Specified

**Date**: 2025-12-31
**Status**: Blocking Spec 35 Deployment
**Severity**: Medium (spec gap, not code bug)

---

## The Problem

Spec 35 requires a "disjoint model" for chunk scoring to avoid circularity, but **we never specified which model to use**.

From `docs/archive/specs/35-offline-chunk-level-phq8-scoring.md`:

```
## Circularity Controls (Required)

1. **Disjoint model**: scorer model must not equal the assessment model
   (enforced by default; override requires --allow-same-model).
```

The script enforces this check:
```python
if not args.allow_same_model and config.scorer_model == settings.model.quantitative_model:
    print(
        "ERROR: scorer model matches quantitative assessment model "
        "(Spec 35 circularity risk)."
    )
    return 2
```

**But we never documented which model to use.**

---

## Current State

| Model | Available | Can Use for Scoring? |
|-------|-----------|---------------------|
| `gemma3:27b-it-qat` | Yes | NO (is the assessment model) |
| `qwen3-embedding:8b` | Yes | NO (embedding model, not chat) |
| `gemma3:27b` | Not pulled | MAYBE (same weights, different quant) |
| `llama3.1:8b-instruct-q4_K_M` | Not pulled | YES (disjoint family, strong baseline) |
| `qwen2.5:7b-instruct-q4_K_M` | Not pulled | YES (disjoint family, strong JSON compliance) |
| `phi4:14b` | Not pulled | YES (disjoint family, higher quality but slower) |

---

## The Circularity Risk

Why can't we just use `gemma3:27b-it-qat --allow-same-model`?

**Circular bias**: The scorer LLM labels chunks → those labels train/guide the assessment LLM → if they're the same model, we're teaching the model to predict its own predictions.

---

## Options

### Option A: Use gemma3:27b (non-QAT)

```bash
ollama pull gemma3:27b  # ~18GB, same as existing
```

**Pros**: Same model family, consistent behavior
**Cons**: Same base weights - is this "disjoint enough"?

### Option B: Use a different family (recommended)

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M   # ~4–5GB class
# or
ollama pull llama3.1:8b-instruct-q4_K_M  # ~4–5GB class
```

**Pros**: Truly different model, no circularity concern
**Cons**: Different model may score differently than Gemma would

### Option C: Use gemma3:27b-it-qat with --allow-same-model

```bash
python scripts/score_reference_chunks.py \
    --scorer-model gemma3:27b-it-qat \
    --allow-same-model
```

**Pros**: No new model needed
**Cons**: Full circularity risk, defeats the purpose of Spec 35

---

## Recommendation

For research integrity, **Option B** is the only defensible default.

**Recommended scorer model (default):** `qwen2.5:7b-instruct-q4_K_M`
- Disjoint from the assessment model family (Gemma)
- Strong instruction following + structured output tendencies (important because the script rejects
  any output that is not *exactly* the expected JSON schema)
- Fast enough to score ~6,837 chunks in a reasonable one-time job on Apple Silicon

**Backup choice:** `llama3.1:8b-instruct-q4_K_M`
- Also disjoint, similar size class, widely used

If JSON compliance or score quality is poor with 7–8B models, consider `phi4:14b` (disjoint but slower).

For practical deployment, **Option A (gemma3:27b)** is acceptable:
- Same family = consistent clinical reasoning
- Different quantization provides SOME separation
- Debatable whether this violates the spirit of "disjoint"

---

## Action Required

1. **Decision**: Which scorer model to use?
2. **Pull the model**: `ollama pull <model>`
3. **Sanity check JSON compliance** (recommended):

```bash
python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train \
  --scorer-backend ollama \
  --scorer-model qwen2.5:7b-instruct-q4_K_M \
  --limit 5
```

4. **Run full scoring** (one-time cost):

```bash
python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train \
  --scorer-backend ollama \
  --scorer-model qwen2.5:7b-instruct-q4_K_M
```

5. **Document the choice** in experiment metadata

---

## Related

- `docs/archive/specs/35-offline-chunk-level-phq8-scoring.md` - Spec 35
- `scripts/score_reference_chunks.py` - The scoring script
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` - Why this matters

---

## First Principles Answer: Is Spec 36 Worth Running Without Spec 35?

**Not as a “fix”, but yes as a (costly) mitigation.**

| Spec | What It Does | Fixes Score Problem? |
|------|--------------|---------------------|
| Spec 34 | Domain filtering (keyword) | No |
| Spec 36 | Relevance/contradiction filtering (LLM) using evidence + reference | No |
| Spec 35 | Chunk-level scoring | **YES** |

Spec 36 is not “just tags again” — it can drop references that contradict the *current* evidence.
But it still cannot make participant-level `reference_score` valid for an arbitrary chunk.

Running Spec 36 without Spec 35 = more LLM calls for (likely) incremental improvement.
Treat it as an ablation while Spec 35 is blocked, not as an endpoint.

**Conclusion**: Pick a Spec 35 scorer model next. If you have runtime budget, Spec 36 is still worth
running as a *separate ablation* while Spec 35 is blocked.
