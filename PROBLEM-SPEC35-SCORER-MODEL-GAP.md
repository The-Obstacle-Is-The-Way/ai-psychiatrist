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
if config.scorer_model == settings.model.quantitative_model:
    print("ERROR: scorer model matches quantitative assessment model")
```

**But we never documented which model to use.**

---

## Current State

| Model | Available | Can Use for Scoring? |
|-------|-----------|---------------------|
| `gemma3:27b-it-qat` | Yes | NO (is the assessment model) |
| `qwen3-embedding:8b` | Yes | NO (embedding model, not chat) |
| `gemma3:27b` | Not pulled | MAYBE (same weights, different quant) |
| `llama3:8b` | Not pulled | YES (different family) |

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

### Option B: Use a different family (llama3:8b)

```bash
ollama pull llama3:8b  # ~4.7GB
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

For research integrity, **Option B (llama3:8b)** is cleanest:
- Truly disjoint model
- Smaller = faster scoring (~4 hours instead of ~9)
- Different family = no circular bias

For practical deployment, **Option A (gemma3:27b)** is acceptable:
- Same family = consistent clinical reasoning
- Different quantization provides SOME separation
- Debatable whether this violates the spirit of "disjoint"

---

## Action Required

1. **Decision**: Which scorer model to use?
2. **Pull the model**: `ollama pull <model>`
3. **Run scoring**: ~4-9 hours one-time cost
4. **Document the choice** in experiment metadata

---

## Related

- `docs/archive/specs/35-offline-chunk-level-phq8-scoring.md` - Spec 35
- `scripts/score_reference_chunks.py` - The scoring script
- `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` - Why this matters

---

## First Principles Answer: Is Spec 36 Worth Running Without Spec 35?

**No.**

| Spec | What It Does | Fixes Score Problem? |
|------|--------------|---------------------|
| Spec 34 | Domain filtering (keyword) | No |
| Spec 36 | Domain filtering (LLM) | No |
| Spec 35 | Chunk-level scoring | **YES** |

Spec 36 is just a more accurate version of Spec 34. Both filter by domain ("is this chunk about Sleep?"). Neither fixes the core problem: **chunks have participant-level scores, not chunk-specific scores**.

Running Spec 36 without Spec 35 = more LLM calls for marginal improvement.

**Conclusion**: Wait for Spec 35 scorer model decision before running ablations.
