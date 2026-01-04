# Problem: Spec 35 Scorer Model Not Specified

**Date**: 2026-01-01
**Status**: ✅ Unblocked (run same-model baseline; ablate disjoint/MedGemma)
**Severity**: Medium (spec gap, not code bug)

---

## The Problem

Spec 35 requires a "disjoint model" for chunk scoring to avoid circularity, but **we never specified which model to use**.

From `docs/reference/chunk-scoring.md`:

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
| `medgemma:27b` (official, HF) | Not downloaded | YES (HuggingFace only; text-only) |

---

## The Circularity Risk

There is **no training** here, so “circularity” is not gradient leakage. The real risk is:

- **Correlated bias**: if the scorer and assessor are the same model, the assessor is more likely to agree with the scorer’s labeling style.
- **Metric inflation**: few-shot can look “better” because the examples match the model’s own priors, not because retrieval is more clinically valid.

That’s why Spec 35 defaults to **disjoint** scorer and assessor models, but still provides
`--allow-same-model` as an explicit, opt-in override for practicality.

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

### Option D: Use official MedGemma via HuggingFace (text-only)

This uses **official weights** via Transformers (not an Ollama community upload).

**Important**:
- Use `medgemma:27b` → `google/medgemma-27b-text-it` (text generation).
- Do **not** use `google/medgemma-27b-it` or `google/medgemma-4b-it` for this script;
  those are **multimodal** (`Gemma3ForConditionalGeneration`) and are not supported by our
  current HuggingFace chat client (`AutoModelForCausalLM`).

Setup (one-time):
1. Install HuggingFace deps: `pip install 'ai-psychiatrist[hf]'` (or `make dev` if available)
2. Accept model terms + login:
   - https://huggingface.co/google/medgemma-27b-text-it
   - `huggingface-cli login`

Run:
```bash
python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train \
  --scorer-backend huggingface \
  --scorer-model medgemma:27b \
  --limit 5
```

**Pros**: Medical tuning; best-aligned “clinical labeling” option; different fine-tune than `gemma3:*` (not paper-parity, but a defensible ablation).
**Cons**: Very large model; may be slow or infeasible on Apple Silicon without aggressive quantization.

---

## Recommendation

Two-tier recommendation (be honest about constraints):

### If you can actually run official MedGemma

Use **Option D** (`--scorer-backend huggingface --scorer-model medgemma:27b`).
This is the most defensible “clinical scorer” choice if runtime/hardware allow.

### If you need a practical, local default (recommended for this Mac)

Use **Option B** with a small disjoint model:

- **Default**: `qwen2.5:7b-instruct-q4_K_M` (best JSON compliance tendencies)
- **Backup**: `llama3.1:8b-instruct-q4_K_M`

### Reality check: you can (and should) ablate

If you’re unconvinced the disjoint-model requirement matters (reasonable), treat it as a **sensitivity analysis**:

1. Generate chunk scores with the assessment model (explicitly opt in):
   `--scorer-model gemma3:27b-it-qat --allow-same-model`
2. Generate chunk scores with a disjoint scorer (or MedGemma if feasible).
3. Run the exact same evaluation pipeline with each artifact and compare AURC/AUGRC/coverage.

Important: `scripts/score_reference_chunks.py` writes to a fixed filename
(`<embeddings>.chunk_scores.json`). To keep both versions:
- Copy/rename the outputs after each run (including `.chunk_scores.meta.json`), then swap them back in before the evaluation run.

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

- `docs/reference/chunk-scoring.md` - Chunk scoring doc (Spec 35)
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
Treat it as an ablation until you have generated chunk scores (Spec 35), not as an endpoint.

**Conclusion**: Pick a Spec 35 scorer model next. If you have runtime budget, Spec 36 is still worth
running as a *separate ablation* before/after chunk scoring is available.
