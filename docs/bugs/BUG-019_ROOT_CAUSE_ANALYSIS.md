# BUG-019: Root Cause Analysis of Reproduction Friction

**Date**: 2025-12-22
**Status**: ANALYSIS COMPLETE (REVISED)
**Author**: Read-only analysis, no changes made

---

## Executive Summary

Multiple systemic issues were identified during reproduction attempt:

1. **Wrong understanding of paper**: Paper uses Gemma 3 27B for ALL agents. MedGemma was ONLY evaluated as an alternative for quantitative agent in Appendix F.
2. **No official MedGemma in Ollama**: There is NO official `ollama.com/library/medgemma`. All MedGemma models are community uploads.
3. **Legacy code uses llama3**: The original researchers' legacy code defaults to `llama3`, NOT Gemma or MedGemma!
4. **Scoring methodology mismatch (initial reproduction run)**: The paper uses item-level MAE excluding "N/A". Our first reproduction run computed **total-score MAE** on the test split because the checked-in test labels contain only total scores. The reproduction script has since been updated to compute paper-style item-level MAE on splits with per-item labels (train/dev).
5. **Config wiring issue**: `.env` silently overrides code defaults

---

## What the Paper ACTUALLY Says

### Section 2.2 (Models)

> "We utilized a state-of-the-art open-weight language model, **Gemma 3 with 27 billion parameters (Gemma 3 27B)**"
> "For the embedding-based few-shot prompting approach, we used **Qwen 3 8B Embedding**"

**Conclusion**: ALL agents use Gemma 3 27B. ONE embedding model (Qwen 3 8B).

### Section 3.2 (Quantitative Assessment) - MedGemma Mention

> "In addition to Gemma 3 27B, we **also evaluated** its variant fine-tuned on medical text, MedGemma 27B"
> "The few-shot approach with MedGemma 27B achieved an improved average MAE of 0.505 **but detected fewer relevant chunks, making fewer predictions overall** (Appendix F)."

**Conclusion**: MedGemma was an ADDITIONAL EVALUATION, NOT the primary model. Results are in APPENDIX F, not the main paper.

### Appendix F (MedGemma Results)

> "MedGemma 27B had an edge over Gemma 3 27B in most categories overall, achieving an average MAE of 0.505, 18% less than Gemma 3 27B, **although the number of subjects detected as having available evidence from the transcripts was smaller with MedGemma**"

**Conclusion**: MedGemma is MORE CONSERVATIVE (more N/A). Better MAE when it scores, but scores LESS.

---

## What the Legacy Code ACTUALLY Uses

### Legacy Agent Model Assignments

| File | Model Default | Notes |
|------|---------------|-------|
| `_legacy/agents/qualitative_assessor_z.py:5` | `model="llama3"` | Qualitative zero-shot |
| `_legacy/agents/qualitative_assessor_f.py:5` | `model="llama3"` | Qualitative few-shot |
| `_legacy/agents/qualitive_evaluator.py:31` | `model="llama3"` | Judge agent |
| `_legacy/agents/meta_reviewer.py:31` | `model="llama3"` | Meta review |
| `_legacy/agents/quantitative_assessor_f.py:548` | `--chat_model llama3` | Quantitative few-shot |
| `_legacy/agents/quantitative_assessor_z.py:5` | `model="llama3"` | Quantitative zero-shot |
| `_legacy/quantitative_assessment/quantitative_analysis.py:14` | `model = "gemma3-optimized:27b"` | Standalone script |

### Key Finding

**The legacy agent code defaults to `llama3`, NOT Gemma or MedGemma!**

Only the standalone `quantitative_analysis.py` script uses `gemma3-optimized:27b`.

This suggests:
1. The paper may have been written with different model configs than shipped code
2. The `llama3` defaults are likely development placeholders
3. The actual paper results may have used command-line overrides

---

## No Official MedGemma in Ollama

### Evidence from [Ollama Search](https://ollama.com/search?q=medgemma)

There is **NO** `ollama.com/library/medgemma` (official library entry).

All MedGemma models are **community uploads**:

| Model | Uploader | Pulls | Notes |
|-------|----------|-------|-------|
| `alibayram/medgemma:27b` | alibayram | 9,916 | Q4_K_M quantization |
| `alibayram/medgemma:4b` | alibayram | - | Multimodal |
| `amsaravi/medgemma-4b-it` | amsaravi | - | With image support |
| `jwang580/medgemma_27b_q8_0` | jwang580 | - | Q8_0 quantization |
| `jwang580/medgemma_27b_text_it` | jwang580 | - | Text instruction-tuned |

### GitHub Issue

[Ollama Issue #10970](https://github.com/ollama/ollama/issues/10970) (opened June 4, 2025):
> "Support for MedGemma" - Community requesting official support

**Status**: Still not in official library as of December 2025.

### Who is alibayram?

- **NOT** affiliated with the paper authors (TReNDS/GSU team)
- **NOT** affiliated with Google
- Community member who converted HuggingFace weights

### Potential Issues with Community Models

1. Q4_K_M quantization may degrade medical reasoning
2. Conversion could introduce subtle bugs
3. No verification against official Google release
4. The excessive N/A behavior could be a quantization artifact

---

## What Our Code Was Configured For

### Current config.py (after changes during reproduction)

| Agent | Config Key | Current Default |
|-------|------------|-----------------|
| Qualitative | `qualitative_model` | `gemma3:27b` |
| Judge | `judge_model` | `gemma3:27b` |
| Meta-review | `meta_review_model` | `gemma3:27b` |
| Quantitative | `quantitative_model` | `gemma3:27b` (was `alibayram/medgemma:27b`) |
| Embedding | `embedding_model` | `qwen3-embedding:8b` |

### What .env Was Set To

Before fix: `MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b`
After fix: `MODEL_QUANTITATIVE_MODEL=gemma3:27b`

### Stale Comment in config.py

Lines 285-286 STILL say:
```python
# NOTE: enable_medgemma removed - use MODEL__QUANTITATIVE_MODEL directly.
# Default quantitative_model is already alibayram/medgemma:27b (Paper Appendix F).
```

But the actual default (line 89) is now `gemma3:27b`.

---

## Scoring Methodology Mismatch (CRITICAL)

### Paper's Methodology

From **Section 3.2** and **Figure 4**:
- Reports **item-level MAE** (each item scored 0-3)
- Shows confusion matrices for **each of the 8 PHQ-8 items separately**
- **Excludes N/A predictions** from MAE calculation
- MAE 0.619 (few-shot), 0.796 (zero-shot)

From **Section 3.2**:
> "in 50% of cases it was unable to provide a prediction due to insufficient evidence"

### Our Code's Methodology

From `src/ai_psychiatrist/domain/entities.py:115-123`:
```python
@property
def total_score(self) -> int:
    """Calculate total PHQ-8 score (0-24).
    N/A scores contribute 0 to the total.
    """
    return sum(item.score_value for item in self.items.values())
```

From `scripts/reproduce_results.py` (current):
- Computes **item-level MAE** on predicted PHQ-8 item scores (0-3)
- **Excludes** items marked "N/A" from MAE (paper-style)
- Reports multiple aggregation views:
  - MAE weighted by number of predicted items
  - Mean of per-item MAEs
  - Mean of per-subject MAEs on available items
- Tracks coverage (% non-N/A predictions)

Important limitation: the checked-in AVEC2017 test split files do not include per-item PHQ-8 labels, so paper-style item-level MAE cannot be computed for test from the repo data alone.

### Legacy Code's Methodology (Correct!)

From `_legacy/quantitative_assessment/quantitative_analysis.py:201-202`:
```python
if n_available > 0:
    avg_difference = sum(differences) / n_available  # ITEM LEVEL
```

**Legacy code correctly excludes N/A from MAE calculation!**

### Why This Matters

| Aspect | Paper | Our Code | Legacy |
|--------|-------|----------|--------|
| Scale | 0-3 per item | 0-24 total | 0-3 per item |
| N/A handling | Excluded | Count as 0 | Excluded |
| MAE range | ~0.5-0.8 | ~4-7 | ~0.5-0.8 |

**Our MAE 4.02 is NOT comparable to paper's 0.619!**

---

## Summary of Root Causes

| Issue | Root Cause | Severity |
|-------|------------|----------|
| MedGemma default | Misread paper - Appendix F ≠ main results | CRITICAL |
| Community model | No official Ollama MedGemma; used untested alibayram | HIGH |
| Scoring methodology | Initial evaluation used total-score MAE due to missing per-item test labels | CRITICAL |
| Legacy uses llama3 | Paper may have used different configs than shipped | HIGH |
| .env override | Pydantic loads .env which overrides code defaults | MEDIUM |
| Stale docs | Comments don't match code | LOW |

---

## Corrected Understanding

### What Paper Says

1. **ALL agents**: Gemma 3 27B
2. **Embeddings**: Qwen 3 8B
3. **MedGemma**: Only evaluated as ALTERNATIVE for quantitative agent (Appendix F)
4. **MAE**: Item-level, excludes N/A

### What We Should Do

1. Use `gemma3:27b` for ALL agents (matches paper Section 2.2)
2. Use `qwen3-embedding:8b` for embeddings
3. If using MedGemma, use official Google weights (not alibayram community conversion)
4. Calculate **item-level MAE** to compare with paper
5. **Exclude N/A** from MAE calculation

### What Legacy Code Suggests

The legacy code defaults to `llama3` everywhere, suggesting:
1. The paper results may have been generated with command-line overrides
2. Or the code shipped doesn't match what produced the paper results
3. Or `llama3` was a development placeholder never updated

---

## Files That Need Fixing (NOT CHANGED)

1. `src/ai_psychiatrist/config.py:285-286` - Stale comment
2. Ensure we have (or can derive) per-item PHQ-8 labels for the AVEC2017 test split if we want a strict, end-to-end reproduction of the paper's reported MAE.
3. Verify model - use official Google MedGemma weights if needed (Ollama has only community uploads)

---

## References

- [Ollama Search for medgemma](https://ollama.com/search?q=medgemma) - Shows community models only
- [GitHub Issue #10970](https://github.com/ollama/ollama/issues/10970) - MedGemma support request
- [Google MedGemma Official](https://huggingface.co/google/medgemma-27b-text-it) - HuggingFace
- [Google DeepMind MedGemma](https://deepmind.google/models/gemma/medgemma/) - Official page
