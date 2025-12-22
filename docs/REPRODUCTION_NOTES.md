# Reproduction Notes: PHQ-8 Assessment Evaluation

**Status**: ⚠️ **PENDING RE-RUN** - Previous results used wrong methodology
**Last Updated**: 2025-12-22

---

## ⚠️ CRITICAL: Previous Results Are Invalid

The reproduction run on 2025-12-22 04:01 AM used **wrong methodology**:
- Computed **total-score MAE** (0-24 scale) instead of **item-level MAE** (0-3 scale)
- Used test split which has NO per-item ground truth labels
- Did NOT exclude N/A items from MAE calculation

The file `data/outputs/reproduction_results_20251222_040100.json` should be **ignored**.

### Corrected Script (Merged to Main)

Commit `1c414e4` rewrote `scripts/reproduce_results.py` to match paper methodology:
- Item-level MAE (0-3 scale)
- N/A exclusion
- Uses train/dev splits (have per-item labels)
- Reports coverage metrics

**The corrected script has NOT been run yet.**

---

## Paper's Reported Results (Target)

From Section 3.2 and Appendix F:

| Model | Mode | Item-Level MAE | Coverage | Notes |
|-------|------|---------------|----------|-------|
| Gemma 3 27B | Few-shot | 0.619 | ~50% | Paper's primary result |
| Gemma 3 27B | Zero-shot | 0.796 | - | Baseline |
| MedGemma 27B | Few-shot | 0.505 | Lower | "Fewer predictions overall" |

---

## To Run Reproduction (REQUIRED NEXT STEP)

```bash
# Ensure on main or dev branch with latest code
git checkout main

# Run with corrected script (uses train/dev splits with per-item labels)
python scripts/reproduce_results.py --split dev

# Or full train+dev (142 participants, matches paper's qualitative figures)
python scripts/reproduce_results.py --split train+dev
```

---

## Previous (Invalid) Results for Reference

**Note**: These numbers are on a different scale and methodology. Do NOT compare to paper.

| Metric | Value | Issue |
|--------|-------|-------|
| Total Participants | 47 | Test split (no per-item labels) |
| MAE | 4.02 | **Wrong scale** (0-24 not 0-3) |
| Failed (timeouts) | 6 | 13% timeout rate |

---

## Bug Summary (Fixed in Code)

### BUG-018a: MedGemma Excessive N/A
- **Problem**: `alibayram/medgemma:27b` produces 8/8 N/A for most participants
- **Fix**: Default changed to `gemma3:27b`
- **Status**: ✅ Fixed in config

### BUG-018i: Wrong MAE Methodology
- **Problem**: Computed total-score MAE instead of item-level MAE
- **Fix**: Complete rewrite of `scripts/reproduce_results.py`
- **Status**: ✅ Fixed in code, ⚠️ NOT re-run

---

## Environment

- **Hardware**: M1 Pro Max (Apple Silicon)
- **Models**:
  - `gemma3:27b` (17GB) - All agents
  - `qwen3-embedding:8b` (4.7GB) - Few-shot embeddings
- **Timeout**: 180s per LLM call (consider 300s for large transcripts)

---

## Configuration (.env)

Current paper-optimal settings:
```bash
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_JUDGE_MODEL=gemma3:27b
MODEL_META_REVIEW_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIMENSION=4096
OLLAMA_TIMEOUT_SECONDS=180  # Increase to 300 for large transcripts
```

---

## Next Steps

1. **Re-run reproduction** with corrected script
2. **Compare results** to paper's item-level MAE values
3. **Investigate** if results differ significantly from paper
4. **Document** final validated results
