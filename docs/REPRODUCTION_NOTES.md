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

### Corrected Script

The `scripts/reproduce_results.py` implements paper methodology:
- Item-level MAE (0-3 scale)
- N/A exclusion
- Uses train/dev splits (have per-item labels)
- Reports coverage metrics

**Note**: Paper uses a custom 58/43/41 stratified split, not the original AVEC2017 splits.
See `scripts/create_paper_split.py` to generate paper-style splits.

**Status**: The corrected script has been smoke-tested (`--limit 1`), but full
paper-parity runs (41 participants, few-shot) have not been completed yet.

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
# Quick sanity check (AVEC dev split; has per-item labels)
uv run python scripts/reproduce_results.py --split dev

# Paper-parity workflow (paper split + paper embeddings + evaluation on paper test)
uv run python scripts/create_paper_split.py --seed 42
uv run python scripts/generate_embeddings.py --split paper-train
uv run python scripts/reproduce_results.py --split paper --few-shot-only
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
- **Timeout**: 300s per LLM call (default in config; adjust via OLLAMA_TIMEOUT_SECONDS)

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
OLLAMA_TIMEOUT_SECONDS=300
```

---

## Paper Unspecified Parameters

Some parameters are NOT explicitly specified in the paper. See `docs/bugs/GAP-001_PAPER_UNSPECIFIED_PARAMETERS.md` for:

- **GAP-001a**: Exact split membership (paper uses custom 58/43/41, not AVEC2017 splits)
- **GAP-001b**: Temperature (paper says "fairly deterministic", we use 0.2)
- **GAP-001c**: top_k / top_p (not specified, we use defaults)
- **GAP-001d**: Model quantization (not specified, we use Ollama defaults)

**Implication**: MAE may differ ±0.1 from paper due to split differences and model variance.

---

## Next Steps

1. **Generate paper-style splits**: `uv run python scripts/create_paper_split.py --seed 42`
2. **Generate paper-style embeddings**: `uv run python scripts/generate_embeddings.py --split paper-train`
3. **Run paper-parity evaluation**: `uv run python scripts/reproduce_results.py --split paper --few-shot-only`
4. **Compare** to paper MAE (Gemma few-shot 0.619; zero-shot 0.796) and coverage notes
5. **Document** final validated results and provenance (seed + artifact paths)
