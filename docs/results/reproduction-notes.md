# Reproduction Notes: PHQ-8 Assessment Evaluation

**Status**: Example run log (local). Results vary by hardware, backend, and model quantization.
**Last Updated**: 2025-12-23

---

## Example Reproduction Run (2025-12-23)

### Summary Table

| Metric | Paper | Our Result | Δ | Notes |
|--------|-------|------------|---|-------|
| **Item MAE** | 0.619 | 0.757 (weighted) / 0.753 (by-item) | +0.14 / +0.13 | Coverage tradeoff (hypothesis) |
| **Coverage** | ~50% | 74.1% | +24% | We predict more items |
| Zero-shot MAE | 0.796 | - | - | Not run yet |
| Participants | 41 | 41 | ✓ | Paper-style split (Appendix C algorithm; membership not published) |
| Runtime | ~1 min (M3 Pro) | ~3.9 hrs | - | Paper reports ~1 min on M3 Pro; our run was on M1 Pro with concurrent training |

### Per-Item Breakdown

| PHQ-8 Item | Our MAE | Our Coverage | Paper Notes |
|------------|---------|--------------|-------------|
| NoInterest | 0.64 | 88% | - |
| Depressed | 0.90 | 100% | Always predicted |
| Sleep | 0.68 | 98% | - |
| Tired | 0.76 | 83% | - |
| Appetite | 0.57 | **34%** | Paper: "no retrieved chunks" |
| Failure | 0.77 | 95% | - |
| Concentrating | 0.81 | 51% | - |
| Moving | 0.89 | **44%** | Paper: "highly variable" |

**Key Insight**: Appetite and Moving have low coverage in this run. The paper reports similar
availability issues (e.g., Appendix E notes "PHQ-8-Appetite had no successfully retrieved reference
chunks during inference").

---

## Analysis: Why Our MAE Differs

### The Coverage-Accuracy Tradeoff

The paper reports that **in ~50% of cases** Gemma 3 27B was unable to provide a prediction due to
insufficient evidence (Section 3.2).

In this example run, overall item prediction coverage was **74.1%** (predicted items / (41 × 8)).

One plausible explanation for the MAE difference is that higher coverage forces the system to score
more items (including harder-to-evidence symptoms). This is a general tradeoff; confirming the cause
requires ablations (e.g., keyword backfill on/off; prompt variants).

**Our MAE (0.757 weighted) falls between:**
- Paper zero-shot (0.796) — worse
- Paper few-shot (0.619) — better

This is compatible with few-shot helping, but a proper conclusion requires running the zero-shot
baseline on the same split with the same backend/model configuration.

### Why Higher Coverage?

Possible reasons (investigation needed):
1. **Prompt differences**: Our prompts may be less conservative about N/A
2. **Keyword backfill**: We add keyword-matched evidence when LLM misses items
3. **Model version**: Ollama gemma3:27b may differ from paper's exact weights

---

## Reproduction Methodology

### Commands Run

```bash
# 1. Create paper-style 58/43/41 split
uv run python scripts/create_paper_split.py --seed 42

# 2. Generate embeddings for paper train set (58 participants, ~65 min)
uv run python scripts/generate_embeddings.py --split paper-train

# 3. Run reproduction on paper test set (41 participants, ~3.9 hrs)
uv run python scripts/reproduce_results.py --split paper --few-shot-only
```

Note: `--seed` provides determinism for our implementation. The paper does not publish the exact
split membership (participant IDs), so results may differ across implementations even when the
methodology matches.

### Artifacts Generated

Note: `data/` is gitignored (DAIC-WOZ licensing). Your local run will write artifacts under `data/`.

| File | Description |
|------|-------------|
| `data/paper_splits/paper_split_*.csv` | 58/43/41 paper-style splits |
| `data/embeddings/paper_reference_embeddings.npz` | Paper-train reference embeddings (NPZ) |
| `data/embeddings/paper_reference_embeddings.json` | Text sidecar for embeddings (JSON) |
| `data/outputs/reproduction_results_<timestamp>.json` | Reproduction output (per-subject and aggregate metrics) |

---

## Configuration Used

```bash
MODEL_QUANTITATIVE_MODEL=gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIMENSION=4096
EMBEDDING_TOP_K_REFERENCES=2
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_CHUNK_STEP=2
OLLAMA_TIMEOUT_SECONDS=300
MODEL_TEMPERATURE=0.2
```

Embedding hyperparameters match paper Appendix D (optimal hyperparameters).
Sampling parameters not specified in the paper (temperature/top-k/top-p) are tracked in
`docs/bugs/gap-001-paper-unspecified-parameters.md`.

---

## Known Gaps and Divergences

### GAP-001: Paper Unspecified Parameters
See `docs/bugs/gap-001-paper-unspecified-parameters.md`:
- Temperature: Paper says "fairly deterministic", we use 0.2
- top_k/top_p: Not specified, we use Ollama defaults
- Model quantization: Not specified (see GAP-002)

### GAP-002: Model Quantization Unspecified
The paper says "Gemma 3 27B" but does NOT specify:
- Bit precision (FP16, Q8, Q4?)
- Quantization method (GPTQ, AWQ, GGUF?)
- GPU VRAM requirements

Ollama's `gemma3:27b` uses quantized GGUF weights (e.g., `quantization Q4_K_M` in `ollama show gemma3:27b`).
Paper may have used different weights.

---

## Runtime Performance

### Our Run (M1 Pro Max, 64GB)
- 41 participants × ~5.7 min/participant = **3.9 hours**
- Concurrent with arc-meshchop training (30s/iteration)

### Paper Reference
- Paper reports a full pipeline run in **~1 minute on a MacBook Pro M3 Pro** (Section 2.3.5).
- The paper does not specify background workload, per-participant timing, or the exact model quantization used.

This is not necessarily a bug: wall-clock runtime is sensitive to hardware, quantization, and concurrent workloads.

---

## Next Steps

1. [ ] Run zero-shot comparison to validate MAE improvement
2. [ ] Investigate N/A threshold differences (why 74% vs 50% coverage)
3. [ ] Consider running with dedicated GPU (no concurrent training)
4. [ ] Compare specific participant predictions with paper (if available)

---

## Previous (Invalid) Results

An earlier run (2025-12-22) was invalid:
- Used wrong methodology (total-score MAE instead of item-level)
- Output `data/outputs/reproduction_results_20251222_040100.json` (if present locally) should be ignored

---

## References

- Paper: `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`
- Section 3.2: Quantitative Assessment results
- Appendix D: Hyperparameter optimization
- Appendix E: Retrieval statistics (confirms Appetite issue)
- Appendix F: MedGemma comparison
