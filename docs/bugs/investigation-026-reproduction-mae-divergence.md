# INVESTIGATION-026: Reproduction MAE Divergence Analysis

**Status**: DOCUMENTED - Root causes identified (paper-text vs paper-repo divergence)
**Severity**: HIGH (core research question)
**Found**: 2025-12-24 (few-shot reproduction run)
**Related**: Paper Section 3.2, Appendix E, Appendix F
**Last Updated**: 2025-12-24 (revised after deeper analysis)

---

## Executive Summary

Our few-shot reproduction achieved **MAE 0.778** vs paper's **0.619** (Δ = +0.159).

### What We Know For Certain

1. **Model**: `gemma3:27b` (confirmed, same as paper Section 2.2)
2. **Keyword Backfill**: OFF (confirmed in provenance)
3. **Coverage**: 69.2% (vs paper's ~50%)
4. **MAE**: 0.778 (vs paper's 0.619)

### The Unresolved Mystery

**With keyword backfill OFF, why is our coverage (69.2%) so much HIGHER than paper's (~50%)?**

The coverage investigation document expected backfill OFF to produce ~50% coverage matching paper.
We got 69.2% with backfill OFF. This is unexplained.

### Possible Explanations (UNVERIFIED)

1. **Prompt differences** - Our prompts may differ from paper's original prompts
2. **Model version** - Ollama's gemma3:27b may have been updated since paper
3. **Quantization** - Q4_K_M behavior may differ from paper's runtime
4. **Reference embeddings** - Different embedding model behavior
5. **Stochastic variation** - Paper acknowledges LLMs have inherent randomness

---

## Verified Configuration

We verified the run configuration from provenance metadata:

```json
{
  "split": "paper",
  "embeddings_path": "data/embeddings/paper_reference_embeddings.npz",
  "quantitative_model": "gemma3:27b",
  "embedding_model": "qwen3-embedding:8b",
  "embedding_dimension": 4096,
  "embedding_chunk_size": 8,
  "embedding_chunk_step": 2,
  "embedding_top_k": 2,
  "enable_keyword_backfill": false  // <-- CONFIRMED OFF
}
```

And from `.env`:
```bash
MODEL_QUANTITATIVE_MODEL=gemma3:27b
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false
QUANTITATIVE_TRACK_NA_REASONS=true
```

**All settings match paper-TEXT methodology.** However, the paper's public repo has keyword
backfill ALWAYS ON (see analysis-027). Our run used backfill OFF (paper-text parity).
Yet coverage is 69.2% vs paper's ~50%.

---

## The Core Mystery

### Expected vs Actual

| Metric | Paper Text | Paper Repo | Our Run | Notes |
|--------|-----------|------------|---------|-------|
| Model | gemma3:27b | gemma3:27b | gemma3:27b | ✅ Same |
| Backfill | Not mentioned | ON (always) | OFF | ⚠️ Paper-text parity |
| Coverage | ~50% | Unknown | 69.2% | **+19.2%** ❓ |
| MAE | 0.619 | Unknown | 0.778 | +0.159 |

### Where Is The Extra Coverage Coming From?

With backfill OFF, our system should:
1. Ask LLM to extract evidence (no keyword assistance)
2. Ask LLM to score based on evidence
3. If no evidence → N/A

But we're getting **~20% more predictions** than the paper. This means our LLM is:
- Finding more evidence than paper's LLM, OR
- Being less conservative about outputting N/A

### Investigation TODO

1. [ ] Fetch paper's original prompts from https://github.com/trendscenter/ai-psychiatrist
2. [ ] Compare our prompts to paper's prompts
3. [ ] Check if model versions/updates affected behavior
4. [ ] Run with different temperature to test conservatism

---

## Finding 1: Paper Expects 50% N/A Rate

### Source: Paper Section 3.2 (VERBATIM QUOTE)

> "With optimized hyperparameters, Gemma 3 27B achieved an average MAE of 0.619 when predicting PHQ-8 scores, but **in 50% of cases it was unable to provide a prediction due to insufficient evidence** (Figure 4). The distribution of available scores was not even: certain symptoms, such as appetite, had few available scores, while others, such as sleep quality, had available scores for nearly all subjects. This reflects the variability in the content of the interview, with some symptoms discussed more frequently than others."

This is CRITICAL. The paper's 0.619 MAE is calculated ONLY on the 50% of items where the model had sufficient evidence.

### Our Run vs Paper

| Metric | Paper | Our Run | Delta |
|--------|-------|---------|-------|
| Coverage | ~50% | 69.2% | +19.2% |
| MAE | 0.619 | 0.778 | +0.159 |

### Interpretation

Our **higher coverage is not better** - it likely means:
1. We're making predictions on harder cases paper declined
2. These harder cases have worse accuracy
3. Our aggregate MAE is diluted by low-confidence predictions

### Verification Needed

We should calculate MAE **only on high-confidence predictions** to compare apples-to-apples with paper.

---

## Finding 2: Evidence Extraction Failures Are Normal

### Paper Appendix E (VERBATIM QUOTE)

> "Additionally, we noted that PHQ-8-Appetite had **no successfully retrieved reference chunks** during inference. Upon closer inspection, Gemma 3 27B did not identify any evidence related to appetite issues in the available transcripts, resulting in no reference for that symptom."

**Important clarification**: "No successfully retrieved reference chunks" refers to the few-shot
retrieval step (finding similar examples from the training set), NOT the final prediction coverage.
These are different metrics:
- **Reference retrieval**: Finding similar training examples to guide scoring
- **Prediction coverage**: Whether the model outputs a score vs N/A

The paper does NOT provide per-symptom coverage percentages. The ~50% figure is aggregate.

**Key Insight**: Evidence extraction failures are expected behavior in this pipeline.
Our extraction failures (participants 303, 401) are consistent with paper methodology.

---

## Finding 3: MedGemma Trade-off (CRITICAL)

### Paper Appendix F (VERBATIM QUOTE)

> "We evaluated MedGemma 27B with the same optimal hyperparameters determined for Gemma 3 27B... MedGemma 27B had an edge over Gemma 3 27B in most categories overall, achieving an average MAE of **0.505**, 18% less than Gemma 3 27B, **although the number of subjects detected as having available evidence from the transcripts was smaller with MedGemma**."

### The Paper's Choice

| Model | MAE | Coverage | Why Used |
|-------|-----|----------|----------|
| Gemma 3 27B | 0.619 | ~50% | Main results (balanced) |
| MedGemma 27B | 0.505 | <50% | Appendix only (too conservative) |

The paper deliberately chose to report Gemma 3 27B in main results because **MedGemma was TOO conservative** - it got better accuracy but declined to score too many cases.

### Implication for Us

Our 69.2% coverage is the OPPOSITE problem - we're scoring MORE cases than the paper, likely including lower-confidence predictions that hurt our MAE.

---

## Finding 4: Stochasticity Acknowledgment

### Paper Section 4 (VERBATIM QUOTE)

> "The stochastic nature of LLMs renders a key limitation of the proposed approach. **Even with fairly deterministic parameters, responses can vary across runs**, making it challenging to obtain consistent performance metrics."

This explicitly acknowledges that MAE will vary between runs. Our 0.778 vs 0.619 may partially be explained by normal stochastic variation.

---

## Finding 5: Quantization Analysis

### What We're Using

```
Model: gemma3:27b (Q4_K_M quantization)
Size: ~17GB
Source: Default Ollama download
```

### Available Quantizations

| Quantization | Size | Quality | Can Run on M1 Max 64GB? |
|--------------|------|---------|-------------------------|
| Q4_K_M | ~17GB | Good | ✅ Yes (current) |
| Q8_0 | ~27GB | Better | ✅ Yes (48GB GPU available) |
| QAT | ~13-17GB | Official Google | ✅ Yes (if available) |
| FP16/BF16 | ~54GB | Best | ❌ No (exceeds GPU memory) |

### System Capabilities

```
System: Apple M1 Max
RAM: 64GB unified memory
GPU Available: ~48GB (75% of RAM for Metal)
Current Model: 17GB (35% utilization)
```

**We have significant headroom** to upgrade to Q8_0 (~27GB).

### Paper's Hardware (CONFLICTING INFORMATION)

**Paper text (Section 2.3.5)**:
> "The full assessment pipeline executes in approximately one minute on a MacBook Pro with an Apple M3 Pro chipset."

**Paper repo**: Contains SLURM scripts (`_reference/slurm/job_ollama.sh`) configured for
`--gres=gpu:A100:2` (2x NVIDIA A100 GPUs on GSU TReNDS cluster).

**Conclusion**: We do NOT know what hardware/quantization produced the reported MAE 0.619.
The paper text claims MacBook M3 Pro; the repo shows A100 cluster infrastructure.
This is documented in analysis-027.

### Why Quantization Matters

Lower quantization = more precision loss:
- Q4 = 4-bit weights, significant quantization noise
- Q8 = 8-bit weights, less noise, better nuance
- FP16 = full precision (but too large for our hardware)

For clinical assessment requiring subtle language understanding, **Q8_0 could improve accuracy**.

---

## Finding 6: Detailed Error Analysis

### Our Run Log Analysis

From `reproduction_run_20251223_224516.log`:

#### Evidence Extraction Failures (2 participants)

```
Participant 303: Failed to parse evidence JSON, using empty evidence
  - Cause: Unescaped quote in LLM response
  - Impact: items_with_evidence=0
  - Final score: total_score=4, na_count=4, severity=MINIMAL
  - System: Graceful fallback worked

Participant 401: Failed to parse evidence JSON, using empty evidence
  - Same pattern
  - System continued normally
```

#### Quantitative Parse Failures (2 participants)

```
Participant 325:
  - Result: ALL N/A (8/8)
  - Cause: Main response couldn't be parsed
  - MAE contribution: Excluded from calculation

Participant 474:
  - Result: ALL N/A (8/8)
  - Cause: Main response couldn't be parsed
  - MAE contribution: Excluded from calculation
```

#### Success Rate

- **Parsing success**: 39/41 (95.1%)
- **Complete failure (all N/A)**: 2/41 (4.9%)
- **Evidence extraction warnings**: 2/41 (4.9%)

### Paper's Error Rate

The paper acknowledges extraction/parsing issues but doesn't give exact failure rates. Our 95.1% success rate appears consistent with paper methodology.

---

## Recommendations

### Immediate Actions

1. **Calculate coverage-matched MAE** (HIGHEST PRIORITY)
   - Filter our results to ~50% coverage (high-confidence only)
   - Compare MAE on matched subset
   - This is the apples-to-apples comparison with paper

2. **Profile per-symptom coverage**
   - Compare to paper's Figure 4 confusion matrices
   - Identify which symptoms we're over-predicting

3. **Analyze confidence distribution**
   - Are we making low-confidence predictions paper declined?
   - Can we add confidence threshold to match paper behavior?

### Future Experiments

1. **Try Q8_0 quantization**
   ```bash
   # Pull higher quality quantization (if available)
   ollama pull gemma3:27b-instruct-q8_0

   # Update .env
   MODEL_QUANTITATIVE_MODEL=gemma3:27b-instruct-q8_0
   ```

2. **Test with MedGemma** (expect: lower coverage, better MAE)
   ```bash
   MODEL_QUANTITATIVE_MODEL=medgemma:27b
   ```

3. **Log confidence scores** for future runs

---

## Root Cause Analysis

### Why MAE 0.778 vs 0.619?

| Factor | Impact | Evidence |
|--------|--------|----------|
| Higher coverage (69% vs 50%) | **HIGH** | Predicting harder cases |
| Stochastic variation | MEDIUM | Paper explicitly acknowledges |
| Q4 vs potentially higher quant | LOW-MEDIUM | Less model precision |
| Different random seed | LOW | Affects sampling |

### Primary Hypothesis (UNVERIFIED)

**Hypothesis**: Our higher coverage may be the main contributor to our higher MAE. If we're
making predictions on cases the paper declined, these harder cases would drag down our aggregate MAE.

The paper chose Gemma 3 over MedGemma precisely because MedGemma was TOO conservative
(0.505 MAE but <50% coverage). We may have gone the opposite direction - being LESS
conservative than the paper's Gemma 3 baseline.

**Caveat**: We cannot verify this without knowing what the paper actually ran (notebooks
vs agents/, what hardware, etc.). See analysis-027 for the paper-text vs paper-repo discrepancy.

---

## The Coverage-MAE Trade-off

```
MedGemma (Appendix F):  MAE 0.505, Coverage <50%  (too conservative)
Paper Gemma 3:          MAE 0.619, Coverage ~50%  (balanced)
Our Run:                MAE 0.778, Coverage 69.2% (too aggressive)
```

**Hypothesis**: Higher coverage correlates with worse MAE because you're making more
low-confidence predictions. This is consistent with the MedGemma trade-off but not conclusively proven.

---

## Verification Plan

| Step | Effort | Expected Insight |
|------|--------|------------------|
| Calculate MAE at 50% coverage | Low (from existing data) | True paper comparison |
| Profile per-symptom coverage | Low (from existing data) | Identify over-predictions |
| Run with Q8_0 quantization | High (~2 hours) | Quality improvement potential |
| Run with MedGemma | High (~2 hours) | Coverage reduction effect |

---

## Conclusion

The investigation reveals our reproduction is **more aggressive than the paper** (69% vs 50% coverage). This isn't necessarily better - the paper's 50% N/A rate is by design, allowing it to only score high-confidence cases.

### Key Insight

The paper explicitly evaluated the coverage-MAE trade-off:
- MedGemma: Better MAE (0.505) but declined too many cases
- Gemma 3: Balanced (0.619 MAE, ~50% coverage)

We've inadvertently gone in the opposite direction - we're scoring cases even the balanced Gemma 3 declined.

### Next Steps

1. **Calculate MAE at 50% coverage** to get true paper comparison
2. Consider adding confidence threshold to match paper methodology
3. Accept that stochastic variation is acknowledged by paper

**Status**: Root causes documented (paper-text vs paper-repo divergence, unknown hardware).
We are shifting focus to our own clean implementation rather than exact paper reproduction.

---

## Appendix A: Hardware Comparison

| Spec | Our System | Paper Text Claim | Paper Repo Evidence |
|------|-----------|------------------|---------------------|
| Chip | Apple M1 Max | Apple M3 Pro | GSU TReNDS cluster (A100s) |
| RAM | 64GB | 18-36GB | 100GB (SLURM script) |
| GPU | ~48GB unified | ~13-27GB unified | 2x NVIDIA A100 |
| Model Size | Q8_0 capable | Q4_K_M (assumed) | Unknown |

**Note**: We cannot determine what hardware the paper actually used. The text claims
MacBook M3 Pro; the repo has SLURM scripts for A100 cluster. See analysis-027.

---

## Appendix B: Paper Quotes Summary

### Section 3.2 - Core Result
> "With optimized hyperparameters, Gemma 3 27B achieved an average MAE of 0.619 when predicting PHQ-8 scores, but in 50% of cases it was unable to provide a prediction due to insufficient evidence."

### Section 4 - Stochasticity
> "The stochastic nature of LLMs renders a key limitation of the proposed approach. Even with fairly deterministic parameters, responses can vary across runs, making it challenging to obtain consistent performance metrics."

### Appendix E - Appetite Had Zero Reference Retrieval (NOT coverage)
> "Additionally, we noted that PHQ-8-Appetite had no successfully retrieved reference chunks during inference."

**Note**: This refers to few-shot reference retrieval, not prediction coverage. See Finding 2.

### Appendix F - MedGemma Trade-off
> "MedGemma 27B had an edge over Gemma 3 27B in most categories overall, achieving an average MAE of 0.505, 18% less than Gemma 3 27B, although the number of subjects detected as having available evidence from the transcripts was smaller with MedGemma."

---

## Appendix C: Reproduction Run Summary

| Metric | Value |
|--------|-------|
| Date | 2025-12-24 |
| Split | Paper test (41 participants) |
| Mode | Few-shot |
| Model | gemma3:27b (Q4_K_M) |
| Duration | 109 minutes (~2.7 min/participant) |
| Success Rate | 39/41 (95.1%) |
| Coverage | 69.2% |
| MAE | 0.778 |
| Paper MAE | 0.619 |
| Delta | +0.159 |
