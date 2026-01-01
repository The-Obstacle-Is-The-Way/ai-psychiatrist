# GAP-001: Paper Unspecified Parameters

**Date**: 2025-12-22
**Status**: ✅ RESOLVED / ARCHIVED
**Severity**: MEDIUM - Affects exact reproducibility but not system validity
**Resolved**: 2025-12-26 - All gaps now documented in dedicated reference docs
**Tracked by**:
- [GitHub Issue #46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46) (sampling parameters)
- [GitHub Issue #47](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/47) (model quantization)

---

## Resolution

This investigation is complete. All gaps have been resolved and documented in dedicated SSOT docs:

| Gap | Resolution | SSOT Location |
|-----|------------|---------------|
| GAP-001a (Data Split) | ✅ Reverse-engineered exact IDs from authors' output files | [`docs/data/paper-split-registry.md`](../../data/paper-split-registry.md) |
| GAP-001b (Temperature) | ✅ Evidence-based: temp=0.0 for clinical AI | [`docs/reference/agent-sampling-registry.md`](../../reference/agent-sampling-registry.md) |
| GAP-001c (top_k/top_p) | ✅ Don't set (irrelevant at temp=0; best practice) | [`docs/reference/agent-sampling-registry.md`](../../reference/agent-sampling-registry.md) |
| GAP-001d (Hardware) | ✅ Documented Q4_K_M vs BF16 quantization | [`docs/models/model-registry.md`](../../models/model-registry.md) |

**The content below is retained for historical context.**

---

## Original Investigation (Historical)

This document captured ALL parameters NOT explicitly specified in the paper, along with our implementation decisions and rationales.

---

## Summary of Paper Gaps

| Gap ID | Parameter | Paper Says | Our Implementation | Status |
|--------|-----------|-----------|-------------------|--------|
| GAP-001a | Data Split Membership | "58/43/41 stratified" but no participant IDs | `scripts/create_paper_split.py` | ✅ Reproducible algorithm |
| GAP-001b | Temperature | "fairly deterministic" | 0.0 (all agents) | ✅ Evidence-based |
| GAP-001c | top_k / top_p | Not in paper text | Not set (irrelevant at temp=0) | ✅ Best practice |
| GAP-001d | Hardware / Quantization | Not specified | Local Ollama defaults | ⚠️ May affect results |

---

## GAP-001a: Exact Data Split Membership Lists

### What the Paper Says

**Section 2.4.1:**
> "We split 142 subjects with eight-item PHQ-8 scores from the DAIC-WOZ database into training, validation, and test sets. [...] We used a 41% training (58 participants), 30% validation (43), and 29% test (41) split"

**Appendix C:**
> "We stratified 142 subjects from the DAIC-WOZ training and development sets into training, validation, and test sets based on PHQ-8 total scores and gender information. [...] For PHQ-8 total scores with two participants, we put one in the validation set and one in the test set. For PHQ-8 total scores with one participant, we put that one participant in the training set."

### What's NOT Specified

1. **Exact participant IDs** for each split
2. **Random seed** used for stratification
3. **Ordering/tie-breaking** when multiple participants have same PHQ-8 score and gender

### Impact

- We cannot guarantee identical splits to the paper
- MAE results may differ due to different test participants
- But the METHODOLOGY is reproducible

### Our Implementation

We implement the algorithm described in Appendix C:

1. Group participants by PHQ-8 total score (Appendix C is described per total score)
2. For groups with 1 participant → assign to training
3. For groups with 2 participants → one to validation, one to test
4. For groups with 3+ participants → initial proportional allocation, then deterministic rebalancing
   to hit **exact 58/43/41** targets while maintaining approximate stratification
5. Gender balancing pass (deterministic swaps on flexible IDs) to better match overall gender ratio

**Location**: `scripts/create_paper_split.py`

### Actual Results (seed=42)

```text
Paper Target:  58 train (41%) / 43 val (30%) / 41 test (29%)
Our Result:    58 train (41%) / 43 val (30%) / 41 test (29%)
```

Exact split membership will still differ from the paper because:
- The paper does not publish the participant ID lists for each split
- The paper does not publish the random seed or tie-breaking rules

### Justification

- Follows the paper's algorithm exactly as described
- Uses fixed random seed (42) for reproducibility
- Documents that exact membership may differ from paper
- Ensures exact reported split sizes (58/43/41)

---

## GAP-001b: Sampling Parameters (Temperature)

### What the Paper Says

**Section 4 (Discussion):**
> "The stochastic nature of LLMs renders a key limitation of the proposed approach. Even with fairly deterministic parameters, responses can vary across runs, making it challenging to obtain consistent performance metrics."

### What's NOT Specified

- Exact temperature value
- Whether different agents use different temperatures
- Any temperature tuning methodology

### Our Implementation (Updated 2025-12-24)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `temperature` | 0.0 | Clinical AI best practice |

**Location**: `src/ai_psychiatrist/config.py` (`ModelSettings.temperature`)

### Justification (Evidence-Based)

- Med-PaLM uses temp=0.0 for clinical answers ([Nature Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC11922739/))
- "Lower temperatures promote diagnostic accuracy" ([medRxiv 2025](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full))
- Anthropic: "temp 0.0 for analytical / multiple choice"
- See [Agent Sampling Registry](../../reference/agent-sampling-registry.md) for full citations

---

## GAP-001c: Sampling Parameters (top_k, top_p)

### What the Paper Says

Nothing. These parameters are not mentioned.

### What Their Code Does (Reference Only)

Their codebase has contradictory values:

| Source | top_k | top_p | Notes |
|--------|-------|-------|-------|
| `basic_quantitative_analysis.ipynb:207` | 1 | 1.0 | Zero-shot |
| `embedding_quantitative_analysis.ipynb:1174` | 20 | 0.8 | Few-shot |
| `qual_assessment.py` | 20 | 0.9 | Wrong model default |
| `meta_review.py` | 20 | 1.0 | Wrong model default |

### Our Implementation (Updated 2025-12-24)

**We do NOT set top_k or top_p.**

### Justification (Evidence-Based)

1. **At temp=0, they're irrelevant** — greedy decoding ignores sampling filters
2. **Best practice: use temperature only** — "top_k is recommended for advanced use cases only. You usually only need to use temperature" ([Anthropic](https://www.prompthub.us/blog/using-anthropic-best-practices-parameters-and-large-context-windows))
3. **Don't use both** — "You should alter either temperature or top_p, but not both" ([AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html))
4. **Claude 4.x enforces this** — Returns error: "temperature and top_p cannot both be specified"
5. **top_k is obsolete** — "not as well-supported, notably missing from OpenAI's API" ([Vellum](https://www.vellum.ai/llm-parameters/temperature))

See [Agent Sampling Registry](../../reference/agent-sampling-registry.md) for full citations

---

## GAP-001d: Hardware / Quantization

### What the Paper Says

**Section 2.2:**
> "We utilized a state-of-the-art open-weight language model, Gemma 3 with 27 billion parameters (Gemma 3 27B)"

No mention of quantization.

### Our Implementation

**Default (Ollama):**
- Chat: `gemma3:27b` (Q4_K_M quantization, ~16GB)
- Embeddings: `qwen3-embedding:8b` (Q4_K_M quantization, ~4.7GB)

**High-Quality (HuggingFace):**
- Chat: `google/medgemma-27b-text-it` (FP16, 18% better MAE per Appendix F)
- Embeddings: `Qwen/Qwen3-Embedding-8B` (FP16, higher precision similarity)

See [Model Registry - High-Quality Setup](../../models/model-registry.md#high-quality-setup-recommended-for-production) and [Issue #42](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/42) for graceful fallback.

### Justification

- The paper states the pipeline can run on a MacBook Pro M3 Pro (Section 2.3.5), but the public repo
  also includes SLURM scripts configured for multi‑GPU nodes (e.g., `_reference/slurm/job_ollama.sh` uses
  `--gres=gpu:A100:2`).
- The hardware and precision/quantization used for the reported MAE/coverage are therefore not uniquely
  determined from the paper text alone.
- Quantization/precision can materially change both MAE and coverage (abstention rate).

### Alternative (For Maximum Fidelity)

Users can use the HuggingFace backend with official weights:
```bash
LLM_BACKEND=huggingface
LLM_HF_QUANTIZATION=int4  # or int8 for higher precision
```

---

## Impact on Reproducibility

### Results We CAN Reproduce

1. ✅ Item-level MAE methodology (documented in Section 3.2)
2. ✅ N/A exclusion behavior (documented throughout)
3. ✅ Hyperparameters: Nchunk=8, Nexample=2, Ndimension=4096 (Appendix D)
4. ✅ Feedback loop: max_iterations=10, threshold<4 (Section 2.3.1)

### Results That May Differ

1. ⚠️ Exact MAE values (due to different splits, model variance)
2. ⚠️ Per-participant predictions (stochastic LLM behavior)
3. ⚠️ Coverage percentage (depends on model confidence thresholds)

### Acceptable Variance

The paper acknowledges stochasticity (“responses can vary across runs”), but it does not define
an explicit tolerance band for reproduction.

For internal sanity-checking only (heuristic, not a paper claim):
- Treat MAE deltas on the order of ~0.1 as “plausibly within run-to-run + implementation drift”
- Treat coverage deltas on the order of ~10% as “plausibly within denominator/behavior drift”

---

## Configuration SSOT

All parameters are now documented with explicit rationales:

```python
# src/ai_psychiatrist/config.py

# PAPER SPECIFIED (Appendix D)
chunk_size: int = 8          # Paper: "Nchunk = 8"
chunk_step: int = 2          # Paper: "step size of 2 lines"
top_k_references: int = 2    # Paper: "Nexample = 2"
dimension: int = 4096        # Paper: "Ndimension = 4096"

# PAPER SPECIFIED (Section 2.3.1)
max_iterations: int = 10     # Paper: "limited to a maximum of 10 iterations"
score_threshold: int = 3     # Paper: "score was below four" → ≤3

# NOT SPECIFIED - Using evidence-based clinical AI defaults
temperature: float = 0.0     # Med-PaLM, medRxiv 2025: temp=0 for clinical AI
# top_k and top_p: NOT SET (irrelevant at temp=0, best practice is temp only)
```

See [Agent Sampling Registry](../../reference/agent-sampling-registry.md) for full citations.

---

## Action Items

1. ✅ **Split algorithm implemented**: `scripts/create_paper_split.py`
2. ✅ **Reproduction CLI supports paper splits**: `scripts/reproduce_results.py --split paper*`
3. ✅ **Embedding generation supports paper train**: `scripts/generate_embeddings.py --split paper-train`
4. **Document split provenance**: keep `paper_split_metadata.json` (seed, IDs) as the SSOT for a run
5. **Consider GitHub issue**: request exact split membership from authors (would enable exact parity)

---

## Related Issues

- [bug-018-reproduction-friction.md](./bug-018-reproduction-friction.md) - Reproduction Friction Log
- [reproduction-results.md](../../results/reproduction-results.md) - Reproduction results

---

## References

- Paper Section 2.2: Model specification
- Paper Section 2.4.1: Data splitting methodology
- Paper Section 4: Discussion of stochasticity
- Paper Appendix C: Stratification algorithm
- Paper Appendix D: Hyperparameter optimization
