# GAP-001: Paper Unspecified Parameters

**Date**: 2025-12-22
**Status**: DOCUMENTED - Implementations proposed with explicit rationales
**Severity**: MEDIUM - Affects exact reproducibility but not system validity
**Updated**: 2025-12-24 - Added paper-repo hardcoded sampling + hardware evidence
**Tracked by**:
- [GitHub Issue #46](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/46) (sampling parameters)
- [GitHub Issue #47](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/47) (model quantization)

This document captures ALL parameters NOT explicitly specified in the paper, along with our implementation decisions and rationales.

---

## Summary of Paper Gaps

| Gap ID | Parameter | Paper Says | Our Implementation | Status |
|--------|-----------|-----------|-------------------|--------|
| GAP-001a | Data Split Membership | "58/43/41 stratified" but no participant IDs | `scripts/create_paper_split.py` | ✅ Reproducible algorithm |
| GAP-001b | Temperature | "fairly deterministic" | 0.2 (few-shot), 0.0 (judge/zero-shot) | ✅ Matches notebooks |
| GAP-001c | top_k / top_p | Not in paper text | Few-shot: 20/0.8, Zero-shot: 1/1.0 | ✅ Matches notebooks |
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

### Our Implementation

| Setting | Value | Rationale |
|---------|-------|-----------|
| `temperature` | 0.2 | Low but not zero; allows slight variation |
| `temperature_judge` | 0.0 | Judge should be deterministic for consistency |

**Location**: `src/ai_psychiatrist/config.py` (`ModelSettings.temperature`, `ModelSettings.temperature_judge`)

### Justification

- "Fairly deterministic" → temperature near 0
- 0.2 is a common "low temperature" choice in clinical NLP
- Judge uses 0.0 for maximum consistency in scoring
- These values are explicit and can be tuned

---

## GAP-001c: Sampling Parameters (top_k, top_p)

### What the Paper Says

Nothing. These parameters are not mentioned.

### What the Public Paper Repo Code Does

Although the paper text does not specify `top_k`/`top_p`, the publicly available repository
does hardcode sampling options. **SSOT: Notebooks take precedence over .py files** (the .py files
have wrong model defaults like `llama3` but sampling params happen to match).

| Mode | Source (SSOT) | Temperature | top_k | top_p | Verified |
|------|---------------|-------------|-------|-------|----------|
| **Zero-shot** | `_reference/quantitative_assessment/basic_quantitative_analysis.ipynb` (line 207) | 0 | 1 | 1.0 | ✅ 2025-12-24 |
| **Few-shot** | `_reference/quantitative_assessment/embedding_quantitative_analysis.ipynb` (line 1174) | 0.2 | 20 | 0.8 | ✅ 2025-12-24 |

**Cross-verified with .py files** (which match for sampling params despite wrong model defaults):
- `_reference/quantitative_assessment/quantitative_analysis.py`: `temp=0, top_k=1, top_p=1.0` ✅
- `_reference/agents/quantitative_assessor_f.py`: `temp=0.2, top_k=20, top_p=0.8` ✅

**Note**: `basic_quantitative_analysis.ipynb` line 599 shows `temp=0.1, top_k=10` which appears
to be experimental, NOT the production setting. Line 207 is labeled "Most deterministic".

### Why Different Settings for Zero-shot vs Few-shot?

From first principles:
- **Zero-shot** (`temp=0, top_k=1, top_p=1.0`): Fully deterministic because no grounding examples.
  Must maximize consistency since the model has no reference points.
- **Few-shot** (`temp=0.2, top_k=20, top_p=0.8`): Slight flexibility allowed because the model is
  grounded by retrieved examples. Small temperature helps navigate edge cases.

### Our Implementation

| Setting | Value | Rationale |
|---------|-------|-----------|
| `top_k` | 20 | Matches few-shot notebook defaults |
| `top_p` | 0.8 | Matches few-shot notebook defaults |

**Location**: `src/ai_psychiatrist/config.py` (`ModelSettings.top_k`, `ModelSettings.top_p`)

### Gap Status: ✅ ADDRESSED (2025-12-24)

Zero-shot specific settings added to config:

| Mode | Paper Repo | Our Implementation | Gap? |
|------|------------|-------------------|------|
| Zero-shot | `temp=0, top_k=1, top_p=1.0` | `MODEL_TEMPERATURE_ZERO_SHOT=0.0`, etc. | **FIXED** |
| Few-shot | `temp=0.2, top_k=20, top_p=0.8` | `temp=0.2, top_k=20, top_p=0.8` | No |

New settings in `config.py`:
- `MODEL_TEMPERATURE_ZERO_SHOT` (default: 0.0)
- `MODEL_TOP_K_ZERO_SHOT` (default: 1)
- `MODEL_TOP_P_ZERO_SHOT` (default: 1.0)

### Justification

- These match the public repo few-shot agent defaults (notebooks as SSOT)
- With low temperature (0.2), these have limited effect but can still shift abstention behavior
- Can be overridden via environment variables for ablations

---

## GAP-001d: Hardware / Quantization

### What the Paper Says

**Section 2.2:**
> "We utilized a state-of-the-art open-weight language model, Gemma 3 with 27 billion parameters (Gemma 3 27B)"

No mention of quantization.

### Our Implementation

- Use local Ollama (quantized GGUF weights for `gemma3:27b`)
- Use local Ollama embeddings (`qwen3-embedding:8b`) with requested dimension 4096

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

	# NOT SPECIFIED - Using justified defaults
	temperature: float = 0.2     # Paper: "fairly deterministic" → low temp
	temperature_judge: float = 0.0  # Deterministic for consistency
	top_k: int = 20              # Matches paper repo few-shot defaults (paper text unspecified)
	top_p: float = 0.8           # Matches paper repo few-shot defaults (paper text unspecified)
```

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
- [reproduction-notes.md](../results/reproduction-notes.md) - Reproduction results

---

## References

- Paper Section 2.2: Model specification
- Paper Section 2.4.1: Data splitting methodology
- Paper Section 4: Discussion of stochasticity
- Paper Appendix C: Stratification algorithm
- Paper Appendix D: Hyperparameter optimization
