# BUG-027: Consistency Sampling Temperature May Be Suboptimal for Clinical Accuracy

**Status**: ✅ Resolved (Implemented)
**Severity**: P2 (Optimization)
**Filed**: 2026-01-04
**Resolved**: 2026-01-04
**Component**: `src/ai_psychiatrist/config.py`, `.env.example`

---

## Summary

The consistency sampling feature (Spec 050) used `CONSISTENCY_TEMPERATURE=0.3`, which is likely higher than optimal for clinical diagnostic accuracy. 2025-2026 medical research suggests that temperatures in the 0.1-0.2 range provide better diagnostic accuracy while still enabling sufficient variance for self-consistency signals.

**Resolution**: Update the baseline to `CONSISTENCY_TEMPERATURE=0.2` (docs + code + tests) to align with low-variance clinical scoring guidance while preserving multi-sample diversity.

---

## Fix Implemented

1. Baseline default updated:
   - `src/ai_psychiatrist/config.py`: `ConsistencySettings.temperature` default `0.3 → 0.2`
2. Runbook/config updated:
   - `.env.example`: `CONSISTENCY_TEMPERATURE=0.2`
   - `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `NEXT-STEPS.md`: updated examples and checklists
3. Regression prevention:
   - `tests/unit/test_bootstrap.py`: asserts `.env.example` contains `CONSISTENCY_TEMPERATURE=0.2`
   - `tests/unit/test_config.py`: asserts schema default is `0.2`

---

## Current Configuration

```bash
# .env.example:114
MODEL_TEMPERATURE=0.0      # Primary inference (CORRECT)

# .env.example:138
CONSISTENCY_TEMPERATURE=0.2  # Consistency sampling (low-variance baseline)
```

**Primary inference at 0.0**: Correct per [Med-PaLM best practices](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full).

**Consistency sampling at 0.3**: May introduce unnecessary diagnostic variance.

---

## Research Evidence

### 2025 Medical Research Findings

| Study | Finding | Source |
|-------|---------|--------|
| Emergency Diagnostic Accuracy (medRxiv 2025) | "At temperature 0.0, GPT-4o achieved 100% leading diagnosis accuracy. As temperature increased, accuracy declined systematically to 89.4% at temperature 1.0." | [medRxiv 2025.06.04.25328288](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full) |
| Same study | "Diagnostic divergence increased 583% from temp 0.0 to 1.0" | Same |
| GPT-4 Clinical Depression Assessment | "Optimal performance observed at lower temperature values (0.0-0.2) for complex prompts. Beyond temperature >= 0.3, the relationship becomes unpredictable." | [arXiv 2501.00199](https://arxiv.org/abs/2501.00199) |

### Self-Consistency Trade-off

| Context | Temperature | Rationale |
|---------|-------------|-----------|
| Original self-consistency papers | 0.5-0.7 | General reasoning tasks, maximize diversity |
| Clinical accuracy (single-shot) | 0.0-0.2 | Maximize diagnostic accuracy |
| **Our consistency sampling** | **0.3** | **Middle ground, but may lean too high** |

---

## The Trade-off

Self-consistency REQUIRES non-zero temperature to generate diverse reasoning paths:

```
If temperature = 0.0:
  All N samples are identical (deterministic)
  No variance = no consistency signal
  Feature becomes useless

If temperature too high (0.5+):
  High variance = diverse paths
  BUT: Introduces diagnostic errors
  Medical research shows accuracy drops significantly
```

**The question**: Is 0.3 the right balance, or should we use 0.1-0.2?

---

## Impact Analysis

### Current Behavior
- 5 samples at temperature 0.3
- Generates variance for agreement-based confidence
- May introduce 5-10% additional diagnostic variance vs. 0.0

### Potential Improvement
- Lower to 0.1-0.2
- Still generates variance for consistency signals
- Better alignment with medical best practices
- Potential accuracy improvement (needs empirical validation)

---

## Proposed Fix

### Recommended: Temperature 0.2

Based on 2025 clinical research, **0.2 is the recommended value**:

```bash
# .env.example
CONSISTENCY_TEMPERATURE=0.2  # Clinical best practice for low-variance sampling
```

**Rationale**:
1. A 2025 clinical reasoning assessment study explicitly categorized temperatures as:
   - **Low: 0.2** (recommended for accuracy)
   - **High: 0.7** (for creativity/exploration)

   Source: [How Model Size, Temperature, and Prompt Style Affect LLM-Human Assessment Score Alignment](https://arxiv.org/html/2509.19329)

2. The Emergency Diagnostic Accuracy study (medRxiv 2025) tested:
   - 0.0, 0.25, 0.50, 0.75, 1.0
   - Best accuracy at 0.0, declining at each step
   - 0.25 is the next tested point after 0.0

   Our current 0.3 sits between 0.25 and 0.50 - slightly suboptimal.

3. ECG diagnostic studies used **temperature 0.2** for consistency across runs.

4. Anthropic's official guidance recommends **0.0-0.2** for analytical tasks.

### Why Not 0.1?
- Too low may not generate sufficient variance across 5 samples
- 0.2 is the established "low" threshold in clinical research
- Provides balance between variance and accuracy

### Why Not Keep 0.3?
- 0.3 is above the "low" threshold used in clinical studies
- Research specifically tested 0.2 vs 0.7 as low/high categories
- GPT-4 depression study noted performance becomes "unpredictable" at >= 0.3

---

## Code Locations

| File | Line | Current Value |
|------|------|---------------|
| `.env.example` | 138 | `CONSISTENCY_TEMPERATURE=0.2` |
| `.env` | 137 | `CONSISTENCY_TEMPERATURE=0.2` |
| `src/ai_psychiatrist/config.py` | 440-444 | `temperature: float = Field(default=0.2, ...)` |

---

## Validation (Optional)

Empirical testing can still be run (0.2 vs 0.3) to quantify impact on both accuracy and calibration, but the baseline is now aligned with the low-temperature clinical guidance cited below.

1. **Empirical testing**: Run comparison at 0.2 vs 0.3 on paper-test split
2. **Variance sufficiency**: Verify 0.2 still produces meaningful variance across 5 samples
3. **Consistency signal quality**: Check if lower temp degrades agreement-based confidence calibration

---

## Decision Points (If Revisited)

- [ ] Keep `0.2` (baseline) and tune other confidence signals first
- [ ] Evaluate adaptive temperature selection (see arXiv 2502.05234)
- [ ] Increase `n_samples` and reduce temperature further (if variance remains sufficient)

---

## References

1. [Temperature-Driven Variability in Emergency Diagnostic Accuracy (medRxiv 2025)](https://www.medrxiv.org/content/10.1101/2025.06.04.25328288v1.full) - Primary clinical evidence
2. [How Model Size, Temperature, and Prompt Style Affect LLM-Human Assessment (arXiv 2509.19329)](https://arxiv.org/html/2509.19329) - Defines 0.2 as "low" clinical threshold
3. [GPT-4 on Clinic Depression Assessment (arXiv 2501.00199)](https://arxiv.org/abs/2501.00199) - PHQ-8 specific, notes unpredictability at >= 0.3
4. [Optimizing Temperature for Multi-Sample Inference (arXiv 2502.05234)](https://arxiv.org/html/2502.05234v1) - TURN automated optimization
5. [Calibration Study in Biomedical Research (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.02.11.637373v1.full) - Self-consistency calibration
6. [Statistical Framework for LLM Consistency in Medical Diagnosis (medRxiv 2025)](https://www.medrxiv.org/content/10.1101/2025.08.06.25333170v1.full)
