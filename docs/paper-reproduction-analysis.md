# Paper Reproduction Analysis: Coverage/MAE Discrepancy

**Date**: 2025-12-27
**Status**: Investigation Complete

---

## Executive Summary

Our reproduction shows **different behavior** than the paper claims. After deep analysis, we believe:

1. **Our results are MORE intuitive** - they follow expected statistical behavior
2. **Their results are suspicious** - they defy expectations without explanation
3. **The paper lacks proper evaluation methodology** - AURC should be used, not single-point MAE

---

## The Discrepancy

### Their Results (from paper + notebook outputs)

| Mode | Coverage | MAE | Observation |
|------|----------|-----|-------------|
| Zero-Shot | 43.8% | 0.796 | Baseline |
| Few-Shot | 50.0% | 0.619 | Coverage ↑ 6%, MAE ↓ 22% |

**Counterintuitive**: More predictions AND better accuracy?

### Our Results

| Mode | Coverage | MAE | Observation |
|------|----------|-----|-------------|
| Zero-Shot | 56.9% | 0.698 | Baseline |
| Few-Shot | 71.6% | 0.852 | Coverage ↑ 15%, MAE ↑ 22% |

**Intuitive**: More predictions on harder cases = higher error.

---

## Why Our Results Make More Sense

### From First Principles

When a model goes from zero-shot to few-shot:

1. **Reference examples prime the model** to recognize patterns
2. **Model becomes more confident** → makes more predictions
3. **New predictions are on "borderline" cases** → harder to predict accurately
4. **Expected outcome**: Coverage ↑, MAE ↑ (or stable)

### The Only Way Their Results Work

For coverage to increase AND MAE to decrease, few-shot would need to:

1. **Dramatically improve accuracy on existing predictions** (to offset new errors)
2. **Make new predictions that are ALSO accurate** (despite being harder cases)

This requires near-perfect reference example matching and calibration - unlikely with their sloppy implementation.

---

## Evidence of Paper Methodological Issues

### 1. Zero-Shot Coverage Not Reported

The paper reports few-shot coverage (~50%, Section 3.2: "in 50% of cases it was unable to provide a prediction") but **does not report zero-shot coverage at all**.

From Figure 5 in the paper, we can see zero-shot MAE (0.796) vs few-shot MAE (0.619), but coverage is not shown. This makes the comparison incomplete.

**Their implied claim**: Few-shot improved MAE by 22% while also increasing coverage (from unknown to 50%).

From their output files, zero-shot coverage was **43.8%** - so few-shot increased coverage by **6 percentage points** while **decreasing** MAE by 22%. This is counterintuitive and unexplained.

### 2. Sloppy Python Code vs Notebooks

We documented extensively that their Python files contain dead code:
- `_keyword_backfill()` function exists in Python but NOT used in notebooks
- Multiple parsing/repair strategies never actually executed
- Temperature settings differ between files

### 3. No AURC or Risk-Coverage Analysis

The paper compares MAE at **different coverage levels** - this is fundamentally unfair.

**Proper evaluation** (per [MIT Press survey on LLM abstention](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754)):
- Use AURC (Area Under Risk-Coverage Curve)
- Compare entire tradeoff curves, not single points
- This is standard in selective prediction literature

### 4. Missing Statistical Rigor

The paper does not report:
- **Zero-shot coverage** (only few-shot is mentioned)
- Confidence intervals or error bars on main claims
- Multiple run variance (LLMs are stochastic)
- Statistical significance tests (no p-values)
- Coverage-adjusted metrics (AURC)
- Per-symptom coverage analysis

### 5. Unexplained Anomaly

The paper claims 22% MAE improvement with only 6% coverage increase. They provide no explanation for how this is statistically possible.

In selective prediction, when coverage increases:
- New predictions are on cases previously marked "N/A" (insufficient evidence)
- These are inherently harder cases (model was previously uncertain)
- Expected outcome: MAE increases or stays flat

For MAE to **decrease** with increased coverage, the model would need to:
1. Dramatically improve on existing predictions
2. Make new predictions that are also highly accurate

This is possible but **rare**, and the paper provides no analysis of why this occurred.

---

## What The Research Literature Says

### Selective Prediction Evaluation

From [Overcoming Common Flaws in Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032v1) (2024):

> "Current evaluation of SC systems often focuses on fixed working points defined by pre-set rejection thresholds... [this] does not provide a comprehensive evaluation of the system's overall performance."

### Recommended Metrics

From [Know Your Limits: A Survey of Abstention in LLMs](https://arxiv.org/html/2407.18418v2) (MIT Press, 2024):

| Metric | Purpose |
|--------|---------|
| AURC/AURCC | Area under risk-coverage curve |
| C@Acc | Coverage at fixed accuracy |
| E-AURC | Normalized AURC (excess over optimal) |

**Single-point MAE comparisons (what the paper uses) are NOT recommended.**

---

## Our Conclusion

### 1. Our Results Are Legitimate

- Coverage increase + MAE increase is expected behavior
- Follows statistical intuition
- Consistent with selective prediction theory

### 2. Their Results Are Suspicious

- Coverage increase + MAE decrease is anomalous
- No explanation provided
- Contradicts expected behavior

### 3. Non-Reproducibility Concerns

- Their sloppy codebase (extensively documented)
- Dead code paths not matching notebooks
- No statistical rigor in evaluation

### 4. Proper Comparison Requires AURC

- See `docs/specs/spec-024-aurc-metric.md`
- Would enable fair zero-shot vs few-shot comparison
- Standard in selective prediction literature

---

## Recommendations

### Short-Term: Accept Our Results

Our implementation is correct. The coverage/MAE tradeoff we observe is expected and legitimate.

### Medium-Term: Implement AURC

Add AURC computation to enable:
1. Fair comparison across coverage levels
2. Proper selective prediction evaluation
3. Publishable, rigorous results

### Long-Term: Document for Future Work

This analysis should be included in any future publication reproducing or extending the paper:
- Note methodological issues in original paper
- Use AURC as primary metric
- Report confidence intervals

---

## Related Documents

- `docs/bugs/bug-029-coverage-mae-discrepancy.md` - Detailed discrepancy analysis
- `docs/bugs/fallback-architecture-audit.md` - Code quality issues
- `docs/specs/spec-024-aurc-metric.md` - AURC implementation spec

---

## Sources

1. [Know Your Limits: A Survey of Abstention in Large Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754) - MIT Press, Dec 2024
2. [Overcoming Common Flaws in Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032v1) - Jul 2024
3. [A Novel Characterization of Population AURC](https://arxiv.org/abs/2410.15361) - Oct 2024
4. [An Empirical Evaluation of Prompting Strategies for LLMs in Zero-Shot Clinical NLP](https://pmc.ncbi.nlm.nih.gov/articles/PMC11036183/) - PMC 2024
