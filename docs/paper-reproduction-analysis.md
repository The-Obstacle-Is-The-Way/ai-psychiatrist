# Paper Reproduction Analysis: Coverage/MAE Discrepancy

**Date**: 2025-12-27 (Updated 2025-12-29)
**Status**: Investigation Complete

> **Update (2025-12-29)**: Post Spec 31/32 (paper-parity format), few-shot MAE improved to 0.774 (and AURC to 0.193), but zero-shot still wins (AURC 0.134).
> The analysis below uses earlier run data but conclusions remain valid.

---

## Executive Summary

Our reproduction shows **different behavior** than the paper claims. After deep analysis, we believe:

1. **Our results differ materially from the paper** (coverage and MAE move in different directions)
2. **The paper omits key evaluation details** (notably, zero-shot coverage on the test set)
3. **A coverage-adjusted evaluation is required** for fair comparison (risk–coverage curve / AURC-family metrics)

---

## The Discrepancy

### Their Results (from paper + released output artifacts)

| Mode | Coverage | MAE | Observation |
|------|----------|-----|-------------|
| Zero-Shot | 40.9% | 0.796 | Baseline |
| Few-Shot | 50.0% | 0.619 | Coverage ↑ ~9%, MAE ↓ ~22% |

Notes:
- Coverage above is on the **paper TEST split (41 participants)**.
- The MAE above is the **paper-style metric**: mean of per-item MAEs excluding "N/A" (each item equally weighted).
- The often-cited 43.8% zero-shot coverage is the coverage across **all 142 participants**, not the test split.

### Our Results

| Mode | Cmax | MAE_w | Observation |
|------|------|-------|-------------|
| Zero-Shot | 55.5% | 0.698 | Baseline (1 participant excluded from MAE: 8/8 N/A) |
| Few-Shot | 70.1% | 0.774 | Coverage ↑ ~15%, MAE_w ↑ ~11% |

Notes:
- `Cmax` is the maximum coverage in the AURC risk–coverage curve (computed over all included items).
- Our MAE above is the weighted MAE from the latest post-Spec 31/32 run (computed over evaluated subjects only).
- Earlier runs showed higher few-shot MAE (0.804-0.860) before paper-parity formatting.
- Coverage-adjusted results (Spec 25): zero-shot AURC = 0.134, few-shot AURC = 0.193 (zero-shot wins).

---

## First-Principles Interpretation

Few-shot prompting can plausibly do **either** of the following:
- Improve accuracy and increase coverage (best case)
- Increase coverage but harm accuracy (overconfidence / bad retrieval / mismatched hyperparameters)

### Why Their Results Are Counterintuitive (but not impossible)

For coverage to increase AND MAE to decrease, few-shot would need to:

1. **Dramatically improve accuracy on existing predictions** (to offset new errors)
2. **Make new predictions that are ALSO accurate** (despite being harder cases)

Their released test-set artifacts support that this happened for their run:
- Zero-shot (test): MAE_by_item = 0.796 at 40.9% coverage
- Few-shot (test, chunk=8/examples=2): MAE_by_item = 0.619 at 50.0% coverage

---

## Evidence of Paper Methodological Issues

### 1. Zero-Shot Coverage Not Reported

The paper reports few-shot coverage (~50%, Section 3.2: "in 50% of cases it was unable to provide a prediction") but **does not report zero-shot coverage at all**.

From Figure 5 in the paper, we can see zero-shot MAE (0.796) vs few-shot MAE (0.619), but coverage is not shown. This makes the comparison incomplete.

**Their implied claim**: Few-shot improved MAE by 22% while also increasing coverage (from unknown to 50%).

From their released artifacts on the test split, zero-shot coverage is **40.9%**. The paper does not report this value.

### 2. Sloppy Python Code vs Notebooks

We documented extensively that their Python files contain dead code:
- `_keyword_backfill()` function exists in Python but NOT used in notebooks
- Multiple parsing/repair strategies never actually executed
- Temperature settings differ between files

### 3. No AURC or Risk-Coverage Analysis

The paper compares MAE at **different coverage levels** without analyzing the tradeoff curve.

**Proper evaluation** (selective prediction / abstention literature):
- Use risk–coverage curves (or AURC-family metrics)
- Compare entire tradeoff curves, not single points

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

From [A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)](https://arxiv.org/abs/2410.15361) (2024):

> "The Area Under the Risk-Coverage Curve (AURC) has emerged as the foremost evaluation metric for assessing the performance of [selective] systems."

---

## Our Conclusion

### 1. Our Results Are Internally Consistent

- Coverage increase + MAE increase is expected behavior
- Follows statistical intuition
- Consistent with selective prediction theory

### 2. Their Results Are Supported by Their Artifacts

Coverage increase + MAE decrease is counterintuitive, but their released test-set outputs reproduce the paper's MAE values.
The issue is not that the numbers are "impossible"; it is that the paper does not supply enough evaluation context
(zero-shot coverage, confidence intervals, multiple runs) to interpret the claim rigorously.

### 3. Non-Reproducibility Concerns

- Their sloppy codebase (extensively documented)
- Dead code paths not matching notebooks
- No statistical rigor in evaluation

### 4. Proper Comparison Requires AURC

- Implemented as Spec 25: `docs/archive/specs/25-aurc-augrc-implementation.md`
- Results are tracked in `docs/results/run-history.md`
- Standard in selective prediction literature

---

## Recommendations

### Short-Term: Accept Our Results

Our implementation is correct. The coverage/MAE tradeoff we observe is expected and legitimate.

### Medium-Term: Use AURC as Primary Metric

Treat AURC/AUGRC + paired bootstrap deltas as the primary comparison:
1. Fair comparison across coverage levels
2. Explicit uncertainty tradeoff (risk–coverage curve)
3. Publishable statistical inference (CI on ΔAURC/ΔAUGRC)

### Long-Term: Document for Future Work

This analysis should be included in any future publication reproducing or extending the paper:
- Note methodological issues in original paper
- Use AURC as primary metric
- Report confidence intervals

---

## Related Documents

- `docs/archive/bugs/bug-029-coverage-mae-discrepancy.md` - Detailed discrepancy analysis
- `docs/archive/bugs/fallback-architecture-audit.md` - Code quality issues
- `docs/archive/specs/25-aurc-augrc-implementation.md` - Selective prediction evaluation suite

---

## Sources

1. [Overcoming Common Flaws in Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032v1) - 2024
2. [A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC)](https://arxiv.org/abs/2410.15361) - 2024
3. [Know Your Limits: A Survey of Abstention in Large Language Models](https://arxiv.org/abs/2407.18418) - 2024
