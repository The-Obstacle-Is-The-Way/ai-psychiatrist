# Spec 046: Improve Selective Prediction Confidence Signals (AURC/AUGRC)

**Status**: Implemented (2026-01-02)
**Primary implementation**: `src/ai_psychiatrist/agents/quantitative.py`, `scripts/reproduce_results.py`, `scripts/evaluate_selective_prediction.py`
**SSOT metric definitions**: `docs/statistics/metrics-and-evaluation.md`

## 0. Problem Statement

This repository evaluates PHQ-8 scoring as a **selective prediction** system: each item can be predicted (`0–3`) or abstained (`N/A`). We compare systems using **risk–coverage curves** and integrated metrics (**AURC / AUGRC**) computed from per-item predictions and a scalar **confidence** ranking signal.

Today, confidence is derived only from **evidence counts**:

- `confidence_llm = llm_evidence_count`
- `confidence_total_evidence = llm_evidence_count + keyword_evidence_count`

This is implemented in `scripts/evaluate_selective_prediction.py` and documented in `docs/statistics/metrics-and-evaluation.md`.

In Run 8, few-shot substantially improves accuracy (MAE_item) but does not materially improve AURC/AUGRC under the current confidence signal, suggesting we are **leaving information on the floor**:

- Few-shot retrieval computes per-item retrieval similarity and (when enabled) chunk-level reference scores, but these signals are not persisted into run outputs and therefore cannot be used as confidence signals.

If we want to improve AURC/AUGRC (i.e., “know when we’re likely wrong”), we must improve the ranking signal used by the risk–coverage curve.

Research basis (validated 2026-01-02):
- UniCR (2025) explicitly targets “calibrated probability → risk-controlled refusal” and reports improvements in area under risk–coverage metrics: https://arxiv.org/abs/2509.01455
- Sufficient Context (ICLR 2025) shows retrieval-augmented context can increase hallucinations when insufficient, motivating retrieval-aware abstention signals: https://arxiv.org/abs/2411.06037
- ACL 2025 highlights that generic UE methods can fail in RAG and motivates retrieval-aware calibration functions: https://aclanthology.org/2025.findings-acl.852/

## 1. Goals / Non-Goals

### 1.1 Goals

- Add **retrieval-grounded confidence signals** to quantitative run outputs (per item, per participant).
- Extend selective prediction evaluation to support **new confidence variants** using those signals.
- Keep changes:
  - deterministic (no sampling required),
  - backward compatible (old run artifacts still evaluable),
  - observable (signals stored for audit; no transcript text in metrics artifacts).
- Provide an ablation path to answer: “Which confidence signal improves AURC/AUGRC on paper-test?”
- (Optional) Enable a calibrated **risk-controlled refusal** policy that can abstain on likely-wrong item scores while preserving coverage when possible.

### 1.2 Non-Goals

- Improving MAE directly (this spec targets confidence/ranking quality).
- Changing the paper-parity prompt format or retrieval content.
- Enabling Spec 36 validation by default (still optional; this spec only consumes its signal if present).

## 2. Baseline (Current Behavior)

### 2.1 Confidence variants (current)

Per `docs/statistics/metrics-and-evaluation.md`:

- `llm`: `confidence = llm_evidence_count`
- `total_evidence`: `confidence = llm_evidence_count + keyword_evidence_count`

### 2.2 Key observation (Run 8)

Run 8 shows large MAE_item improvement for few-shot but similar AURC/AUGRC with the current confidence signal. This indicates the confidence ranking is not improving alongside accuracy.

## 3. Proposed Solution (Phase 1: Retrieval Similarity Signals)

### 3.1 Persist per-item retrieval similarity statistics

When building the few-shot `ReferenceBundle`, we already have per-item retrieved matches:

- `SimilarityMatch.similarity` (float in `[0, 1]`)
- `SimilarityMatch.reference_score` (int/None; when chunk scoring is enabled, this is per-chunk item score)

Add aggregated retrieval stats to `ItemAssessment` so they can be exported by `scripts/reproduce_results.py`:

- `retrieval_reference_count: int`
- `retrieval_similarity_mean: float | None`
- `retrieval_similarity_max: float | None`

Rules:

- If no references exist for that item: count=0, mean/max=`None`.
- Statistics are computed from the **final** matches used for prompt construction (after min-similarity filtering and optional validation).

Primary change:

- `src/ai_psychiatrist/agents/quantitative.py`: keep the `ReferenceBundle` (not only `reference_text`) and attach per-item stats when constructing `ItemAssessment`.

Supporting change:

- `src/ai_psychiatrist/domain/value_objects.py`: extend `ItemAssessment` with the new optional fields.

### 3.2 Export the new signals in run output JSON

Extend `scripts/reproduce_results.py` to include retrieval stats under `item_signals`:

- `retrieval_reference_count`
- `retrieval_similarity_mean`
- `retrieval_similarity_max`

Type safety:

- Update the internal typing of `EvaluationResult.item_signals` to allow floats:
  - `int | float | str | None` (or a named type alias).

Backwards compatibility:

- For older run artifacts, these keys will be absent and retrieval-based confidence variants must fail fast with a clear error.

Forward compatibility (recommended):
- For runs produced after this spec, write these keys for **both** modes:
  - zero-shot: `retrieval_reference_count=0`, `retrieval_similarity_mean=null`, `retrieval_similarity_max=null`
  - few-shot: computed values from the final references used in the prompt

### 3.3 Add new confidence variants in selective prediction evaluation

Extend `scripts/evaluate_selective_prediction.py`:

- Add `--confidence retrieval_similarity_mean`
- Add `--confidence retrieval_similarity_max`
- Add `--confidence hybrid_evidence_similarity` (deterministic combination)

Default formula for the hybrid signal (chosen for simplicity + monotonicity):

```text
e = min(llm_evidence_count, 3) / 3            # normalize to [0, 1] with a cap
s = retrieval_similarity_mean or 0.0          # in [0, 1]
confidence = 0.5 * e + 0.5 * s
```

Rationale:
- `llm_evidence_count` is available in both modes and correlates with evidence presence.
- `retrieval_similarity_mean` is retrieval-grounded and continuous, reducing plateaus.
- The combination is deterministic, bounded, and easy to audit.

CLI behavior:

- If a retrieval-based confidence is requested but required signals are missing:
  - Raise a clear error pointing to the run artifact and required keys.
  - Do not silently treat missing as 0.0 (this would bias results).

Applicability guidance:
- `retrieval_similarity_mean` / `retrieval_similarity_max` are primarily meaningful for **few-shot** runs.
- `hybrid_evidence_similarity` is the recommended cross-mode comparison signal because it degrades gracefully for zero-shot (similarity term = 0 when absent).

Documentation updates:

- Update `docs/statistics/metrics-and-evaluation.md` “Confidence Variants” with the new options and their exact formulas.
- Update `docs/results/run-output-schema.md` to list the new `item_signals` keys.

## 4. Optional Extensions (Phase 2+)

### 4.1 Reference-score dispersion (when chunk scoring enabled)

If `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk` and retrieved matches carry per-chunk item scores:

- Add per-item dispersion features:
  - `retrieval_reference_score_mean`
  - `retrieval_reference_score_std`

Hypothesis:
- high disagreement among retrieved reference scores → higher uncertainty.

### 4.2 Supervised calibrator (paper-val → paper-test)

Train a calibrator that maps signals → predicted correctness (or expected loss), then use the calibrated score as the confidence ranking signal.

Implementation sketch:
- New script: `scripts/calibrate_confidence.py`
- Inputs: a run artifact from `paper-val` (or cross-validated folds), selecting a mode.
- Features: evidence counts, retrieval similarity stats, evidence_source, (optional) reference-score dispersion.
- Target:
  - either `abs_error_norm` regression, or
  - `correct = 1{abs_error == 0}` classification.
- Output: JSON calibrator artifact (weights + schema + training metadata; no transcript text).

The calibrator is evaluated by re-running `scripts/evaluate_selective_prediction.py` on `paper-test` using the calibrator-produced confidence.

### 4.3 Risk-controlled refusal (conformal; runtime behavior)

If we want the system to “know when not to answer” (not just rank confidence post-hoc), add an optional runtime refusal layer:

1. Train a calibrator on `paper-val` (Section 4.2) to output `p_correct` per `(participant, item)`.
2. Fit a conformal risk-control threshold `τ` for a user-specified error budget (e.g., expected normalized absolute error) or for a correctness target.
3. At inference time: if `p_correct < τ`, override `score -> None` and set `na_reason = "low_confidence"` (new enum value).

This approach is aligned with UniCR’s “calibrated probability → risk-controlled decision” framing (see 2509.01455).

## 5. Test Plan (TDD)

### 5.1 Unit tests: retrieval stats extraction

Add unit tests for a pure helper that computes retrieval stats from `ReferenceBundle.item_references[item]`:

- empty list → count=0, mean/max=None
- non-empty list → correct count/mean/max

### 5.2 Unit tests: evaluation confidence parsing

Add unit tests for `scripts/evaluate_selective_prediction.py:parse_items()` confidence selection:

- retrieval confidence variants error on missing keys (clear message)
- hybrid confidence bounded in `[0, 1]` and deterministic

### 5.3 Integration tests: run artifact schema

Update or add an integration test that:

- runs a mocked few-shot assessment producing known retrieval stats,
- writes a minimal run JSON,
- evaluates AURC/AUGRC with retrieval-based confidence variants successfully.

## 6. Acceptance Criteria

- `scripts/reproduce_results.py` exports the new retrieval stats for few-shot runs without breaking existing schema consumers.
- `scripts/evaluate_selective_prediction.py` supports the new confidence variants and fails fast on missing signals.
- Documentation updated:
  - `docs/statistics/metrics-and-evaluation.md`
  - `docs/results/run-output-schema.md`
  - `docs/_specs/index.md` lists this spec under “Archived (Implemented)”
- Tests / lint / types pass:
  - `uv run pytest tests/ -v --tb=short`
  - `uv run ruff check`
  - `uv run mypy src tests scripts --strict`
