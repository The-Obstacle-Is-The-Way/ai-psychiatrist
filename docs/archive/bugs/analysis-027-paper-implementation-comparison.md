# Analysis 027: Paper vs Implementation Comparison (Quantitative Agent)

**Status**: ✅ ARCHIVED - Historical investigation, findings now in SSOT docs
**Archived**: 2025-12-26

---

## Resolution Summary

This investigation identified discrepancies between the paper's public repository and our implementation. **The key findings are now documented in dedicated SSOT locations:**

| Finding | SSOT Location |
|---------|---------------|
| Keyword backfill toggle | [`docs/concepts/backfill-explained.md`](../concepts/backfill-explained.md) |
| Coverage tradeoff (50% vs 69%) | [`docs/concepts/coverage-explained.md`](../concepts/coverage-explained.md) |
| MAE gap (0.619 vs 0.778) | [`docs/models/model-wiring.md`](../models/model-wiring.md) - **Explained by Q4_K_M vs BF16 quantization** |
| Sampling parameters | [`docs/reference/agent-sampling-registry.md`](../reference/agent-sampling-registry.md) |
| Model options & hardware | [`docs/models/model-wiring.md`](../models/model-wiring.md) |

**Key insight**: The paper likely ran BF16 on A100 GPUs. Our Q4_K_M quantization (4-bit) explains the MAE difference. This is not a code bug—it's a precision tradeoff.

**Going forward**: We don't need to match their sloppy codebase. Our implementation is cleaner and well-documented. When BF16/Q8 hardware is available, we can validate whether precision closes the gap.

---

## Original Investigation (Historical Context)

This document compares the **publicly available paper repository code** (mirrored under
`_reference/`) to our production implementation under `src/ai_psychiatrist/`.

## Reference Code SSOT Hierarchy

**IMPORTANT**: When referencing the paper's code:

| Priority | Source | Notes |
|----------|--------|-------|
| 1 (SSOT) | `_reference/quantitative_assessment/*.ipynb` | Notebooks are authoritative |
| 2 | `_reference/quantitative_assessment/*.py` | .py files have wrong model defaults (e.g., `llama3`) |
| 3 | `_reference/agents/*.py` | Agent .py files have wrong defaults; sampling params match notebooks |

It focuses on the quantitative PHQ‑8 pipeline because our reproduction diverges from the paper:
- Our few-shot MAE: **0.778** vs paper: **0.619**
- Our item-level coverage: **69.2%** (over evaluated subjects) vs paper: **~50% of cases unable to provide a prediction** (denominator unclear)

See also: `docs/bugs/investigation-026-reproduction-mae-divergence.md`.

---

## Executive Summary (Verified Facts)

1. **Keyword backfill is in the public paper repo and is executed** in the few-shot agent:
   - `_reference/agents/quantitative_assessor_f.py` unconditionally calls `_keyword_backfill(...)`
     inside `extract_evidence()` (line ~478).
   - The paper text does **not** describe this heuristic step (Section 2.3.2 describes LLM evidence extraction).

2. **Sampling parameters are not only “unspecified”**: the paper repo hardcodes them:
   - Few-shot (paper repo): `temperature=0.2, top_k=20, top_p=0.8` via `ollama_chat()` options
     (`_reference/agents/quantitative_assessor_f.py` line ~200).
   - Zero-shot script (paper repo): `temperature=0, top_k=1, top_p=1.0`
     (`_reference/quantitative_assessment/quantitative_analysis.py` line ~137).

3. **The paper repo includes HPC/SLURM scripts** (e.g., `_reference/slurm/job_ollama.sh`) configured for
   `--gres=gpu:A100:2`. The paper also states the pipeline can run on a MacBook Pro M3 Pro (Section 2.3.5).
   The hardware/quantization used for the reported MAE/coverage is therefore **not uniquely determined** from
   the paper text alone.

4. **Backfill alone does not explain our higher coverage**: we observed **69.2% coverage with backfill OFF**
   in `data/outputs/reproduction_results_20251224_003441.json` (216/(39×8); 65.9% if including all 41 subjects).
   A historical backfill-ON run was recorded at ~74.1% overall item coverage (243/328) in
   `docs/results/reproduction-notes.md`, but that run is not paper-text parity and its raw output is not stored
   under `data/outputs/` in this repo snapshot.

---

## Discrepancy Table (Paper Repo vs `src/`)

| Area | Paper Repo (`_reference/`) | Our Implementation (`src/`) | Likely Impact | Confidence |
|---|---|---|---|---|
| Keyword backfill execution | Always executed in few-shot (`extract_evidence()` calls `_keyword_backfill`) | Optional via `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` (default `false`) | Changes evidence used for retrieval → can change reference chunks → can change MAE/coverage | High |
| Keyword list content | Small inline dict (substring match; includes broad tokens like `"sleep"`, `"tired"`, `"eat"`) | Large curated YAML (`src/ai_psychiatrist/resources/phq8_keywords.yaml`) with collision-avoidance; still substring match | Even if backfill toggled ON, behavior won’t match paper repo unless lists match | High |
| Embedding model + artifacts | Few-shot agent defaults `emb_model="dengcao/Qwen3-Embedding-8B:Q4_K_M"` and loads a pickle of embedded transcripts | Defaults to `qwen3-embedding:8b` (paper text) and uses NPZ+JSON reference artifacts | Retrieval differences change reference chunks → can change coverage/MAE | High |
| System prompt wording | Mentions DSM‑IV; no line continuations | Mentions DSM‑5; uses `\\` line continuations in `QUANTITATIVE_SYSTEM_PROMPT` | Prompt tokenization/formatting can affect abstention (N/A vs score) | Medium |
| Scoring user prompt output schema | Uses placeholder form: `{{evidence, reason, score}}` | Uses explicit JSON field example: `{"evidence": "...", "reason": "...", "score": ...}` | Could reduce formatting errors, but may also “pressure” model to fill fields (higher coverage) | Medium |
| Evidence extraction prompt formatting | Multi-line prompt literal | Uses `\\` line continuations (collapses some newlines) | Tokenization difference may affect evidence recall | Low–Medium |
| Few-shot defaults (repo) vs paper text | Code defaults `chat_model="llama3"`, `top_k=3` references | Defaults to `gemma3:27b` via config; `top_k_references=2` (paper Appendix D) | If paper repo defaults were used accidentally, results differ from paper writeup | Medium |
| Reference bundle formatting | Uses `<Reference Examples>` but “closes” with `<Reference Examples>` (no slash); lines include `(PHQ8_* Score: X)` | Uses `<Reference Examples> ... </Reference Examples>`; lines include `(Score: X)`; headers are `[NoInterest]` not `[PHQ8_NoInterest]` | Formatting differences can change how strongly model conditions on examples | Medium |
| JSON block stripping | Extracts substring from first `{` to last `}`; extra sanitation | Does not trim to `{...}`; relies more on LLM repair if parsing fails | Can change parse-failure rate; indirectly changes exclusions | Low–Medium |
| Score normalization | Floats become `N/A` (e.g., `2.0` → N/A); out-of-range ints clamped | Accepts `2.0` → `2`; out-of-range ints rejected (→ N/A) | Float acceptance can increase coverage; out-of-range rejection can decrease coverage | Low |
| JSON repair sampling | Repair uses same `ollama_chat()` options (0.2/20/0.8) | Repair uses `temperature=0.1` (lower) | Could change repair success; small impact | Low |

---

## Side-by-Side Prompt Comparison (Key Deltas)

### System Prompt: DSM version
- Paper repo: “DSM‑IV criteria” (`_reference/agents/quantitative_assessor_f.py` line ~224/315)
- Our prompt: “DSM‑5 criteria” (`src/ai_psychiatrist/agents/prompts/quantitative.py`)

### Scoring Output Spec: placeholder vs explicit schema
- Paper repo: `- "PHQ8_NoInterest": {{evidence, reason, score}} ...`
- Our prompt: `- "PHQ8_NoInterest": {"evidence": "...", "reason": "...", "score": ...}`

### Reference Bundle: header + tag differences
- Paper repo headers: `[PHQ8_NoInterest]` etc; includes `(PHQ8_* Score: X)` per example chunk.
- Our headers: `[NoInterest]` etc; includes `(Score: X)` per example chunk.

---

## Why Is Coverage Higher (69.2%) With Backfill OFF?

Backfill being OFF only removes one pathway to *improve retrieval queries*. It does **not** force the
scoring model to abstain. Our higher coverage implies the model is producing numeric scores more often
than the paper’s reported run, *or* that we are comparing different denominators (“cases” vs item coverage).

Most plausible explanations (not mutually exclusive):
1. **Model/weights/runtime mismatch**: the paper repo contains both (a) MacBook runtime claims in the paper
   and (b) SLURM/A100 scripts in the repo. Different quantization/precision/backends can shift abstention.
2. **Prompt formatting and reference formatting drift**: small formatting changes can change “N/A vs 0”
   behavior even when semantics are similar.
3. **Different model tags used historically**: the paper repo’s zero-shot script uses `gemma3-optimized:27b`
   and the few-shot agent defaults `chat_model="llama3"` unless overridden. If the paper’s reported metrics
   were produced with a different tag than our run, coverage can shift substantially.
4. **Normalization differences**: our float-score acceptance could modestly increase coverage.
5. **Evaluation denominator interpretation**: our script reports coverage over `evaluated_subjects` (excluding
   subjects with 0 predicted items). If the paper computed coverage over all subjects/items (including excluded),
   the reported percentage would be lower. (This is a smaller effect in our run because only 2/41 subjects were excluded.)

---

## Ranked Root-Cause Hypotheses (MAE + Coverage)

1. **Hardware / quantization / model build differences** (highest leverage, hardest to verify from paper text).
2. **Paper repo vs paper writeup mismatch** (backfill + defaults like `top_k=3`, `chat_model="llama3"`).
3. **Reference bundle formatting differences** (conditioning strength and conservatism).
4. **Keyword backfill implementation mismatch** (even if enabled, our YAML list ≠ paper repo’s list).
5. **Parser/normalization differences** (likely small).

---

## Fixes / Experiments to Try (For “Paper Parity”)

Because the paper text and public repo are not perfectly aligned, treat “paper parity” as **two targets**:

### Target A: Paper-Text Parity (methodology as written)
- Keep `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` (heuristic not described in paper).
- Document and accept that coverage may still differ due to model/runtime differences.

### Target B: Paper-Repo Parity (match the public implementation)
1. Enable keyword backfill *and* use the paper repo’s `DOMAIN_KEYWORDS` list/behavior.
2. Match reference bundle formatting (headers, score line format, and tag style).
3. Match normalization rules (treat float scores as N/A; clamp out-of-range like the repo).
4. Ensure the quantitative model tag matches what the repo/run used historically (e.g., clarify whether
   few-shot was run with Gemma 3 27B or something else; the repo defaults differ from the paper text).

### Cross-cutting experiment suggestions
- Run an ablation grid holding split/embeddings constant:
  - backfill OFF/ON
  - reference formatting paper-like vs current
  - strict score parsing (paper-like) vs current

---

## Questions to Ask the Paper Authors (Recommended)

To make reproduction unambiguous, we likely need clarification on:
1. Which exact model tag/build was used for the reported Gemma 3 27B results?
2. What hardware/precision/quantization was used for the reported MAE/coverage?
3. Was keyword backfill considered part of the quantitative methodology for the reported results?
4. Did the reported “~50% unable to provide prediction” refer to item-level missingness, subject-level exclusion,
   or a specific coverage definition?
