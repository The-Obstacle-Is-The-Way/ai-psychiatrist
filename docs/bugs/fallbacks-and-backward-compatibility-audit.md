# Fallbacks & Backward-Compatibility Audit (ai-psychiatrist)

**Date**: 2025-12-27
**Scope**: Runtime fallbacks + “legacy/compat” shims that can affect **paper reproduction** correctness, reproducibility, and interpretability.

This document is intentionally **root-level** (not under `docs/`) to act as a “reset point” when other docs/specs flip‑flop.

---

## Executive Summary (first principles)

1. **The Pydantic AI “fallback to legacy” does *not* switch to a different model or a non‑LLM method.**
   Both paths still call Ollama with the same `model` string (from `ModelSettings`) and the same research prompts. The difference is **the Python wrapper layer and parsing/repair behavior**, not the model family.

2. **The fallback is mostly “backward compatibility + belt-and-suspenders,” not a primary reliability mechanism.**
   It can help with *rare* “wrapper/validation/library” edge cases, but for the most common runtime failure in this repo (slow inference / timeout), it’s often **not helpful** and can **waste time**.

3. **There is a real timeout misalignment risk, but the earlier “Pydantic AI timeout is not configurable” claim is not strictly true.**
   - Pydantic AI’s Ollama path (via `openai` client) defaults to **600s** (OpenAI-like) because it uses `pydantic_ai.models.cached_async_http_client(timeout=600)`.
   - Our legacy `OllamaClient` defaults to **300s** via `OllamaSettings.timeout_seconds`.
   - However, **Pydantic AI *can* accept a per-request timeout** via `model_settings={"timeout": ...}` (supported by `pydantic_ai.models.openai.OpenAIChatModel` and `openai` client). We simply **don’t pass it today**, so it’s not configurable via our `Settings` yet.

4. **The largest “research validity” risk is not that we call a different model — it’s that we may change the *algorithmic pipeline* per participant depending on transient failures.**
   Even when the same model is used, “primary path vs legacy path” can imply different parsing/repair prompts and thus can cause *run-to-run drift* if not recorded.

---

## What “fallback” means in this repo (it’s easy to misunderstand)

### Primary path (Pydantic AI)

**Where it’s created**
- `src/ai_psychiatrist/agents/pydantic_agents.py`

**How it talks to Ollama**
- Uses `pydantic_ai.models.openai.OpenAIChatModel` + `pydantic_ai.providers.ollama.OllamaProvider`
- Hits Ollama’s **OpenAI‑compatible endpoint**: `/v1/chat/completions` (you’ll see log lines like “Retrying request to /chat/completions”)

**How it enforces structure**
- Uses `TextOutput(...)` extractors in `src/ai_psychiatrist/agents/extractors.py`
- Extractors raise `ModelRetry(...)` on parse/validation errors → Pydantic AI performs a retry loop up to `PYDANTIC_AI_RETRIES` (default 3).

### Legacy path (“fallback to legacy”)

**How it talks to Ollama**
- Uses the project’s `SimpleChatClient` (typically `src/ai_psychiatrist/infrastructure/llm/ollama.py::OllamaClient`)
- Hits Ollama’s **native endpoint**: `/api/chat`

**Key clarification**
- This is *still the same LLM*. “Legacy” is not “non-LLM.” It’s just our older wrapper.

---

## Inventory: Runtime fallbacks that can affect experiment semantics

### 1) Pydantic AI → Legacy fallback (all 4 agents)

All four agents implement:
- “If Pydantic AI is enabled, try `Agent.run()`”
- If it throws **any** exception (except `asyncio.CancelledError`), log and fall back to legacy.

**Locations**
- Quantitative: `src/ai_psychiatrist/agents/quantitative.py::_score_items()`
- Qualitative: `src/ai_psychiatrist/agents/qualitative.py::assess()` and `refine()`
- Judge: `src/ai_psychiatrist/agents/judge.py::_evaluate_metric()`
- Meta-review: `src/ai_psychiatrist/agents/meta_review.py::review()`

**Why it exists**
- Historical: legacy code pre-dated Pydantic AI.
- Safety: avoid “new framework causes total failure.”
- Practicality: preserve behavior while we gain confidence.

**When it’s actually helpful**
- A framework/library bug in `pydantic_ai` or `openai` that doesn’t affect `/api/chat`.
- A validation/extraction failure where legacy parsing heuristics succeed.

**When it’s probably *not* helpful**
- True slow inference / overload / GPU throttling. Both paths still need the same inference.
- Connection-level failures that affect the whole Ollama server.

**Research implication**
- This can cause per-participant pipeline divergence (primary vs legacy) based on transient issues.
- If not recorded in experiment provenance, it can make runs harder to compare.

### 2) Legacy quantitative parsing “repair ladder”

In the quantitative legacy path, parsing includes an explicit multi-stage ladder:
1) Parse directly (strip `<answer>`/markdown + tolerant fixups)
2) **LLM repair prompt** (`_llm_repair`)
3) Fallback skeleton

**Location**
- `src/ai_psychiatrist/agents/quantitative.py::_parse_response()` + `_llm_repair()`

**Research implication**
- Stage (2) is a *different* prompt (“repair this JSON…”) vs the original scoring prompt.
- This is still the same model, but it’s an additional “algorithmic behavior” that can shift outputs.

### 3) Meta-review severity fallback (non-LLM “derived” fallback)

If the legacy meta-review response can’t be parsed into an integer 0–4, it falls back to quantitative severity:

**Location**
- `src/ai_psychiatrist/agents/meta_review.py::_parse_response()` → “quantitative fallback”

**Research implication**
- This is a genuine semantic fallback (not just parsing): a missing/wrong `<severity>` can cause the system to output a different severity source than intended.

### 4) Judge default score on LLM failure

If judge LLM call fails, the judge returns a default `score=3` (which triggers refinement thresholds):

**Location**
- `src/ai_psychiatrist/agents/judge.py::_evaluate_metric()` → returns `EvaluationScore(score=3, ...)` on `LLMError`

**Research implication**
- This influences the feedback loop behavior and can change refinement frequency.

### 5) Batch evaluation “continue-on-error”

The reproduction script treats participant evaluation as best-effort:
- Any exception → `success=False` for that participant → run continues.

**Location**
- `scripts/reproduce_results.py::evaluate_participant()`

**Research implication**
- This is usually correct for long research runs (partial data is better than none).
- It must be paired with provenance that clearly reports failure counts and which participants failed.

---

## Inventory: Backward-compatibility shims (usually good, but must be explicit)

These are “compat” shims that generally **reduce breakage** without changing model semantics.

### 1) Reference embeddings metadata: semantic vs strict validation + legacy fallbacks

**Location**
- `src/ai_psychiatrist/services/reference_store.py`

**Behavior**
- Newer artifacts use `split_ids_hash` (semantic list-of-IDs hash) for validation.
- Older artifacts can fall back to `split_csv_hash` (raw bytes hash).
- If `split_ids_hash` is missing, code can derive an IDs hash from the JSON sidecar for legacy artifacts.

**Why it matters**
- This is exactly the “avoid false positives from harmless CSV rewrites” lesson.
- It is a good compatibility pattern: validate on semantics, keep strict hash for audit.

### 2) API request mode: accept legacy integer 0/1

**Location**
- `server.py::AssessmentRequest.parse_mode()`

**Why it matters**
- This is backwards compatible for earlier client code; does not affect reproduction correctness.

### 3) Domain enum string values match legacy repo format

**Location**
- `src/ai_psychiatrist/domain/enums.py::PHQ8Item`

**Why it matters**
- This keeps parity with older artifacts and logs (e.g., “NoInterest”, “Concentrating”).

---

## The real root cause behind “fallback confusion”

The confusion usually comes from mixing two different meanings:

1) **“Fallback = different model / different experiment”** → unacceptable for research parity
2) **“Fallback = same model, different wrapper/repair path”** → may be acceptable, but must be measured and recorded

In this repo today, the agent fallback is (2), not (1).

The actual risk is not “we used a different model” — it’s:
- we may have **per-participant pipeline drift** (Pydantic AI path vs legacy path vs repair ladder)
- which can change outcomes while still using the same model.

---

## Timeout behavior: verified from library + code

### Legacy timeout (project code)
- `src/ai_psychiatrist/config.py::OllamaSettings.timeout_seconds` defaults to **300**
- `src/ai_psychiatrist/infrastructure/llm/ollama.py::OllamaClient` uses that timeout in `httpx.AsyncClient(...)`

### Pydantic AI timeout (vendor/library behavior)
Verified by reading the installed library source:
- `pydantic_ai==1.39.0`
- `openai==2.14.0`
- `pydantic_ai.models.cached_async_http_client(timeout=600)` default is **600 seconds**

Important nuance:
- `pydantic_ai.models.openai.OpenAIChatModel` passes `timeout=model_settings.get('timeout', NOT_GIVEN)` through to `openai`.
- Therefore, **Pydantic AI per-request timeouts are configurable**, but our agent code currently does not pass a timeout.

**Implication for paper runs**
- If the model is genuinely slow on a given transcript, retry+fallback can multiply wasted time without improving success rate.

---

## Recommendations (aligned to paper reproduction intent)

These are not “change model / change experiment” recommendations; they’re about making the experiment *cleaner*.

### A) Make fallbacks explicit and measurable
- Record, per participant and per agent, whether we used:
  - `pydantic_ai_primary`
  - `legacy_primary` (if Pydantic AI disabled)
  - `legacy_fallback_after_pydantic_failure`
  - `legacy_llm_repair_used` (quantitative parse ladder)
- This can be part of experiment tracking outputs (so run-to-run comparisons remain valid).

### B) Don’t treat timeouts like “recoverable parser errors”
- Consider distinguishing “timeout/connection” from “validation/parser” errors:
  - Parser/validation: fallback may help.
  - Timeout: fallback is usually wasted time (same model still needs to run).
- If we keep fallback, a reasonable rule is: **don’t fallback on timeouts** (or at least don’t fallback with a smaller timeout).

### C) Unify timeout configuration across both wrappers
- Drive both Pydantic AI and legacy calls from one configured timeout (single SSOT).
- Pydantic AI supports `model_settings={"timeout": ...}`; using it would remove the current mismatch.

### D) Decide what “paper parity” means in this repo
If the goal is strict reproduction:
- Prefer one pipeline (either Pydantic AI *or* legacy) and treat failures as failures (but continue-on-error).
If the goal is “best possible robust reproduction”:
- Keep fallback + repair ladder, but record it as part of the method and provenance.

---

## TL;DR

- The current fallback mechanism is **not inherently wrong** for a long-running research pipeline.
- The thing to be suspicious about is **unrecorded pipeline divergence**, not “switching to a different model.”
- If we want “DeepMind-grade” rigor: **make fallbacks explicit, unify timeouts, and discriminate timeout vs validation failures**.
