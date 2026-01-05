# Ralph Wiggum Loop Prompt — ai-psychiatrist Master Bug Audit

This prompt is designed for the **Ralph Wiggum loop** in **Claude Code** to perform a comprehensive, end-to-end audit of the `ai-psychiatrist` repository and produce a **single master bug report**.

## How to Run (human operator)

**Claude Code plugin** (official Ralph Wiggum plugin):

```bash
/ralph-loop "$(cat prompt.md)" --completion-promise "AUDIT_COMPLETE" --max-iterations 50
```

If the plugin is unavailable, use an **external Ralph bash loop** (see `anthropics/claude-code/plugins/ralph-wiggum` and community workarounds like `ParkerRex/ralph-loop`). In that case, copy this file to whatever your loop expects (often `.claude/RALPH_PROMPT.md`).

## Iteration Awareness (IMPORTANT)

Each Ralph iteration sees the file system from previous iterations.

**If `MASTER_BUG_AUDIT.md` already exists**:
- Read it first to understand progress
- Continue from the last incomplete phase
- Do NOT restart from Phase 0 unless explicitly incomplete

**If starting fresh**: Begin at Phase 0.

---

## Mission (do this before you stop)

Produce `MASTER_BUG_AUDIT.md` at the repository root.

Goal: identify **every** meaningful bug / validity threat / incorrect assumption that could invalidate research conclusions or waste multi-hour runs, and document it so it can be fixed with TDD + specs.

You must:
- Run the repo’s CI checks and record what you ran.
- Audit the **data pipeline invariants** (embeddings, transcripts preprocessing outputs, run artifacts).
- Audit the **agent pipeline** (evidence extraction → retrieval → scoring → metrics) for brittleness and silent degradations.
- Perform **web research** (2025–2026 best practices) where it materially improves rigor, but do not “handwave”; cite sources.

You must **not** implement fixes in this loop. Only document.

---

## Hard Constraints (non-negotiable)

### 1) DAIC-WOZ licensing / sensitive text

DAIC-WOZ transcripts are restricted.

**Never** paste or quote transcript text, evidence quotes, or chunk previews into:
- `MASTER_BUG_AUDIT.md`
- console logs
- issues/PR descriptions

If you must reference content to prove a bug, use:
- participant id(s)
- file path(s)
- row/word counts
- hashes (e.g., `sha256`)
- offsets/lengths

### 2) No secrets

Do not print or record `.env` values, API keys, or tokens. You may verify that required **keys exist** (names only), and that `.env` is derived from `.env.example`, but never disclose values.

### 3) No data modification

Do not modify any files under `data/` (including `data/transcripts*`, `data/embeddings`, `data/outputs`). Treat `data/` as read-only.

### 4) No “fixes”

Do not change production code, tests, or docs in this loop. This is an audit pass. You may propose fixes and test plans, but do not apply patches.

---

## Repository Reality (read this first)

This repo intentionally diverges from the original paper due to documented flaws. Root agent instructions are SSOT and intentionally redundant:
- `AGENTS.md`
- `CLAUDE.md`
- `GEMINI.md`

Also: item-level PHQ-8 scoring from transcripts is often **underdetermined** because PHQ-8 is a 2-week **frequency** instrument. This is a core validity constraint, not automatically a bug.

SSOT for task validity: `docs/clinical/task-validity.md`.

---

## Audit Workflow (Ralph loop steps)

Follow this sequence. Do not reorder unless you explain why in the audit doc.

### Phase 0 — Baseline integrity

1) Record repository metadata:
- current git branch + commit SHA
- OS + Python version (`python -c "import sys; print(sys.version)"`)

2) Run code quality + tests:
- `make ci`
- `uv run mkdocs build --strict`

In `MASTER_BUG_AUDIT.md`, record:
- exact commands
- pass/fail status
- any warnings (and whether they matter)

### Phase 1 — Data invariants (read-only)

Audit the `data/` tree and ensure all **required artifacts** exist for a valid run (no expensive compute).

Minimum checks (do not print transcript text):
- `data/transcripts_participant_only/` exists and contains expected participant folders.
- `data/embeddings/` contains the active HuggingFace artifacts:
  - `huggingface_qwen3_8b_paper_train_participant_only.npz`
  - `huggingface_qwen3_8b_paper_train_participant_only.json`
  - `huggingface_qwen3_8b_paper_train_participant_only.meta.json`
  - `huggingface_qwen3_8b_paper_train_participant_only.tags.json`
  - `huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json`
  - `huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.meta.json`
- `data/outputs/RUN_LOG.md` exists and is append-only.

Additionally:
- Verify that “preprocessed transcripts” are truly a separate directory (no overwrites of `data/transcripts/`).
- Identify obvious corruption signals (0-byte files, missing columns, wrong CSV headers) **without** copying text.

### Phase 2 — Config invariants (no secrets)

Verify the repo is configured to avoid known “wasted run” failures:
- HuggingFace embeddings are used by default *and* HF deps are installed (torch/transformers/sentence-transformers).
- Few-shot uses chunk scores (`EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`).
- Evidence quote validation is enabled (if that’s the validated baseline).

Rules:
- You may compare `.env` and `.env.example` by **key names only**, not values.
- If `.env` is missing keys from `.env.example`, that’s a P0 run-preflight bug.

### Phase 3 — Pipeline dry-run validation (no long runs)

Run *fast* smoke checks to ensure the pipeline fails fast instead of wasting hours.

Required:
- `uv run python scripts/reproduce_results.py --split paper-test --dry-run`
- `uv run python scripts/reproduce_results.py --split paper-test --limit 1 --few-shot-only`
- `uv run python scripts/reproduce_results.py --split paper-test --limit 1 --zero-shot-only`

If any of these fail, capture:
- exact exception type
- where it originated (file:line)
- why it wasn’t caught earlier (missing preflight, missing validation, etc.)

### Phase 4 — Metrics correctness + reproducibility

Pick the most recent **valid** run artifact in `data/outputs/` (do not re-run full pipelines) and compute selective prediction metrics:

```bash
uv run python scripts/evaluate_selective_prediction.py --input <RUN_JSON> --mode few_shot --confidence all --bootstrap-resamples 1000
uv run python scripts/evaluate_selective_prediction.py --input <RUN_JSON> --mode zero_shot --confidence all --bootstrap-resamples 1000
```

Validate from first principles:
- MAE comparisons are only meaningful when coverage is similar.
- AURC/AUGRC are computed over risk–coverage and should be stable under bootstrap.
- Any "improvement" claim must be paired with coverage deltas and confidence intervals.

**Telemetry check (Spec 060)**:
- Check `data/outputs/telemetry_*.json` for retry/repair spikes
- High retry counts may indicate JSON parsing brittleness
- Document any concerning patterns in the audit

### Phase 5 — Static code audit (bugs & anti-patterns)

Do a wide search for high-risk patterns and document each with file:line references:

**P0/P1 patterns**:
- Silent fallbacks that change experimental conditions (e.g., “few-shot silently becomes zero-shot”).
- Broad exception catches that return defaults (`except Exception: return {}` / `return []`).
- Non-idempotent “fixups” that can corrupt JSON.
- Multiple implementations of the same parsing logic (fragmentation).
- Embedding space mismatch risks (HF vs Ollama artifacts).
- Any logging that could leak DAIC-WOZ text into shareable artifacts.

Search suggestions:
- `rg -n "except Exception|pass\\b|return \\{\\}|return \\[\\]" src scripts`
- `rg -n "json\\.loads\\(|ast\\.literal_eval\\(|json_repair" src`
- `rg -n "chunk_preview|transcript.*preview|evidence.*preview" src`
- `rg -n "paper-parity|paper optimal" -S .` (excluding archives)

### Phase 6 — Web research (2025–2026 best practices)

Only do web research that materially informs:
- structured output / JSON reliability for LLMs
- selective prediction evaluation (AURC/AUGRC, e-AURC, calibration)
- PHQ-8 psychometrics & what’s inferable from interviews
- DAIC-WOZ prior art and validity threats (interviewer prompt confounds)

For each external claim in the audit doc, include a URL.

---

## `MASTER_BUG_AUDIT.md` Required Structure (SSOT)

Write the report with this exact structure:

1) **Executive Summary**
   - counts by severity: P0–P4
   - top 3 “wastes-hours” failure modes
   - top 3 “invalidates-conclusions” validity threats

2) **Environment + Commands Run**
   - git SHA
   - `make ci` result
   - `mkdocs build --strict` result
   - any warnings that matter

3) **Known Non-Bugs / Expected Limitations**
   - must include the task-validity constraint and link `docs/clinical/task-validity.md`

4) **Findings (table)**
   Columns:
   - ID (BUG-XXX)
   - Severity (P0–P4)
   - Category (data / config / parsing / retrieval / evaluation / docs / observability)
   - Symptom
   - Root cause (with file:line)
   - Impact (what breaks / what becomes invalid)
   - Repro steps (minimal)
   - Proposed fix (high-level)
   - Test plan (tests-first)

5) **Deep Dives (one section per P0/P1)**
   - include evidence without leaking transcript text
   - include why the current guardrails failed

6) **Prioritized Fix Roadmap**
   - ordered list of specs to write (or existing specs to implement)
   - “definition of done” per item

7) **Open Questions**

---

## Escape Hatch (if stuck)

If after **40 iterations** you cannot complete all phases:

1. Document in `MASTER_BUG_AUDIT.md` → **Open Questions** section:
   - What is blocking progress
   - What was attempted
   - Suggested alternative approaches or human intervention needed

2. Mark incomplete phases with `[INCOMPLETE]` in the doc

3. Output `AUDIT_COMPLETE` anyway—partial audit is better than infinite loop

---

## Completion Rule

You may stop only when:
- `MASTER_BUG_AUDIT.md` exists at repo root and matches the required structure, and
- you have run Phase 0–Phase 6 (or documented why any step was impossible/incomplete).

Then output exactly (no markdown, no quotes):

AUDIT_COMPLETE
