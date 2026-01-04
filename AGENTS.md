# Agent Runbook (SSOT)

This repository is consumed by multiple AI coding assistants. The contents of `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` are intentionally kept **identical** for redundancy. If you update one, update all three.

## Critical Context

**IMPORTANT**: This is a **robust, independent implementation** that fixes severe methodological flaws in the original research paper (Greene et al.). Do **not** use “paper-parity” terminology in code, docs, issues, or PRs. Prefer: **validated configuration**, **baseline defaults**, **conservative defaults**.

Documented paper failures (closed issues):
- **#81**: Participant-level PHQ-8 scores assigned to individual chunks (semantic mismatch)
- **#69**: Few-shot retrieval attaches participant scores to arbitrary text chunks
- **#66**: Invalid statistical comparison (MAE compared at different coverages)
- **#47, #46**: Quantization/sampling parameters unspecified
- **#45**: Undocumented custom split

## GitHub Repository

**IMPORTANT**: This is a fork. Use the fork for issues and PRs:
- https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist

```bash
gh issue create --repo The-Obstacle-Is-The-Way/ai-psychiatrist ...
gh pr create --repo The-Obstacle-Is-The-Way/ai-psychiatrist ...
```

## Non-Negotiables

- Use the repo virtualenv: prefer `uv run …` for all Python commands.
- Do not commit DAIC-WOZ data or secrets; configure via `.env`.
- Always run `make ci` before/after non-trivial changes.
- Treat silent degradations as bugs; prefer loud failures over “best effort”.

## Setup (Never-Again Defaults)

1. Install dependencies (robust default):
   - `make dev`
   - This installs dev + docs + HuggingFace extras (`torch`, `transformers`, `sentence-transformers`) and pre-commit hooks.

2. Configure environment:
   - `cp .env.example .env`

3. Verify HuggingFace deps load (required for HF embeddings and HF LLM backend):
   - `uv run python -c "import torch, transformers, sentence_transformers; print(torch.__version__)"`

### Why HF Deps Matter Even If Embeddings Exist

Precomputed `data/embeddings/*.npz` files are **reference embeddings** only. Few-shot retrieval must compute **query embeddings at runtime** from the participant’s evidence text in the **same embedding space**. If `EMBEDDING_BACKEND=huggingface`, the runtime query embedder requires HF deps.

**Prevention**:
- `make dev` installs HF deps by default.
- `scripts/reproduce_results.py` fails fast with `MissingHuggingFaceDependenciesError` if HF deps are missing (before wasting hours).

## Validated Baseline Configuration (Must Be Active)

These are the “known-good” defaults from `.env.example`:

| Setting | Value | Purpose |
|---------|-------|---------|
| `DATA_TRANSCRIPTS_DIR` | `data/transcripts_participant_only` | Deterministic transcript variant (reduces protocol leakage) |
| `EMBEDDING_BACKEND` | `huggingface` | Higher-quality FP16 similarity |
| `EMBEDDING_EMBEDDINGS_FILE` | `huggingface_qwen3_8b_paper_train_participant_only` | Must match backend + artifact |
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `chunk` | Chunk-level scoring (Spec 35) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `true` | Domain filtering (Spec 34) |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.3` | Guardrail (Spec 33) |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `500` | Prompt budget guardrail |
| `QUANTITATIVE_EVIDENCE_QUOTE_VALIDATION_ENABLED` | `true` | Prevent ungrounded quotes contaminating retrieval |

## Run Preflight (Don’t Skip)

Run these before long tmux runs:

```bash
# 1) Sanity: artifacts exist
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.npz
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.tags.json
ls data/embeddings/huggingface_qwen3_8b_paper_train_participant_only.chunk_scores.json

# 2) Sanity: transcripts exist (expected ~190 dirs)
ls -d data/transcripts_participant_only/*_P | wc -l

# 3) Dry-run prints an exact header (verify “FOUND” and score source=chunk)
uv run python scripts/reproduce_results.py --split paper-test --dry-run
```

If the header is wrong, **stop** and fix config before starting a multi-hour run.

## Running Reproduction (tmux)

```bash
tmux new -s run11

# Full reproduction (zero-shot + few-shot) with consistency signals (Spec 050)
uv run python scripts/reproduce_results.py \
  --split paper-test \
  --consistency-samples 5 \
  --consistency-temperature 0.3 \
  2>&1 | tee data/outputs/run11_$(date +%Y%m%d_%H%M%S).log
```

To make consistency “always on” without remembering CLI flags, set in `.env`:
- `CONSISTENCY_ENABLED=true`
- `CONSISTENCY_N_SAMPLES=5`
- `CONSISTENCY_TEMPERATURE=0.3`

Other useful flags:
- `--zero-shot-only` / `--few-shot-only`
- `--limit 1` (smoke test)
- `--embedding-backend huggingface` (override if you suspect env drift)

## Evaluation Metrics (CRITICAL)

Primary metrics are **coverage-aware**:
- `AURC` and `AUGRC` (preferred) + `Cmax`

**MAE comparisons are only valid when coverage is similar** between conditions.

Compute selective prediction metrics:

```bash
# Evaluate all base confidence variants (Spec 046+ suite)
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<run>.json \
  --mode few_shot \
  --confidence all \
  --bootstrap-resamples 1000
```

Token-level CSFs (Spec 051) require token logprobs from the backend. To remove uncertainty:

```bash
uv run python scripts/reproduce_results.py --split paper-test --few-shot-only --limit 1
rg -n '\"token_msp\"|\"token_pe\"|\"token_energy\"' data/outputs/both_*.json | head
```

## Common Failure Modes (And How We Prevent Them)

- **HF deps missing**: `make dev` (and reproduction now fails fast).
- **Embedding space mismatch**: `EMBEDDING_BACKEND` and `EMBEDDING_EMBEDDINGS_FILE` must match (`huggingface_*` with HF backend; `ollama_*` with Ollama backend).
- **Chunk scores missing**: ensure `...chunk_scores.json` exists and `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`.
- **Token CSFs missing**: backend didn’t return logprobs; skip token variants for that run.
- **Evidence grounding rejects everything**: this is a loud failure to prevent silent corruption; fix prompts/parsing rather than disabling validation.

## Pointers (SSOT)

- `NEXT-STEPS.md` (run checklist + exact commands)
- `PIPELINE-BRITTLENESS.md` (failure taxonomy + how we avoid silent corruption)
- `docs/statistics/metrics-and-evaluation.md` (definitions for AURC/AUGRC and confidence variants)
