# Repository Guidelines

## Critical Context

**IMPORTANT**: This is a **robust, independent implementation** that fixes severe methodological flaws in the original research paper (Greene et al.). Do NOT use "paper-parity" terminology.

The original paper has documented failures:
- **#81**: Participant-level PHQ-8 scores assigned to individual chunks (semantic mismatch)
- **#69**: Few-shot retrieval attaches participant scores to arbitrary text chunks
- **#66**: Paper uses invalid statistical comparison (MAE at different coverages)
- **#47, #46**: Paper does not specify quantization or sampling parameters
- **#45**: Paper uses undocumented custom split

Our implementation fixes these issues. Use "baseline defaults" or "validated configuration" instead of "paper-parity".

## Project Structure

- `src/ai_psychiatrist/`: Production package (agents, services, domain, infrastructure, config).
- `server.py`: FastAPI app + orchestration endpoints (`/health`, `/assess/*`, `/full_pipeline`).
- `scripts/`: Utilities (e.g., dataset prep, embeddings generation, reproduction helpers).
- `tests/`: `unit/`, `integration/`, and `e2e/` tests (e2e is opt-in).
- `docs/`: MkDocs documentation; `docs/_archive/` contains historical artifacts.
- `data/`: Local-only DAIC-WOZ artifacts (gitignored due to licensing).
- `_legacy/`, `_reference/`: Archived original research code (reference only; NOT production).

## Build, Test, and Development Commands

Use `uv` and the Makefile targets:

- `make dev`: Install dev + docs deps and pre-commit hooks.
- `make ci`: Run formatting check, lint, mypy, and full test suite.
- `make test`: Run all tests with coverage.
- `make serve`: Run the API locally via Uvicorn.
- `make docs` / `make docs-serve`: Build/serve MkDocs.

E2E tests require a running Ollama instance:
- `AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e`

## Configuration

**CRITICAL**: Copy `.env.example` to `.env` before running evaluations. Code defaults are conservative baselines for testing only.

### Recommended Settings (from .env.example)

| Setting | Value | Purpose |
|---------|-------|---------|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `chunk` | Use chunk-level scores (fixes core flaw) |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `true` | Filter refs by PHQ-8 item domain |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.3` | Drop low-quality refs |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `500` | Limit ref context per item |
| `DATA_TRANSCRIPTS_DIR` | `data/transcripts_participant_only` | Participant-only preprocessing |
| `EMBEDDING_BACKEND` | `huggingface` | FP16 embeddings (better quality) |

**Important (HuggingFace embeddings)**: Even with precomputed reference embeddings (`*.npz`), few-shot must embed each query (participant evidence) at runtime in the *same* embedding space. If `EMBEDDING_BACKEND=huggingface`, install HF deps first (`make dev`) or few-shot will fail when it tries to embed queries.

## Coding Style & Naming

- Format + lint: `ruff format` and `ruff check` (don't hand-format).
- Typing: `mypy` is strict across `src/`, `tests/`, and `scripts/`.
- Prefer explicit names and small, single-purpose functions/classes.
- Public interfaces live in `src/ai_psychiatrist/...`; avoid shipping test doubles in `src/`.

## Testing Guidelines

- Framework: `pytest` with markers (`unit`, `integration`, `e2e`).
- Add tests for behavior (not "smoke-only" calls). Prefer deterministic unit tests with mocks at I/O boundaries.
- Keep coverage ≥ 80% (CI enforces).

## Commit & Pull Requests

- Use conventional commits (e.g., `feat(spec-09): …`, `fix(embedding): …`, `docs: …`).
- PRs should include: clear summary, linked bug/spec docs when relevant, and `make ci` passing.

## Security & Configuration

- Do not commit DAIC-WOZ data or secrets; configure via `.env` (see `.env.example`).
- Prefer running commands via `uv run …` to ensure the repo's venv is used.

## Few-Shot RAG Pipeline (Specs 33-46)

Our implementation fixes the original methodology's core flaw: participant-level PHQ-8 scores were assigned to arbitrary chunks.

### Spec Status

| Spec | Description | Status |
|------|-------------|--------|
| 33 | Retrieval guardrails | Enabled (`EMBEDDING_MIN_REFERENCE_SIMILARITY=0.3`) |
| 34 | Item-tag filtering | Enabled (`EMBEDDING_ENABLE_ITEM_TAG_FILTER=true`) |
| 35 | Chunk-level scoring | Enabled (`EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`) |
| 36 | CRAG runtime validation | Disabled by default (runtime cost) |
| 46 | Retrieval similarity confidence signals | Implemented (new confidence variants) |

### Running the Full Evaluation

```bash
# In tmux for long runs (~2-3 hours):
tmux new -s run9
uv run python scripts/reproduce_results.py --split paper-test \
  2>&1 | tee data/outputs/run9_$(date +%Y%m%d_%H%M%S).log
```

## Evaluation Metrics (CRITICAL)

**WARNING: The original paper's methodology is not reproducible. Our implementation fixes the flaws.**

### Primary Metrics: AURC and AUGRC (NOT MAE)

**You cannot compare MAE values at different coverage levels.** Use coverage-aware metrics:

| Metric | What It Measures | Lower = Better |
|--------|------------------|----------------|
| **AURC** | Area Under Risk-Coverage curve | ✓ |
| **AUGRC** | Area Under Generalized Risk-Coverage curve (preferred) | ✓ |
| **Cmax** | Maximum coverage | Higher = more predictions |

**Why AUGRC over AURC?** AURC puts excessive weight on high-confidence failures. AUGRC provides holistic silent failure risk assessment. See [Traub et al. 2024](https://arxiv.org/html/2407.01032v1).

### When MAE Is Acceptable

MAE comparisons are ONLY valid when coverage is similar between conditions.

```bash
# Compute AURC/AUGRC with new confidence signals
uv run python scripts/evaluate_selective_prediction.py \
  --input data/outputs/<your_run>.json \
  --mode few_shot \
  --confidence hybrid_evidence_similarity \
  --bootstrap-resamples 1000
```
