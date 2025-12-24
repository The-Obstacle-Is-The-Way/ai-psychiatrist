# Spec Validation & Gap Analysis (Paper ↔ Code ↔ Specs)

This report audits the research paper (`_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`), the current repo implementation, and the spec series under `docs/specs/`.

## Target Strategy: Paper-Optimal

**All specs now target paper-optimal behavior**, not as-is code parity. As-is behavior is documented only for migration context.

| Target | Value | Paper Reference |
|--------|-------|-----------------|
| Qualitative/Judge/Meta chat | `gemma3:27b` | Section 2.2 (paper baseline) |
| Quantitative chat | `gemma3:27b` (Appendix F: MedGemma as optional alternative via HuggingFace) | Section 2.2; Appendix F |
| Embedding model family | Qwen 3 8B Embedding (example Ollama tag: `qwen3-embedding:8b`; quantization not specified in paper) | Section 2.2 |
| Feedback threshold | 3 (scores < 4 trigger) | Section 2.3.1 |
| Max iterations | 10 | Section 2.3.1 |
| top_k references | 2 | Appendix D |
| chunk_size | 8 | Appendix D |
| dimension | 4096 | Appendix D |

## DISCREPANCY REPORT

### Current Status

The production pipeline is implemented in `src/ai_psychiatrist/` and exposed via
`server.py`. The original research prototype is archived under `_legacy/` and is
kept only for historical comparison and paper-parity auditing.

### Critical Issues (Historical / Already Addressed)

This table captures issues that existed in the original prototype but are no longer
blocking because the production pipeline has replaced the legacy runtime path.

| Area | Legacy location | Historical issue | Status in production |
|------|-----------------|------------------|----------------------|
| Transcript loading | `_legacy/agents/interview_simulator.py` | Hardcoded transcript path expectations | Replaced by `TranscriptService` + request-driven transcript resolution |
| Few-shot artifacts | `_legacy/agents/quantitative_assessor_f.py` | Missing reference artifact paths | Production uses `data/embeddings/reference_embeddings.{npz,json}` |
| Prompt formatting | `_legacy/agents/qualitative_assessor_f.py` | Malformed tag templates | Production prompts live in `src/ai_psychiatrist/agents/prompts/` |
| Feedback loop | `_legacy/qualitative_assessment/feedback_loop.py` | Not wired in demo API | Production wires `FeedbackLoopService` into `/full_pipeline` |
| Environment | `_legacy/assets/env_reqs.yml` | Conda env drift vs imports | Production uses `pyproject.toml` + `uv` |

### Missing Coverage

After the spec edits on 2025-12-18, the repo has an explicit code→spec mapping in `docs/specs/00-overview.md`.
Remaining “coverage gaps” are primarily depth gaps (e.g., not every notebook cell is specified), not missing file ownership.

| Codebase File | Functionality | Spec Gap |
|---------------|---------------|----------|
| `_legacy/visualization/quan_visualization.ipynb` | Exact MAE/N/A aggregation formulas per symptom | Only summarized; not fully line-by-line specified |
| `_legacy/quantitative_assessment/embedding_batch_script.py` | Full hyperparameter sweep CLI and JSONL schema variants | Only summarized; not fully specified |

### Prompt Mismatches

| Spec | Codebase Prompt | Spec Prompt | Diff |
|------|-----------------|-------------|------|
| 06 | `_legacy/agents/qualitative_assessor_f.py` | `docs/archive/specs/06_QUALITATIVE_AGENT.md` | Legacy prompt formatting differs from production prompt templates |
| 08 | `_legacy/agents/quantitative_assessor_f.py` | `docs/archive/specs/08_EMBEDDING_SERVICE.md` | Legacy prompt structure differs from production prompt templates |
| 10 | `_legacy/agents/meta_reviewer.py` | `docs/archive/specs/10_META_REVIEW_AGENT.md` | Legacy prompt differs from production prompt templates |

### Undocumented Constants

| Constant | Value in Code | Location | Spec Status |
|----------|---------------|----------|-------------|
| `TRANSCRIPT_PATH` | env var | `_legacy/agents/interview_simulator.py` | Legacy-only (archived) |
| `VERBOSE` | `True` (CLI `--quiet` disables) | `_legacy/agents/quantitative_assessor_f.py` | Legacy-only (archived) |
| `DOMAIN_KEYWORDS` | PHQ-8 keyword lists | `src/ai_psychiatrist/agents/prompts/quantitative.py` | Production (configurable via YAML) |

### Paper vs Spec vs Code Matrix

| Parameter | Paper Value | Production Default | Legacy Prototype (archived) | Status |
|-----------|------------|--------------------|-----------------------------|--------|
| Chat models | Gemma 3 27B (Section 2.2); MedGemma 27B evaluated as optional quantitative alternative (Appendix F) | Production defaults use `gemma3:27b` for all agents; optional `MODEL_QUANTITATIVE_MODEL=medgemma:27b` when using the HuggingFace backend | Varies across `_legacy/*`; not used in production | **Aligned** |
| Embedding model | Qwen 3 8B Embedding | `qwen3-embedding:8b` + `EMBEDDING_DIMENSION=4096` | Varies across `_legacy/*`; not used in production | **Aligned** |
| `chunk_size` | 8 (optimal) | `EMBEDDING_CHUNK_SIZE=8` | Varies; see `_legacy/quantitative_assessment/*` | **Aligned** |
| `step_size` | 2 | `EMBEDDING_CHUNK_STEP=2` | Varies; see `_legacy/quantitative_assessment/*` | **Aligned** |
| `top_k` examples | 2 (optimal) | `EMBEDDING_TOP_K_REFERENCES=2` | Varies; see `_legacy/quantitative_assessment/*` | **Aligned** |
| Embedding dimension | 4096 (optimal) | `EMBEDDING_DIMENSION=4096` | Varies; see `_legacy/*` | **Aligned** |
| Feedback threshold | scores < 4 trigger refinement | `FEEDBACK_SCORE_THRESHOLD=3` and `FEEDBACK_TARGET_SCORE=4` | Varies; see `_legacy/qualitative_assessment/*` | **Aligned** |
| Feedback max iters | 10 | `FEEDBACK_MAX_ITERATIONS=10` | Varies; see `_legacy/qualitative_assessment/*` | **Aligned** |

## Specific Questions (Answers)

1. **Exact XML tag structures in prompts captured?**
   - Yes for the production prompt templates in `src/ai_psychiatrist/agents/prompts/`.
   - Legacy prompt variants are preserved for auditing in `docs/archive/specs/` and `_legacy/`.

2. **Is the JSON repair cascade complete?**
   - Yes in `src/ai_psychiatrist/agents/quantitative.py`: tolerant fixups → `<answer>` extraction → LLM repair → fallback skeleton.

3. **Do visualization notebooks reveal missed processing?**
   - Yes: severity/diagnosis mapping + metric calculations live in `_legacy/visualization/meta_review_heatmap.ipynb`, and MAE/N/A aggregation + retrieval diagnostics live in `_legacy/visualization/quan_visualization.ipynb` and `_legacy/quantitative_assessment/embedding_quantitative_analysis.ipynb`.

4. **Does `server.py` show orchestration patterns not in Spec 11?**
   - `server.py` is the production orchestration layer and is the SSOT for endpoints; see `docs/reference/api/endpoints.md`.

5. **Environment variables in code not documented in Spec 01?**
   - Yes originally (`TRANSCRIPT_PATH`, plus many HPC `OLLAMA_*`/`GGML_*` exports); now documented in Spec 01/05/11.

6. **SLURM job configuration reflected?**
   - Legacy SLURM scripts are archived under `_legacy/slurm/` and referenced for historical context only.

7. **Are `analysis_output` CSV/JSONL structures documented?**
   - Legacy artifacts (if present) are archived under `_legacy/analysis_output/`.

8. **Does `_legacy/agents/interview_simulator.py` serve a purpose worth preserving?**
   - It remains useful for reproducing the legacy prototype behavior, but production uses `TranscriptService` and request-driven transcript resolution.

## 2025 Tooling Verification (Quick Check)

Verified that the referenced docs are live as of 2025-12-18:

- `uv`: https://docs.astral.sh/uv/
- `Ruff`: https://docs.astral.sh/ruff/
- `pydantic-settings v2`: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- `structlog`: https://www.structlog.org/en/stable/
- `pytest-asyncio`: redirects to https://pytest-asyncio.readthedocs.io/en/stable/
- `Ollama API`: https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md (confirms `/api/generate`, `/api/chat`, `/api/embeddings`)

## Spec Confidence Scores (0–100%)

Scores reflect **accuracy + completeness** for paper-optimal targeting with as-is documentation for migration.

| Spec | Score | Notes |
|------|-------|-------|
| 00 | 98% | Explicit code→spec coverage map; distinguishes as-is vs paper-optimal target |
| 01 | 98% | Paper-optimal `.env.example` (Gemma baseline; MedGemma documented as optional Appendix F alternative); documents SLURM/conda for migration |
| 02 | 96% | Domain entities aligned with paper; as-is conventions documented |
| 03 | 98% | **Paper-optimal defaults**: per-agent models, threshold=3, top_k=2, dim=4096 |
| 04 | 98% | Paper-optimal target table added; documents Ollama endpoints |
| 05 | 95% | Captures both file-based loader and DAIC-WOZ research ingestion |
| 06 | 95% | Includes as-is qualitative prompt (malformed tags documented for migration) |
| 07 | 98% | Paper-optimal feedback threshold (< 4); as-is prompts documented |
| 08 | 98% | **Paper-optimal target table**: top_k=2, dim=4096, chunk_size=8 |
| 09 | 98% | **Paper-optimal baseline**: Gemma3 few-shot, top_k=2; Appendix F optional MedGemma evaluation (lower MAE but lower coverage) |
| 10 | 95% | As-is prompt documented; target uses paper models via config |
| 11 | 95% | Documents as-is `server.py` endpoint + known runtime blockers |
| 12 | 95% | As-is observability documented; target uses structlog |
