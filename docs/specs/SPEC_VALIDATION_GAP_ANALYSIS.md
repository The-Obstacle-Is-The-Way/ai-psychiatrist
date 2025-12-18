# Spec Validation & Gap Analysis (Paper ↔ Code ↔ Specs)

This report audits the research paper (`_literature/markdown/ai_psychiatrist/ai_psychiatrist.md`), the current repo implementation, and the spec series under `docs/specs/`.

## Target Strategy: Paper-Optimal

**All specs now target paper-optimal behavior**, not as-is code parity. As-is behavior is documented only for migration context.

| Target | Value | Paper Reference |
|--------|-------|-----------------|
| Qualitative/Judge/Meta chat | `gemma3:27b` | Section 2.2 (paper baseline) |
| Quantitative chat | MedGemma 27B (example Ollama tag: `alibayram/medgemma:27b`) | Appendix F (MAE 0.505; fewer predictions) |
| Embedding model family | Qwen 3 8B Embedding (example Ollama tag: `dengcao/Qwen3-Embedding-8B:Q8_0`; quantization not specified in paper) | Section 2.2 |
| Feedback threshold | 3 (scores < 4 trigger) | Section 2.3.1 |
| Max iterations | 10 | Section 2.3.1 |
| top_k references | 2 | Appendix D |
| chunk_size | 8 | Appendix D |
| dimension | 4096 | Appendix D |

## DISCREPANCY REPORT

### Critical Issues (Must Fix)

| Spec | File | Line | Issue | Evidence | Fix Required |
|------|------|------|-------|----------|--------------|
| 11 | `server.py` | 29 | API contract differs from spec target | Current endpoint is `POST /full_pipeline` and loads transcript from disk; Spec 11 target API accepts transcript text | Documented as-is API in Spec 11; future refactor can implement target API |
| 11 | `agents/interview_simulator.py` | 14 | Default transcript file missing from repo | Default path is `agents/transcript.txt` but file is not present | Provide sample transcript or require `TRANSCRIPT_PATH` in `.env.example`/docs |
| 11/09/08 | `agents/quantitative_assessor_f.py` | 421 | Few-shot defaults rely on missing artifacts | Defaults reference `agents/chunk_8_step_2_participant_embedded_transcripts.pkl` and `agents/*_split_Depression_AVEC2017.csv` (not in repo) | Commit small sample artifacts or document required external paths clearly |
| 10/11 | `server.py` | 55 | Meta-review input type mismatch | For few-shot, quantitative agent returns a dict; `agents/meta_reviewer.py` interpolates it as text | Normalize quantitative output for meta-review (e.g., render tag-based format consistently) |
| 06 | `agents/qualitative_assessor_f.py` | 54 | Malformed XML template in prompt | Closing tag typo: `</little_interest or pleasure>` and unclosed `<exact_quotes>` under `<social_factors>` | Fix prompt template in code (if XML is required), or remove XML requirement |
| 08 | `agents/quantitative_assessor_f.py` | 411 | Reference “XML” is not valid XML | Uses `<Reference Examples>` as both open and close marker | Decide on a valid tag format and update both prompt + downstream parsing |
| 07 | Paper vs demo pipeline | n/a | Paper feedback loop not implemented in server | Paper: score <4 triggers loop; `server.py` does no iteration | Either implement loop in server or explicitly keep as research-only behavior |
| 01 | `assets/env_reqs.yml` | 1 | Conda environment does not match imports | Repo code imports `fastapi`, `pydantic`, `pandas`, `numpy`, `sklearn` but env file mostly includes `requests`/Jupyter basics | Update env file or add `pyproject.toml`/`uv` per Spec 01 target |

### Missing Coverage

After the spec edits on 2025-12-18, the repo has an explicit code→spec mapping in `docs/specs/00_OVERVIEW.md`.
Remaining “coverage gaps” are primarily depth gaps (e.g., not every notebook cell is specified), not missing file ownership.

| Codebase File | Functionality | Spec Gap |
|---------------|---------------|----------|
| `visualization/quan_visualization.ipynb` | Exact MAE/N/A aggregation formulas per symptom | Only summarized; not fully line-by-line specified |
| `quantitative_assessment/embedding_batch_script.py` | Full hyperparameter sweep CLI and JSONL schema variants | Only summarized; not fully specified |

### Prompt Mismatches

| Spec | Codebase Prompt | Spec Prompt | Diff |
|------|-----------------|-------------|------|
| 06 | `agents/qualitative_assessor_f.py` | `docs/specs/06_QUALITATIVE_AGENT.md` | Spec now contains verbatim as-is prompt; target prompt may intentionally fix malformed tags |
| 08 | `agents/quantitative_assessor_f.py` | `docs/specs/08_EMBEDDING_SERVICE.md` | As-is uses `<Reference Examples>` “open=open”; target uses valid close tag `</Reference Examples>` |
| 10 | `agents/meta_reviewer.py` | `docs/specs/10_META_REVIEW_AGENT.md` | As-is prompt is embedded as a single user message; target may separate system/user prompts |

### Undocumented Constants

| Constant | Value in Code | Location | Spec Status |
|----------|---------------|----------|-------------|
| `TRANSCRIPT_PATH` | env var | `agents/interview_simulator.py:16` | Documented (Spec 01/05/11) |
| `VERBOSE` | `True` (CLI `--quiet` disables) | `agents/quantitative_assessor_f.py:20` | Partially documented (Spec 12 notes verbose logs) |
| `DOMAIN_KEYWORDS` | PHQ8 keyword lists | `agents/quantitative_assessor_f.py:29` | Documented (Spec 09) |

### Paper vs Spec vs Code Matrix

| Parameter | Paper Value | Spec Target | As-Is Code | Status |
|-----------|------------|-------------|------------|--------|
| Chat model | Gemma 3 27B (Section 2.2); MedGemma 27B evaluated for quantitative (Appendix F) | `gemma3:27b` (qual/judge/meta), MedGemma 27B for quantitative | `llama3` (demo), `gemma3*` (scripts) | **Spec aligned; demo deviates** |
| Embedding model | Qwen 3 8B Embedding (quantization not specified) | Qwen 3 8B Embedding (example tag `:Q8_0`) | `:Q4_K_M` (demo), `:Q8_0` (scripts) | **Aligned (tag is impl choice)** |
| `chunk_size` | 8 (optimal) | 8 | 8 (pickle naming) | **Aligned** |
| `step_size` | 2 | 2 | 2 | **Aligned** |
| `top_k` examples | 2 (optimal) | 2 | 3 (demo default) | **Spec aligned with paper** |
| Embedding dimension | 4096 (optimal) | 4096 | 4096 (notebooks) | **Aligned** |
| Feedback threshold | < 4 (Section 2.3.1) | 3 (scores ≤ 3 trigger) | ≤ 2 (scripts) | **Spec aligned with paper** |
| Feedback max iters | 10 | 10 | 10 (scripts), none (demo) | **Aligned** |
| Expected MAE | 0.619 (Gemma 3 few-shot); 0.505 (MedGemma few-shot) | 0.505 target (MedGemma) with 0.619 baseline (Gemma 3) | 0.619 (Gemma 3 reported) | **Spec targets paper-validated MedGemma improvement** |
| Meta-review accuracy | 0.780 | 0.780 | Computed in notebooks | **Aligned** |

## Specific Questions (Answers)

1. **Exact XML tag structures in prompts captured?**  
   - Yes for the demo agent prompts in Specs 06/09/10 (verbatim blocks added).  
   - Additional tag schemas used by cluster scripts (`<assessment>`, `<quotes>`) are noted in Spec 06/07.

2. **Is the JSON repair cascade complete?**  
   - Yes in `agents/quantitative_assessor_f.py`: tolerant fixups → `<answer>` extraction → LLM repair → fallback skeleton.

3. **Do visualization notebooks reveal missed processing?**  
   - Yes: severity/diagnosis mapping + metric calculations live in `visualization/meta_review_heatmap.ipynb`, and MAE/N/A aggregation + retrieval diagnostics live in `visualization/quan_visualization.ipynb` and `quantitative_assessment/embedding_quantitative_analysis.ipynb`.

4. **Does `server.py` show orchestration patterns not in Spec 11?**  
   - Yes: fixed transcript loader, `mode` only, no feedback loop, type mismatches; documented in Spec 11 as-is section.

5. **Environment variables in code not documented in Spec 01?**  
   - Yes originally (`TRANSCRIPT_PATH`, plus many HPC `OLLAMA_*`/`GGML_*` exports); now documented in Spec 01/05/11.

6. **SLURM job configuration reflected?**  
   - Yes: documented under Spec 01 and referenced in Spec 00 mapping; `slurm/job_ollama.sh` env exports are now listed.

7. **Are `analysis_output` CSV/JSONL structures documented?**  
   - Yes: Spec 12 now summarizes schemas and filenames present in `analysis_output/`.

8. **Does `interview_simulator.py` serve a purpose worth preserving?**  
   - Yes: it is the only transcript ingestion path used by the demo API (`server.py`) and supports `TRANSCRIPT_PATH`.

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
| 01 | 98% | Paper-optimal .env.example (Gemma baseline + MedGemma quantitative); documents SLURM/conda for migration |
| 02 | 96% | Domain entities aligned with paper; as-is conventions documented |
| 03 | 98% | **Paper-optimal defaults**: per-agent models, threshold=3, top_k=2, dim=4096 |
| 04 | 98% | Paper-optimal target table added; documents Ollama endpoints |
| 05 | 95% | Captures both file-based loader and DAIC-WOZ research ingestion |
| 06 | 95% | Includes as-is qualitative prompt (malformed tags documented for migration) |
| 07 | 98% | Paper-optimal feedback threshold (< 4); as-is prompts documented |
| 08 | 98% | **Paper-optimal target table**: top_k=2, dim=4096, chunk_size=8 |
| 09 | 98% | **Paper-optimal target**: MedGemma, few-shot, top_k=2; expected MAE 0.505 |
| 10 | 95% | As-is prompt documented; target uses paper models via config |
| 11 | 95% | Documents as-is `server.py` endpoint + known runtime blockers |
| 12 | 95% | As-is observability documented; target uses structlog |
