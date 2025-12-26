# AI-Psychiatrist: Engineering Refactor Spec Overview

## Executive Summary

This document series describes the plan used to transform the AI-Psychiatrist research codebase into a maintainable, testable Python system. The specs follow **vertical slice architecture**, enabling incremental delivery of working features while maintaining paper parity where the paper is explicit (and tracking gaps where it is not).

## Paper Reference

**Title:** AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews

**Key Contributions:**
1. LLM-based multi-agent system with four collaborative agents
2. Embedding-based few-shot prompting for PHQ-8 prediction
3. Iterative self-refinement via judge agent feedback loop
4. Meta-review agent for severity integration

## Current State Analysis

### What Exists (Current Repo)
- Modern, testable implementation under `src/ai_psychiatrist/` (agents, services, domain, infrastructure)
- FastAPI server entrypoint (`server.py`) exposing `/health`, `/assess/*`, and `/full_pipeline`
- Archived original research/prototype code under `_reference/` (scripts, notebooks, SLURM jobs, example outputs)
- Local-only data artifacts under `data/` (DAIC-WOZ transcripts, generated embeddings); `data/` is gitignored due to licensing

### Codebase → Spec Coverage Map (No Orphaned Files)

This spec series covers both:
1) **As-is repo code** (current behavior and prompts), and
2) **Target refactor** (planned `src/ai_psychiatrist/` architecture).

| Codebase file | Purpose | Covered by spec |
|---|---|---|
| `server.py` | FastAPI orchestration endpoint | Spec 11 |
| `_reference/agents/interview_simulator.py` | Fixed transcript loader (`TRANSCRIPT_PATH`) | Spec 05 |
| `_reference/agents/qualitative_assessor_f.py` / `_reference/agents/qualitative_assessor_z.py` | Qualitative assessor prompts (F/Z variants) | Spec 06 |
| `_reference/agents/qualitive_evaluator.py` | Judge/evaluator (4 metrics) | Spec 07 |
| `_reference/agents/quantitative_assessor_f.py` / `_reference/agents/quantitative_assessor_z.py` | Quantitative scoring (few/zero-shot) | Spec 09 |
| `_reference/agents/meta_reviewer.py` | Meta-review prompt + severity output | Spec 10 |
| `_reference/agents/interview_evaluator.py` | Conversation quality evaluator (non-paper) | Spec 11 (as-is extras) |
| `_reference/qualitative_assessment/qual_assessment.py` | Cluster script for qualitative runs (Gemma 3 27B) | Spec 06 (as-is research) |
| `_reference/qualitative_assessment/feedback_loop.py` | Cluster script feedback loop (<=2 threshold, 10 iters) | Spec 07 (as-is research) |
| `_reference/quantitative_assessment/quantitative_analysis.py` | Cluster script zero-shot quantitative analysis | Spec 09 (as-is research) |
| `_reference/quantitative_assessment/embedding_batch_script.py` | Cluster script + sweeps for embeddings few-shot runs | Spec 08/09 (as-is research) |
| `_reference/quantitative_assessment/basic_quantitative_analysis.ipynb` | Zero-shot notebook analysis | Spec 09 (as-is research) |
| `_reference/quantitative_assessment/embedding_quantitative_analysis.ipynb` | Few-shot notebook (splits, t-SNE, retrieval stats) | Spec 08/09 (as-is research) |
| `_reference/visualization/qual_boxplot.ipynb` | Qual judge boxplots + mean/SD | Spec 07 (as-is research) |
| `_reference/visualization/quan_visualization.ipynb` | Quant MAE/confusion + N/A rates + t-SNE | Spec 08/09 (as-is research) |
| `_reference/visualization/meta_review_heatmap.ipynb` | Severity/diagnosis confusion matrices + metrics | Spec 10 (as-is research) |
| `_reference/analysis_output/*` | Example outputs (CSV/JSONL) | Spec 11/12 (as-is validation artifacts) |
| `_reference/slurm/job_ollama.sh` / `_reference/slurm/job_assess.sh` | HPC deployment scripts | Spec 01/04 |
| `_reference/assets/env_reqs.yml` / `_reference/assets/ollama_example.py` | Conda env + Ollama usage example | Spec 01/04 |
| `_reference/assets/overview.png` | System overview image (non-code artifact) | Spec 00 |

### Paper → Spec Traceability Map (All Figures + Appendices)

This table ensures every paper figure/image extracted under `_literature/markdown/ai_psychiatrist/` is explicitly owned by one or more specs.

| Paper element | Evidence (repo) | What it establishes | Covered by spec(s) |
|---|---|---|---|
| Figure 1: Multi-agent system overview | `_literature/markdown/ai_psychiatrist/_page_2_Figure_1.jpeg` | Four-agent orchestration + human review loop | Spec 11 (pipeline), Spec 06/07/09/10 (agents) |
| Figure 2: Qual scores pre/post feedback | `_literature/markdown/ai_psychiatrist/_page_5_Figure_1.jpeg` | Feedback loop improves judge metrics | Spec 07 |
| Figure 3: Human vs LLM judge scores | `_literature/markdown/ai_psychiatrist/_page_5_Figure_3.jpeg` | Judge rubric is meaningful vs human | Spec 07 |
| Figure 4: PHQ-8 confusion matrices (few-shot) | `_literature/markdown/ai_psychiatrist/_page_6_Figure_1.jpeg` | Item-wise quantitative performance + “N/A” availability | Spec 09 (agent), Spec 12 (validation artifacts) |
| Figure 5: Few-shot vs zero-shot bar chart | `_literature/markdown/ai_psychiatrist/_page_6_Figure_7.jpeg` | Few-shot improves MAE vs zero-shot | Spec 09 |
| Figure 6: Severity confusion matrices | `_literature/markdown/ai_psychiatrist/_page_7_Figure_1.jpeg` | Meta-review severity performance + comparisons | Spec 10 |
| Table 1: Severity metrics | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Accuracy/BA/P/R/F1 targets | Spec 10 |
| Figure A2: Chunk size × examples sweep | `_literature/markdown/ai_psychiatrist/_page_15_Figure_8.jpeg` | Hyperparameter optimization (chunk_size, N_example) | Spec 08/09 |
| Figure A3: Embedding dimension sweep | `_literature/markdown/ai_psychiatrist/_page_15_Figure_10.jpeg` | Dimension selection (4096 optimal) | Spec 08 |
| Figure A4: t-SNE embedding clusters | `_literature/markdown/ai_psychiatrist/_page_16_Figure_1.jpeg` | Retrieval embedding space sanity check | Spec 08 |
| Figure A5: Retrieval error histograms | `_literature/markdown/ai_psychiatrist/_page_16_Figure_2.jpeg` + `_literature/markdown/ai_psychiatrist/_page_16_Figure_3.jpeg` | Retrieval agreement statistics; symptom gaps (e.g., appetite) | Spec 08/09 |
| Figure A6: MedGemma confusion matrices | `_literature/markdown/ai_psychiatrist/_page_17_Figure_1.jpeg` | MedGemma quantitative performance (MAE 0.505) | Spec 09 |
| Figure A7: MedGemma few vs zero bar chart | `_literature/markdown/ai_psychiatrist/_page_17_Figure_3.jpeg` | MedGemma few-shot vs zero-shot comparison | Spec 09 |

| Paper appendix | Evidence (paper markdown) | What it establishes | Covered by spec(s) |
|---|---|---|---|
| Appendix B | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Judge metric definitions + mistake→score mapping | Spec 07 |
| Appendix C | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Split strategy + rare-score handling | Spec 05 |
| Appendix D | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Hyperparameter search space + chosen optima | Spec 03/08/09 |
| Appendix E | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Retrieval statistics + appetite evidence failure case | Spec 08/09 |
| Appendix F | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | MedGemma quantitative improvement + tradeoffs | Spec 09 |
| Appendix G | `_literature/markdown/ai_psychiatrist/ai_psychiatrist.md` | Single-prompt experiment failure mode | Spec 11 |

### Critical Gaps (Legacy Research Code)

These gaps describe the original research implementation archived under `_reference/`. The modern
refactor under `src/ai_psychiatrist/` addresses most of them; remaining paper-reproduction gaps are
tracked in `docs/results/reproduction-notes.md` and `docs/bugs/`.

| Gap (legacy) | Impact | Status in refactor |
|-----|--------|----------|
| No tests | Cannot verify correctness | ✅ Addressed (extensive unit/integration/e2e tests) |
| Inconsistent error handling | Some paths crash; others silently degrade | ✅ Improved (typed exceptions + defensive parsing) |
| No structured logging/observability | Difficult to debug and measure | ✅ Addressed (structlog; deeper observability deferred) |
| Hardcoded paths/configs | Cannot deploy | ✅ Addressed (Pydantic settings; `.env.example`) |
| No type safety | Runtime errors | ✅ Addressed (mypy strict across `src/`, `tests/`, `scripts/`, `server.py`) |
| No dependency injection | Untestable | ✅ Addressed (protocols + DI in `server.py`) |
| No API documentation | Unusable by others | ✅ Addressed (FastAPI OpenAPI + docs) |
| Naming inconsistencies | Confusing codebase | ✅ Improved (consistent module and doc paths) |
| No CI/CD pipeline | Manual deployment | ✅ Addressed (Make targets + CI workflow) |

### Remaining Gaps (Current)

| Gap | Impact | Tracking |
|-----|--------|----------|
| Paper MAE parity not yet achieved | Cannot claim reproduction within tolerance | `docs/results/reproduction-notes.md` |
| Paper sampling/quantization parameters unspecified | Can affect MAE/coverage and runtime | `docs/bugs/gap-001-paper-unspecified-parameters.md` |

## Modern Python Tooling Stack (2025)

### Core Tooling
| Tool | Purpose | Replaces |
|------|---------|----------|
| [uv](https://docs.astral.sh/uv/) | Package/project management | pip, poetry, pyenv, virtualenv |
| [Ruff](https://docs.astral.sh/ruff/) | Linting + formatting | flake8, black, isort |
| [pytest](https://docs.pytest.org/) | Testing framework | unittest |
| [Pydantic v2](https://docs.pydantic.dev/) | Data validation | manual validation |
| [structlog](https://www.structlog.org/) | Structured logging | print(), logging |
| [mypy](https://mypy-lang.org/) | Static type checking | runtime errors |

### Project Structure (Target)
```
ai-psychiatrist/
├── pyproject.toml           # Single source of truth (PEP 621)
├── uv.lock                   # Locked dependencies
├── Makefile                  # Developer workflow automation
├── .env.example              # Environment template
├── src/
│   └── ai_psychiatrist/
│       ├── __init__.py
│       ├── config.py         # Pydantic BaseSettings
│       ├── api/              # FastAPI routes
│       ├── agents/           # Agent implementations
│       ├── domain/           # Business logic / entities
│       ├── services/         # External integrations
│       └── infrastructure/   # DB, logging, etc.
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
```

## Vertical Slice Architecture

Each spec represents a **vertical slice** - a complete feature from API to storage that can be independently developed, tested, and deployed.

```
┌─────────────────────────────────────────────────────────┐
│                     VERTICAL SLICE                       │
├─────────────────────────────────────────────────────────┤
│  API Layer      │  FastAPI endpoint + Pydantic models   │
├─────────────────────────────────────────────────────────┤
│  Service Layer  │  Business logic orchestration         │
├─────────────────────────────────────────────────────────┤
│  Domain Layer   │  Entities + Value Objects             │
├─────────────────────────────────────────────────────────┤
│  Infra Layer    │  LLM client, persistence, logging     │
├─────────────────────────────────────────────────────────┤
│  Tests          │  Unit + Integration + E2E             │
└─────────────────────────────────────────────────────────┘
```

## Spec Index

Note: Specs **01–11.5** and **12.5** are archived under `docs/archive/specs/` (historical record).
Active specs live under `docs/specs/`.

| Spec | Title | Deliverable |
|------|-------|-------------|
| **01** | Project Bootstrap | pyproject.toml, uv, Makefile, CI/CD |
| **02** | Core Domain | PHQ8, Transcript, Assessment entities |
| **03** | Configuration & Logging | Pydantic settings, structlog |
| **04** | LLM Infrastructure | Ollama client abstraction |
| **04A** | Data Organization | DAIC-WOZ dataset preparation script |
| **04.5** | Integration Checkpoint | Foundation verification, bug hunt |
| **05** | Transcript Loader | Interview data ingestion |
| **06** | Qualitative Agent | PHQ-8 symptom analysis |
| **07** | Judge Agent | Self-refinement feedback loop |
| **07.5** | Integration Checkpoint | Qualitative path verification |
| **08** | Embedding Service | Vector similarity search |
| **09** | Quantitative Agent | Few-shot PHQ-8 scoring |
| **09.5** | Integration Checkpoint | Quantitative path verification |
| **10** | Meta-Review Agent | Severity integration |
| **11** | Full Pipeline API | Complete assessment endpoint |
| **11.5** | Integration Checkpoint | Full pipeline verification |
| **12** | Observability | Metrics, tracing, health checks |
| **12.5** | Final Cleanup | Legacy removal, cruft cleanup |
| **13** | Pydantic AI Integration | Full framework migration with TextOutput mode |
| **14** | Keyword Matching Improvements | Word-boundary regex, negation detection |
| **15** | Experiment Tracking | Full provenance, semantic naming, registry |
| **16** | Log Output Improvements | ANSI auto-detection, clean log files |

## Integration Checkpoints

The `.5` specs are **mandatory pause points** for quality review:

```text
Spec 01-04A: Foundation
      │
      ▼
┌─────────────────────────────┐
│  CHECKPOINT 04.5            │  Verify foundation before services
│  Bug hunt: P0/P1/P2 issues  │
└─────────────────────────────┘
      │
      ▼
Spec 05-07: Qualitative Path
      │
      ▼
┌─────────────────────────────┐
│  CHECKPOINT 07.5            │  First complete vertical slice
│  Verify qual → judge flow   │
└─────────────────────────────┘
      │
      ▼
Spec 08-09: Quantitative Path
      │
      ▼
┌─────────────────────────────┐
│  CHECKPOINT 09.5            │  Both paths ready for merge
│  Verify embeddings + quant  │
└─────────────────────────────┘
      │
      ▼
Spec 10-11: Integration
      │
      ▼
┌─────────────────────────────┐
│  CHECKPOINT 11.5            │  Functional complete
│  Paper metrics verified     │
└─────────────────────────────┘
      │
      ▼
Spec 12: Observability (Polish)
      │
      ▼
┌─────────────────────────────┐
│  CHECKPOINT 12.5            │  FINAL: Remove legacy code
│  Clean codebase only        │
└─────────────────────────────┘
```

Each checkpoint includes:
- **Bug Hunt Protocol**: P0/P1/P2/P3/P4 issue detection
- **Quality Gates**: CI, coverage, type checking
- **Technical Debt Inventory**: What's acceptable vs. must-fix
- **Exit Criteria**: Must pass before proceeding

## Design Principles

### SOLID Principles
- **S**ingle Responsibility: Each agent has one job
- **O**pen/Closed: Extend via protocols, not modification
- **L**iskov Substitution: Agents implement common interface
- **I**nterface Segregation: Small, focused protocols
- **D**ependency Inversion: Depend on abstractions

### Gang of Four Patterns Applied
- **Strategy**: Swappable LLM providers
- **Template Method**: Base agent with hooks
- **Factory**: Agent instantiation
- **Observer**: Event-driven logging

### Clean Architecture Layers
1. **Entities**: PHQ8Score, TranscriptChunk, Assessment
2. **Use Cases**: AssessTranscript, RefineQualitative
3. **Interface Adapters**: FastAPI routes, Pydantic models
4. **Infrastructure**: Ollama client, file loaders

### DRY (Don't Repeat Yourself)
- Shared prompt templates
- Common JSON parsing utilities
- Unified error handling

### Test-Driven Development
- Write tests BEFORE implementation
- Red → Green → Refactor cycle
- 80%+ code coverage target

### Testing Philosophy: No Mock Abuse

**CRITICAL**: This codebase explicitly forbids the "mock everything" anti-pattern.

**Acceptable Mocking** (I/O boundaries only):
- LLM boundary:
  - For `OllamaClient` tests: mock HTTP with `respx` (httpx)
  - For agent/service tests: use `tests/fixtures/mock_llm.py` (protocol-compatible fake), not ad-hoc mocks
- File system operations for tests that shouldn't touch disk
- Time-dependent operations (sparingly)

**Forbidden Mocking**:
- Business logic (if you mock it, your design is wrong)
- Domain models (use real instances with test data)
- Internal functions (test through public API)
- Mocking to make tests pass (fix the code instead)

**Test Data vs Mocks**:
- `sample_transcript`, `sample_phq8_scores`: GOOD (real data structures)
- `sample_ollama_response`: GOOD (real API response for parsing tests)
- `Mock()` objects replacing real behavior: BAD unless at I/O boundary

See **Spec 01** (`docs/archive/specs/01_PROJECT_BOOTSTRAP.md`) for detailed examples and rationale.

## Success Criteria

### Functional Parity
- [ ] Qualitative assessment matches paper Section 2.3.1
- [ ] Judge agent feedback loop matches paper Section 2.3.1
- [ ] Few-shot prompting matches paper Section 2.4.2
- [ ] Meta-review severity prediction matches paper Section 2.3.3
- [ ] MAE metrics reproducible (0.619 few-shot vs 0.796 zero-shot)

### Production Quality
- [ ] 80%+ test coverage
- [ ] Type hints throughout (mypy strict)
- [ ] Structured JSON logging
- [ ] Health check endpoints
- [ ] Graceful error handling
- [ ] API documentation (OpenAPI)
- [ ] Docker deployment ready
- [ ] CI/CD pipeline green

### Performance
- [ ] Full pipeline < 2 minutes on M3 Pro (paper: ~1 minute)
- [ ] Embedding retrieval < 100ms
- [ ] API response streaming for long operations

## Implementation Order

The specs are ordered for **maximum early value** with **mandatory checkpoints**:

1. **Spec 01-04A**: Foundation (can't build without it)
2. **CHECKPOINT 04.5**: Verify foundation, bug hunt
3. **Spec 05-07**: Qualitative path (first testable feature)
4. **CHECKPOINT 07.5**: Verify qualitative pipeline
5. **Spec 08-09**: Quantitative path (second testable feature)
6. **CHECKPOINT 09.5**: Verify quantitative pipeline
7. **Spec 10-11**: Integration (full pipeline)
8. **CHECKPOINT 11.5**: Verify paper metrics reproduced
9. **Spec 12**: Polish (observability)
10. **CHECKPOINT 12.5**: Remove legacy code, clean codebase
11. **Spec 13 (Implemented - Partial)**: Pydantic AI `TextOutput` integration (quantitative scoring path; `PYDANTIC_AI_ENABLED` opt-in)
12. **Spec 14 (Deferred)**: Word-boundary regex + negation detection (precision refinement)
13. **Spec 15 (Implemented)**: Experiment tracking with full provenance
14. **Spec 16 (Implemented)**: Log output improvements (ANSI auto-detection)

Each spec produces a **working increment** that can be demoed and tested.
Each checkpoint produces a **quality gate** that must pass before proceeding.

**Implementation Priority**:
- Spec 13: Extend Pydantic AI beyond quantitative scoring (judge/meta-review) if desired
- Spec 14: **Deferred** (precision refinement)

## References

### Paper
- Greene et al. "AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews"

### Modern Python
- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [structlog Documentation](https://www.structlog.org/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

### Architecture
- Robert C. Martin, "Clean Architecture"
- Gang of Four, "Design Patterns"
- Martin Fowler, "Patterns of Enterprise Application Architecture"
