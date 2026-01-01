# AI Psychiatrist Documentation

**LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews**

---

## What is AI Psychiatrist?

AI Psychiatrist is an engineering-focused, reproducible implementation of a research paper that uses large language models (LLMs) in a multi-agent architecture to assess depression severity from clinical interview transcripts. The system analyzes interview transcripts and predicts PHQ-8 depression scores using a four-agent pipeline.

> **Clinical disclaimer**: This repository is intended for paper reproduction and experimentation. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.

### Key Features

- **Four-Agent Pipeline**: Qualitative, Judge, Quantitative, and Meta-Review agents collaborate for comprehensive assessment
- **Embedding-Based Few-Shot Retrieval**: Optional few-shot references; retrieval quality is controlled by guardrails, item-tag filtering, chunk-level score attachment, and CRAG validation (see results docs)
- **Iterative Self-Refinement**: Judge agent feedback loop improves assessment quality
- **Selective Prediction Evaluation**: AURC/AUGRC + bootstrap confidence intervals (coverage-aware evaluation)
- **Engineering-Focused Architecture**: Clean architecture, type safety, structured logging, and comprehensive testing

### Paper Reference

> Greene et al. "AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews"
> [OpenReview](https://openreview.net/forum?id=mV0xJpO7A0)

---

## Quick Navigation

### Getting Started

| Document | Description |
|----------|-------------|
| [Quickstart](getting-started/quickstart.md) | Get running in 5 minutes |
| [Zero-Shot Preflight](preflight-checklist/preflight-checklist-zero-shot.md) | Pre-run verification for zero-shot reproduction |
| [Few-Shot Preflight](preflight-checklist/preflight-checklist-few-shot.md) | Pre-run verification for few-shot reproduction |

### Architecture

| Document | Description |
|----------|-------------|
| [Architecture](architecture/architecture.md) | System layers and design patterns |
| [Pipeline](architecture/pipeline.md) | How the 4-agent pipeline works |
| [Future Architecture](architecture/future-architecture.md) | LangGraph integration roadmap |

### Clinical Domain

| Document | Description |
|----------|-------------|
| [PHQ-8](clinical/phq8.md) | Understanding PHQ-8 depression assessment |
| [Clinical Understanding](clinical/clinical-understanding.md) | How the system works clinically |
| [Glossary](clinical/glossary.md) | Terms and definitions |

### Configuration

| Document | Description |
|----------|-------------|
| [Configuration Reference](configs/configuration.md) | All configuration options |
| [Configuration Philosophy](configs/configuration-philosophy.md) | Why defaults are what they are |
| [Agent Sampling Registry](configs/agent-sampling-registry.md) | Sampling parameters per agent |

### Models

| Document | Description |
|----------|-------------|
| [Model Registry](models/model-registry.md) | Supported models and backends |
| [Model Wiring](models/model-wiring.md) | How agents connect to models |

### Embeddings & Few-Shot

| Document | Description |
|----------|-------------|
| [Embeddings Explained](embeddings/embeddings-explained.md) | Core embedding concepts |
| [Embedding Generation](embeddings/embedding-generation.md) | Fail-fast embedding artifacts |
| [Few-Shot Design Considerations](embeddings/few-shot-design-considerations.md) | Design rationale and tradeoffs |
| [Few-Shot Prompt Format](embeddings/few-shot-prompt-format.md) | How reference examples are formatted |
| [Chunk Scoring](embeddings/chunk-scoring.md) | Chunk-level PHQ-8 scoring (Spec 35) |
| [Item Tagging Setup](embeddings/item-tagging-setup.md) | Item-tag filtering (Spec 34) |
| [Retrieval Debugging](embeddings/debugging-retrieval-quality.md) | Interpret retrieval logs |
| [Batch Query Embedding](embeddings/batch-query-embedding.md) | Query-embedding timeout fix (Spec 37) |

### Data

| Document | Description |
|----------|-------------|
| [DAIC-WOZ Schema](data/daic-woz-schema.md) | Dataset schema for development without data access |
| [DAIC-WOZ Preprocessing](data/daic-woz-preprocessing.md) | Transcript cleaning + participant-only variants |
| [Data Splits Overview](data/data-splits-overview.md) | AVEC2017 vs paper splits explained |
| [Paper Split Registry](data/paper-split-registry.md) | Exact participant IDs for paper splits |
| [Artifact Namespace Registry](data/artifact-namespace-registry.md) | Embedding artifact naming conventions |
| [Patch Missing PHQ-8 Values](data/patch-missing-phq8-values.md) | Deterministic ground-truth repair |

### Pipeline Internals

| Document | Description |
|----------|-------------|
| [Feature Reference](pipeline-internals/features.md) | Implemented features + defaults |
| [Evidence Extraction](pipeline-internals/evidence-extraction.md) | How quotes are extracted from transcripts |

### Statistics & Evaluation

| Document | Description |
|----------|-------------|
| [Metrics and Evaluation](statistics/metrics-and-evaluation.md) | Exact metric definitions |
| [Coverage Explained](statistics/coverage.md) | What coverage means and why it matters |
| [AURC/AUGRC Methodology](statistics/statistical-methodology-aurc-augrc.md) | Selective prediction metrics |
| [CRAG Validation Guide](statistics/crag-validation-guide.md) | Reference validation (Spec 36) |

### Results & Reproduction

| Document | Description |
|----------|-------------|
| [Run History](results/run-history.md) | Canonical history of reproduction runs |
| [Reproduction Results](results/reproduction-results.md) | Current reproduction status |
| [Run Output Schema](results/run-output-schema.md) | Output JSON format |

### Developer Reference

| Document | Description |
|----------|-------------|
| [API Endpoints](developer/api-endpoints.md) | REST API reference |
| [Testing](developer/testing.md) | Markers, fixtures, and test-doubles policy |
| [Error Handling](developer/error-handling.md) | Exception handling patterns |
| [Exceptions](developer/exceptions.md) | Exception class hierarchy |
| [Dependency Registry](developer/dependency-registry.md) | Third-party dependencies |

### Archive

| Document | Description |
|----------|-------------|
| [Spec 20: Keyword Fallback](_archive/specs/20-keyword-fallback-improvements.md) | Deferred — intentionally not implementing |

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI PSYCHIATRIST PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌─────────────────────────────────────────────┐   │
│   │  TRANSCRIPT  │───►│              QUALITATIVE AGENT              │   │
│   │   (Input)    │    │  Analyzes social, biological, risk factors  │   │
│   └──────────────┘    └──────────────────────┬──────────────────────┘   │
│                                              │                          │
│                                              ▼                          │
│                       ┌─────────────────────────────────────────────┐   │
│                       │                JUDGE AGENT                  │   │
│                       │  Evaluates coherence, completeness,         │   │
│           ┌──────────►│  specificity, accuracy (1-5 scale)          │   │
│           │           └──────────────────────┬──────────────────────┘   │
│           │                                  │                          │
│           │           ┌──────────────────────▼──────────────────────┐   │
│           │           │            FEEDBACK LOOP SERVICE            │   │
│           └───────────┤  If score < 4: refine and re-evaluate       │   │
│                       │  Max 10 iterations per paper                │   │
│                       └──────────────────────┬──────────────────────┘   │
│                                              │                          │
│   ┌──────────────┐    ┌──────────────────────▼──────────────────────┐   │
│   │  EMBEDDINGS  │───►│            QUANTITATIVE AGENT               │   │
│   │ (Few-Shot)   │    │  Predicts PHQ-8 item scores (0-3 each)      │   │
│   └──────────────┘    └──────────────────────┬──────────────────────┘   │
│                                              │                          │
│                                              ▼                          │
│                       ┌─────────────────────────────────────────────┐   │
│                       │             META-REVIEW AGENT               │   │
│                       │  Integrates all assessments                 │   │
│                       │  Outputs final severity (0-4)               │   │
│                       └──────────────────────┬──────────────────────┘   │
│                                              │                          │
│                                              ▼                          │
│                       ┌─────────────────────────────────────────────┐   │
│                       │              FINAL ASSESSMENT               │   │
│                       │  Severity: MINIMAL|MILD|MODERATE|           │   │
│                       │            MOD_SEVERE|SEVERE                │   │
│                       └─────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Category | Tool | Purpose |
|----------|------|---------|
| **Package Management** | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management |
| **LLM Backend** | [Ollama](https://ollama.ai/) / HuggingFace (optional) | Local inference via Ollama; optional Transformers backend for official weights |
| **Framework** | FastAPI | REST API server |
| **Validation** | Pydantic v2 | Configuration and data validation |
| **Logging** | structlog | Structured JSON logging |
| **Testing** | pytest | Unit, integration, and E2E tests |
| **Linting** | Ruff | Fast Python linting and formatting |
| **Types** | mypy | Static type checking (strict mode) |

---

## Project Status

This codebase is an **engineering-focused refactor** of the original research implementation. Key improvements:

- Full test coverage (80%+ target)
- Type hints throughout (mypy strict mode)
- Clean architecture with dependency injection
- Structured logging for observability
- Comprehensive configuration management
- Local-first deployment (Ollama + FastAPI); containerization TBD

---

## Contributing

See `CLAUDE.md` in the repository root for development guidelines and commands.

```bash
# Quick development setup
make dev          # Install dependencies + pre-commit hooks
make test         # Run all tests with coverage
make ci           # Full CI pipeline (format, lint, typecheck, test)
```

---

## License

Licensed under Apache 2.0. See `LICENSE` and `NOTICE` in the repository root for details and attribution.

This project is a clean-room reimplementation based on research from Georgia State University. See the [paper](https://openreview.net/forum?id=mV0xJpO7A0) for academic citation.
