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

| Document | Description | Time |
|----------|-------------|------|
| [Quickstart](getting-started/quickstart.md) | Get running in 5 minutes | 5 min |

### Understanding the System

| Document | Description |
|----------|-------------|
| [Architecture](architecture/architecture.md) | System layers and design patterns |
| [Pipeline](architecture/pipeline.md) | How the 4-agent pipeline works |
| [PHQ-8](concepts/phq8.md) | Understanding PHQ-8 depression assessment |

### Reference

| Document | Description |
|----------|-------------|
| [Configuration](configs/configuration.md) | All configuration options |
| [Feature Reference](reference/features.md) | Implemented features + defaults (non-archive canonical) |
| [Run Output Schema](reference/run-output-schema.md) | Output JSON + experiment registry format |
| [API Endpoints](reference/api-endpoints.md) | REST API reference |
| [Testing](reference/testing.md) | Markers, fixtures, and test-doubles policy |
| [Glossary](reference/glossary.md) | Terms and definitions |

### Data Documentation

| Document | Description |
|----------|-------------|
| [DAIC-WOZ Schema](data/daic-woz-schema.md) | Dataset schema for development without data access |

### Architecture Evolution

| Document | Description |
|----------|-------------|
| [Future Architecture](architecture/future-architecture.md) | LangGraph integration roadmap (Pydantic AI is already integrated) |
| [Spec 20: Keyword Fallback Improvements](_archive/specs/20-keyword-fallback-improvements.md) | Deferred — intentionally not implementing (see spec) |

### Debugging / Reproduction

| Document | Description |
|----------|-------------|
| [Run History](results/run-history.md) | Canonical history of reproduction runs |
| [Reproduction Results](results/reproduction-results.md) | Current reproduction status + known issues |
| [Retrieval Debugging](guides/debugging-retrieval-quality.md) | How to interpret retrieval logs and diagnose few-shot |
| [Batch Query Embedding](guides/batch-query-embedding.md) | Query-embedding timeout fix (Spec 37) |
| [Embedding Generation](guides/embedding-generation.md) | Fail-fast embedding artifacts + debug partial mode |
| [Patch Missing PHQ-8 Values](guides/patch-missing-phq8-values.md) | Deterministic ground-truth repair for missing item cells |

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
