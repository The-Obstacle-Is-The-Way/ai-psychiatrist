# AI-Psychiatrist: Production-Ready Conversion Spec Overview

## Executive Summary

This document series outlines a comprehensive plan to transform the AI-Psychiatrist research codebase into a production-ready, maintainable Python system. The specs follow **vertical slice architecture**, enabling incremental delivery of fully-functional features while maintaining 100% parity with the research paper.

## Paper Reference

**Title:** AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews

**Key Contributions:**
1. LLM-based multi-agent system with four collaborative agents
2. Embedding-based few-shot prompting for PHQ-8 prediction
3. Iterative self-refinement via judge agent feedback loop
4. Meta-review agent for severity integration

## Current State Analysis

### What Exists (Research Code)
- 13 Python modules (~3,860 LOC)
- FastAPI server with single endpoint
- Four agent implementations (qual/quant assessors, judge, meta-reviewer)
- Embedding-based few-shot retrieval system
- Iterative feedback loop for quality improvement
- SLURM job scripts for HPC execution
- Conda environment configuration

### Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No tests | Cannot verify correctness | P0 |
| No error handling | Production crashes | P0 |
| No logging/observability | Cannot debug issues | P0 |
| Hardcoded paths/configs | Cannot deploy | P0 |
| No type safety | Runtime errors | P1 |
| No dependency injection | Untestable | P1 |
| No API documentation | Unusable by others | P1 |
| Naming inconsistencies | Confusing codebase | P2 |
| No CI/CD pipeline | Manual deployment | P2 |

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

| Spec | Title | Deliverable |
|------|-------|-------------|
| **01** | Project Bootstrap | pyproject.toml, uv, Makefile, CI/CD |
| **02** | Core Domain | PHQ8, Transcript, Assessment entities |
| **03** | Configuration & Logging | Pydantic settings, structlog |
| **04** | LLM Infrastructure | Ollama client abstraction |
| **05** | Transcript Loader | Interview data ingestion |
| **06** | Qualitative Agent | PHQ-8 symptom analysis |
| **07** | Judge Agent | Self-refinement feedback loop |
| **08** | Embedding Service | Vector similarity search |
| **09** | Quantitative Agent | Few-shot PHQ-8 scoring |
| **10** | Meta-Review Agent | Severity integration |
| **11** | Full Pipeline API | Complete assessment endpoint |
| **12** | Observability | Metrics, tracing, health checks |

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

The specs are ordered for **maximum early value**:

1. **Spec 01-03**: Foundation (can't build without it)
2. **Spec 04-05**: Infrastructure (LLM + data)
3. **Spec 06-07**: Qualitative path (first testable feature)
4. **Spec 08-09**: Quantitative path (second testable feature)
5. **Spec 10-11**: Integration (full pipeline)
6. **Spec 12**: Polish (observability)

Each spec produces a **working increment** that can be demoed and tested.

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
