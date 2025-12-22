# AI Psychiatrist

**LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

---

## Overview

AI Psychiatrist implements a research paper's methodology for automated depression assessment using a four-agent LLM pipeline. The system analyzes clinical interview transcripts to predict PHQ-8 depression scores and severity levels.

### Key Features

- **Four-Agent Pipeline**: Qualitative, Judge, Quantitative, and Meta-Review agents collaborate for comprehensive assessment
- **Embedding-Based Few-Shot Learning**: 22% improvement in accuracy over zero-shot approaches
- **Iterative Self-Refinement**: Judge agent feedback loop improves assessment quality
- **Production-Ready**: Clean architecture, type safety, structured logging, 80%+ test coverage

### Paper Reference

> Greene et al. "AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews"
> [OpenReview](https://openreview.net/forum?id=mV0xJpO7A0)

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- 16GB+ RAM (for 27B models)

### Installation

```bash
# Clone repository
git clone https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist.git
cd ai-psychiatrist

# Install dependencies (uses uv)
make dev

# Pull required models
ollama pull gemma3:27b
ollama pull alibayram/medgemma:27b
ollama pull qwen3-embedding:8b

# Configure (uses paper-optimal defaults)
cp .env.example .env

# Start server
make serve
```

### Run Your First Assessment

```bash
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_text": "Ellie: How are you doing today?\nParticipant: I have been feeling really down lately."
  }'
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [**Quickstart**](docs/getting-started/quickstart.md) | Get running in 5 minutes |
| [**Architecture**](docs/concepts/architecture.md) | System design and layers |
| [**Pipeline**](docs/concepts/pipeline.md) | How the 4-agent pipeline works |
| [**PHQ-8**](docs/concepts/phq8.md) | Understanding depression assessment |
| [**Configuration**](docs/reference/configuration.md) | All configuration options |
| [**API Reference**](docs/reference/api/endpoints.md) | REST API documentation |
| [**Glossary**](docs/reference/glossary.md) | Terms and definitions |

### For Developers

| Document | Description |
|----------|-------------|
| [**CLAUDE.md**](CLAUDE.md) | Development guidelines |
| [**Specs**](docs/specs/00-overview.md) | Implementation specifications |
| [**Data Schema**](docs/data/daic-woz-schema.md) | Dataset format documentation |

---

## Project Structure

```text
ai-psychiatrist/
├── src/ai_psychiatrist/
│   ├── agents/           # Four assessment agents
│   ├── domain/           # Entities, enums, value objects
│   ├── services/         # Business logic (feedback loop, embeddings)
│   ├── infrastructure/   # Ollama client, logging
│   └── config.py         # Pydantic settings
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── docs/
│   ├── getting-started/  # Tutorials
│   ├── concepts/         # Explanations
│   ├── reference/        # API and config reference
│   ├── data/             # Dataset documentation
│   └── specs/            # Implementation specs
└── data/                 # DAIC-WOZ dataset (gitignored)
```

---

## Development

```bash
# Full CI pipeline
make ci

# Individual commands
make test           # Run all tests with coverage
make test-unit      # Fast unit tests only
make lint-fix       # Auto-fix linting issues
make typecheck      # mypy strict mode
make format         # Format code with ruff

# Development server with hot reload
make serve
```

### Testing with Real Ollama

```bash
# Enable Ollama integration tests
AI_PSYCHIATRIST_OLLAMA_TESTS=1 make test-e2e
```

---

## Configuration

All settings via environment variables or `.env` file:

```bash
# Models (paper-optimal defaults)
MODEL_QUALITATIVE_MODEL=gemma3:27b
MODEL_QUANTITATIVE_MODEL=alibayram/medgemma:27b    # 18% better MAE
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b

# Few-shot retrieval (Appendix D optimal)
EMBEDDING_DIMENSION=4096
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_TOP_K_REFERENCES=2

# Feedback loop (Section 2.3.1)
FEEDBACK_MAX_ITERATIONS=10
FEEDBACK_SCORE_THRESHOLD=3
```

See [Configuration Reference](docs/reference/configuration.md) for all options.

---

## Paper Results

| Metric | Zero-Shot | Few-Shot | MedGemma Few-Shot |
|--------|-----------|----------|-------------------|
| PHQ-8 MAE | 0.796 | 0.619 | **0.505** |
| Severity Accuracy | - | - | **78%** |

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Package management |
| [Ollama](https://ollama.ai/) | Local LLM inference |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API |
| [Pydantic v2](https://docs.pydantic.dev/) | Configuration & validation |
| [structlog](https://www.structlog.org/) | Structured logging |
| [pytest](https://docs.pytest.org/) | Testing |
| [Ruff](https://docs.astral.sh/ruff/) | Linting & formatting |
| [mypy](https://mypy-lang.org/) | Type checking |

---

## License

This project implements research from Georgia State University. See the [paper](https://openreview.net/forum?id=mV0xJpO7A0) for academic citation.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following [CLAUDE.md](CLAUDE.md) guidelines
4. Run `make ci` to verify
5. Submit a pull request
