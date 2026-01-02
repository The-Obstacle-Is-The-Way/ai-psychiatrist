# AI Psychiatrist

**LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

---

## Overview

AI Psychiatrist implements a research paper's methodology for automated depression assessment using a four-agent LLM pipeline. The system analyzes clinical interview transcripts to predict PHQ-8 depression scores and severity levels.

### Key Features

- **Four-Agent Pipeline**: Qualitative, Judge, Quantitative, and Meta-Review agents collaborate for comprehensive assessment
- **Embedding-Based Few-Shot Learning**: Paper reports 22% lower item-level MAE vs zero-shot (0.796 → 0.619, Section 3.2); this repo tracks coverage-adjusted metrics (AURC/AUGRC/Cmax) in run artifacts
- **Iterative Self-Refinement**: Judge agent feedback loop improves assessment quality
- **Engineering-Focused**: Clean architecture, strict type checking, structured logging, 80%+ test coverage

### Paper Reference

> Greene et al. "AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews"
> [OpenReview](https://openreview.net/forum?id=mV0xJpO7A0)

> **Clinical disclaimer**: This repository is a research/engineering implementation intended for paper reproduction and experimentation.
> It is not a medical device and should not be used for clinical diagnosis or treatment decisions.

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
make dev  # or: make dev-hf (to enable HuggingFace backends)

# Pull required models
ollama pull gemma3:27b-it-qat  # or gemma3:27b
ollama pull qwen3-embedding:8b

# Configure (uses paper-optimal defaults)
cp .env.example .env

# Start server
make serve
```

> **Note (Embeddings backend)**: Chat and embeddings can use different backends:
> - `LLM_BACKEND` controls chat for agents (default: `ollama`)
> - `EMBEDDING_BACKEND` controls embeddings (default: `huggingface`)
>
> If you installed `make dev` (no HF deps), set `EMBEDDING_BACKEND=ollama` in `.env` for a pure-Ollama setup.

> **Optional (Appendix F)**: The paper evaluates MedGemma 27B as an alternative model for the
> quantitative agent. There is no official MedGemma model in the Ollama library; use the HuggingFace
> backend (`make dev-hf`, `LLM_BACKEND=huggingface`, `MODEL_QUANTITATIVE_MODEL=medgemma:27b`) to load
> the official gated weights.

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
| [**Architecture**](docs/architecture/architecture.md) | System design and layers |
| [**Pipeline**](docs/architecture/pipeline.md) | How the 4-agent pipeline works |
| [**PHQ-8**](docs/clinical/phq8.md) | Understanding depression assessment |
| [**Configuration**](docs/configs/configuration.md) | All configuration options |
| [**API Reference**](docs/developer/api-endpoints.md) | REST API documentation |
| [**Glossary**](docs/clinical/glossary.md) | Terms and definitions |
| [**Reproduction Results**](docs/results/reproduction-results.md) | Current-state reproduction summary |
| [**Run History**](docs/results/run-history.md) | Canonical timeline + per-run statistics |

### For Developers

| Document | Description |
|----------|-------------|
| [**CLAUDE.md**](CLAUDE.md) | Development guidelines |
| [**Specs**](docs/_specs/index.md) | Specs index (implemented specs are distilled into canonical docs) |
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
│   ├── specs/            # Active specs
│   └── archive/          # Historical artifacts (not required for active docs)
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
# Models (recommended defaults; see `.env.example`)
MODEL_QUALITATIVE_MODEL=gemma3:27b-it-qat  # or gemma3:27b
MODEL_QUANTITATIVE_MODEL=gemma3:27b-it-qat  # or gemma3:27b
MODEL_EMBEDDING_MODEL=qwen3-embedding:8b

# Backends (chat vs embeddings)
LLM_BACKEND=ollama
EMBEDDING_BACKEND=huggingface

# Few-shot retrieval (Appendix D optimal)
EMBEDDING_DIMENSION=4096
EMBEDDING_CHUNK_SIZE=8
EMBEDDING_TOP_K_REFERENCES=2

# Reference embeddings selection (NPZ + JSON sidecar)
# Default: FP16 HuggingFace embeddings (paper-train)
EMBEDDING_EMBEDDINGS_FILE=huggingface_qwen3_8b_paper_train_participant_only
# Transcript source must match how embeddings were built
DATA_TRANSCRIPTS_DIR=data/transcripts_participant_only
# Alternative: legacy Ollama embeddings (paper-train)
# EMBEDDING_EMBEDDINGS_FILE=paper_reference_embeddings
# DATA_EMBEDDINGS_PATH=/absolute/or/relative/path/to/artifact.npz  # full-path override

# Chunk scoring (Spec 35; requires {name}.chunk_scores.json sidecar)
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk

# Feedback loop (Section 2.3.1)
FEEDBACK_MAX_ITERATIONS=10
FEEDBACK_SCORE_THRESHOLD=3

# Appendix F (optional): use official MedGemma via HuggingFace backend
# LLM_BACKEND=huggingface
# MODEL_QUANTITATIVE_MODEL=medgemma:27b
```

See [Configuration Reference](docs/configs/configuration.md) for all options.

---

## Paper Results (Reported)

From the paper:
- **Quantitative (PHQ-8 item scoring, 0–3 per item)**: MAE 0.796 (zero-shot) vs 0.619 (few-shot)
- **Appendix F (optional)**: MedGemma 27B MAE 0.505, with lower coverage (“fewer predictions overall”)
- **Meta-review (binary classification)**: 78% accuracy (comparable to the human expert)

Note: The MAE values are **item-level** (per PHQ-8 item) and exclude items marked “N/A”.

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

Licensed under **Apache 2.0**. See `LICENSE` and `NOTICE`.

This repository is a clean-room, production-grade reimplementation of the paper’s method. It does
not distribute the DAIC-WOZ dataset. The original research code is referenced in the paper under
“Data and Code Availability”.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following [CLAUDE.md](CLAUDE.md) guidelines
4. Run `make ci` to verify
5. Submit a pull request
