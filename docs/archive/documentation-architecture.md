# Documentation Architecture Proposal

**Created**: 2025-12-21
**Status**: PROPOSAL (awaiting approval)
**Audience**: Maintainers, contributors, future developers

---

## Executive Summary

This document proposes a comprehensive documentation structure for the AI Psychiatrist codebase, following the **Diátaxis framework** (the 2025 gold standard for technical documentation) while preserving existing internal documentation.

### Current State

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Specs | 20 | 10,970 | Implementation blueprints |
| Bugs | 22 | ~2,000 | Issue tracking |
| Brainstorming | 2 | ~300 | Future work |
| Models | 1 | 74 | Model registry |
| **Total internal docs** | 45 | ~13,000 | |

**Production code**: 4,120 lines
**Test code**: 1,767 lines
**Doc-to-code ratio**: 3.2:1 (excellent for internal docs, but missing user-facing layer)

### The Gap

The existing documentation is **implementation-focused** (specs for building the system). What's missing is **user-focused** documentation (how to use, understand, and operate the system).

---

## Diátaxis Framework

[Diátaxis](https://diataxis.fr/) organizes documentation into four modes based on user needs:

```
                    PRACTICAL                     THEORETICAL
                    (doing)                       (understanding)
                        │                              │
    LEARNING      ┌─────┴─────┐                ┌───────┴───────┐
    (acquiring)   │ TUTORIALS │                │  EXPLANATION  │
                  │           │                │               │
                  │ Learning- │                │Understanding- │
                  │ oriented  │                │   oriented    │
                  └───────────┘                └───────────────┘
                        │                              │
    WORKING       ┌─────┴─────┐                ┌───────┴───────┐
    (applying)    │  HOW-TO   │                │   REFERENCE   │
                  │  GUIDES   │                │               │
                  │           │                │ Information-  │
                  │  Task-    │                │   oriented    │
                  │ oriented  │                │               │
                  └───────────┘                └───────────────┘
```

### Why Diátaxis?

Used by: Django, Stripe, GitLab, Cloudflare, NumPy, Gatsby, BBC, and hundreds of production systems.

**Key insight**: Different documentation serves different purposes. A tutorial is not a reference. A how-to is not an explanation. Mixing them creates confusion.

---

## Proposed Structure

```
docs/
├── index.md                          # Landing page
│
├── getting-started/                  # TUTORIALS (learning-oriented)
│   ├── quickstart.md                 # 5-minute first run
│   ├── installation.md               # Detailed setup (uv, Ollama, models)
│   └── first-assessment.md           # Run your first transcript assessment
│
├── guides/                           # HOW-TO GUIDES (task-oriented)
│   ├── configuration.md              # How to configure the system
│   ├── running-pipeline.md           # How to run the full 4-agent pipeline
│   ├── few-shot-setup.md             # How to set up few-shot mode
│   ├── generating-embeddings.md      # How to generate reference embeddings
│   ├── adding-transcripts.md         # How to add new transcripts
│   ├── model-selection.md            # How to choose/change LLM models
│   └── troubleshooting.md            # How to debug common issues
│
├── concepts/                         # EXPLANATION (understanding-oriented)
│   ├── architecture.md               # System architecture overview
│   ├── pipeline.md                   # How the 4-agent pipeline works
│   ├── agents.md                     # Agent responsibilities and interactions
│   ├── phq8.md                       # Understanding PHQ-8 assessment
│   ├── few-shot-retrieval.md         # Embedding-based retrieval explained
│   ├── feedback-loop.md              # Judge agent refinement loop
│   └── domain-model.md               # Entities, value objects, enums
│
├── reference/                        # REFERENCE (information-oriented)
│   ├── api/                          # API documentation
│   │   ├── endpoints.md              # All endpoints with schemas
│   │   └── errors.md                 # Error codes and handling
│   ├── configuration.md              # All configuration options
│   ├── cli.md                        # CLI commands and options
│   ├── environment.md                # Environment variables reference
│   ├── models.md                     # Supported models (link to MODEL_REGISTRY.md)
│   └── glossary.md                   # Terms and definitions
│
├── data/                             # DATA DOCUMENTATION (research-specific)
│   ├── README.md                     # Data documentation overview
│   ├── daic-woz-schema.md            # DAIC-WOZ dataset schema (our discussion)
│   ├── transcript-format.md          # Transcript file format
│   ├── ground-truth.md               # PHQ-8 ground truth CSVs
│   ├── splits.md                     # Train/dev/test split methodology
│   └── embeddings.md                 # Pre-computed embeddings format
│
├── contributing/                     # DEVELOPER EXPERIENCE
│   ├── development.md                # Dev environment setup
│   ├── testing.md                    # Testing philosophy and how to test
│   ├── code-style.md                 # Linting, formatting, type hints
│   ├── architecture-decisions.md     # ADRs (Architecture Decision Records)
│   └── release.md                    # Release process
│
├── operations/                       # OPERATIONAL (production)
│   ├── deployment.md                 # How to deploy
│   ├── monitoring.md                 # Observability (when Spec 12 is done)
│   ├── performance.md                # Performance tuning
│   └── security.md                   # Security considerations
│
├── research/                         # RESEARCH-SPECIFIC
│   ├── paper-reference.md            # Paper citations and methodology
│   ├── replication.md                # How to replicate paper results
│   └── experiments.md                # Running experiments
│
├── specs/                            # [EXISTING] Implementation specs
│   └── (20 existing spec files)
│
├── bugs/                             # [EXISTING] Bug tracking
│   └── (existing bug files)
│
├── brainstorming/                    # [EXISTING] Future work
│   └── (existing brainstorming files)
│
└── models/                           # [EXISTING] Model registry
    └── MODEL_REGISTRY.md
```

---

## What Each Section Contains

### 1. Getting Started (Tutorials)

**Purpose**: Get a new user from zero to running in 15 minutes.

| File | Content | Time |
|------|---------|------|
| `quickstart.md` | Clone, install, run with sample data | 5 min |
| `installation.md` | Detailed setup: uv, Ollama, model pulls, verification | 10 min |
| `first-assessment.md` | Run a real transcript through the pipeline | 5 min |

**Key principle**: No explanation, just steps. "Do this, then this, then this."

### 2. Guides (How-To)

**Purpose**: Solve specific problems. Task-oriented.

Examples:
- "I need to change the LLM model" → `model-selection.md`
- "I need to run with my own data" → `adding-transcripts.md`
- "Something isn't working" → `troubleshooting.md`

**Key principle**: Assumes the reader already knows what they want to do.

### 3. Concepts (Explanation)

**Purpose**: Provide understanding. Answer "why" and "how does this work?"

| File | Explains |
|------|----------|
| `architecture.md` | Clean architecture layers, why this structure |
| `pipeline.md` | The 4-agent flow, what each agent does |
| `phq8.md` | What PHQ-8 is, why it matters clinically |
| `feedback-loop.md` | Why we iterate, threshold logic |
| `few-shot-retrieval.md` | Embedding similarity, reference selection |

**Key principle**: Discursive, connects ideas, explains tradeoffs.

### 4. Reference

**Purpose**: Look up specific information. Comprehensive and precise.

| File | Content |
|------|---------|
| `api/endpoints.md` | All API endpoints, request/response schemas |
| `configuration.md` | Every config option with type, default, paper reference |
| `cli.md` | Every CLI command with all flags |
| `glossary.md` | Terms: MDD, PHQ-8, DAIC-WOZ, severity levels, etc. |

**Key principle**: No narrative. Just facts. Like a dictionary.

### 5. Data Documentation

**Purpose**: Enable coding agents and developers to work with gated data without having it.

| File | Content |
|------|---------|
| `daic-woz-schema.md` | Full DAIC-WOZ schema, column types, value ranges, examples |
| `transcript-format.md` | Transcript CSV format, speaker labels, timestamp format |
| `ground-truth.md` | PHQ-8 CSV schema, item scores, binary classification |
| `splits.md` | AVEC2017 splits, paper re-splits, participant ID gaps |
| `embeddings.md` | NPZ format, JSON sidecar, dimension requirements |

**Key principle**: Everything you need to write code against the data, without the data.

### 6. Contributing

**Purpose**: Enable external contributors and maintain code quality.

| File | Content |
|------|---------|
| `development.md` | Dev setup, Makefile targets, IDE config |
| `testing.md` | Testing philosophy (no mock abuse), how to run tests |
| `code-style.md` | Ruff config, mypy strict mode, naming conventions |
| `architecture-decisions.md` | Why we chose X over Y (ADRs) |

### 7. Operations

**Purpose**: Run the system in production.

| File | Content |
|------|---------|
| `deployment.md` | Docker, HPC/SLURM, cloud deployment |
| `monitoring.md` | Structured logging, metrics, alerts |
| `performance.md` | Tuning Ollama, batch processing, caching |

### 8. Research

**Purpose**: Maintain connection to the paper and enable reproducibility.

| File | Content |
|------|---------|
| `paper-reference.md` | Section-by-section mapping to code |
| `replication.md` | Steps to reproduce paper metrics |
| `experiments.md` | How to run ablations, hyperparameter sweeps |

---

## What Gets Updated

### README.md (Complete Rewrite)

Current README is outdated (references conda, old paths). New README should be:

```markdown
# AI Psychiatrist

LLM-based Multi-Agent System for Depression Assessment from Clinical Interviews.

## Quick Start

\`\`\`bash
# Install
git clone https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist
cd ai-psychiatrist
make dev

# Pull models
ollama pull gemma3:27b
ollama pull qwen3-embedding:8b

# Run
make serve
\`\`\`

## Documentation

- [Getting Started](docs/getting-started/quickstart.md)
- [Architecture](docs/concepts/architecture.md)
- [API Reference](docs/reference/api/endpoints.md)
- [Contributing](docs/contributing/development.md)

## Paper

[AI Psychiatrist Assistant: An LLM-based Multi-Agent System for Depression Assessment](https://openreview.net/forum?id=mV0xJpO7A0)
```

### CLAUDE.md (Minor Updates)

Add link to documentation structure. Keep the agent-focused instructions.

---

## Implementation Priority

### Phase 1: Foundation (Immediate)
1. `docs/index.md` - Landing page
2. `docs/getting-started/quickstart.md` - 5-minute start
3. `docs/reference/glossary.md` - Terms and definitions
4. `docs/data/daic-woz-schema.md` - Data schema (enable async coding)
5. Update `README.md`

### Phase 2: Concepts (High Value)
1. `docs/concepts/architecture.md`
2. `docs/concepts/pipeline.md`
3. `docs/concepts/phq8.md`

### Phase 3: Reference (Comprehensive)
1. `docs/reference/configuration.md` (consolidate from .env.example + config.py)
2. `docs/reference/api/endpoints.md`
3. `docs/reference/cli.md`

### Phase 4: Guides (As Needed)
1. Add how-to guides based on common questions
2. Add troubleshooting as issues arise

### Phase 5: Operations (Pre-Production)
1. `docs/operations/deployment.md`
2. `docs/operations/monitoring.md`

---

## Tooling Recommendation

### MkDocs with Material Theme

Already in `pyproject.toml` as optional dependency:
```toml
docs = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.27.0",
]
```

**Benefits**:
- Beautiful default theme (Material)
- Automatic API docs from docstrings (mkdocstrings)
- Search, dark mode, mobile-friendly
- Deploy to GitHub Pages with one command
- Diátaxis-friendly navigation structure

**Setup**:
```bash
uv pip install -e ".[docs]"
mkdocs new .  # Creates mkdocs.yml
mkdocs serve  # Local preview at :8000
mkdocs gh-deploy  # Deploy to GitHub Pages
```

---

## Migration Strategy

### Keep Existing Structure

All existing docs stay where they are:
- `docs/specs/` - Implementation blueprints (internal)
- `docs/bugs/` - Issue tracking (internal)
- `docs/brainstorming/` - Future work (internal)
- `docs/models/` - Model registry (becomes part of reference)

### Add User-Facing Layer

New directories are additive, not replacements:
- `docs/getting-started/` - NEW
- `docs/guides/` - NEW
- `docs/concepts/` - NEW
- `docs/reference/` - NEW
- `docs/data/` - NEW
- `docs/contributing/` - NEW
- `docs/operations/` - NEW
- `docs/research/` - NEW

### Cross-Reference

New docs link to specs where appropriate:
- `architecture.md` → links to `specs/00_OVERVIEW.md` for implementation details
- `pipeline.md` → links to individual agent specs
- `data/` → links to `specs/04A_DATA_ORGANIZATION.md`

---

## Success Criteria

A developer should be able to:

1. **Start** (< 15 min): Clone, install, run first assessment
2. **Understand** (< 30 min): Know what the system does and how it works
3. **Use** (< 5 min per task): Find how to do any specific task
4. **Reference** (< 1 min): Look up any configuration option, API endpoint, or term
5. **Contribute** (< 1 hour): Set up dev environment, run tests, make a PR
6. **Operate** (documented): Deploy, monitor, troubleshoot

A coding agent should be able to:

1. **Understand data** (without access): Work with data schemas
2. **Navigate codebase**: Find where things are implemented
3. **Follow conventions**: Know code style, testing patterns

---

## Next Steps

1. **Approve** this proposal (or suggest changes)
2. **Create** `docs/index.md` and basic structure
3. **Write** Phase 1 docs (quickstart, glossary, data schema)
4. **Update** README.md
5. **Configure** MkDocs for local preview
6. **Iterate** on Phase 2-5 as development continues

---

## References

- [Diátaxis Framework](https://diataxis.fr/)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Write the Docs](https://www.writethedocs.org/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)
