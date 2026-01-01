# Architecture

This document explains the software architecture of AI Psychiatrist, including layer organization, design patterns, and key abstractions.

---

## Overview

AI Psychiatrist follows **Clean Architecture** principles with a **vertical slice** implementation approach. The codebase is organized into distinct layers with explicit dependency rules.

```text
┌─────────────────────────────────────────────────────────────────┐
│                          API Layer                              │
│                    (FastAPI routes, CLI)                        │
├─────────────────────────────────────────────────────────────────┤
│                         Agents Layer                            │
│      (Qualitative, Judge, Quantitative, Meta-Review)            │
├─────────────────────────────────────────────────────────────────┤
│                        Services Layer                           │
│   (FeedbackLoop, Embedding, Transcript, GroundTruth, Chunking)  │
├─────────────────────────────────────────────────────────────────┤
│                         Domain Layer                            │
│            (Entities, Value Objects, Enums, Exceptions)         │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                       │
│        (OllamaClient, HuggingFaceClient, Logging, Protocols)    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Rule**: Dependencies only point inward. Domain knows nothing about infrastructure.

---

## Directory Structure

```text
src/ai_psychiatrist/
├── __init__.py
├── config.py              # Pydantic settings (all configuration)
├── cli.py                 # Command-line interface
│
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── qualitative.py     # QualitativeAssessmentAgent
│   ├── judge.py           # JudgeAgent
│   ├── quantitative.py    # QuantitativeAssessmentAgent
│   ├── meta_review.py     # MetaReviewAgent
│   └── prompts/           # LLM prompt templates
│       ├── qualitative.py
│       ├── judge.py
│       ├── quantitative.py
│       └── meta_review.py
│
├── domain/                # Core business logic
│   ├── __init__.py
│   ├── entities.py        # Mutable domain objects with identity
│   ├── value_objects.py   # Immutable domain objects
│   ├── enums.py           # PHQ8Item, SeverityLevel, etc.
│   └── exceptions.py      # Domain-specific exceptions
│
├── services/              # Application services
│   ├── __init__.py
│   ├── feedback_loop.py   # Iterative refinement orchestration
│   ├── embedding.py       # Embedding generation and similarity
│   ├── reference_store.py # Pre-computed reference management
│   ├── transcript.py      # Transcript loading
│   ├── ground_truth.py    # PHQ-8 ground truth loading
│   └── chunking.py        # Transcript chunking for embeddings
│
├── infrastructure/        # External integrations
│   ├── __init__.py
│   ├── logging.py         # Structured logging setup
│   └── llm/
│       ├── __init__.py
│       ├── factory.py      # Backend factory (Ollama vs HuggingFace)
│       ├── huggingface.py  # HuggingFaceClient implementation (optional)
│       ├── model_aliases.py # Canonical → backend-specific model IDs
│       ├── protocols.py   # ChatClient, EmbeddingClient protocols
│       ├── ollama.py      # OllamaClient implementation
│       └── responses.py   # Response parsing utilities
│
└── api/                   # HTTP API module
    └── __init__.py
```

---

## Layer Details

### Domain Layer (`src/ai_psychiatrist/domain/`)

The innermost layer containing pure business logic with no external dependencies.

#### Entities (`src/ai_psychiatrist/domain/entities.py`)

Mutable objects with identity (UUID). Represent core business concepts.

| Entity | Purpose | Key Properties |
|--------|---------|----------------|
| `Transcript` | Interview transcript | `participant_id`, `text`, `word_count` |
| `PHQ8Assessment` | Quantitative assessment | `items`, `total_score`, `severity` |
| `QualitativeAssessment` | Narrative assessment | `overall`, `phq8_symptoms`, `social_factors` |
| `QualitativeEvaluation` | Judge scores | `scores`, `average_score`, `needs_improvement` |
| `MetaReview` | Final integration | `severity`, `explanation`, `is_mdd` |
| `FullAssessment` | Complete result | Combines all assessment types |

```python
@dataclass
class Transcript:
    participant_id: int
    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)
```

#### Value Objects (`src/ai_psychiatrist/domain/value_objects.py`)

Immutable objects without identity. Equal if all attributes equal.

| Value Object | Purpose |
|--------------|---------|
| `TranscriptChunk` | Segment of transcript for embedding |
| `EmbeddedChunk` | Chunk with its embedding vector |
| `ItemAssessment` | Single PHQ-8 item result |
| `EvaluationScore` | Single metric score from Judge |
| `SimilarityMatch` | Reference match from embedding search |

```python
@dataclass(frozen=True, slots=True)
class ItemAssessment:
    item: PHQ8Item
    evidence: str
    reason: str
    score: int | None  # None = N/A
```

#### Enums (`src/ai_psychiatrist/domain/enums.py`)

Type-safe constants for domain concepts.

| Enum | Values |
|------|--------|
| `PHQ8Item` | `NO_INTEREST ("NoInterest")`, `DEPRESSED ("Depressed")`, `SLEEP ("Sleep")`, `TIRED ("Tired")`, `APPETITE ("Appetite")`, `FAILURE ("Failure")`, `CONCENTRATING ("Concentrating")`, `MOVING ("Moving")` |
| `PHQ8Score` | `NOT_AT_ALL (0)`, `SEVERAL_DAYS (1)`, `MORE_THAN_HALF (2)`, `NEARLY_EVERY_DAY (3)` |
| `SeverityLevel` | `MINIMAL`, `MILD`, `MODERATE`, `MOD_SEVERE`, `SEVERE` |
| `EvaluationMetric` | `COHERENCE`, `COMPLETENESS`, `SPECIFICITY`, `ACCURACY` |
| `AssessmentMode` | `ZERO_SHOT`, `FEW_SHOT` |

---

### Agents Layer (`src/ai_psychiatrist/agents/`)

Each agent encapsulates a specific LLM interaction pattern.

#### QualitativeAssessmentAgent (`src/ai_psychiatrist/agents/qualitative.py`)

Generates narrative assessments from transcripts.

**Responsibilities:**
- Parse transcripts for clinical insights
- Extract PHQ-8 symptoms, social factors, biological factors, risk factors
- Support refinement based on Judge feedback

**Interface:**
```python
async def assess(transcript: Transcript) -> QualitativeAssessment
async def refine(assessment: QualitativeAssessment, feedback: dict, transcript: Transcript) -> QualitativeAssessment
```

#### JudgeAgent (`src/ai_psychiatrist/agents/judge.py`)

Evaluates qualitative assessment quality using LLM-as-judge pattern.

**Responsibilities:**
- Score assessments on 4 metrics (1-5 Likert scale)
- Extract feedback for low-scoring metrics
- Use deterministic temperature (0.0) for consistency

**Interface:**
```python
async def evaluate(assessment: QualitativeAssessment, transcript: Transcript, iteration: int) -> QualitativeEvaluation
def get_feedback_for_low_scores(evaluation: QualitativeEvaluation, threshold: int) -> dict[str, str]
```

#### QuantitativeAssessmentAgent (`src/ai_psychiatrist/agents/quantitative.py`)

Predicts PHQ-8 item scores (0-3) using evidence extraction and few-shot retrieval.

**Responsibilities:**
- Extract evidence quotes for each PHQ-8 item
- Build reference bundle from embeddings (few-shot mode)
- Score with multi-level JSON repair for robustness

**Interface:**
```python
async def assess(transcript: Transcript) -> PHQ8Assessment
```

#### MetaReviewAgent (`src/ai_psychiatrist/agents/meta_review.py`)

Integrates qualitative and quantitative assessments into final severity.

**Responsibilities:**
- Combine all assessment outputs
- Predict final severity level (0-4)
- Generate explanation for the determination

**Interface:**
```python
async def review(transcript: Transcript, qualitative: QualitativeAssessment, quantitative: PHQ8Assessment) -> MetaReview
```

---

### Services Layer (`src/ai_psychiatrist/services/`)

Application-level orchestration and external data management.

#### FeedbackLoopService (`src/ai_psychiatrist/services/feedback_loop.py`)

Orchestrates iterative refinement between Qualitative and Judge agents.

**Algorithm (Paper Section 2.3.1):**
1. Generate initial qualitative assessment
2. Evaluate with Judge agent
3. If any metric score ≤ threshold: extract feedback, refine, re-evaluate
4. Repeat until all scores acceptable OR max iterations reached

**Configuration:**
- `max_iterations`: 10 (paper default)
- `score_threshold`: 3 (scores ≤ 3 trigger refinement)

#### EmbeddingService (`src/ai_psychiatrist/services/embedding.py`)

Manages embedding generation and similarity search for few-shot retrieval.

**Features:**
- Generates embeddings via the configured embedding backend (`EMBEDDING_BACKEND`), which may differ from `LLM_BACKEND`
- Computes cosine similarity with reference store
- Builds reference bundles per PHQ-8 item
- Handles dimension validation

#### TranscriptService (`src/ai_psychiatrist/services/transcript.py`)

Loads and parses DAIC-WOZ format transcripts.

**Format:** Tab-separated CSV with columns: `start_time`, `stop_time`, `speaker`, `value`

#### ReferenceStore (`src/ai_psychiatrist/services/reference_store.py`)

Manages pre-computed reference embeddings (NPZ format with JSON sidecar + optional `.tags.json` item tags sidecar).

---

### Infrastructure Layer (`src/ai_psychiatrist/infrastructure/`)

External system integrations.

#### OllamaClient (`src/ai_psychiatrist/infrastructure/llm/ollama.py`)

HTTP client for Ollama LLM API.

**Implements:**
- `ChatClient` / `EmbeddingClient` / `LLMClient` protocols in `src/ai_psychiatrist/infrastructure/llm/protocols.py`
- Convenience helpers `simple_chat()` / `simple_embed()` used by agents/services

**Features:**
- Configurable timeout (Ollama default: 600s via `OLLAMA_TIMEOUT_SECONDS`; kept aligned with `PYDANTIC_AI_TIMEOUT_SECONDS` by default)
- Model-specific temperature and sampling parameters
- L2 normalization for embeddings

#### Protocols (`src/ai_psychiatrist/infrastructure/llm/protocols.py`)

Type-safe abstractions for LLM clients.

```python
class ChatClient(Protocol):
    async def chat(self, request: ChatRequest) -> ChatResponse: ...

class EmbeddingClient(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse: ...

class LLMClient(ChatClient, EmbeddingClient, Protocol):
    pass
```

For the common “single prompt → text response” path, we also define a lightweight
`SimpleChatClient` protocol in `src/ai_psychiatrist/infrastructure/llm/responses.py` (used by several agents). The
concrete `OllamaClient` supports both the full request/response APIs and these
convenience helpers.

---

## Design Patterns

### Dependency Injection

Agents and services receive their dependencies via constructor injection.

```python
class QuantitativeAssessmentAgent:
    def __init__(
        self,
        llm_client: SimpleChatClient,
        embedding_service: EmbeddingService | None = None,
        mode: AssessmentMode = AssessmentMode.FEW_SHOT,
        model_settings: ModelSettings | None = None,
    ) -> None:
```

This enables:
- Easy testing with mock clients
- Swappable LLM backends
- Configuration flexibility

### Protocol-Based Abstractions

Python `Protocol` classes define expected interfaces without inheritance.

Benefits:
- Structural typing (duck typing with type safety)
- No coupling to concrete implementations
- Enables mock injection for testing

### Strategy Pattern

Assessment mode (`ZERO_SHOT` vs `FEW_SHOT`) changes behavior without modifying agent code.

### Template Method Pattern

Agents follow a common structure:
1. Prepare prompt
2. Call LLM
3. Parse response
4. Return domain entity

---

## Configuration

All settings centralized in `src/ai_psychiatrist/config.py` using Pydantic Settings.

**Setting Groups:**
- `OllamaSettings`: Host, port, timeout
- `ModelSettings`: Model names, temperature, sampling
- `EmbeddingSettings`: Dimension, chunk size, top-k
- `FeedbackLoopSettings`: Max iterations, threshold
- `DataSettings`: File paths
- `LoggingSettings`: Format, level
- `APISettings`: Host, port, CORS

**Environment Variable Override:**
```bash
OLLAMA_HOST=192.168.1.100
MODEL_QUANTITATIVE_MODEL=gemma3:27b
EMBEDDING_TOP_K_REFERENCES=3
```

See [Configuration Reference](../configs/configuration.md) for complete documentation.

---

## Testing Philosophy

### No Mock Abuse

**Acceptable mocking** (I/O boundaries only):
- HTTP calls to Ollama API
- File system operations
- Time-dependent operations

**Forbidden mocking:**
- Business logic
- Domain models
- Internal functions

### Test Data vs Mocks

```python
# GOOD: Real data structures
sample_transcript = Transcript(participant_id=300, text="...")
sample_assessment = PHQ8Assessment(items=..., mode=AssessmentMode.ZERO_SHOT, participant_id=300)

# GOOD: Mock at I/O boundary
mock_client = MockLLMClient(chat_responses=["<assessment>...</assessment>"])

# BAD: Mocking internal behavior
mock_agent = Mock()
mock_agent.assess.return_value = ...  # Don't do this
```

---

## See Also

- [Pipeline](pipeline.md) - How agents collaborate
- [Configuration](../configs/configuration.md) - All settings
- [Feature Reference](../pipeline-internals/features.md) - Implemented features + defaults
