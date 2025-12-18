# Spec 11: Full Pipeline API

## Objective

Implement the complete assessment pipeline as a FastAPI application, orchestrating all agents into a single endpoint.

## Paper Reference

- **Section 2.3.5**: Agentic System (~1 minute on M3 Pro)
- **Figure 1**: System overview

## Deliverables

1. `src/ai_psychiatrist/api/main.py` - FastAPI application
2. `src/ai_psychiatrist/api/routes/assessment.py` - Assessment endpoints
3. `src/ai_psychiatrist/api/models.py` - Pydantic request/response models
4. `src/ai_psychiatrist/api/dependencies.py` - Dependency injection
5. `tests/e2e/test_api.py` - End-to-end tests

## Implementation

### API Application (api/main.py)

```python
"""FastAPI application for AI Psychiatrist."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_psychiatrist.api.routes import assessment
from ai_psychiatrist.config import get_settings
from ai_psychiatrist.infrastructure.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    setup_logging(settings.logging)
    yield


app = FastAPI(
    title="AI Psychiatrist API",
    description="LLM-based Multi-Agent System for Depression Assessment",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router, prefix="/api/v1", tags=["assessment"])


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}
```

### Request/Response Models (api/models.py)

```python
"""API request and response models."""

from pydantic import BaseModel, Field

from ai_psychiatrist.domain.enums import AssessmentMode, SeverityLevel


class AssessmentRequest(BaseModel):
    """Request for full assessment pipeline."""

    transcript: str = Field(..., min_length=50, description="Interview transcript")
    participant_id: int = Field(default=0, ge=0)
    mode: AssessmentMode = Field(default=AssessmentMode.FEW_SHOT)
    enable_feedback_loop: bool = Field(default=True)


class ItemScore(BaseModel):
    """Single PHQ-8 item score."""

    evidence: str
    reason: str
    score: int | None
    is_available: bool


class PHQ8Response(BaseModel):
    """PHQ-8 assessment response."""

    items: dict[str, ItemScore]
    total_score: int
    severity: SeverityLevel
    mode: AssessmentMode


class QualitativeResponse(BaseModel):
    """Qualitative assessment response."""

    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    supporting_quotes: list[str]


class EvaluationResponse(BaseModel):
    """Qualitative evaluation response."""

    coherence: int
    completeness: int
    specificity: int
    accuracy: int
    average_score: float
    iteration: int


class MetaReviewResponse(BaseModel):
    """Meta-review response."""

    severity: SeverityLevel
    severity_label: str
    explanation: str
    is_mdd: bool


class FullAssessmentResponse(BaseModel):
    """Complete assessment pipeline response."""

    participant_id: int
    quantitative: PHQ8Response
    qualitative: QualitativeResponse
    evaluation: EvaluationResponse
    meta_review: MetaReviewResponse
    processing_time_seconds: float
```

### Assessment Route (api/routes/assessment.py)

```python
"""Assessment API endpoints."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ai_psychiatrist.api.dependencies import (
    get_feedback_loop_service,
    get_meta_review_agent,
    get_quantitative_agent,
    get_transcript_service,
)
from ai_psychiatrist.api.models import (
    AssessmentRequest,
    FullAssessmentResponse,
    PHQ8Response,
)
from ai_psychiatrist.domain.exceptions import DomainError
from ai_psychiatrist.infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/assess", response_model=FullAssessmentResponse)
async def run_full_assessment(
    request: AssessmentRequest,
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    feedback_loop: Annotated[FeedbackLoopService, Depends(get_feedback_loop_service)],
    quantitative_agent: Annotated[QuantitativeAgent, Depends(get_quantitative_agent)],
    meta_review_agent: Annotated[MetaReviewAgent, Depends(get_meta_review_agent)],
) -> FullAssessmentResponse:
    """Run complete depression assessment pipeline.

    Steps:
    1. Load/create transcript
    2. Run qualitative assessment with feedback loop
    3. Run quantitative PHQ-8 assessment
    4. Run meta-review integration
    5. Return combined results
    """
    start_time = time.monotonic()

    try:
        # Create transcript entity
        transcript = transcript_service.load_transcript_from_text(
            participant_id=request.participant_id,
            text=request.transcript,
        )

        # Qualitative with feedback loop
        loop_result = await feedback_loop.run(transcript)

        # Quantitative
        quantitative = await quantitative_agent.assess(transcript)

        # Meta-review
        meta_review = await meta_review_agent.review(
            transcript=transcript,
            qualitative=loop_result.final_assessment,
            quantitative=quantitative,
        )

        elapsed = time.monotonic() - start_time

        logger.info(
            "Assessment complete",
            participant_id=request.participant_id,
            severity=meta_review.severity.name,
            processing_time=f"{elapsed:.2f}s",
        )

        return _build_response(
            participant_id=request.participant_id,
            quantitative=quantitative,
            qualitative=loop_result.final_assessment,
            evaluation=loop_result.final_evaluation,
            meta_review=meta_review,
            processing_time=elapsed,
        )

    except DomainError as e:
        logger.error("Assessment failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/assess/quantitative", response_model=PHQ8Response)
async def run_quantitative_only(
    request: AssessmentRequest,
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    quantitative_agent: Annotated[QuantitativeAgent, Depends(get_quantitative_agent)],
) -> PHQ8Response:
    """Run quantitative PHQ-8 assessment only."""
    transcript = transcript_service.load_transcript_from_text(
        participant_id=request.participant_id,
        text=request.transcript,
    )
    result = await quantitative_agent.assess(transcript)
    return _build_phq8_response(result)
```

### Dependency Injection (api/dependencies.py)

```python
"""FastAPI dependency injection."""

from functools import lru_cache

from ai_psychiatrist.agents.judge import JudgeAgent
from ai_psychiatrist.agents.meta_review import MetaReviewAgent
from ai_psychiatrist.agents.qualitative import QualitativeAssessmentAgent
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.config import get_settings
from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient
from ai_psychiatrist.services.embedding import EmbeddingService
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService
from ai_psychiatrist.services.reference_store import ReferenceStore
from ai_psychiatrist.services.transcript import TranscriptService


@lru_cache
def get_llm_client() -> OllamaClient:
    """Get cached LLM client."""
    settings = get_settings()
    return OllamaClient(settings.ollama, settings.model)


@lru_cache
def get_transcript_service() -> TranscriptService:
    """Get transcript service."""
    return TranscriptService(get_settings().data)


@lru_cache
def get_reference_store() -> ReferenceStore:
    """Get reference store."""
    settings = get_settings()
    return ReferenceStore(settings.data, settings.embedding)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get embedding service."""
    settings = get_settings()
    return EmbeddingService(
        get_llm_client(),
        get_reference_store(),
        settings.embedding,
    )


def get_qualitative_agent() -> QualitativeAssessmentAgent:
    """Get qualitative agent (not cached - may have state)."""
    return QualitativeAssessmentAgent(get_llm_client())


def get_judge_agent() -> JudgeAgent:
    """Get judge agent."""
    return JudgeAgent(get_llm_client())


def get_feedback_loop_service() -> FeedbackLoopService:
    """Get feedback loop service."""
    return FeedbackLoopService(
        get_qualitative_agent(),
        get_judge_agent(),
        get_settings().feedback,
    )


def get_quantitative_agent() -> QuantitativeAssessmentAgent:
    """Get quantitative agent."""
    settings = get_settings()
    embedding = get_embedding_service() if settings.enable_few_shot else None
    return QuantitativeAssessmentAgent(
        get_llm_client(),
        embedding,
        mode=AssessmentMode.FEW_SHOT if settings.enable_few_shot else AssessmentMode.ZERO_SHOT,
    )


def get_meta_review_agent() -> MetaReviewAgent:
    """Get meta-review agent."""
    return MetaReviewAgent(get_llm_client())
```

## Acceptance Criteria

- [ ] Single endpoint runs complete pipeline
- [ ] Proper error handling with meaningful responses
- [ ] Dependency injection for testability
- [ ] Health check endpoint
- [ ] CORS configuration
- [ ] Structured logging throughout
- [ ] Processing time tracking
- [ ] Paper performance (~1 minute on M3 Pro)

## Dependencies

- **Spec 03**: Configuration
- **Spec 05-10**: All agents and services

## Specs That Depend on This

- **Spec 12**: Observability
