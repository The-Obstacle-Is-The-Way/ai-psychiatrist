# server.py
"""Modern API server using the refactored ai_psychiatrist agents.

This replaces the legacy server that used deprecated agents from the root agents/ directory.
Now uses the paper-aligned implementations from src/ai_psychiatrist/agents/.

BUG-012, BUG-014: Fixes split-brain architecture and missing transcript issues.
BUG-016, BUG-017: Adds MetaReviewAgent and wires FeedbackLoopService.
"""

import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from importlib.util import find_spec
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ai_psychiatrist.agents import (
    JudgeAgent,
    MetaReviewAgent,
    QualitativeAssessmentAgent,
    QuantitativeAssessmentAgent,
)
from ai_psychiatrist.config import ModelSettings, Settings, get_settings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, EvaluationMetric
from ai_psychiatrist.infrastructure.llm import create_llm_client
from ai_psychiatrist.infrastructure.llm.factory import create_embedding_client
from ai_psychiatrist.infrastructure.llm.responses import SimpleChatClient
from ai_psychiatrist.services import EmbeddingService, ReferenceStore, TranscriptService
from ai_psychiatrist.services.feedback_loop import FeedbackLoopService

# Reserved participant ID for ad-hoc transcripts sent as raw text.
# Must be positive to satisfy domain constraints (Transcript.participant_id > 0).
AD_HOC_PARTICIPANT_ID = 999_999


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle resources."""
    settings = get_settings()

    # Create separate clients for chat (agents) and embeddings
    llm_client = create_llm_client(settings)
    embedding_client = create_embedding_client(settings)

    chat_client = cast("SimpleChatClient", llm_client)
    try:
        app.state.settings = settings
        app.state.model_settings = settings.model
        app.state.llm_client = llm_client
        app.state.embedding_client = embedding_client

        app.state.transcript_service = TranscriptService(settings.data)

        reference_store = ReferenceStore(settings.data, settings.embedding)
        app.state.embedding_service = EmbeddingService(
            llm_client=embedding_client,
            reference_store=reference_store,
            settings=settings.embedding,
            model_settings=app.state.model_settings,
        )

        qual_agent = QualitativeAssessmentAgent(
            llm_client=chat_client,
            model_settings=app.state.model_settings,
        )
        judge_agent = JudgeAgent(
            llm_client=chat_client,
            model_settings=app.state.model_settings,
        )
        app.state.feedback_loop_service = FeedbackLoopService(
            qualitative_agent=qual_agent,
            judge_agent=judge_agent,
            settings=settings.feedback,
        )

        yield
    finally:
        # Close chat client
        close_chat = getattr(llm_client, "close", None)
        if callable(close_chat):
            maybe_awaitable = close_chat()
            if inspect.isawaitable(maybe_awaitable):
                await cast("Awaitable[object]", maybe_awaitable)

        # Close embedding client
        close_embed = getattr(embedding_client, "close", None)
        if callable(close_embed):
            maybe_awaitable = close_embed()
            if inspect.isawaitable(maybe_awaitable):
                await cast("Awaitable[object]", maybe_awaitable)


app = FastAPI(
    title="AI Psychiatrist Pipeline",
    version="2.0.0",
    description="Modern API using paper-aligned agent implementations",
    lifespan=lifespan,
)


# --- Dependency Injection ---
def get_llm_client(request: Request) -> SimpleChatClient:
    """Get initialized LLM client (SimpleChatClient)."""
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    return cast("SimpleChatClient", client)


def get_transcript_service(request: Request) -> TranscriptService:
    """Get initialized TranscriptService."""
    service = getattr(request.app.state, "transcript_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Transcript service not initialized")
    return cast("TranscriptService", service)


def get_embedding_service(request: Request) -> EmbeddingService:
    """Get initialized EmbeddingService."""
    service = getattr(request.app.state, "embedding_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return cast("EmbeddingService", service)


def get_model_settings(request: Request) -> ModelSettings:
    """Get initialized ModelSettings."""
    settings = getattr(request.app.state, "model_settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Model settings not initialized")
    return cast("ModelSettings", settings)


def get_app_settings(request: Request) -> Settings:
    """Get initialized Settings."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")
    return cast("Settings", settings)


def get_feedback_loop_service(request: Request) -> FeedbackLoopService:
    """Get initialized FeedbackLoopService."""
    service = getattr(request.app.state, "feedback_loop_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Feedback loop service not initialized")
    return cast("FeedbackLoopService", service)


# --- Request/Response Models ---
class AssessmentRequest(BaseModel):
    """Assessment request with participant ID or transcript text."""

    participant_id: int | None = Field(
        default=None,
        description="DAIC-WOZ participant ID (loads transcript from file)",
    )
    transcript_text: str | None = Field(
        default=None,
        description="Raw transcript text (alternative to participant_id)",
    )
    mode: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="0=zero-shot, 1=few-shot. If not specified, uses settings.enable_few_shot.",
    )

    def get_mode(self, enable_few_shot: bool = True) -> AssessmentMode:
        """Convert mode to AssessmentMode enum.

        Args:
            enable_few_shot: Fallback from settings when mode is not specified.

        Returns:
            AssessmentMode based on request mode or settings fallback.
        """
        if self.mode is not None:
            return AssessmentMode.ZERO_SHOT if self.mode == 0 else AssessmentMode.FEW_SHOT
        return AssessmentMode.FEW_SHOT if enable_few_shot else AssessmentMode.ZERO_SHOT


class QuantitativeResult(BaseModel):
    """Quantitative assessment result."""

    total_score: int
    severity: str
    na_count: int
    items: dict[str, dict[str, str | int | None]]


class QualitativeResult(BaseModel):
    """Qualitative assessment result."""

    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    supporting_quotes: list[str]


class EvaluationResult(BaseModel):
    """Qualitative evaluation result from judge agent."""

    coherence: int
    completeness: int
    specificity: int
    accuracy: int
    average_score: float
    iteration: int


class MetaReviewResult(BaseModel):
    """Meta-review result integrating all assessments."""

    severity: int
    severity_label: str
    explanation: str
    is_mdd: bool


class FullPipelineResponse(BaseModel):
    """Full pipeline response with all assessments.

    BUG-016, BUG-017: Now includes evaluation and meta-review.
    """

    participant_id: int
    mode: str
    quantitative: QuantitativeResult
    qualitative: QualitativeResult
    evaluation: EvaluationResult
    meta_review: MetaReviewResult


# --- Endpoints ---
@app.get("/health")
async def health_check(
    request: Request,
    app_settings: Annotated[Settings, Depends(get_app_settings)],
) -> dict[str, Any]:
    """Health check endpoint."""
    backend = app_settings.backend.backend.value
    if backend == "ollama":
        client = getattr(request.app.state, "llm_client", None)
        ping_fn = getattr(client, "ping", None)
        if not callable(ping_fn):
            return {"status": "degraded", "backend": backend, "error": "ping not available"}
        try:
            is_healthy = await ping_fn()
        except Exception as e:
            return {"status": "degraded", "backend": backend, "error": str(e)}
        return {
            "status": "healthy" if is_healthy else "degraded",
            "backend": backend,
            "ollama": is_healthy,
        }

    # HuggingFace backend: verify required deps are installed.
    deps_ok = all(
        find_spec(name) is not None for name in ("torch", "transformers", "sentence_transformers")
    )
    return {
        "status": "healthy" if deps_ok else "degraded",
        "backend": backend,
        "deps_installed": deps_ok,
    }


@app.post("/assess/quantitative", response_model=QuantitativeResult)
async def assess_quantitative(
    request: AssessmentRequest,
    llm: Annotated[SimpleChatClient, Depends(get_llm_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    model_settings: Annotated[ModelSettings, Depends(get_model_settings)],
    app_settings: Annotated[Settings, Depends(get_app_settings)],
) -> QuantitativeResult:
    """Run quantitative (PHQ-8) assessment."""
    # Load or create transcript
    transcript = _resolve_transcript(request, transcript_service)

    # Create agent with appropriate mode (request overrides settings.enable_few_shot)
    mode = request.get_mode(app_settings.enable_few_shot)
    agent = QuantitativeAssessmentAgent(
        llm_client=llm,
        embedding_service=embedding_service if mode == AssessmentMode.FEW_SHOT else None,
        mode=mode,
        model_settings=model_settings,
        quantitative_settings=app_settings.quantitative,
        pydantic_ai_settings=app_settings.pydantic_ai,
        ollama_base_url=app_settings.ollama.base_url,
    )

    try:
        assessment = await agent.assess(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {e}") from e

    return QuantitativeResult(
        total_score=assessment.total_score,
        severity=assessment.severity.name,
        na_count=assessment.na_count,
        items={
            item.value: {
                "score": data.score,
                "evidence": data.evidence,
                "reason": data.reason,
            }
            for item, data in assessment.items.items()
        },
    )


@app.post("/assess/qualitative", response_model=QualitativeResult)
async def assess_qualitative(
    request: AssessmentRequest,
    llm: Annotated[SimpleChatClient, Depends(get_llm_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    model_settings: Annotated[ModelSettings, Depends(get_model_settings)],
) -> QualitativeResult:
    """Run qualitative assessment (single-pass, no feedback loop).

    Note: This endpoint intentionally bypasses the FeedbackLoopService to provide
    a quick single-pass assessment. For iterative refinement with judge feedback
    per Paper Section 2.3.1-2.3.2, use the /full_pipeline endpoint instead.
    """
    transcript = _resolve_transcript(request, transcript_service)

    agent = QualitativeAssessmentAgent(llm_client=llm, model_settings=model_settings)

    try:
        assessment = await agent.assess(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {e}") from e

    return QualitativeResult(
        overall=assessment.overall,
        phq8_symptoms=assessment.phq8_symptoms,
        social_factors=assessment.social_factors,
        biological_factors=assessment.biological_factors,
        risk_factors=assessment.risk_factors,
        supporting_quotes=assessment.supporting_quotes,
    )


@app.post("/full_pipeline", response_model=FullPipelineResponse)
async def run_full_pipeline(
    request: AssessmentRequest,
    llm: Annotated[SimpleChatClient, Depends(get_llm_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    model_settings: Annotated[ModelSettings, Depends(get_model_settings)],
    app_settings: Annotated[Settings, Depends(get_app_settings)],
    feedback_loop: Annotated[FeedbackLoopService, Depends(get_feedback_loop_service)],
) -> FullPipelineResponse:
    """Run full assessment pipeline.

    Pipeline order (Paper Section 2.3):
    1. Qualitative assessment with feedback loop (Section 2.3.1-2.3.2)
    2. Quantitative PHQ-8 assessment (Section 2.3.3)
    3. Meta-review integration (Section 2.3.4)

    BUG-016, BUG-017: Now includes FeedbackLoopService and MetaReviewAgent.
    """
    transcript = _resolve_transcript(request, transcript_service)

    # Mode from request, falling back to settings.enable_few_shot
    mode = request.get_mode(app_settings.enable_few_shot)

    try:
        # Step 1: Qualitative assessment with feedback loop (BUG-017 fix)
        loop_result = await feedback_loop.run(transcript)
        qual_result = loop_result.final_assessment
        eval_result = loop_result.final_evaluation

        # Step 2: Quantitative PHQ-8 assessment
        quant_agent = QuantitativeAssessmentAgent(
            llm_client=llm,
            embedding_service=embedding_service if mode == AssessmentMode.FEW_SHOT else None,
            mode=mode,
            model_settings=model_settings,
            quantitative_settings=app_settings.quantitative,
            pydantic_ai_settings=app_settings.pydantic_ai,
            ollama_base_url=app_settings.ollama.base_url,
        )
        quant_result = await quant_agent.assess(transcript)

        # Step 3: Meta-review integration (BUG-016 fix)
        meta_review_agent = MetaReviewAgent(
            llm_client=llm,
            model_settings=model_settings,
        )
        meta_review = await meta_review_agent.review(
            transcript=transcript,
            qualitative=qual_result,
            quantitative=quant_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e

    return FullPipelineResponse(
        participant_id=transcript.participant_id,
        mode=mode.value,
        quantitative=QuantitativeResult(
            total_score=quant_result.total_score,
            severity=quant_result.severity.name,
            na_count=quant_result.na_count,
            items={
                item.value: {
                    "score": data.score,
                    "evidence": data.evidence,
                    "reason": data.reason,
                }
                for item, data in quant_result.items.items()
            },
        ),
        qualitative=QualitativeResult(
            overall=qual_result.overall,
            phq8_symptoms=qual_result.phq8_symptoms,
            social_factors=qual_result.social_factors,
            biological_factors=qual_result.biological_factors,
            risk_factors=qual_result.risk_factors,
            supporting_quotes=qual_result.supporting_quotes,
        ),
        evaluation=EvaluationResult(
            coherence=eval_result.scores[EvaluationMetric.COHERENCE].score,
            completeness=eval_result.scores[EvaluationMetric.COMPLETENESS].score,
            specificity=eval_result.scores[EvaluationMetric.SPECIFICITY].score,
            accuracy=eval_result.scores[EvaluationMetric.ACCURACY].score,
            average_score=eval_result.average_score,
            iteration=eval_result.iteration,
        ),
        meta_review=MetaReviewResult(
            severity=meta_review.severity.value,
            severity_label=meta_review.severity.name,
            explanation=meta_review.explanation,
            is_mdd=meta_review.is_mdd,
        ),
    )


# --- Helper Functions ---
def _resolve_transcript(
    request: AssessmentRequest,
    transcript_service: TranscriptService,
) -> Transcript:
    """Resolve transcript from request (participant_id or raw text).

    Args:
        request: Assessment request.
        transcript_service: Service for loading transcripts.

    Returns:
        Resolved Transcript entity.

    Raises:
        HTTPException: If transcript cannot be resolved.
    """
    if request.participant_id is not None:
        try:
            return transcript_service.load_transcript(request.participant_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load transcript for participant {request.participant_id}: {e}",
            ) from e

    if request.transcript_text:
        try:
            # Use a reserved, positive participant ID for ad-hoc transcripts to satisfy
            # domain constraints (Transcript requires participant_id > 0) while avoiding
            # collision with real DAIC-WOZ participants (300-492).
            return transcript_service.load_transcript_from_text(
                participant_id=AD_HOC_PARTICIPANT_ID,
                text=request.transcript_text,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transcript text: {e}",
            ) from e

    raise HTTPException(
        status_code=400,
        detail="Either participant_id or transcript_text must be provided",
    )
