# server.py
"""Modern API server using the refactored ai_psychiatrist agents.

This replaces the legacy server that used deprecated agents from the root agents/ directory.
Now uses the paper-aligned implementations from src/ai_psychiatrist/agents/.

BUG-012, BUG-014: Fixes split-brain architecture and missing transcript issues.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_psychiatrist.agents import QualitativeAssessmentAgent, QuantitativeAssessmentAgent
from ai_psychiatrist.config import get_settings
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import AssessmentMode
from ai_psychiatrist.infrastructure.llm import OllamaClient
from ai_psychiatrist.services import EmbeddingService, ReferenceStore, TranscriptService

# --- Shared State (initialized at startup) ---
_ollama_client: OllamaClient | None = None
_transcript_service: TranscriptService | None = None
_embedding_service: EmbeddingService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifecycle resources."""
    global _ollama_client, _transcript_service, _embedding_service  # noqa: PLW0603

    settings = get_settings()

    # Initialize OllamaClient
    _ollama_client = OllamaClient(settings.ollama)

    # Initialize TranscriptService
    _transcript_service = TranscriptService(settings.data)

    # Initialize EmbeddingService (for few-shot mode)
    reference_store = ReferenceStore(settings.data, settings.embedding)
    _embedding_service = EmbeddingService(
        llm_client=_ollama_client,
        reference_store=reference_store,
        settings=settings.embedding,
    )

    yield

    # Cleanup
    if _ollama_client:
        await _ollama_client.close()


app = FastAPI(
    title="AI Psychiatrist Pipeline",
    version="2.0.0",
    description="Modern API using paper-aligned agent implementations",
    lifespan=lifespan,
)


# --- Dependency Injection ---
def get_ollama_client() -> OllamaClient:
    """Get initialized OllamaClient."""
    if _ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    return _ollama_client


def get_transcript_service() -> TranscriptService:
    """Get initialized TranscriptService."""
    if _transcript_service is None:
        raise HTTPException(status_code=503, detail="Transcript service not initialized")
    return _transcript_service


def get_embedding_service() -> EmbeddingService:
    """Get initialized EmbeddingService."""
    if _embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return _embedding_service


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
    mode: int = Field(
        default=1,
        ge=0,
        le=1,
        description="0=zero-shot, 1=few-shot (default, paper-optimal)",
    )

    def get_mode(self) -> AssessmentMode:
        """Convert mode int to AssessmentMode enum."""
        return AssessmentMode.ZERO_SHOT if self.mode == 0 else AssessmentMode.FEW_SHOT


class QuantitativeResult(BaseModel):
    """Quantitative assessment result."""

    total_score: int
    severity: str
    na_count: int
    items: dict[str, dict]


class QualitativeResult(BaseModel):
    """Qualitative assessment result."""

    overall: str
    phq8_symptoms: str
    social_factors: str
    biological_factors: str
    risk_factors: str
    supporting_quotes: list[str]


class FullPipelineResponse(BaseModel):
    """Full pipeline response with both assessments."""

    participant_id: int
    mode: str
    quantitative: QuantitativeResult
    qualitative: QualitativeResult


# --- Endpoints ---
@app.get("/health")
async def health_check(ollama: Annotated[OllamaClient, Depends(get_ollama_client)]):
    """Health check endpoint."""
    try:
        is_healthy = await ollama.ping()
        return {"status": "healthy" if is_healthy else "degraded", "ollama": is_healthy}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/assess/quantitative", response_model=QuantitativeResult)
async def assess_quantitative(
    request: AssessmentRequest,
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
):
    """Run quantitative (PHQ-8) assessment."""
    # Load or create transcript
    transcript = _resolve_transcript(request, transcript_service)

    # Create agent with appropriate mode
    mode = request.get_mode()
    agent = QuantitativeAssessmentAgent(
        llm_client=ollama,
        embedding_service=embedding_service if mode == AssessmentMode.FEW_SHOT else None,
        mode=mode,
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
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
):
    """Run qualitative assessment."""
    transcript = _resolve_transcript(request, transcript_service)

    agent = QualitativeAssessmentAgent(llm_client=ollama)

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
    ollama: Annotated[OllamaClient, Depends(get_ollama_client)],
    transcript_service: Annotated[TranscriptService, Depends(get_transcript_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
):
    """Run full assessment pipeline (quantitative + qualitative).

    This is the main endpoint replacing the legacy /full_pipeline.
    Now uses modern agents from src/ai_psychiatrist/agents/.
    """
    transcript = _resolve_transcript(request, transcript_service)

    mode = request.get_mode()

    # Create agents
    quant_agent = QuantitativeAssessmentAgent(
        llm_client=ollama,
        embedding_service=embedding_service if mode == AssessmentMode.FEW_SHOT else None,
        mode=mode,
    )
    qual_agent = QualitativeAssessmentAgent(llm_client=ollama)

    try:
        # Run assessments in parallel for better performance
        quant_result, qual_result = await asyncio.gather(
            quant_agent.assess(transcript),
            qual_agent.assess(transcript),
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
            # Use synthetic participant ID (-1) for ad-hoc transcripts
            # to avoid collision with real DAIC-WOZ participants (300-492)
            return transcript_service.load_transcript_from_text(
                participant_id=-1,
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
