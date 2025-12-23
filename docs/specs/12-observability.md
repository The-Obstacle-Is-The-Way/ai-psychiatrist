# Spec 12: Observability

> **STATUS: DEFERRED**
>
> This spec is deferred until real E2E testing with actual models is complete.
> The core pipeline (Specs 01-11.5) is fully implemented and working. Basic
> structured logging via `structlog` is already in place. Full observability
> (Prometheus metrics, OpenTelemetry tracing, health endpoints) will be added
> when preparing for production deployment.
>
> **Tracked by**: [GitHub Issue #27](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/27)
>
> **Last Updated**: 2025-12-21

---

## Objective

Implement comprehensive observability infrastructure including metrics, distributed tracing, and health checks for production deployment.

## Paper Reference

- **Section 2.3.5**: Performance characteristics (~1 minute on M3 Pro)
- **Section 3**: Experimental results requiring metric tracking

## As-Is Observability (Repo)

- **Modern implementation (`src/`)**: structured logging via `structlog` is in place (see `src/ai_psychiatrist/infrastructure/logging.py`) and is used by the API/server and services.
- **Legacy implementation (`_legacy/`)**: most scripts use `print` (no structured logging).
- No metrics/tracing beyond basic FastAPI behavior; `/health` exists in `server.py` but deeper runtime metrics/tracing are deferred.
- One notable exception: `_legacy/agents/quantitative_assessor_f.py` has a `VERBOSE` flag and prints timestamped `[STEP]`, `[CHAT]`, and `[EMB]` logs, including the "exact" user prompt for chat calls.
- The repo contains evaluation artifacts in `_legacy/analysis_output/` and plotting code in notebooks, but those are offline—not runtime observability.

## As-Is Validation Artifacts (`_legacy/analysis_output/`)

These files are used by notebooks and are useful for validating spec parity:

- `_legacy/analysis_output/quan_gemma_zero_shot.jsonl`: JSONL of per-item predictions
  - Top-level: `participant_id`, `timestamp`, and `PHQ8_*` keys
  - Each `PHQ8_*` value: `{ "evidence": str, "reason": str, "score": int | "N/A" }`
- `_legacy/analysis_output/quan_medgemma_few_shot.jsonl` (and similar few-shot runs): JSONL with additional retrieval metadata
  - Some items include fields like `cosine_similarity` and lists like `reference_evidence_scores` / `reference_evidence_total_scores`
- `_legacy/analysis_output/qual_gemma.csv`: qualitative outputs
  - Columns: `participant_id`, `dataset_type`, `qualitative_assessment` (often contains fenced ```xml blocks)
- `_legacy/analysis_output/metareview_gemma_few_shot.csv`: meta-review outputs
  - Columns: `participant_id`, `response`, `severity`, `explanation`

## Deliverables

Note: The deliverables below are **planned** (not implemented in `src/` yet). They are listed here
to make the intended production hardening explicit and reviewable.

1. `src/ai_psychiatrist/infrastructure/metrics.py` - Prometheus metrics
2. `src/ai_psychiatrist/infrastructure/tracing.py` - OpenTelemetry integration
3. `src/ai_psychiatrist/api/routes/health.py` - Health check endpoints
4. `src/ai_psychiatrist/api/middleware.py` - Observability middleware
5. `tests/unit/infrastructure/test_observability.py` - Tests

## Implementation

### Metrics (infrastructure/metrics.py)

```python
"""Prometheus metrics for AI Psychiatrist."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info("ai_psychiatrist", "Application information")

# Request metrics
REQUEST_COUNT = Counter(
    "ai_psychiatrist_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ai_psychiatrist_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Pipeline metrics
PIPELINE_DURATION = Histogram(
    "ai_psychiatrist_pipeline_duration_seconds",
    "Full pipeline execution time",
    buckets=[10.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0],
)

PIPELINE_IN_PROGRESS = Gauge(
    "ai_psychiatrist_pipeline_in_progress",
    "Number of pipelines currently executing",
)

# Agent metrics
AGENT_CALLS = Counter(
    "ai_psychiatrist_agent_calls_total",
    "Total agent invocations",
    ["agent_name", "status"],
)

AGENT_DURATION = Histogram(
    "ai_psychiatrist_agent_duration_seconds",
    "Agent execution time",
    ["agent_name"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
)

# LLM metrics
LLM_CALLS = Counter(
    "ai_psychiatrist_llm_calls_total",
    "Total LLM API calls",
    ["model", "status"],
)

LLM_DURATION = Histogram(
    "ai_psychiatrist_llm_duration_seconds",
    "LLM call latency",
    ["model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
)

LLM_TOKENS = Counter(
    "ai_psychiatrist_llm_tokens_total",
    "Total tokens processed",
    ["model", "direction"],  # direction: input/output
)

# Feedback loop metrics
FEEDBACK_ITERATIONS = Histogram(
    "ai_psychiatrist_feedback_iterations",
    "Number of feedback loop iterations",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)

FEEDBACK_CONVERGENCE = Counter(
    "ai_psychiatrist_feedback_convergence_total",
    "Feedback loop convergence outcomes",
    ["outcome"],  # converged, max_iterations, error
)

# Assessment metrics
ASSESSMENT_SCORES = Histogram(
    "ai_psychiatrist_phq8_total_score",
    "Distribution of PHQ-8 total scores",
    buckets=[0, 4, 9, 14, 19, 24],
)

SEVERITY_DISTRIBUTION = Counter(
    "ai_psychiatrist_severity_total",
    "Severity level distribution",
    ["severity"],
)

NA_ITEMS = Histogram(
    "ai_psychiatrist_na_items_count",
    "Number of N/A items per assessment",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

# Embedding metrics
EMBEDDING_CACHE_HITS = Counter(
    "ai_psychiatrist_embedding_cache_total",
    "Embedding cache operations",
    ["result"],  # hit, miss
)

REFERENCE_RETRIEVAL_DURATION = Histogram(
    "ai_psychiatrist_reference_retrieval_seconds",
    "Reference retrieval latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)


def init_app_info(version: str, model: str) -> None:
    """Initialize application info metric."""
    APP_INFO.info({
        "version": version,
        "model": model,
        "environment": "production",
    })
```

### Tracing (infrastructure/tracing.py)

```python
"""OpenTelemetry distributed tracing."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ai_psychiatrist.config import TracingSettings

F = TypeVar("F", bound=Callable[..., Any])

_tracer: trace.Tracer | None = None


def setup_tracing(settings: TracingSettings, app: FastAPI | None = None) -> None:
    """Initialize OpenTelemetry tracing."""
    global _tracer

    if not settings.enabled:
        return

    resource = Resource.create({
        "service.name": settings.service_name,
        "service.version": "2.0.0",
        "deployment.environment": settings.environment,
    })

    provider = TracerProvider(resource=resource)

    if settings.otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)

    # Auto-instrument FastAPI
    if app:
        FastAPIInstrumentor.instrument_app(app)

    # Auto-instrument HTTP client
    HTTPXClientInstrumentor().instrument()


def get_tracer() -> trace.Tracer:
    """Get the configured tracer."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Create a traced span context manager."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        try:
            yield s
        except Exception as e:
            s.set_status(Status(StatusCode.ERROR, str(e)))
            s.record_exception(e)
            raise


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator to trace a function."""

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name) as s:
                s.set_attribute("function.name", func.__qualname__)
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name) as s:
                s.set_attribute("function.name", func.__qualname__)
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class TracingContext:
    """Context for adding attributes to current span."""

    @staticmethod
    def set_participant_id(participant_id: int) -> None:
        """Set participant ID on current span."""
        current = trace.get_current_span()
        current.set_attribute("participant.id", participant_id)

    @staticmethod
    def set_agent(agent_name: str) -> None:
        """Set agent name on current span."""
        current = trace.get_current_span()
        current.set_attribute("agent.name", agent_name)

    @staticmethod
    def set_severity(severity: str) -> None:
        """Set predicted severity on current span."""
        current = trace.get_current_span()
        current.set_attribute("assessment.severity", severity)

    @staticmethod
    def add_event(name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to current span."""
        current = trace.get_current_span()
        current.add_event(name, attributes=attributes or {})
```

### Health Check Routes (api/routes/health.py)

```python
"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

if TYPE_CHECKING:
    from ai_psychiatrist.infrastructure.llm.ollama import OllamaClient

router = APIRouter(tags=["health"])


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Individual component health."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Complete health check response."""

    status: HealthStatus
    version: str
    timestamp: datetime
    components: list[ComponentHealth]
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool
    checks: dict[str, bool]


# Track startup time
_startup_time: datetime | None = None


def set_startup_time() -> None:
    """Record application startup time."""
    global _startup_time
    _startup_time = datetime.now(timezone.utc)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    llm_client: OllamaClient = Depends(get_llm_client),
) -> HealthResponse:
    """Comprehensive health check endpoint.

    Checks:
    - LLM service connectivity
    - Embedding service availability
    - Reference store accessibility
    """
    import time

    components: list[ComponentHealth] = []

    # Check LLM
    llm_health = await _check_llm(llm_client)
    components.append(llm_health)

    # Determine overall status
    statuses = [c.status for c in components]
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED

    uptime = 0.0
    if _startup_time:
        uptime = (datetime.now(timezone.utc) - _startup_time).total_seconds()

    return HealthResponse(
        status=overall,
        version="2.0.0",
        timestamp=datetime.now(timezone.utc),
        components=components,
        uptime_seconds=uptime,
    )


@router.get("/health/live")
async def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe.

    Simple check that the application is running.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe(
    llm_client: OllamaClient = Depends(get_llm_client),
) -> ReadinessResponse:
    """Kubernetes readiness probe.

    Checks if application is ready to accept traffic.
    """
    checks = {}

    # Check LLM connectivity
    try:
        await llm_client.ping()
        checks["llm"] = True
    except Exception:
        checks["llm"] = False

    ready = all(checks.values())

    return ReadinessResponse(ready=ready, checks=checks)


async def _check_llm(client: OllamaClient) -> ComponentHealth:
    """Check LLM service health."""
    import time

    start = time.monotonic()
    try:
        await client.ping()
        latency = (time.monotonic() - start) * 1000

        return ComponentHealth(
            name="ollama",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        return ComponentHealth(
            name="ollama",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


# Dependency injection helper
def get_llm_client() -> OllamaClient:
    """Get LLM client for health checks."""
    from ai_psychiatrist.api.dependencies import get_llm_client as _get_client
    return _get_client()
```

### Observability Middleware (api/middleware.py)

```python
"""FastAPI middleware for observability."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ai_psychiatrist.infrastructure.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from ai_psychiatrist.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Record request metrics."""
        start_time = time.monotonic()

        # Extract path template for consistent labeling
        path = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.monotonic() - start_time

            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status,
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path,
            ).observe(duration)

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Log request details."""
        import uuid

        request_id = str(uuid.uuid4())[:8]
        start_time = time.monotonic()

        # Bind request context
        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info("Request started")

        try:
            response = await call_next(request)

            duration = time.monotonic() - start_time
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.exception(
                "Request failed",
                error=str(e),
                duration_ms=round(duration * 1000, 2),
            )
            raise


def setup_middleware(app) -> None:
    """Configure all observability middleware."""
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
```

### Agent Instrumentation Example

```python
"""Example of instrumented agent."""

from ai_psychiatrist.infrastructure.metrics import (
    AGENT_CALLS,
    AGENT_DURATION,
    PIPELINE_DURATION,
    PIPELINE_IN_PROGRESS,
)
from ai_psychiatrist.infrastructure.tracing import span, TracingContext


class InstrumentedQualitativeAgent:
    """Qualitative agent with observability."""

    async def assess(self, transcript: Transcript) -> QualitativeAssessment:
        """Instrumented assessment."""
        with span("qualitative_assessment") as s:
            TracingContext.set_participant_id(transcript.participant_id)
            TracingContext.set_agent("qualitative")

            start = time.monotonic()
            try:
                result = await self._do_assessment(transcript)

                AGENT_CALLS.labels(
                    agent_name="qualitative",
                    status="success",
                ).inc()

                return result

            except Exception as e:
                AGENT_CALLS.labels(
                    agent_name="qualitative",
                    status="error",
                ).inc()
                raise

            finally:
                duration = time.monotonic() - start
                AGENT_DURATION.labels(
                    agent_name="qualitative",
                ).observe(duration)
```

### Prometheus Endpoint Integration

```python
"""Add to api/main.py"""

from prometheus_client import make_asgi_app

# Create metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## Configuration

```python
"""Add to config.py"""

class TracingSettings(BaseModel):
    """Tracing configuration."""

    enabled: bool = Field(default=True)
    service_name: str = Field(default="ai-psychiatrist")
    environment: str = Field(default="development")
    otlp_endpoint: str | None = Field(default=None)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class MetricsSettings(BaseModel):
    """Metrics configuration."""

    enabled: bool = Field(default=True)
    port: int = Field(default=9090)
    path: str = Field(default="/metrics")
```

## Acceptance Criteria

- [ ] Prometheus metrics exposed at `/metrics`
- [ ] Request count, latency, and error rate tracked
- [ ] Agent-level metrics (calls, duration)
- [ ] LLM metrics (calls, tokens, latency)
- [ ] Feedback loop iteration tracking
- [ ] OpenTelemetry tracing integration
- [ ] Span propagation across agents
- [ ] Health check endpoint (`/health`)
- [ ] Liveness probe (`/health/live`)
- [ ] Readiness probe (`/health/ready`)
- [ ] Structured request logging with correlation IDs
- [ ] Paper performance metrics trackable (~1 minute pipeline)

## Grafana Dashboard (Reference)

Key panels to create:
1. **Request Rate** - `rate(ai_psychiatrist_requests_total[5m])`
2. **Latency P95** - `histogram_quantile(0.95, ai_psychiatrist_request_duration_seconds_bucket)`
3. **Pipeline Duration** - `histogram_quantile(0.5, ai_psychiatrist_pipeline_duration_seconds_bucket)`
4. **Agent Breakdown** - `ai_psychiatrist_agent_duration_seconds` by agent
5. **Severity Distribution** - `ai_psychiatrist_severity_total`
6. **Feedback Iterations** - `histogram_quantile(0.5, ai_psychiatrist_feedback_iterations_bucket)`
7. **Error Rate** - `rate(ai_psychiatrist_requests_total{status=~"5.."}[5m])`

## Dependencies

- **Spec 03**: Configuration (TracingSettings, MetricsSettings)
- **Spec 04**: LLM infrastructure (ping method)
- **Spec 11**: Full Pipeline (middleware integration)

## Specs That Depend on This

- None (final spec in chain)
