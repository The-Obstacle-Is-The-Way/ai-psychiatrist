# API Reference

REST API endpoints for AI Psychiatrist.

---

## Overview

The API is built with FastAPI and provides endpoints for depression assessment from interview transcripts.

**Base URL:** `http://localhost:8000` (default)

**Documentation:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

---

## Endpoints

### Health Check

#### `GET /health`

Check API and Ollama connectivity.

**Response:**
```json
{
  "status": "healthy",
  "ollama": "connected",
  "models": ["gemma3:27b", "alibayram/medgemma:27b", "qwen3-embedding:8b"]
}
```

**Status Codes:**
- `200 OK`: System healthy
- `503 Service Unavailable`: Ollama not reachable

---

### Full Assessment

#### `POST /assess`

Run complete 4-agent pipeline on a transcript.

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript": "Ellie: How are you doing today?\nParticipant: I have been feeling really down lately..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `participant_id` | int | Yes | Unique participant identifier |
| `transcript` | string | Yes | Interview transcript text |

**Response:**
```json
{
  "participant_id": 300,
  "severity": "MODERATE",
  "severity_level": 2,
  "is_mdd": true,
  "phq8": {
    "total_score": 12,
    "items": {
      "NO_INTEREST": {
        "score": 2,
        "evidence": "i don't enjoy anything anymore",
        "reason": "Clear anhedonia expressed"
      },
      "DEPRESSED": {
        "score": 2,
        "evidence": "feeling really down",
        "reason": "Direct statement of depressed mood"
      },
      "SLEEP": {
        "score": null,
        "evidence": "No relevant evidence found",
        "reason": "Not discussed in transcript"
      }
    },
    "available_count": 5,
    "na_count": 3
  },
  "qualitative": {
    "overall": "Participant shows moderate depressive symptoms...",
    "phq8_symptoms": "Reports anhedonia and low mood...",
    "social_factors": "Limited social support network...",
    "biological_factors": "No family history mentioned...",
    "risk_factors": "Recent job loss identified..."
  },
  "evaluation": {
    "coherence": 4,
    "completeness": 4,
    "specificity": 4,
    "accuracy": 4,
    "average": 4.0,
    "iterations_used": 1
  },
  "explanation": "Based on the qualitative assessment and PHQ-8 scores...",
  "processing_time_seconds": 45.2
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `severity` | string | `MINIMAL`, `MILD`, `MODERATE`, `MOD_SEVERE`, `SEVERE` |
| `severity_level` | int | 0-4 numeric level |
| `is_mdd` | bool | Major Depressive Disorder indicator (score ≥ 10) |
| `phq8.total_score` | int | Sum of item scores (0-24) |
| `phq8.items` | object | Per-item assessment details |
| `qualitative` | object | Narrative assessment sections |
| `evaluation` | object | Judge agent scores |
| `explanation` | string | Meta-review reasoning |

**Status Codes:**
- `200 OK`: Assessment complete
- `400 Bad Request`: Invalid transcript format
- `500 Internal Server Error`: LLM or processing error
- `504 Gateway Timeout`: Ollama timeout

---

### Assess by Participant ID

#### `POST /assess/{participant_id}`

Assess transcript from data directory.

**Path Parameters:**
- `participant_id` (int): DAIC-WOZ participant ID (e.g., 300, 402)

**Request Body:** None (uses transcript from `data/transcripts/{id}_P/{id}_TRANSCRIPT.csv`)

**Response:** Same as `POST /assess`

**Status Codes:**
- `200 OK`: Assessment complete
- `404 Not Found`: Participant transcript not found
- `500 Internal Server Error`: Processing error

**Example:**
```bash
curl -X POST http://localhost:8000/assess/300
```

---

### Qualitative Assessment Only

#### `POST /assess/qualitative`

Run only the qualitative assessment agent (with optional feedback loop).

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript": "...",
  "enable_feedback_loop": true
}
```

**Response:**
```json
{
  "participant_id": 300,
  "assessment": {
    "overall": "...",
    "phq8_symptoms": "...",
    "social_factors": "...",
    "biological_factors": "...",
    "risk_factors": "..."
  },
  "evaluation": {
    "coherence": 4,
    "completeness": 4,
    "specificity": 5,
    "accuracy": 4,
    "iterations_used": 2
  }
}
```

---

### Quantitative Assessment Only

#### `POST /assess/quantitative`

Run only the quantitative assessment agent.

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript": "...",
  "mode": "few_shot"
}
```

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `mode` | string | `few_shot` | `few_shot`, `zero_shot` |

**Response:**
```json
{
  "participant_id": 300,
  "mode": "few_shot",
  "total_score": 12,
  "severity": "MODERATE",
  "items": {
    "NO_INTEREST": {"score": 2, "evidence": "...", "reason": "..."},
    "DEPRESSED": {"score": 2, "evidence": "...", "reason": "..."}
  }
}
```

---

### Batch Assessment

#### `POST /assess/batch`

Assess multiple participants in sequence.

**Request Body:**
```json
{
  "participant_ids": [300, 301, 302],
  "mode": "few_shot"
}
```

**Response:**
```json
{
  "results": [
    {"participant_id": 300, "severity": "MODERATE", "total_score": 12},
    {"participant_id": 301, "severity": "MILD", "total_score": 7},
    {"participant_id": 302, "severity": "SEVERE", "total_score": 21}
  ],
  "summary": {
    "total": 3,
    "completed": 3,
    "failed": 0,
    "processing_time_seconds": 135.6
  }
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-12-21T10:00:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `TRANSCRIPT_NOT_FOUND` | 404 | Participant transcript file not found |
| `TRANSCRIPT_EMPTY` | 400 | Transcript text is empty |
| `INVALID_PARTICIPANT_ID` | 400 | Participant ID must be positive integer |
| `OLLAMA_CONNECTION_ERROR` | 503 | Cannot connect to Ollama server |
| `OLLAMA_TIMEOUT` | 504 | Ollama request timed out |
| `LLM_ERROR` | 500 | LLM returned error or unparseable response |
| `EMBEDDING_DIMENSION_MISMATCH` | 500 | Query/reference embedding dimensions differ |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Request/Response Models

### TranscriptRequest

```python
class TranscriptRequest(BaseModel):
    participant_id: int = Field(..., gt=0)
    transcript: str = Field(..., min_length=10)
```

### AssessmentResponse

```python
class AssessmentResponse(BaseModel):
    participant_id: int
    severity: str  # MINIMAL, MILD, MODERATE, MOD_SEVERE, SEVERE
    severity_level: int  # 0-4
    is_mdd: bool
    phq8: PHQ8Response
    qualitative: QualitativeResponse
    evaluation: EvaluationResponse
    explanation: str
    processing_time_seconds: float
```

### PHQ8ItemResponse

```python
class PHQ8ItemResponse(BaseModel):
    score: int | None  # 0-3 or null for N/A
    evidence: str
    reason: str
```

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider:

- API key authentication via header
- OAuth2/JWT tokens
- Rate limiting

---

## CORS

CORS is configured via `API_CORS_ORIGINS` environment variable:

```bash
# Development (allow all)
API_CORS_ORIGINS=["*"]

# Production (restrict)
API_CORS_ORIGINS=["https://myapp.com", "https://admin.myapp.com"]
```

---

## Rate Limiting

Not currently implemented. Consider adding for production:
- Per-client rate limits
- Request queuing for batch operations
- Backpressure handling

---

## WebSocket (Future)

Planned for streaming responses during long assessments:

```
WS /ws/assess/{participant_id}
```

Would stream progress updates during pipeline execution.

---

## Examples

### cURL

```bash
# Full assessment
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "participant_id": 300,
    "transcript": "Ellie: How are you?\nParticipant: Not great..."
  }'

# Assess by ID (transcript from data directory)
curl -X POST http://localhost:8000/assess/300

# Health check
curl http://localhost:8000/health
```

### Python (httpx)

```python
import httpx

async with httpx.AsyncClient(timeout=300) as client:
    response = await client.post(
        "http://localhost:8000/assess",
        json={
            "participant_id": 300,
            "transcript": "Ellie: How are you?..."
        }
    )
    result = response.json()
    print(f"Severity: {result['severity']}")
    print(f"PHQ-8 Total: {result['phq8']['total_score']}")
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/assess",
    json={
        "participant_id": 300,
        "transcript": "Ellie: How are you?..."
    },
    timeout=300
)
result = response.json()
```

---

## See Also

- [Quickstart](../../getting-started/quickstart.md) - Getting started
- [Configuration](../configuration.md) - API settings
- [Pipeline](../../concepts/pipeline.md) - Processing flow
