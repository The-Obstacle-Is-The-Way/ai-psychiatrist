# API Reference

REST API endpoints for AI Psychiatrist.

---

## Overview

The API is built with FastAPI and provides endpoints for depression assessment from interview transcripts. The main server is implemented in `server.py` at the project root.

**Base URL:** `http://localhost:8000` (default)

**Documentation:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

---

## Endpoints

### Health Check

#### `GET /health`

Check API and chat backend availability (`LLM_BACKEND`).

**Note**: This endpoint does not currently check embedding backend health (`EMBEDDING_BACKEND`).

**Response:**
```json
{
  "status": "healthy",
  "backend": "ollama",
  "ollama": true
}
```

If `LLM_BACKEND=huggingface`:
```json
{
  "status": "healthy",
  "backend": "huggingface",
  "deps_installed": true
}
```

**Status Codes:**
- `200 OK`: System healthy
- `200 OK` with `"status": "degraded"`: backend not reachable / deps missing

---

### Full Pipeline Assessment

#### `POST /full_pipeline`

Run complete 4-agent pipeline on a transcript. This is the recommended endpoint for full assessments as it includes:
1. Qualitative assessment with judge-driven refinement (Paper Section 2.3.1)
2. Quantitative PHQ-8 assessment (Paper Section 2.3.2)
3. Meta-review integration (Paper Section 2.3.3)

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript_text": null,
  "mode": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `participant_id` | int \| null | No* | DAIC-WOZ participant ID (loads transcript from file) |
| `transcript_text` | string \| null | No* | Raw transcript text (alternative to participant_id) |
| `mode` | int \| null | No | 0=zero-shot, 1=few-shot. If null, uses `settings.enable_few_shot` |

*One of `participant_id` or `transcript_text` must be provided.

**Response:**
```json
{
  "participant_id": 300,
  "mode": "few_shot",
  "quantitative": {
    "total_score": 12,
    "severity": "MODERATE",
    "na_count": 3,
    "items": {
      "NoInterest": {
        "score": 2,
        "evidence": "i don't enjoy anything anymore",
        "reason": "Clear anhedonia expressed"
      },
      "Depressed": {
        "score": 2,
        "evidence": "feeling really down",
        "reason": "Direct statement of depressed mood"
      },
      "Sleep": {
        "score": null,
        "evidence": "No relevant evidence found",
        "reason": "Not discussed in transcript"
      }
    }
  },
  "qualitative": {
    "overall": "Participant shows moderate depressive symptoms...",
    "phq8_symptoms": "Reports anhedonia and low mood...",
    "social_factors": "Limited social support network...",
    "biological_factors": "No family history mentioned...",
    "risk_factors": "Recent job loss identified...",
    "supporting_quotes": ["i don't enjoy anything anymore", "feeling really down"]
  },
  "evaluation": {
    "coherence": 4,
    "completeness": 4,
    "specificity": 4,
    "accuracy": 4,
    "average_score": 4.0,
    "iteration": 0
  },
  "meta_review": {
    "severity": 2,
    "severity_label": "MODERATE",
    "explanation": "Based on the qualitative assessment and PHQ-8 scores...",
    "is_mdd": true
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `zero_shot` or `few_shot` |
| `quantitative.severity` | string | `MINIMAL`, `MILD`, `MODERATE`, `MOD_SEVERE`, `SEVERE` |
| `quantitative.total_score` | int | Sum of item scores (0-24) |
| `quantitative.na_count` | int | Number of items without scores |
| `qualitative` | object | Narrative assessment sections |
| `evaluation` | object | Judge agent scores (1-5 Likert scale) |
| `evaluation.iteration` | int | Refinement iteration for the returned qualitative assessment (`0` means first-pass evaluation) |
| `meta_review.severity` | int | Final severity level (0-4) |
| `meta_review.is_mdd` | bool | Major Depressive Disorder indicator (severity >= 2) |

**Status Codes:**
- `200 OK`: Assessment complete
- `400 Bad Request`: Invalid request (missing transcript)
- `500 Internal Server Error`: LLM or processing error

---

### Quantitative Assessment Only

#### `POST /assess/quantitative`

Run only the quantitative assessment agent (PHQ-8 scoring).

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript_text": null,
  "mode": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `participant_id` | int \| null | No* | DAIC-WOZ participant ID |
| `transcript_text` | string \| null | No* | Raw transcript text |
| `mode` | int \| null | No | 0=zero-shot, 1=few-shot |

**Response:**
```json
{
  "total_score": 12,
  "severity": "MODERATE",
  "na_count": 3,
  "items": {
    "NoInterest": {"score": 2, "evidence": "...", "reason": "..."},
    "Depressed": {"score": 2, "evidence": "...", "reason": "..."},
    "Sleep": {"score": null, "evidence": "No relevant evidence found", "reason": "..."},
    "Tired": {"score": 2, "evidence": "...", "reason": "..."},
    "Appetite": {"score": null, "evidence": "...", "reason": "..."},
    "Failure": {"score": 1, "evidence": "...", "reason": "..."},
    "Concentrating": {"score": 1, "evidence": "...", "reason": "..."},
    "Moving": {"score": null, "evidence": "...", "reason": "..."}
  }
}
```

---

### Qualitative Assessment Only

#### `POST /assess/qualitative`

Run only the qualitative assessment agent (single-pass, no feedback loop).

> **Note:** This endpoint bypasses the FeedbackLoopService for speed. For iterative refinement per Paper Section 2.3.1, use `/full_pipeline` instead.

**Request Body:**
```json
{
  "participant_id": 300,
  "transcript_text": null
}
```

**Response:**
```json
{
  "overall": "The participant shows signs of moderate depression...",
  "phq8_symptoms": "Anhedonia (several days), low mood (most days)...",
  "social_factors": "Limited support network, lives alone...",
  "biological_factors": "No family history mentioned...",
  "risk_factors": "Recent stressors including job loss...",
  "supporting_quotes": ["i don't enjoy anything anymore", "feeling really down"]
}
```

---

## Request Models

### AssessmentRequest

All assessment endpoints accept the same request model:

```python
class AssessmentRequest(BaseModel):
    participant_id: int | None = None  # DAIC-WOZ participant ID
    transcript_text: str | None = None  # Raw transcript text
    mode: int | None = None  # 0=zero-shot, 1=few-shot
```

**Transcript Resolution:**
1. If `participant_id` is provided: Loads transcript from `data/transcripts/{id}_P/{id}_TRANSCRIPT.csv`
2. If `transcript_text` is provided: Uses the raw text directly
3. If neither: Returns 400 error

**Mode Resolution:**
1. If `mode=0`: Zero-shot assessment
2. If `mode=1`: Few-shot assessment with embeddings
3. If `mode=null`: Uses `settings.enable_few_shot` (default: true)

---

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error description"
}
```

### Common Error Codes

| HTTP Status | Cause |
|-------------|-------|
| `400 Bad Request` | Missing both `participant_id` and `transcript_text` |
| `400 Bad Request` | Failed to load transcript for participant ID |
| `400 Bad Request` | Invalid transcript text |
| `500 Internal Server Error` | LLM or pipeline processing error |
| `503 Service Unavailable` | Ollama client not initialized |

---

## Examples

### cURL

```bash
# Full pipeline assessment with participant ID
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{"participant_id": 300}'

# Full pipeline with raw transcript text
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_text": "Ellie: How are you doing today?\nParticipant: I have been feeling really down lately."
  }'

# Quantitative-only assessment
curl -X POST http://localhost:8000/assess/quantitative \
  -H "Content-Type: application/json" \
  -d '{"participant_id": 300, "mode": 1}'

# Qualitative-only assessment
curl -X POST http://localhost:8000/assess/qualitative \
  -H "Content-Type: application/json" \
  -d '{"participant_id": 300}'

# Health check
curl http://localhost:8000/health
```

### Python (httpx)

```python
import httpx

async with httpx.AsyncClient(timeout=300) as client:
    # Full pipeline
    response = await client.post(
        "http://localhost:8000/full_pipeline",
        json={"participant_id": 300}
    )
    result = response.json()
    print(f"Severity: {result['meta_review']['severity_label']}")
    print(f"PHQ-8 Total: {result['quantitative']['total_score']}")
    print(f"Is MDD: {result['meta_review']['is_mdd']}")
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/full_pipeline",
    json={"participant_id": 300},
    timeout=300
)
result = response.json()
```

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider:

- API key authentication via header
- OAuth2/JWT tokens
- Rate limiting

---

## CORS

`API_CORS_ORIGINS` exists in configuration, but `server.py` does not currently install
FastAPI/Starlette `CORSMiddleware`. If you need CORS today, configure it at a reverse
proxy (recommended) or add `CORSMiddleware` in `server.py`.

---

## Timeout Considerations

LLM inference can be slow, especially on first request when models are loading:

- **First request**: 30-60 seconds (model loading)
- **Subsequent requests**: 45-100 seconds depending on feedback loop iterations
- **Recommended client timeout**: 300 seconds (5 minutes)

---

## See Also

- [Quickstart](../getting-started/quickstart.md) - Getting started
- [Configuration](../configs/configuration.md) - API and model settings
- [Pipeline](../architecture/pipeline.md) - Processing flow
