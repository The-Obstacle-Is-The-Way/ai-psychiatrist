# Quickstart Guide

Get AI Psychiatrist running in 5 minutes.

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- 16GB+ RAM (for 27B models)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist.git
cd ai-psychiatrist
```

### 2. Install Dependencies

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
make dev
```

Or manually:

```bash
pip install uv
uv sync --all-extras
uv run pre-commit install
```

### 3. Pull Required Models

```bash
# Primary LLM (Gemma 3 27B) - used by Qualitative, Judge, and Meta-Review agents
ollama pull gemma3:27b

# Quantitative LLM (MedGemma 27B) - 18% better PHQ-8 prediction accuracy
ollama pull alibayram/medgemma:27b

# Embedding model (Qwen 3 8B) - for few-shot retrieval
ollama pull qwen3-embedding:8b
```

> **Note**: These are large models. Ensure you have sufficient disk space (~50GB total).

### 4. Configure Environment

```bash
cp .env.example .env
```

Default configuration uses paper-optimal settings. Edit `.env` to customize.

---

## Verify Installation

### Check Ollama Connection

```bash
curl http://localhost:11434/api/tags
```

Should return a list of installed models.

### Run Tests

```bash
make test-unit  # Fast unit tests
```

All tests should pass.

---

## Start the Server

```bash
make serve
```

The API will be available at `http://localhost:8000`.

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Your First Assessment

### Using the API

```bash
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_text": "Ellie: How are you doing today?\nParticipant: I have been feeling really down lately.\nEllie: Can you tell me more about that?\nParticipant: I just do not have energy for anything. I used to love hiking but now I cannot even get out of bed some days."
  }'
```

### Expected Response

```json
{
  "participant_id": 999999,
  "mode": "few_shot",
  "quantitative": {
    "total_score": 12,
    "severity": "MODERATE",
    "na_count": 4,
    "items": {
      "NoInterest": {"score": 2, "evidence": "I used to love hiking but now I cannot even get out of bed", "reason": "Clear anhedonia"},
      "Depressed": {"score": 2, "evidence": "I have been feeling really down lately", "reason": "Direct statement"},
      "Tired": {"score": 3, "evidence": "I just do not have energy for anything", "reason": "Severe fatigue"}
    }
  },
  "qualitative": {
    "overall": "The participant shows clear signs of depression including anhedonia and fatigue...",
    "phq8_symptoms": "Reports loss of interest, low mood, and severe fatigue..."
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
    "explanation": "Based on PHQ-8 scores and qualitative assessment...",
    "is_mdd": true
  }
}
```

> **Note:** When using `transcript_text`, the system assigns a placeholder `participant_id` of 999999.

---

## Using the DAIC-WOZ Dataset

If you have access to the DAIC-WOZ dataset:

### 1. Prepare the Data

```bash
python scripts/prepare_dataset.py --input /path/to/daic-woz --output data/
```

This extracts transcripts and ground truth files.

### 2. Generate Embeddings (Optional, for Few-Shot Mode)

```bash
python scripts/generate_embeddings.py
```

Creates `data/embeddings/reference_embeddings.npz`.

### 3. Assess a Participant

```bash
curl -X POST http://localhost:8000/full_pipeline \
  -H "Content-Type: application/json" \
  -d '{"participant_id": 300}'
```

Uses the transcript from `data/transcripts/300_P/300_TRANSCRIPT.csv`.

---

## Common Issues

### Ollama Connection Failed

```
Error: Connection refused on localhost:11434
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

### Out of Memory

```
Error: CUDA out of memory
```

**Solution**: Use smaller quantized models:
```bash
# In .env
MODEL_QUALITATIVE_MODEL=gemma3:27b-q4_K_M
MODEL_QUANTITATIVE_MODEL=gemma3:27b-q4_K_M
```

### Slow Inference

The first request may take 30-60 seconds as models load into memory. Subsequent requests are faster.

---

## Next Steps

- [Architecture](../concepts/architecture.md) - Understand system design
- [Pipeline](../concepts/pipeline.md) - Learn how agents collaborate
- [Configuration](../reference/configuration.md) - Customize settings
