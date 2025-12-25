"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os

import pytest

# Real Ollama tests are opt-in and must be able to read developer `.env` / env vars.
# Default test runs stay isolated from local configuration to keep unit tests deterministic.
_RUN_REAL_OLLAMA_TESTS = os.environ.get("AI_PSYCHIATRIST_OLLAMA_TESTS") == "1"

# Set TESTING mode BEFORE any app imports to prevent .env file loading (default).
if _RUN_REAL_OLLAMA_TESTS:
    os.environ.pop("TESTING", None)
else:
    os.environ["TESTING"] = "1"

# Clear environment variables BEFORE any imports that might use Pydantic Settings
# This runs at conftest import time, before test collection
_ENV_VARS_TO_CLEAR = [
    "LLM_BACKEND",
    "LLM_HF_CACHE_DIR",
    "LLM_HF_DEVICE",
    "LLM_HF_QUANTIZATION",
    "LLM_HF_TOKEN",
    "OLLAMA_HOST",
    "OLLAMA_PORT",
    "MODEL_EMBEDDING_MODEL",
    "MODEL_JUDGE_MODEL",
    "MODEL_META_REVIEW_MODEL",
    "MODEL_QUALITATIVE_MODEL",
    "MODEL_QUANTITATIVE_MODEL",
    "MODEL_TEMPERATURE",
    "EMBEDDING_BACKEND",
    "EMBEDDING_CHUNK_SIZE",
    "EMBEDDING_CHUNK_STEP",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_EMBEDDINGS_FILE",
    "EMBEDDING_MIN_EVIDENCE_CHARS",
    "EMBEDDING_TOP_K_REFERENCES",
    "FEEDBACK_ENABLED",
    "FEEDBACK_MAX_ITERATIONS",
    "FEEDBACK_SCORE_THRESHOLD",
    "FEEDBACK_TARGET_SCORE",
    "QUANTITATIVE_ENABLE_KEYWORD_BACKFILL",
    "QUANTITATIVE_KEYWORD_BACKFILL_CAP",
    "QUANTITATIVE_TRACK_NA_REASONS",
    "DATA_BASE_DIR",
    "DATA_DEV_CSV",
    "DATA_EMBEDDINGS_PATH",
    "DATA_TRAIN_CSV",
    "DATA_TRANSCRIPTS_DIR",
    "API_HOST",
    "API_PORT",
    "OLLAMA_TIMEOUT_SECONDS",
    "LOG_FORMAT",
    "LOG_LEVEL",
]

# Clear at import time for test isolation from local .env (default).
if not _RUN_REAL_OLLAMA_TESTS:
    for _var in _ENV_VARS_TO_CLEAR:
        os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear environment variables that might be set by .env file.

    This ensures tests use code defaults, not local developer overrides.
    Also clears any cached settings to force re-read of defaults.
    """
    if _RUN_REAL_OLLAMA_TESTS:
        return

    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)

    # Clear the settings cache to ensure fresh reads
    from ai_psychiatrist.config import get_settings  # noqa: PLC0415

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def sample_transcript() -> str:
    """Return a sample interview transcript for testing."""
    return """
    Ellie: How are you doing today?
    Participant: Not great, I've been feeling really down lately.
    Ellie: Can you tell me more about that?
    Participant: I just can't seem to enjoy anything anymore.
    I used to love going out with friends, but now I can't be bothered.
    Ellie: How long has this been going on?
    Participant: A few months now. I'm also not sleeping well.
    I wake up at 3 or 4 in the morning and can't get back to sleep.
    """.strip()


@pytest.fixture(scope="session")
def sample_phq8_scores() -> dict[str, int]:
    """Return sample PHQ-8 ground truth scores."""
    return {
        "PHQ8_NoInterest": 2,
        "PHQ8_Depressed": 2,
        "PHQ8_Sleep": 2,
        "PHQ8_Tired": 1,
        "PHQ8_Appetite": 0,
        "PHQ8_Failure": 1,
        "PHQ8_Concentrating": 0,
        "PHQ8_Moving": 0,
    }


@pytest.fixture
def sample_ollama_response() -> dict[str, object]:
    """Return sample Ollama API response structure for testing.

    NOTE: This is TEST DATA, not a mock. We use real data structures to test
    parsing/validation logic. Only mock external I/O boundaries (HTTP calls),
    never business logic.
    """
    content = (
        '{"PHQ8_NoInterest": '
        '{"evidence": "can\'t be bothered", "reason": "clear anhedonia", "score": 2}}'
    )
    return {
        "model": "alibayram/medgemma:27b",  # Paper-optimal (Appendix F)
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }
