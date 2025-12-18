"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest


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
def sample_ollama_response() -> dict:
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
