"""Stable hashing helpers for privacy-safe observability.

We use short SHA-256 prefixes in logs and JSON artifacts to correlate failures
without leaking transcript text or full LLM outputs.
"""

from __future__ import annotations

import hashlib
from typing import Final

HASH_PREFIX_LENGTH: Final[int] = 12


def stable_text_hash(text: str) -> str:
    """Return a stable short hash for a text payload (no raw text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_PREFIX_LENGTH]


def stable_bytes_hash(payload: bytes) -> str:
    """Return a stable short hash for a bytes payload (no raw bytes)."""
    return hashlib.sha256(payload).hexdigest()[:HASH_PREFIX_LENGTH]
