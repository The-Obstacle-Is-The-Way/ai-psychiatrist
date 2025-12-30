"""Spec 35: Offline chunk-level PHQ-8 scoring prompt + provenance helpers."""

from __future__ import annotations

import hashlib

PHQ8_ITEM_KEYS: tuple[str, ...] = (
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
)

PHQ8_ITEM_KEY_SET = set(PHQ8_ITEM_KEYS)

CHUNK_SCORING_PROMPT_TEMPLATE = """\
You are labeling a single transcript chunk for PHQ-8 item frequency evidence.

Task:
- For each PHQ-8 item key below, output an integer 0-3 if the chunk explicitly supports that
  frequency.
- If the chunk does not mention the symptom or frequency is unclear, output null.
- Do not guess or infer beyond the text.

Keys (must be present exactly):
PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired,
PHQ8_Appetite, PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving

Chunk:
<<<CHUNK_TEXT>>>
{chunk_text}

Return JSON only in this exact shape:
{{
  "PHQ8_NoInterest": 0|1|2|3|null,
  "PHQ8_Depressed": 0|1|2|3|null,
  ...
}}
"""


def chunk_scoring_prompt_hash() -> str:
    """Return a stable short hash for the scorer prompt (Spec 35 protocol lock)."""
    return hashlib.sha256(CHUNK_SCORING_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12]


def render_chunk_scoring_prompt(*, chunk_text: str) -> str:
    """Render the scorer prompt for a single transcript chunk."""
    return CHUNK_SCORING_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
