"""Domain constants and clinical criteria configuration.

Centralizes domain-specific knowledge and magic strings used across agents.
Moved from ai_psychiatrist.agents.prompts.quantitative to improve maintainability.
"""

from __future__ import annotations

# Domain keywords for keyword backfill in quantitative assessment.
# Used to catch evidence when LLM extraction misses relevant sentences.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "PHQ8_NoInterest": [
        "can't be bothered",
        "no interest",
        "nothing really",
        "not enjoy",
        "no pleasure",
        "what's the point",
        "cant be bothered",
    ],
    "PHQ8_Depressed": [
        "fed up",
        "miserable",
        "depressed",
        "very black",
        "hopeless",
        "low",
    ],
    "PHQ8_Sleep": [
        "sleep",
        "fall asleep",
        "wake up",
        "insomnia",
        "clock",
        "tired in the morning",
    ],
    "PHQ8_Tired": [
        "exhausted",
        "tired",
        "little energy",
        "fatigue",
        "no energy",
    ],
    "PHQ8_Appetite": [
        "appetite",
        "weight",
        "lost weight",
        "eat",
        "eating",
        "don't bother",
        "dont bother",
        "looser",
    ],
    "PHQ8_Failure": [
        "useless",
        "failure",
        "bad about myself",
        "burden",
    ],
    "PHQ8_Concentrating": [
        "concentrat",
        "memory",
        "forgot",
        "thinking of something else",
        "focus",
    ],
    "PHQ8_Moving": [
        "moving slowly",
        "restless",
        "fidget",
        "speaking slowly",
        "psychomotor",
    ],
}
