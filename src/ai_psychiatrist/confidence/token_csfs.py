"""Token-level confidence signal extraction (Spec 051).

These utilities are pure functions that operate on the `logprobs` structure returned
by PydanticAI's OpenAIChatModel (and backed by Ollama when supported).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _require_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")


def _logsumexp(values: np.ndarray) -> float:
    """Stable logsumexp for 1D arrays."""
    if values.size == 0:
        raise ValueError("logsumexp requires at least one value")
    max_val = float(np.max(values))
    return max_val + float(np.log(np.sum(np.exp(values - max_val))))


def compute_token_msp(logprobs: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean maximum softmax probability (MSP) over generated tokens."""
    if not logprobs:
        raise ValueError("logprobs cannot be empty")
    probs = []
    for idx, lp in enumerate(logprobs):
        logprob = _require_float(lp.get("logprob"), field_name=f"logprobs[{idx}].logprob")
        probs.append(math.exp(logprob))
    return float(np.mean(np.array(probs, dtype=float)))


def compute_token_pe(logprobs: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean predictive entropy over generated tokens.

    Returns raw entropy (lower = more confident). Convert to a confidence scalar
    at call sites (e.g., 1/(1+entropy)).
    """
    if not logprobs:
        raise ValueError("logprobs cannot be empty")

    entropies = []
    for idx, lp in enumerate(logprobs):
        top = lp.get("top_logprobs")
        if not isinstance(top, Sequence) or not top:
            raise ValueError(f"logprobs[{idx}].top_logprobs must be a non-empty list")

        logits = np.array(
            [_require_float(t.get("logprob"), field_name="top_logprobs.logprob") for t in top],
            dtype=float,
        )
        probs = np.exp(logits)
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            raise ValueError(f"logprobs[{idx}].top_logprobs probabilities sum to zero")
        probs = probs / probs_sum
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        entropies.append(entropy)

    return float(np.mean(np.array(entropies, dtype=float)))


def compute_token_energy(logprobs: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean token energy over generated tokens.

    Energy is computed as logsumexp over the top token log-probabilities.
    When the inputs are log-probabilities, `exp(energy)` is the cumulative mass
    captured by the `top_logprobs` list.
    """
    if not logprobs:
        raise ValueError("logprobs cannot be empty")

    energies = []
    for idx, lp in enumerate(logprobs):
        top = lp.get("top_logprobs")
        if not isinstance(top, Sequence) or not top:
            raise ValueError(f"logprobs[{idx}].top_logprobs must be a non-empty list")

        logits = np.array(
            [_require_float(t.get("logprob"), field_name="top_logprobs.logprob") for t in top],
            dtype=float,
        )
        energies.append(_logsumexp(logits))

    return float(np.mean(np.array(energies, dtype=float)))
