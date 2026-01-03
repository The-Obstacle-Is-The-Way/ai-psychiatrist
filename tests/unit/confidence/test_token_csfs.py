"""Unit tests for token-level confidence scoring functions (Spec 051)."""

from __future__ import annotations

import math

import pytest

from ai_psychiatrist.confidence.token_csfs import (
    compute_token_energy,
    compute_token_msp,
    compute_token_pe,
)

pytestmark = pytest.mark.unit


def test_compute_token_msp_mean_probability() -> None:
    # log(0.9) and log(0.6) -> mean = 0.75
    logprobs = [
        {"logprob": math.log(0.9), "top_logprobs": [{"logprob": math.log(0.9)}]},
        {"logprob": math.log(0.6), "top_logprobs": [{"logprob": math.log(0.6)}]},
    ]
    assert compute_token_msp(logprobs) == pytest.approx(0.75)


def test_compute_token_pe_entropy_of_top_distribution() -> None:
    # Top distribution: [0.5, 0.5] => entropy = ln(2)
    logprobs = [
        {
            "logprob": math.log(0.5),
            "top_logprobs": [{"logprob": math.log(0.5)}, {"logprob": math.log(0.5)}],
        }
    ]
    assert compute_token_pe(logprobs) == pytest.approx(math.log(2.0))


def test_compute_token_energy_logsumexp_of_top_logprobs() -> None:
    # logsumexp([log(0.5), log(0.25)]) = log(0.75)
    logprobs = [
        {
            "logprob": math.log(0.5),
            "top_logprobs": [{"logprob": math.log(0.5)}, {"logprob": math.log(0.25)}],
        }
    ]
    assert compute_token_energy(logprobs) == pytest.approx(math.log(0.75))
