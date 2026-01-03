"""Confidence scoring functions (CSFs) for selective prediction.

This package centralizes confidence signal computation used by
`scripts/evaluate_selective_prediction.py` and related tooling.
"""

from ai_psychiatrist.confidence.csf_registry import (
    CSFRegistry,
    get_csf,
    parse_csf_variant,
    register_csf,
)

__all__ = [
    "CSFRegistry",
    "get_csf",
    "parse_csf_variant",
    "register_csf",
]
