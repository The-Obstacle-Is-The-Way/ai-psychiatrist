"""Registry for confidence scoring functions (CSFs).

Spec 051 ports a small subset of the fd-shifts "confid_scores" pattern to support
portable confidence signals and simple signal combinations.

Key design choice: CSFs return a scalar where *higher means more confident*.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

CsfFunc: TypeAlias = Callable[[Mapping[str, Any]], float]


def _require_key(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required item_signals['{key}']")
    return mapping[key]


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = _require_key(mapping, key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"item_signals['{key}'] must be a number or null, got {type(value).__name__}")


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = _require_key(mapping, key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"item_signals['{key}'] must be an int or null, got bool")
    if isinstance(value, float) and value == int(value):
        value = int(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"item_signals['{key}'] must be an int or null, got {type(value).__name__}")


@dataclass(slots=True)
class CSFRegistry:
    """In-memory registry for CSFs."""

    funcs: dict[str, CsfFunc] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[CsfFunc], CsfFunc]:
        """Decorator to register a CSF under `name`."""

        def wrapper(func: CsfFunc) -> CsfFunc:
            if name in self.funcs:
                raise ValueError(f"CSF '{name}' is already registered")
            self.funcs[name] = func
            return func

        return wrapper

    def get(self, name: str) -> CsfFunc:
        """Get a CSF by name."""
        try:
            return self.funcs[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.funcs))
            raise ValueError(f"Unknown CSF: {name}. Available: [{available}]") from exc

    def create_secondary(self, csf1: str, csf2: str, *, combine: str) -> CsfFunc:
        """Create a secondary CSF combining two base CSFs."""
        func1 = self.get(csf1)
        func2 = self.get(csf2)

        if combine == "average":
            return lambda sig: (func1(sig) + func2(sig)) / 2.0
        if combine == "product":
            return lambda sig: func1(sig) * func2(sig)

        raise ValueError(f"Unknown combine method: {combine}")

    def parse_variant(self, variant: str) -> CsfFunc:
        """Parse a confidence variant into a CSF callable.

        Supported syntax:
        - `<csf_name>`: base CSF
        - `secondary:<csf1>+<csf2>:<average|product>`: secondary combination
        """
        if variant.startswith("secondary:"):
            rest = variant[len("secondary:") :]
            try:
                csfs_part, combine = rest.rsplit(":", 1)
                csf1, csf2 = csfs_part.split("+", 1)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "Invalid secondary CSF variant format; expected "
                    "'secondary:<csf1>+<csf2>:<average|product>'"
                ) from exc
            return self.create_secondary(csf1, csf2, combine=combine)

        return self.get(variant)


DEFAULT_REGISTRY = CSFRegistry()


def register_csf(name: str) -> Callable[[CsfFunc], CsfFunc]:
    return DEFAULT_REGISTRY.register(name)


def get_csf(name: str) -> CsfFunc:
    return DEFAULT_REGISTRY.get(name)


def parse_csf_variant(variant: str) -> CsfFunc:
    return DEFAULT_REGISTRY.parse_variant(variant)


@register_csf("llm")
@register_csf("total_evidence")
def csf_llm(item_signals: Mapping[str, Any]) -> float:
    return float(item_signals.get("llm_evidence_count", 0))


@register_csf("retrieval_similarity_mean")
def csf_retrieval_similarity_mean(item_signals: Mapping[str, Any]) -> float:
    s = _optional_float(item_signals, "retrieval_similarity_mean")
    return float(s) if s is not None else 0.0


@register_csf("retrieval_similarity_max")
def csf_retrieval_similarity_max(item_signals: Mapping[str, Any]) -> float:
    s = _optional_float(item_signals, "retrieval_similarity_max")
    return float(s) if s is not None else 0.0


@register_csf("hybrid_evidence_similarity")
def csf_hybrid_evidence_similarity(item_signals: Mapping[str, Any]) -> float:
    llm_count = float(item_signals.get("llm_evidence_count", 0))
    e = min(llm_count, 3.0) / 3.0
    e = max(0.0, min(1.0, e))

    s = _optional_float(item_signals, "retrieval_similarity_mean")
    s_val = float(s) if s is not None else 0.0
    s_val = max(0.0, min(1.0, s_val))

    return 0.5 * e + 0.5 * s_val


def _normalize_verbalized_confidence(v: int | None) -> float:
    if v is None:
        return 0.5
    # 1-5 -> [0, 1]
    normalized = (float(v) - 1.0) / 4.0
    return max(0.0, min(1.0, normalized))


@register_csf("verbalized")
def csf_verbalized(item_signals: Mapping[str, Any]) -> float:
    v_int = _optional_int(item_signals, "verbalized_confidence")
    return _normalize_verbalized_confidence(v_int)


@register_csf("hybrid_verbalized")
def csf_hybrid_verbalized(item_signals: Mapping[str, Any]) -> float:
    v_int = _optional_int(item_signals, "verbalized_confidence")
    v = _normalize_verbalized_confidence(v_int)

    llm_count = float(item_signals.get("llm_evidence_count", 0))
    e = min(llm_count, 3.0) / 3.0
    e = max(0.0, min(1.0, e))

    s = _optional_float(item_signals, "retrieval_similarity_mean")
    s_val = float(s) if s is not None else 0.0
    s_val = max(0.0, min(1.0, s_val))

    conf = 0.4 * v + 0.3 * e + 0.3 * s_val
    return max(0.0, min(1.0, conf))


@register_csf("token_msp")
def csf_token_msp(item_signals: Mapping[str, Any]) -> float:
    v = _optional_float(item_signals, "token_msp")
    if v is None:
        raise ValueError("token_msp is required for CSF 'token_msp'")
    return max(0.0, min(1.0, float(v)))


@register_csf("token_pe")
def csf_token_pe(item_signals: Mapping[str, Any]) -> float:
    # Stored value is entropy (lower is more confident). Map to a confidence scalar.
    entropy = _optional_float(item_signals, "token_pe")
    if entropy is None:
        raise ValueError("token_pe is required for CSF 'token_pe'")
    return 1.0 / (1.0 + float(entropy))


@register_csf("token_energy")
def csf_token_energy(item_signals: Mapping[str, Any]) -> float:
    # Stored value is logsumexp(logprobs). If logprobs are log-probabilities, exp(energy) ∈ (0, 1].
    energy = _optional_float(item_signals, "token_energy")
    if energy is None:
        raise ValueError("token_energy is required for CSF 'token_energy'")
    conf = math.exp(float(energy))
    return max(0.0, min(1.0, conf))


@register_csf("consistency")
def csf_consistency(item_signals: Mapping[str, Any]) -> float:
    c = _optional_float(item_signals, "consistency_modal_confidence")
    if c is None:
        raise ValueError("consistency_modal_confidence is required for CSF 'consistency'")
    return max(0.0, min(1.0, float(c)))


@register_csf("consistency_inverse_std")
def csf_consistency_inverse_std(item_signals: Mapping[str, Any]) -> float:
    std = _optional_float(item_signals, "consistency_score_std")
    if std is None:
        raise ValueError("consistency_score_std is required for CSF 'consistency_inverse_std'")
    return 1.0 / (1.0 + float(std))


@register_csf("hybrid_consistency")
def csf_hybrid_consistency(item_signals: Mapping[str, Any]) -> float:
    c = _optional_float(item_signals, "consistency_modal_confidence")
    if c is None:
        raise ValueError("consistency_modal_confidence is required for CSF 'hybrid_consistency'")
    c_val = max(0.0, min(1.0, float(c)))

    llm_count = float(item_signals.get("llm_evidence_count", 0))
    e = min(llm_count, 3.0) / 3.0
    e = max(0.0, min(1.0, e))

    s = _optional_float(item_signals, "retrieval_similarity_mean")
    s_val = float(s) if s is not None else 0.0
    s_val = max(0.0, min(1.0, s_val))

    conf = 0.4 * c_val + 0.3 * e + 0.3 * s_val
    return max(0.0, min(1.0, conf))
