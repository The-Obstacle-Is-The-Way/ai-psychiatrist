"""Evidence extraction validation helpers.

These helpers enforce a strict contract for evidence JSON produced by LLMs.
They intentionally fail loudly on schema violations to prevent silent corruption
of downstream retrieval and confidence signals.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS
from ai_psychiatrist.infrastructure.hashing import stable_text_hash
from ai_psychiatrist.infrastructure.logging import get_logger

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _rapidfuzz_fuzz = None

logger = get_logger(__name__)

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_SMART_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",  # NBSP
    }
)
_ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")


@dataclass(frozen=True, slots=True)
class EvidenceSchemaError(ValueError):
    """Raised when evidence JSON does not match the expected schema.

    Note: This error MUST NOT include transcript text or quote strings in the
    message/violations, because it may be surfaced in logs.
    """

    message: str
    violations: dict[str, str]

    def __str__(self) -> str:
        return self.message


def validate_evidence_schema(obj: object) -> dict[str, list[str]]:
    """Validate and normalize evidence extraction JSON schema (Spec 054).

    Expected schema (keys may be missing):
        {
            "PHQ8_NoInterest": ["quote1", "quote2", ...],
            ...
        }

    Rules:
    - Top-level MUST be an object (dict).
    - Each present key MUST map to a list of strings (list[str]).
    - Missing keys (or explicit null) are treated as an empty list.
    - Quotes are stripped and empty strings removed.
    - De-duplicate while preserving order.

    Args:
        obj: Parsed JSON value from the LLM.

    Returns:
        Dict with all PHQ8_DOMAIN_KEYS present.

    Raises:
        EvidenceSchemaError: If the top-level is not an object or any present value
            violates list[str].
    """
    if not isinstance(obj, dict):
        raise EvidenceSchemaError(
            "Expected JSON object at top level",
            violations={"__root__": f"expected_object_got_{type(obj).__name__}"},
        )

    violations: dict[str, str] = {}
    validated: dict[str, list[str]] = {}

    for key in PHQ8_DOMAIN_KEYS:
        value = obj.get(key)

        if value is None:
            validated[key] = []
            continue

        if not isinstance(value, list):
            violations[key] = f"expected_list_got_{type(value).__name__}"
            continue

        normalized: list[str] = []
        for index, element in enumerate(value):
            if not isinstance(element, str):
                violations[key] = f"expected_list_str_got_{type(element).__name__}_at_{index}"
                break
            stripped = element.strip()
            if stripped:
                normalized.append(stripped)

        if key in violations:
            continue

        seen: set[str] = set()
        deduped: list[str] = []
        for quote in normalized:
            if quote in seen:
                continue
            seen.add(quote)
            deduped.append(quote)

        validated[key] = deduped

    if violations:
        raise EvidenceSchemaError(
            f"Evidence schema violations in {len(violations)} field(s)",
            violations=violations,
        )

    return validated


def normalize_for_quote_match(text: str) -> str:
    """Normalize text for conservative substring grounding checks (Spec 053)."""
    normalized = unicodedata.normalize("NFKC", text).translate(_SMART_QUOTES)
    for ch in _ZERO_WIDTH:
        normalized = normalized.replace(ch, "")
    normalized = _TAG_RE.sub(" ", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip().lower()
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceGroundingStats:
    extracted_count: int
    validated_count: int
    rejected_count: int
    rejected_by_domain: dict[str, int]


class EvidenceGroundingError(ValueError):
    """Raised when extracted evidence cannot be grounded in the transcript."""


def validate_evidence_grounding(
    evidence: dict[str, list[str]],
    transcript_text: str,
    *,
    mode: str = "substring",
    fuzzy_threshold: float = 0.85,
    log_rejections: bool = True,
) -> tuple[dict[str, list[str]], EvidenceGroundingStats]:
    """Validate extracted evidence quotes against the source transcript.

    A quote is accepted iff it can be grounded in the transcript after conservative
    normalization. Default mode is normalized substring matching.

    Args:
        evidence: Dict mapping PHQ-8 domain keys to lists of quote strings.
        transcript_text: Source transcript text.
        mode: "substring" (default) or "fuzzy" (requires rapidfuzz).
        fuzzy_threshold: Similarity threshold in [0, 1] for fuzzy mode.
        log_rejections: Emit privacy-safe logs for rejected quotes.

    Returns:
        (validated_evidence, stats)
    """
    transcript_norm = normalize_for_quote_match(transcript_text)
    transcript_hash = stable_text_hash(transcript_text)

    validated: dict[str, list[str]] = {}
    rejected_by_domain: dict[str, int] = {}
    extracted_count = 0
    validated_count = 0

    for domain, quotes in evidence.items():
        validated[domain] = []
        rejected_by_domain[domain] = 0

        for quote in quotes:
            extracted_count += 1
            quote_norm = normalize_for_quote_match(quote)
            grounded = bool(quote_norm) and (quote_norm in transcript_norm)

            if not grounded and mode == "fuzzy":
                if _rapidfuzz_fuzz is None:
                    raise RuntimeError("evidence_quote_validation_mode='fuzzy' requires rapidfuzz")

                ratio = _rapidfuzz_fuzz.partial_ratio(quote_norm, transcript_norm) / 100.0
                grounded = ratio >= fuzzy_threshold

            if grounded:
                validated_count += 1
                validated[domain].append(quote)
            else:
                rejected_by_domain[domain] += 1
                if log_rejections:
                    logger.warning(
                        "evidence_quote_rejected",
                        domain=domain,
                        quote_hash=stable_text_hash(quote),
                        quote_len=len(quote),
                        transcript_hash=transcript_hash,
                        transcript_len=len(transcript_text),
                        mode=mode,
                    )

    stats = EvidenceGroundingStats(
        extracted_count=extracted_count,
        validated_count=validated_count,
        rejected_count=(extracted_count - validated_count),
        rejected_by_domain=rejected_by_domain,
    )

    if stats.rejected_count > 0 and log_rejections:
        logger.info(
            "evidence_grounding_complete",
            extracted_count=stats.extracted_count,
            validated_count=stats.validated_count,
            rejected_count=stats.rejected_count,
            rejected_by_domain=stats.rejected_by_domain,
            transcript_hash=transcript_hash,
            transcript_len=len(transcript_text),
            mode=mode,
        )
    return validated, stats
