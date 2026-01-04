"""Unit tests for evidence schema validation and grounding helpers."""

from __future__ import annotations

import pytest

from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS
from ai_psychiatrist.services.evidence_validation import (
    EvidenceSchemaError,
    validate_evidence_grounding,
    validate_evidence_schema,
)


class TestValidateEvidenceSchema:
    def test_valid_schema_passes_and_dedupes_preserving_order(self) -> None:
        obj: dict[str, object] = {key: [] for key in PHQ8_DOMAIN_KEYS}
        obj["PHQ8_NoInterest"] = [" quote 1 ", "quote 2", "quote 1", "quote 2", "quote 3"]

        result = validate_evidence_schema(obj)
        assert result["PHQ8_NoInterest"] == ["quote 1", "quote 2", "quote 3"]

    def test_missing_keys_filled_with_empty(self) -> None:
        result = validate_evidence_schema({"PHQ8_NoInterest": ["quote"]})
        assert result["PHQ8_NoInterest"] == ["quote"]
        assert result["PHQ8_Depressed"] == []

    def test_root_not_object_raises(self) -> None:
        with pytest.raises(EvidenceSchemaError, match="Expected JSON object"):
            validate_evidence_schema(["not", "an", "object"])

    def test_value_not_list_raises(self) -> None:
        with pytest.raises(EvidenceSchemaError) as exc:
            validate_evidence_schema({"PHQ8_NoInterest": "not a list"})
        assert "PHQ8_NoInterest" in exc.value.violations

    def test_list_with_non_string_raises(self) -> None:
        with pytest.raises(EvidenceSchemaError) as exc:
            validate_evidence_schema({"PHQ8_NoInterest": ["ok", 123]})
        assert "PHQ8_NoInterest" in exc.value.violations


class TestValidateEvidenceGrounding:
    def test_exact_match_accepted(self) -> None:
        evidence = {"PHQ8_Sleep": ["I can't sleep at night"]}
        transcript = "Patient said: I can't sleep at night. Very tired."
        validated, stats = validate_evidence_grounding(evidence, transcript)
        assert validated["PHQ8_Sleep"] == ["I can't sleep at night"]
        assert stats.rejected_count == 0

    def test_hallucination_rejected(self) -> None:
        evidence = {"PHQ8_Depressed": ["I feel hopeless and worthless"]}
        transcript = "I've been feeling okay lately. Work is going well."
        validated, stats = validate_evidence_grounding(evidence, transcript)
        assert validated["PHQ8_Depressed"] == []
        assert stats.rejected_by_domain["PHQ8_Depressed"] == 1

    def test_minor_whitespace_difference_accepted(self) -> None:
        evidence = {"PHQ8_Tired": ["I   feel  tired"]}  # Extra spaces
        transcript = "I feel tired all the time"
        validated, _stats = validate_evidence_grounding(evidence, transcript)
        assert validated["PHQ8_Tired"] == ["I   feel  tired"]

    def test_case_insensitive_matching(self) -> None:
        evidence = {"PHQ8_Appetite": ["I HAVE NO APPETITE"]}
        transcript = "I have no appetite lately"
        validated, stats = validate_evidence_grounding(evidence, transcript)
        assert validated["PHQ8_Appetite"] == ["I HAVE NO APPETITE"]
        assert stats.validated_count == 1

    def test_smart_quotes_normalized(self) -> None:
        evidence = {"PHQ8_Failure": ['I feel "worthless"']}
        transcript = "I feel \u201cworthless\u201d when I can't do things."  # smart quotes
        validated, stats = validate_evidence_grounding(evidence, transcript)
        assert validated["PHQ8_Failure"] == ['I feel "worthless"']
        assert stats.validated_count == 1
